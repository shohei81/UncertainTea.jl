# Release-grade simulation-based calibration run (issue #18). Not part of the
# CI suite -- run manually when validating a new sampler, adaptation change,
# or distribution family:
#
#   julia --project=. bench/sbc_validation.jl
#
# Runtime is dominated by num_simulations * (num_warmup + num_draws * thin)
# NUTS iterations per model; the defaults below take a few minutes.

using UncertainTea
using UncertainTea.Inference
using UncertainTea.Diagnostics
using Random
using KernelAbstractions: CPU

@tea static function sbc_bench_conjugate()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

@tea static function sbc_bench_hierarchical()
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    for i = 1:4
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea static function sbc_bench_positive()
    rate ~ gamma(2.0, 2.0)
    for i = 1:3
        {:y => i} ~ exponential(rate)
    end
    return rate
end

# issue #226: release-grade calibration of the constrained-transform Jacobian
# layer (bounded / simplex / correlation), the class where recent confirmed
# bugs lived (#105 Logit, #99/#104 LKJ tanh, #101 Dirichlet).

@tea static function sbc_bench_bounded(n)
    p ~ beta(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ bernoulli(p)
    end
    return p
end

@tea static function sbc_bench_simplex(n)
    theta ~ dirichlet([2.0, 3.0, 4.0])
    for i = 1:n
        {:c => i} ~ categorical(theta)
    end
    return theta
end

const SBC_BENCH_ZEROS2 = [0.0, 0.0]
const SBC_BENCH_ONES2 = [1.0, 1.0]
@tea static function sbc_bench_correlation(zeros2_arg, ones2_arg, n)
    Omega ~ lkjcholesky(2, 2.0)
    Ltril = scale_cholesky(ones2_arg, Omega)
    for i = 1:n
        {:y => i} ~ mvnormaldense(zeros2_arg, Ltril)
    end
    return Omega
end

const SBC_BENCH_MODELS = [
    ("conjugate normal-normal", sbc_bench_conjugate, ()),
    ("hierarchical normal", sbc_bench_hierarchical, ()),
    ("gamma-exponential", sbc_bench_positive, ()),
    ("bounded beta (Logit)", sbc_bench_bounded, (8,)),
    ("simplex dirichlet (stick-breaking)", sbc_bench_simplex, (12,)),
    ("correlation lkjcholesky (Cholesky tanh)", sbc_bench_correlation,
        (SBC_BENCH_ZEROS2, SBC_BENCH_ONES2, 16)),
]

rng = MersenneTwister(20260709)
for (name, model, args) in SBC_BENCH_MODELS
    result = sbc(
        model,
        args;
        num_simulations=300,
        num_samples=63,
        num_warmup=200,
        thin=2,
        rng=rng,
    )
    println("== ", name)
    show(stdout, MIME"text/plain"(), result)
    println()
end

# issue #225: release-grade SBC of the device tree kernels, which are only
# statistically (not bitwise) equivalent to the host path and get no other
# rank-calibration gate. On CPU() at Float64 here; where a functional Metal GPU
# exists, re-run with `backend=Metal.MetalBackend(), precision=Float32` for the
# on-device SBC leg (the #154 "on-device SBC" follow-up).
for strategy in (:masked, :persistent)
    result = sbc(
        sbc_bench_conjugate;
        num_simulations=300,
        num_samples=63,
        num_warmup=200,
        thin=2,
        sampler=:batched_nuts,
        tree_strategy=strategy,
        backend=CPU(),
        precision=Float64,
        num_chains=4,
        rng=rng,
    )
    println("== batched_nuts tree_strategy=", strategy, " (CPU / Float64)")
    show(stdout, MIME"text/plain"(), result)
    println()
end

# issue #222: batched execution -- the ENTIRE study is one sampler run, all
# replications as chains (num_chains = num_simulations). The device leg makes the
# whole SBC study a single device dispatch; where a functional Metal GPU exists,
# re-run with `backend=Metal.MetalBackend(), precision=Float32`.
for (label, kwargs) in (
    ("host NUTS", (;)),
    ("device masked (CPU/Float64)", (; tree_strategy=:masked, backend=CPU(), precision=Float64)),
    ("device persistent (CPU/Float64)", (; tree_strategy=:persistent, backend=CPU(), precision=Float64)),
)
    result = sbc(
        sbc_bench_conjugate;
        num_simulations=300,
        num_samples=63,
        num_warmup=200,
        thin=2,
        execution=:batched,
        rng=rng,
        kwargs...,
    )
    println("== execution=:batched, ", label)
    show(stdout, MIME"text/plain"(), result)
    println()
end

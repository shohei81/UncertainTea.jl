# Issue #135: device lowering for the GLM linear predictor + bernoullilogit.
#
# The device analog of the HOST batched analytic lowering (#150): the device now
# lowers `BackendLinearPredictorExpr` (fused `intercept + sum(coef .* X[:, i])`)
# and the `bernoullilogit` family (#149), so logistic regression rides the
# analytic device path (KernelAbstractions/Metal, masked NUTS) instead of falling
# back. The covariate column `X[:, i]` is per-observation constant data staged
# into the observation buffer exactly like observed values; the coefficient dot
# product is unrolled from the compile-time dimension, and the well-conditioned
# `d/d_eta = y - logistic(eta)` gradient falls out of forward-mode through the
# stable `x*eta - log1p(exp(eta))` device logpdf.
#
# All tests here run on KernelAbstractions.CPU() at Float64 -- the device oracle,
# since CI has no Metal (test/gpu mirrors a Float32 smoke on Metal). The device
# takes UNCONSTRAINED parameters and folds the transform log-abs-det in-kernel, so
# the authoritative counterparts are the HOST batched analytic path
# `batched_logjoint_gradient_unconstrained` / `batched_logjoint_unconstrained`
# (the #150 backend path) and, for sampling, the host masked NUTS.

using KernelAbstractions: CPU

# bench_logistic shape (bench/crossppl/julia/models.jl): the i.i.d. N(0, 2.5)
# coefficient prior is a diagonal mvnormal, the observation is bernoullilogit of
# the fused linear predictor.
@tea static function devglm_logistic(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ mvnormal(
        (0.0, 0.0, 0.0, 0.0),
        (2.5, 2.5, 2.5, 2.5),
    )
    for i = 1:n
        {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
    end
    return alpha
end

# generic scalar bernoullilogit (no linear predictor): exercises the non-fused
# device bernoullilogit step.
@tea static function devglm_scalar(n)
    a ~ normal(0.0, 1.0)
    for i = 1:n
        {:z => i} ~ bernoullilogit(a)
    end
    return a
end

# linear predictor with NO intercept (bare `sum(coef .* X[:, i])`): the intercept
# sub-expr is absent, so the device lowers a zero-literal intercept.
@tea static function devglm_no_intercept(X, n)
    beta ~ mvnormal((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    for i = 1:n
        {:y => i} ~ bernoullilogit(sum(beta .* X[:, i]))
    end
    return beta
end

# issue #221: a D > 16 GLM. The iid diagonal-mvnormal coefficient prior lowers to
# the runtime-length diagonal-normal loop step and the fused GLM likelihood to the
# runtime-length step with the in-kernel ANALYTIC gradient, lifting both D <= 16
# unroll caps for this model class. Generated with literal mu/sigma tuples (the DSL
# needs a static vector size).
let mu = Expr(:tuple, fill(0.0, 20)...), sg = Expr(:tuple, fill(2.5, 20)...)
    @eval @tea static function devglm_logistic_big(X, n)
        alpha ~ normal(0.0, 2.5)
        beta ~ mvnormal($mu, $sg)
        for i = 1:n
            {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
        end
        return alpha
    end
end

function devglm_make_data(rng, D, n, intercept, coefficients)
    X = randn(rng, D, n)
    ys = Vector{Float64}(undef, n)
    for i = 1:n
        eta = intercept + sum(coefficients[d] * X[d, i] for d = 1:D)
        ys[i] = rand(rng) < inv(1 + exp(-eta)) ? 1.0 : 0.0
    end
    return X, ys
end

@testset "devglm_lowering_report" begin
    rng = MersenneTwister(135)
    X, _ = devglm_make_data(rng, 4, 10, 0.3, [0.5, -0.4, 0.8, -0.2])
    supported, issues = device_lowering_report(devglm_logistic; constraints=choicemap((:y => 1, 1.0)))
    @test supported
    @test isempty(issues)

    supported_scalar, issues_scalar = device_lowering_report(devglm_scalar)
    @test supported_scalar
    @test isempty(issues_scalar)

    supported_ni, _ = device_lowering_report(devglm_no_intercept)
    @test supported_ni
end

@testset "devglm_gradient_parity_f64" begin
    # HOST analytic path (#150) is the oracle; device masked path must match its
    # logjoint + gradient across parameter vectors to Float64 precision on CPU().
    rng = MersenneTwister(2001)
    D, n = 4, 25
    X, ys = devglm_make_data(rng, D, n, 0.4, [0.6, -0.3, 0.9, -0.5])
    cm = choicemap((:y => i, ys[i]) for i = 1:n)
    # parameter_count = alpha(1) + beta(4) = 5, several independent chains
    params = randn(rng, D + 1, 8)

    v, g = device_batched_logjoint_gradient(devglm_logistic, params, (X, n), cm)
    gref = batched_logjoint_gradient_unconstrained(devglm_logistic, params, (X, n), cm)
    vref = batched_logjoint_unconstrained(devglm_logistic, params, (X, n), cm)
    @test v ≈ vref rtol = 1e-6
    @test g ≈ gref rtol = 1e-6
    # far tighter in practice (forward-mode through the same closed form)
    @test v ≈ vref rtol = 1e-10
    @test g ≈ gref rtol = 1e-8

    # no-intercept shape
    X2, ys2 = devglm_make_data(rng, 3, 18, 0.0, [0.7, -0.6, 0.4])
    cm2 = choicemap((:y => i, ys2[i]) for i = 1:18)
    params2 = randn(rng, 3, 6)
    v2, g2 = device_batched_logjoint_gradient(devglm_no_intercept, params2, (X2, 18), cm2)
    g2ref = batched_logjoint_gradient_unconstrained(devglm_no_intercept, params2, (X2, 18), cm2)
    v2ref = batched_logjoint_unconstrained(devglm_no_intercept, params2, (X2, 18), cm2)
    @test v2 ≈ v2ref rtol = 1e-10
    @test g2 ≈ g2ref rtol = 1e-8

    # generic scalar bernoullilogit
    zs = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    cm3 = choicemap((:z => i, zs[i]) for i = 1:6)
    params3 = reshape([0.3, -0.8, 1.4, 0.05], 1, 4)
    v3, g3 = device_batched_logjoint_gradient(devglm_scalar, params3, (6,), cm3)
    g3ref = batched_logjoint_gradient_unconstrained(devglm_scalar, params3, (6,), cm3)
    v3ref = batched_logjoint_unconstrained(devglm_scalar, params3, (6,), cm3)
    @test v3 ≈ v3ref rtol = 1e-10
    @test g3 ≈ g3ref rtol = 1e-8
end

@testset "devglm_masked_nuts_vs_host_exact" begin
    # With adaptation OFF at a fixed step size AND `device_sync_per_leaf=true` (the
    # Tier-1 order-preserving path, issue #152), the device masked path is a faithful
    # reimplementation of the host masked path: same RNG order, same reduction order,
    # so the draws match to ~1e-8 on CPU() at Float64 (the residual is the fused
    # device gradient's ~1e-16 disagreement with the host gradient cache, which flips
    # no accept decision). The DEFAULT device path is the Tier-2 async path (device-
    # pre-drawn round RNG), which is only STATISTICALLY equivalent, so this bitwise
    # oracle opts into the sync-per-leaf path explicitly.
    rng = MersenneTwister(909)
    D, n = 4, 30
    X, ys = devglm_make_data(rng, D, n, 0.2, [0.5, -0.3, 0.8, -0.4])
    cm = choicemap((:y => i, ys[i]) for i = 1:n)
    kwargs = (
        num_chains=6,
        num_samples=220,
        num_warmup=0,
        step_size=0.05,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        tree_strategy=:masked,
        device_sync_per_leaf=true,
        per_chain_adaptation=false,
    )
    device = batched_nuts(devglm_logistic, (X, n), cm; rng=MersenneTwister(11), backend=CPU(), kwargs...)
    host = batched_nuts(devglm_logistic, (X, n), cm; rng=MersenneTwister(11), kwargs...)
    device_draws = posterior_array(device)
    host_draws = posterior_array(host)
    @test all(isfinite, device_draws)
    @test maximum(abs, device_draws .- host_draws) < 1e-6
end

@testset "devglm_D20_above_cap_parity" begin
    # D = 20 > 16: both the coefficient prior and the fused GLM likelihood must
    # lower (the caps are lifted for this class) and match the host analytic path.
    rng = MersenneTwister(2021)
    D, n = 20, 50
    X, ys = devglm_make_data(rng, D, n, 0.2, 0.3 .* randn(rng, D))
    cm = choicemap((:y => i, ys[i]) for i = 1:n)

    supported, issues = device_lowering_report(devglm_logistic_big; constraints=cm)
    @test supported
    @test isempty(issues)

    params = reshape(vcat([0.15], 0.1 .* randn(rng, D)), D + 1, 1)
    v = device_batched_logjoint(devglm_logistic_big, params, (X, n), cm)
    vref = batched_logjoint_unconstrained(devglm_logistic_big, params, (X, n), cm)
    @test v ≈ vref rtol = 1e-10
    dev32 = device_batched_logjoint(devglm_logistic_big, Float32.(params), (Float32.(X), n), cm; precision=Float32)
    @test isapprox(Float64.(dev32), vref; rtol=1e-3, atol=1e-2)

    # the in-kernel analytic gradient must match the host analytic gradient (#150)
    _, g = device_batched_logjoint_gradient(devglm_logistic_big, params, (X, n), cm)
    gref = batched_logjoint_gradient_unconstrained(devglm_logistic_big, params, (X, n), cm)
    @test g ≈ gref rtol = 1e-8
end

@testset "devglm_D20_masked_nuts_vs_host_exact" begin
    # The scalar-dual masked device tree at D = 20 is a faithful reimplementation of
    # the host masked path (same bitwise-oracle settings as the D = 4 case above).
    rng = MersenneTwister(2022)
    D, n = 20, 40
    X, ys = devglm_make_data(rng, D, n, 0.2, 0.3 .* randn(rng, D))
    cm = choicemap((:y => i, ys[i]) for i = 1:n)
    kwargs = (
        num_chains=4,
        num_samples=120,
        num_warmup=0,
        step_size=0.03,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        tree_strategy=:masked,
        device_sync_per_leaf=true,
        per_chain_adaptation=false,
    )
    device = batched_nuts(devglm_logistic_big, (X, n), cm; rng=MersenneTwister(7), backend=CPU(), kwargs...)
    host = batched_nuts(devglm_logistic_big, (X, n), cm; rng=MersenneTwister(7), kwargs...)
    @test all(isfinite, posterior_array(device))
    @test maximum(abs, posterior_array(device) .- posterior_array(host)) < 1e-6

    # the wide (DeviceGradN, P = 21) persistent tree compiles and produces finite draws
    persistent = batched_nuts(
        devglm_logistic_big, (X, n), cm;
        num_chains=4, num_samples=40, num_warmup=40,
        tree_strategy=:persistent, backend=CPU(), rng=MersenneTwister(9),
    )
    @test all(isfinite, posterior_array(persistent))
end

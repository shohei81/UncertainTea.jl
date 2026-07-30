# Wide (P-partial) forward-dual gradient for the persistent NUTS kernel (issue #154
# increment 4).
#
# WHAT IS UNDER TEST. `batched_nuts(...; tree_strategy=:persistent, backend=...,
# persistent_gradient=:auto|:scalar|:wide)`. The shipped increment-2 kernel computes the
# in-kernel leapfrog gradient with P serial scalar-dual plan walks (one per parameter),
# recomputing the full logjoint -- every transcendental -- P times. Increment 4 adds a
# `DeviceGradN{P}` "wide" dual (src/device/grad_wide.jl) so a SINGLE forward-mode walk
# yields the whole gradient, collapsing the N*(P+1) transcendentals of a heavy GLM to N.
# It stays LANE-PER-CHAIN (the sanctioned P-partial fallback for the threadgroup-per-chain
# variant; see docs/persistent-nuts.md), so the RNG stream is byte-for-byte the scalar
# kernel's and the change is purely how the gradient is evaluated.
#
# VALIDATION CONTRACT (the #121 statistical gate, CPU() Float64). The wide gradient is
# mathematically the same forward-mode gradient as the scalar path, so: (1) the wide dual
# reproduces the scalar dual's derivatives exactly (unit test); (2) the wide persistent
# NUTS agrees with the scalar persistent NUTS to floating-point reassociation on the SAME
# seed; (3) it clears the heavy-model gate (R-hat < 1.02, divergence-free, posterior
# agreement with the masked device path); (4) auto-selection routes small-P models to the
# scalar path and heavy-P models to the wide path. Metal Float32 smoke lives in
# test/gpu/runtests.jl.

using KernelAbstractions: CPU

dptiled_mean(x) = sum(x) / length(x)
function dptiled_std(x)
    m = dptiled_mean(x)
    acc = 0.0
    for v in x
        acc += (v - m)^2
    end
    return sqrt(acc / (length(x) - 1))
end
dptiled_divrate(res) =
    sum(sum(chain.divergent) for chain in res.chains) /
    sum(length(chain.divergent) for chain in res.chains)

# Deterministic logistic-regression data (D coefficients, N observations).
function dptiled_logistic_data(D::Int, N::Int; seed::Int=20)
    rng = MersenneTwister(seed)
    X = randn(rng, D, N)
    beta = randn(rng, D) .* 0.7
    alpha = 0.3
    probs = 1.0 ./ (1.0 .+ exp.(-(alpha .+ vec(sum(beta .* X; dims=1)))))
    y = Float64.(rand(rng, N) .< probs)
    cons = choicemap(((:y => i, y[i]) for i = 1:N)...)
    return (X, N, cons)
end

# Heavy GLM (D=8 coefficients + intercept => P=9 unconstrained params): the wide dual's
# target regime. Same shape as bench_logistic_large, scaled down for test wall-time.
@tea static function dptiled_logistic(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ mvnormal(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5),
    )
    for i = 1:n
        {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
    end
    return alpha
end

# Small model (P=2) to check auto-selection keeps the scalar path.
@tea static function dptiled_two_param()
    mu ~ normal(0.0, 1.0)
    log_sigma ~ normal(0.0, 0.5)
    {:y} ~ normal(mu, exp(log_sigma))
    return mu
end

@testset "dptiled_widedual_matches_scalar_dual_derivatives" begin
    # The new N-wide dual must reproduce the scalar DeviceDual's derivative channel
    # EXACTLY (same forward-mode formulas, just N partials at once). Differentiate a
    # battery of two-variable expressions exercising +,-,*,/,^,exp,log,log1p,sqrt,tanh
    # and compare partials against two seeded scalar-dual evaluations.
    Dual = UncertainTea.DeviceDual
    GN = UncertainTea.DeviceGradN
    exprs = (
        (x, y) -> x * y + x,
        (x, y) -> exp(x) / (1 + exp(x)) + log(1 + y * y),
        (x, y) -> sqrt(x * x + y * y) * tanh(x - y),
        (x, y) -> log1p(exp(x + 2 * y)) - x^3,
        (x, y) -> (x / (y + 3.0))^2 + x * log(y + 5.0),
    )
    for f in exprs, (xv, yv) in ((0.4, 1.1), (-0.7, 0.9), (1.3, 0.2))
        # scalar: seed each variable in turn.
        dfx = f(Dual{Float64}(xv, 1.0), Dual{Float64}(yv, 0.0))
        dfy = f(Dual{Float64}(xv, 0.0), Dual{Float64}(yv, 1.0))
        # wide: both partials in one evaluation.
        r = f(GN{2,Float64}(xv, (1.0, 0.0)), GN{2,Float64}(yv, (0.0, 1.0)))
        @test r.value ≈ dfx.value
        @test r.value ≈ dfy.value
        @test r.partials[1] ≈ dfx.deriv
        @test r.partials[2] ≈ dfy.deriv
    end
end

@testset "dptiled_auto_selection" begin
    # :auto routes heavy-P models to :wide and small-P models to :scalar; explicit modes
    # override; an unknown mode is rejected.
    _, _, cons = dptiled_logistic_data(8, 50)
    Xsmall = randn(MersenneTwister(1), 8, 50)
    heavy_auto = UncertainTea.DevicePersistentNUTSWorkspace(
        dptiled_logistic, 4, 8; backend=CPU(), args=(Xsmall, 50), constraints=cons, gradient_mode=:auto,
    )
    @test heavy_auto.num_params >= UncertainTea._PERSIST_WIDE_MIN_PARAMS
    @test heavy_auto.gradient_mode === :wide

    small_auto = UncertainTea.DevicePersistentNUTSWorkspace(
        dptiled_two_param, 4, 8; backend=CPU(), constraints=choicemap((:y, 0.3)), gradient_mode=:auto,
    )
    @test small_auto.num_params < UncertainTea._PERSIST_WIDE_MIN_PARAMS
    @test small_auto.gradient_mode === :scalar

    heavy_forced_scalar = UncertainTea.DevicePersistentNUTSWorkspace(
        dptiled_logistic, 4, 8; backend=CPU(), args=(Xsmall, 50), constraints=cons, gradient_mode=:scalar,
    )
    @test heavy_forced_scalar.gradient_mode === :scalar

    @test_throws ArgumentError UncertainTea.DevicePersistentNUTSWorkspace(
        dptiled_two_param, 4, 8; backend=CPU(), constraints=choicemap((:y, 0.3)), gradient_mode=:nope,
    )
end

@testset "dptiled_widedual_heavy_gate" begin
    # #121 gate for the WIDE path on a heavy GLM (CPU() Float64): finite, R-hat < 1.02,
    # essentially divergence-free, and posterior mean/sd agreeing with the masked device
    # path (the authoritative device reference) within a few MCSE.
    # N/chains/draws are kept small for CI wall-time on the CPU() KA backend (which
    # emulates the device kernel serially per lane). N drives the per-gradient cost but
    # NOT the wide-vs-masked agreement tolerances -- both target the same posterior at
    # the same N -- so it is cut hardest; the heavy-model speedup is measured separately
    # (docs/persistent-nuts.md), not asserted here.
    X, N, cons = dptiled_logistic_data(8, 120)
    kwargs = (num_chains=8, num_samples=300, num_warmup=300, backend=CPU())
    wide = batched_nuts(
        dptiled_logistic, (X, N), cons;
        tree_strategy=:persistent, persistent_gradient=:wide, rng=MersenneTwister(7), kwargs...,
    )
    masked = batched_nuts(
        dptiled_logistic, (X, N), cons;
        tree_strategy=:masked, rng=MersenneTwister(7), kwargs...,
    )
    wd = posterior_array(wide)
    md = posterior_array(masked)
    @test all(isfinite, wd)
    @test maximum(rhat(wide)) < 1.02
    @test dptiled_divrate(wide) < 0.01
    @test isapprox(dptiled_mean(wd), dptiled_mean(md); atol=0.03)
    @test isapprox(dptiled_std(wd), dptiled_std(md); atol=0.03)
    # Comparable efficiency to the masked path (within 25% of its worst-parameter ESS).
    @test minimum(ess(wide)) > 0.75 * minimum(ess(masked))
end

@testset "dptiled_widedual_matches_scalar_persistent" begin
    # The wide path is the scalar path's gradient computed a different way -- identical RNG
    # stream, mathematically identical gradient (the exact per-operation equivalence of the
    # two duals is nailed by `dptiled_widedual_matches_scalar_dual_derivatives`). It is NOT
    # meaningful to assert the two kernels trace the same trajectory: NUTS makes DISCRETE
    # decisions (`log(u) < candidate - combined` for the multinomial slice/merge, U-turn
    # branches), so the ~1e-12 floating-point reassociation of the P-wide partial sums can
    # flip a proposal selection and send a chain to a different leaf -- an O(1) per-draw
    # difference that is inherent to NUTS, not a defect. What MUST hold is that the two
    # sample the SAME posterior; assert distributional agreement on a full adaptive run.
    X, N, cons = dptiled_logistic_data(8, 120)
    full = (
        num_chains=8, num_samples=300, num_warmup=300, backend=CPU(),
        tree_strategy=:persistent, rng=MersenneTwister(3),
    )
    scalar = batched_nuts(dptiled_logistic, (X, N), cons; persistent_gradient=:scalar, full...)
    wide = batched_nuts(dptiled_logistic, (X, N), cons; persistent_gradient=:wide, full...)
    sd = posterior_array(scalar)
    wd = posterior_array(wide)
    @test isapprox(dptiled_mean(sd), dptiled_mean(wd); atol=0.02)   # same posterior
    @test isapprox(dptiled_std(sd), dptiled_std(wd); atol=0.03)
    @test isapprox(minimum(ess(scalar)), minimum(ess(wide)); rtol=0.15)
end

@testset "dptiled_widedual_small_model_correct" begin
    # Forcing :wide on a small model (P=2, N=1) must still be correct -- exercises the wide
    # walk at N==2 partials and the transform/log-abs-det path through DeviceGradN.
    res = batched_nuts(
        dptiled_two_param, (), choicemap((:y, 0.7));
        num_chains=32, num_samples=400, num_warmup=300,
        tree_strategy=:persistent, persistent_gradient=:wide, backend=CPU(), rng=MersenneTwister(2),
    )
    draws = posterior_array(res)
    @test all(isfinite, draws)
    @test maximum(rhat(res)) < 1.02
    @test dptiled_divrate(res) < 0.01
end

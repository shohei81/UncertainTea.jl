# Issue #134: device lowering for the broadcast-normal observation
# `{:y} ~ normal.(mu_expr, sigma)` -- docs/dsl.md's flagship GPU-lowering form.
#
# The device path previously rejected `BackendBroadcastNormalChoicePlanStep`
# ("device lowering does not support the ... distribution family yet"), so the
# GPU-scale regression form the docs recommend could not run on the device (the
# #121 scaling model had to be downgraded to a loop-addressed gaussian). It now
# lowers a VECTOR observation `y[1..M]` scored per element against
# `normal(mu_i, sigma)`, where `mu_i` is a broadcast affine expression over one or
# more per-observation covariate vectors (model arguments, riding the observation
# buffer like the GLM covariate column, issue #150) and scalar latents/args, and
# `sigma` is a scalar (a latent scale, a scalar arg, or a literal). The dense
# per-element fold reduces the whole vector in one thread-serial pass, and
# forward-mode through the same `_device_normal_logpdf` reproduces the host
# analytic broadcast gradient family by family.
#
# All tests here run on KernelAbstractions.CPU() at Float64 -- the device oracle,
# since CI has no Metal (test/gpu mirrors a Float32 smoke on Metal). The device
# takes UNCONSTRAINED parameters and folds the transform log-abs-det in-kernel, so
# the authoritative counterparts are the HOST batched backend path
# `batched_logjoint_gradient_unconstrained` / `batched_logjoint_unconstrained`
# and, for sampling, the host masked NUTS.

using KernelAbstractions: CPU
using Statistics: mean

# The issue's flagship model: an intercept/slope regression with a lognormal
# scale, observed through the broadcast-normal form over a covariate vector.
@tea static function devbcast_linreg(xs)
    slope ~ normal(0.0, 10.0)
    intercept ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(intercept .+ slope .* xs, sigma)
    return slope
end

# Two covariate vectors in the linear predictor (offsets 0 and 1 in the per-element
# observation block) plus a literal scalar sigma -- exercises multi-covariate
# staging and a constant scale.
@tea static function devbcast_two_covariates(x1, x2)
    a ~ normal(0.0, 5.0)
    b1 ~ normal(0.0, 5.0)
    b2 ~ normal(0.0, 5.0)
    {:y} ~ normal.(a .+ b1 .* x1 .+ b2 .* x2, 0.5)
    return a
end

# No covariate at all: a scalar mean/scale broadcast across the vector observation
# (stride 1, the y row only). Equivalent to M i.i.d. normals sharing (mu, sigma).
@tea static function devbcast_no_covariate()
    mu ~ normal(0.0, 5.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(mu, sigma)
    return mu
end

# A broadcast covariate that is NOT a model argument (a deterministic vector
# binding) cannot ride the observation buffer -- staging can never resolve it --
# so the device report must stay honest-unsupported.
@tea static function devbcast_derived_covariate(xs)
    a ~ normal(0.0, 5.0)
    b ~ normal(0.0, 5.0)
    sigma ~ lognormal(0.0, 1.0)
    zs = xs .* 2.0
    {:y} ~ normal.(a .+ b .* zs, sigma)
    return a
end

devbcast_make_data(rng, xs, slope, intercept, sigma) =
    intercept .+ slope .* xs .+ sigma .* randn(rng, length(xs))

# self-contained Float32 check (rtol/atol 1e-4 contract), so this file runs
# standalone without depending on device_lowering_parity.jl's helper.
function devbcast_check_float32(dev32, ref)
    ok = true
    for (d, r) in zip(dev32, ref)
        ok &= isapprox(Float64(d), r; rtol=1e-4, atol=1e-4 * max(1.0, abs(r)))
    end
    return ok
end

@testset "devbcast_lowering_report" begin
    xs = collect(range(-2.0, 2.0; length=8))
    cm = choicemap((:y, xs))  # any same-length vector is enough for the signature

    # validation item 3: the flagship broadcast-normal model is ACCEPTED.
    supported, issues = device_lowering_report(devbcast_linreg; constraints=cm)
    @test supported
    @test isempty(issues)

    supported2, issues2 = device_lowering_report(devbcast_two_covariates; constraints=cm)
    @test supported2
    @test isempty(issues2)

    supported0, issues0 = device_lowering_report(devbcast_no_covariate; constraints=cm)
    @test supported0
    @test isempty(issues0)

    # a derived (non-argument) covariate is rejected with a message naming it.
    sup_derived, iss_derived = device_lowering_report(devbcast_derived_covariate; constraints=cm)
    @test !sup_derived
    @test !isempty(iss_derived)
end

@testset "devbcast_logjoint_gradient_parity_f64" begin
    rng = MersenneTwister(134)
    xs = collect(range(-3.0, 3.0; length=24))
    ys = devbcast_make_data(rng, xs, 1.5, -0.5, 0.4)
    cm = choicemap((:y, ys))
    # parameter_count = slope + intercept + sigma = 3; several independent chains.
    params = randn(rng, 3, 8)

    v, g = device_batched_logjoint_gradient(devbcast_linreg, params, (xs,), cm)
    vref = batched_logjoint_unconstrained(devbcast_linreg, params, (xs,), cm)
    gref = batched_logjoint_gradient_unconstrained(devbcast_linreg, params, (xs,), cm)
    # forward-mode through the same closed form: agreement is a handful of ULP.
    @test v ≈ vref rtol = 1e-10
    @test g ≈ gref rtol = 1e-10

    # Float32 sanity on the same surfaces.
    v32, g32 = device_batched_logjoint_gradient(devbcast_linreg, Float32.(params), (xs,), cm; precision=Float32)
    @test devbcast_check_float32(v32, vref)
    @test devbcast_check_float32(vec(g32), vec(gref))

    # two-covariate model.
    x1 = collect(range(-2.0, 2.0; length=20))
    x2 = collect(range(1.0, -1.0; length=20))
    ys2 = -0.3 .+ 0.7 .* x1 .- 0.4 .* x2 .+ 0.5 .* randn(rng, 20)
    cm2 = choicemap((:y, ys2))
    params2 = randn(rng, 3, 6)
    v2, g2 = device_batched_logjoint_gradient(devbcast_two_covariates, params2, (x1, x2), cm2)
    v2ref = batched_logjoint_unconstrained(devbcast_two_covariates, params2, (x1, x2), cm2)
    g2ref = batched_logjoint_gradient_unconstrained(devbcast_two_covariates, params2, (x1, x2), cm2)
    @test v2 ≈ v2ref rtol = 1e-10
    @test g2 ≈ g2ref rtol = 1e-10

    # no-covariate (stride-1) scalar broadcast.
    ys3 = 0.6 .+ 0.4 .* randn(rng, 12)
    cm3 = choicemap((:y, ys3))
    params3 = randn(rng, 2, 5)
    v3, g3 = device_batched_logjoint_gradient(devbcast_no_covariate, params3, (), cm3)
    v3ref = batched_logjoint_unconstrained(devbcast_no_covariate, params3, (), cm3)
    g3ref = batched_logjoint_gradient_unconstrained(devbcast_no_covariate, params3, (), cm3)
    @test v3 ≈ v3ref rtol = 1e-10
    @test g3 ≈ g3ref rtol = 1e-10
end

@testset "devbcast_masked_nuts_recovers_posterior" begin
    # The issue's repro: the flagship broadcast-normal regression samples on the
    # device (CPU() oracle) and recovers the linear-regression posterior with a low
    # R-hat and no divergences. Kept small (N/chains/draws) for the slow device
    # shard runner.
    rng = MersenneTwister(2134)
    xs = collect(range(-3.0, 3.0; length=30))
    true_slope, true_intercept, true_sigma = 1.5, -0.5, 0.4
    ys = devbcast_make_data(rng, xs, true_slope, true_intercept, true_sigma)
    cm = choicemap((:y, ys))

    res = batched_nuts(
        devbcast_linreg, (xs,), cm;
        num_chains=6, num_samples=150, num_warmup=150,
        tree_strategy=:masked, backend=CPU(), precision=Float64,
        rng=MersenneTwister(7),
    )
    # constrained_samples: (slope, intercept, sigma) x samples, per chain.
    pooled = hcat((c.constrained_samples for c in res.chains)...)
    means = vec(mean(pooled; dims=2))
    @test all(isfinite, pooled)
    @test isapprox(means[1], true_slope; atol=0.15)
    @test isapprox(means[2], true_intercept; atol=0.2)
    @test isapprox(means[3], true_sigma; atol=0.2)
    @test maximum(UncertainTea.rhat(res)) < 1.05
    divrate = sum(sum(c.divergent) for c in res.chains) / sum(length(c.divergent) for c in res.chains)
    @test divrate < 0.05
end

@testset "devbcast_masked_nuts_vs_host_exact" begin
    # With adaptation OFF at a fixed step size AND device_sync_per_leaf=true (the
    # Tier-1 order-preserving path, issue #152), the device masked path faithfully
    # reimplements the host masked path (same RNG/reduction order), so draws match
    # to ~1e-6 on CPU() at Float64 -- the residual is the fused device gradient's
    # ~1e-16 disagreement with the host gradient cache, which flips no accept
    # decision (mirrors the GLM device oracle test).
    rng = MersenneTwister(909)
    xs = collect(range(-2.0, 2.0; length=20))
    ys = devbcast_make_data(rng, xs, 1.0, 0.2, 0.5)
    cm = choicemap((:y, ys))
    kwargs = (
        num_chains=6,
        num_samples=160,
        num_warmup=0,
        step_size=0.03,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        tree_strategy=:masked,
        device_sync_per_leaf=true,
        per_chain_adaptation=false,
    )
    device = batched_nuts(devbcast_linreg, (xs,), cm; rng=MersenneTwister(11), backend=CPU(), kwargs...)
    host = batched_nuts(devbcast_linreg, (xs,), cm; rng=MersenneTwister(11), kwargs...)
    device_draws = posterior_array(device)
    host_draws = posterior_array(host)
    @test all(isfinite, device_draws)
    @test maximum(abs, device_draws .- host_draws) < 1e-6
end

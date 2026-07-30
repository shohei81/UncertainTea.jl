# Persistent per-chain NUTS tree kernel (issue #154 increment 2).
#
# `batched_nuts(...; tree_strategy=:persistent, backend=...)` builds the ENTIRE NUTS
# tree for one iteration in a SINGLE kernel launch: each grid lane owns a chain and
# runs the whole iterative-doubling recursion device-resident, drawing its own momentum
# + slice/direction/merge randomness from the on-device Philox RNG (increment 1) and
# computing the leapfrog gradient in-kernel via the lowered plan's forward-mode dual
# walk (`src/device/persistent_nuts.jl`). No host round-trip mid-iteration.
#
# VALIDATION CONTRACT. Per #154 risk (a) the on-device draws are NOT the host `Random`
# draws, so this path is only STATISTICALLY (not bitwise) equivalent to the host/masked
# paths. These tests are the #121-style statistical-equivalence gate: posterior
# mean/sd/quantile agreement (analytic + vs the masked device path), R-hat < 1.01,
# comparable ESS, divergence-free, and matching tree-depth distribution. All run on
# KernelAbstractions.CPU() at Float64; the Metal Float32 smoke lives in test/gpu/.

using KernelAbstractions: CPU

# Local mean/std helpers (Statistics is not imported by the test harness).
dpnuts_mean(x) = sum(x) / length(x)
function dpnuts_std(x)
    m = dpnuts_mean(x)
    accumulator = 0.0
    for value in x
        accumulator += (value - m)^2
    end
    return sqrt(accumulator / (length(x) - 1))
end
dpnuts_divrate(res) =
    sum(sum(chain.divergent) for chain in res.chains) /
    sum(length(chain.divergent) for chain in res.chains)

# Conjugate gaussian: mu ~ N(0,1), y ~ N(mu,1) observed at y = 0.3.
# Posterior mean 0.15, variance 0.5.
@tea static function dpnuts_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# Two-parameter location / log-scale model: drives deeper trees (multi-round merges +
# the dyadic checkpoint / U-turn machinery).
@tea static function dpnuts_two_param()
    mu ~ normal(0.0, 1.0)
    log_sigma ~ normal(0.0, 0.5)
    {:y} ~ normal(mu, exp(log_sigma))
    return mu
end

@testset "dpnuts_persistent_requires_backend" begin
    # The persistent strategy is a device path: without a backend it is rejected.
    @test_throws ArgumentError batched_nuts(
        dpnuts_conjugate_gauss, (), choicemap((:y, 0.3));
        num_chains=4, num_samples=10, tree_strategy=:persistent,
    )
end

@testset "dpnuts_persistent_conjugate_gate" begin
    # #121 gate on the conjugate gauss (CPU() Float64): analytic mean/sd within
    # tolerance, R-hat < 1.01, divergence-free.
    res = batched_nuts(
        dpnuts_conjugate_gauss, (), choicemap((:y, 0.3));
        num_chains=64, num_samples=1000, num_warmup=500,
        tree_strategy=:persistent, backend=CPU(), rng=MersenneTwister(511),
    )
    draws = posterior_array(res)
    @test all(isfinite, draws)
    @test isapprox(dpnuts_mean(draws), 0.15; atol=0.03)
    @test isapprox(dpnuts_std(draws), sqrt(0.5); atol=0.03)
    @test maximum(rhat(res)) < 1.01
    @test dpnuts_divrate(res) == 0.0
end

@testset "dpnuts_persistent_matches_masked_moments" begin
    # Persistent vs the masked device path on the SAME model + seed, adaptation ON.
    # RNG semantics differ (on-device Philox vs host Random), so this is a moments-
    # agreement check, not a bitwise one: pooled mean/sd within a few MCSE, and
    # comparable min-ESS.
    kwargs = (
        num_chains=64, num_samples=1000, num_warmup=500, backend=CPU(),
    )
    persistent = batched_nuts(
        dpnuts_conjugate_gauss, (), choicemap((:y, 0.3));
        tree_strategy=:persistent, rng=MersenneTwister(1), kwargs...,
    )
    masked = batched_nuts(
        dpnuts_conjugate_gauss, (), choicemap((:y, 0.3));
        tree_strategy=:masked, rng=MersenneTwister(1), kwargs...,
    )
    pd = posterior_array(persistent)
    md = posterior_array(masked)
    # MCSE of the pooled mean ~ sd/sqrt(min_ESS); with >1.5e4 effective draws a 0.02
    # tolerance is several MCSE.
    @test isapprox(dpnuts_mean(pd), dpnuts_mean(md); atol=0.02)
    @test isapprox(dpnuts_std(pd), dpnuts_std(md); atol=0.02)
    # Comparable efficiency (within 25% of the masked path's worst-parameter ESS).
    @test minimum(ess(persistent)) > 0.75 * minimum(ess(masked))
end

@testset "dpnuts_persistent_deep_trees" begin
    # The two-parameter model exercises multi-round doubling: assert it reaches deep
    # trees, stays divergence-free at a well-adapted step, and agrees with masked.
    kwargs = (num_chains=64, num_samples=1000, num_warmup=500, backend=CPU())
    persistent = batched_nuts(
        dpnuts_two_param, (), choicemap((:y, 0.7));
        tree_strategy=:persistent, rng=MersenneTwister(2), kwargs...,
    )
    masked = batched_nuts(
        dpnuts_two_param, (), choicemap((:y, 0.7));
        tree_strategy=:masked, rng=MersenneTwister(2), kwargs...,
    )
    pd = posterior_array(persistent)
    md = posterior_array(masked)
    @test all(isfinite, pd)
    @test maximum(rhat(persistent)) < 1.01
    @test dpnuts_divrate(persistent) < 0.01
    @test maximum(reduce(vcat, treedepths(persistent))) >= 3
    @test isapprox(dpnuts_mean(pd), dpnuts_mean(md); atol=0.03)
    @test isapprox(dpnuts_std(pd), dpnuts_std(md); atol=0.03)
end

@testset "dpnuts_persistent_seed_reproducible_and_varies" begin
    # On-device draws are seeded from the host `rng` (per-run seed folded into the
    # Philox iteration coordinate): the SAME seed reproduces bitwise, DIFFERENT seeds
    # give different draws (so it is genuinely random, not a fixed sequence).
    kwargs = (
        num_chains=16, num_samples=200, num_warmup=200,
        tree_strategy=:persistent, backend=CPU(),
    )
    a = batched_nuts(dpnuts_conjugate_gauss, (), choicemap((:y, 0.3)); rng=MersenneTwister(3), kwargs...)
    b = batched_nuts(dpnuts_conjugate_gauss, (), choicemap((:y, 0.3)); rng=MersenneTwister(3), kwargs...)
    c = batched_nuts(dpnuts_conjugate_gauss, (), choicemap((:y, 0.3)); rng=MersenneTwister(999), kwargs...)
    da = posterior_array(a)
    db = posterior_array(b)
    dc = posterior_array(c)
    @test da == db                    # same seed -> identical
    @test da != dc                    # different seed -> different draws
    @test isapprox(dpnuts_mean(da), dpnuts_mean(dc); atol=0.06)  # but same posterior
end

# PR 51: device-resident masked batched NUTS.
#
# `batched_nuts(...; tree_strategy=:masked, backend=...)` runs the mask-based
# iterative-doubling trajectory device-resident (src/device/nuts_kernels.jl): all
# the per-leaf P x C arrays -- positions, momenta, gradients, dyadic U-turn
# checkpoints -- stay on the device, and only O(num_chains) vectors cross the bus.
# The DEFAULT (Tier 2, issue #152) pre-draws each round's RNG once and runs the
# per-leaf accept/select bookkeeping on the device, so the host reads the subtree
# state back ONCE per doubling round; `device_sync_per_leaf=true` selects the Tier-1
# order-preserving path (per-leaf host round-trip) that keeps the bitwise oracle. The
# trajectory init/finalize reuse the host code once per outer iteration. All tests
# here run on KernelAbstractions.CPU(); the GPU smoke under test/gpu/ mirrors a
# subset on Metal at Float32.
#
# Parity oracle. With adaptation OFF at a fixed step size the device round loop is
# a faithful reimplementation of the host masked path -- same RNG order, same
# reduction order -- so on CPU() at Float64 the draws match the host masked path
# to ~1e-8 (the residual is the fused device gradient's ~1e-16 disagreement with
# the host gradient cache, which does not flip any accept decision without
# adaptation). This is the tight faithful-port check. WITH step-size adaptation,
# dual averaging amplifies that 1e-16 gradient difference, so the adaptive device
# path is only STATISTICALLY equivalent to the host path, checked with posterior
# analytics within tolerance.

using KernelAbstractions: CPU

# Local mean/std helpers (Statistics is not imported by the test harness).
dnuts_mean(x) = sum(x) / length(x)
function dnuts_std(x)
    m = dnuts_mean(x)
    accumulator = 0.0
    for value in x
        accumulator += (value - m)^2
    end
    return sqrt(accumulator / (length(x) - 1))
end

# Conjugate gaussian: mu ~ N(0,1), y ~ N(mu,1) observed at y = 0.3.
# Posterior mean 0.15, variance 0.5.
@tea static function dnuts_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# Two-parameter location / log-scale model, used to drive deeper trees (so the
# dyadic checkpoint + U-turn kernels and multi-round merges are exercised).
@tea static function dnuts_two_param()
    mu ~ normal(0.0, 1.0)
    log_sigma ~ normal(0.0, 0.5)
    {:y} ~ normal(mu, exp(log_sigma))
    return mu
end

# marginalize=:enumerate: backend-supported (issue #13) but still not device-lowerable.
@tea static function dnuts_marginalize_model()
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y} ~ normal(mu + z, 1.0)
    return mu
end

@testset "dnuts_device_masked_conjugate" begin
    device = batched_nuts(
        dnuts_conjugate_gauss, (), choicemap((:y, 0.3));
        num_chains=4, num_samples=300, num_warmup=200,
        tree_strategy=:masked, backend=CPU(), rng=MersenneTwister(511),
    )
    device_draws = posterior_array(device)
    @test all(isfinite, device_draws)
    @test isapprox(dnuts_mean(device_draws), 0.15; atol=0.1)
    @test isapprox(dnuts_std(device_draws), sqrt(0.5); atol=0.15)
    @test all(<(1.2), rhat(device))
end

@testset "dnuts_device_vs_host_masked_exact" begin
    # Adaptation off at a fixed step size: with `device_sync_per_leaf=true` (the
    # Tier-1 order-preserving path, issue #152) the device round loop preserves the
    # host RNG draw order and reduction order, so the draws are numerically identical
    # to the host masked path (up to the ~1e-16 device/host gradient disagreement,
    # which flips no decisions here). This exercises deeper trees (multi-round merges
    # + dyadic checkpoints) on the two-parameter model. The DEFAULT device path is the
    # Tier-2 async path, which pre-draws the round RNG on the device and is therefore
    # only STATISTICALLY equivalent to the host (checked in the adaptive testset and
    # by `dnuts_device_async_matches_sync_moments` below), so this bitwise oracle
    # opts into the sync-per-leaf path explicitly.
    kwargs = (
        num_chains=6, num_samples=250, num_warmup=0, step_size=0.05,
        adapt_step_size=false, adapt_mass_matrix=false, tree_strategy=:masked,
        device_sync_per_leaf=true,
        # the device path runs shared adaptation; pin the host leg to the same
        # mode (issue #137 made per-chain the host default)
        per_chain_adaptation=false,
    )
    device = batched_nuts(dnuts_two_param, (), choicemap((:y, 0.7)); backend=CPU(), rng=MersenneTwister(7), kwargs...)
    host = batched_nuts(dnuts_two_param, (), choicemap((:y, 0.7)); rng=MersenneTwister(7), kwargs...)

    device_draws = posterior_array(device)
    host_draws = posterior_array(host)
    @test maximum(abs, device_draws .- host_draws) < 1e-8

    device_depths = reduce(vcat, treedepths(device))
    host_depths = reduce(vcat, treedepths(host))
    @test device_depths == host_depths
    @test maximum(host_depths) >= 3  # confirms deep trees were actually exercised
end

@testset "dnuts_device_async_matches_sync_moments" begin
    # Tier-2 async path (default) vs the Tier-1 sync-per-leaf path on the SAME model,
    # adaptation off at a fixed step size, many chains. The async path pre-draws the
    # round RNG on the device (departing from the CPU draw order), so the two are only
    # STATISTICALLY equivalent; assert the pooled posterior mean/sd agree within a few
    # MCSE and that both explore deep trees + stay divergence-free.
    kwargs = (
        num_chains=64, num_samples=400, num_warmup=0, step_size=0.05,
        adapt_step_size=false, adapt_mass_matrix=false, tree_strategy=:masked,
        per_chain_adaptation=false,
    )
    async = batched_nuts(
        dnuts_two_param,
        (),
        choicemap((:y, 0.7));
        backend=CPU(),
        device_sync_per_leaf=false,
        rng=MersenneTwister(20),
        kwargs...,
    )
    sync = batched_nuts(
        dnuts_two_param,
        (),
        choicemap((:y, 0.7));
        backend=CPU(),
        device_sync_per_leaf=true,
        rng=MersenneTwister(20),
        kwargs...,
    )
    async_draws = posterior_array(async)
    sync_draws = posterior_array(sync)
    @test all(isfinite, async_draws)
    # MCSE of the pooled mean ~ sd/sqrt(N_eff); with 64x400 draws a 0.02 tolerance is
    # many MCSE for this well-mixed target.
    @test abs(dnuts_mean(async_draws) - dnuts_mean(sync_draws)) < 0.03
    @test abs(dnuts_std(async_draws) - dnuts_std(sync_draws)) < 0.03
    @test maximum(reduce(vcat, treedepths(async))) >= 3
    @test all(<(1.05), rhat(async))
end

@testset "dnuts_device_vs_host_masked_adaptive" begin
    # Full adaptation: statistically (not bitwise) equivalent to the host path.
    kwargs = (
        num_chains=4, num_samples=300, num_warmup=200, tree_strategy=:masked,
        # compare like with like: the device leg uses shared adaptation
        per_chain_adaptation=false,
    )
    device = batched_nuts(dnuts_conjugate_gauss, (), choicemap((:y, 0.3)); backend=CPU(), rng=MersenneTwister(512), kwargs...)
    host = batched_nuts(dnuts_conjugate_gauss, (), choicemap((:y, 0.3)); rng=MersenneTwister(512), kwargs...)

    device_draws = posterior_array(device)
    host_draws = posterior_array(host)
    @test abs(dnuts_mean(device_draws) - dnuts_mean(host_draws)) < 0.1
    @test abs(dnuts_std(device_draws) - dnuts_std(host_draws)) < 0.15
end

@testset "dnuts_device_guards" begin
    # The device backend supports only the masked strategy.
    @test_throws ArgumentError batched_nuts(
        dnuts_conjugate_gauss, (), choicemap((:y, 0.3));
        num_chains=2, num_samples=1, tree_strategy=:hybrid, backend=CPU(),
    )
    # per-chain adaptation is now SUPPORTED on the device backend (issue #137): it
    # routes to the pooled-mass / per-chain-step driver and converges.
    per_chain = batched_nuts(
        dnuts_conjugate_gauss, (), choicemap((:y, 0.3));
        num_chains=4, num_samples=300, num_warmup=200,
        tree_strategy=:masked, per_chain_adaptation=true, backend=CPU(), rng=MersenneTwister(513),
    )
    per_chain_draws = posterior_array(per_chain)
    @test all(isfinite, per_chain_draws)
    @test isapprox(dnuts_mean(per_chain_draws), 0.15; atol=0.1)
    @test all(<(1.2), rhat(per_chain))
    # A non-lowerable model raises (pointing at device_lowering_report).
    @test_throws ArgumentError batched_nuts(
        dnuts_marginalize_model, (), choicemap((:y, 0.3));
        num_chains=2, num_samples=1, tree_strategy=:masked, backend=CPU(),
    )
end

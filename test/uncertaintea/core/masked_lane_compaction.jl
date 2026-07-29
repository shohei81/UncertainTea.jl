# Issue #160: masked-NUTS lane compaction. When most chains have finished or
# diverged the masked doubling loop still runs a FULL-WIDTH batched gradient at
# every leapfrog leaf. Compaction gathers the active columns, evaluates the
# analytic backend gradient over just those k columns, and scatters back. The
# gradient is per-column independent, so this is a pure permutation: the active
# lanes must get BITWISE-identical logjoint/gradient values, and the inactive
# lanes (downstream don't-cares) must be left untouched.
#
# This file pins the compaction machinery directly: the gather/scatter equals the
# full-width gradient bit-for-bit on the active lanes, the analytic backend is
# invoked over EXACTLY count(active) columns, the inactive lanes are never
# written, and the active-fraction gate fires only below the threshold. The
# end-to-end draw-level guarantee (host masked draws stay bitwise identical) is
# carried by the device-vs-host oracle in device_masked_nuts.jl and the masked
# determinism test in masked_batched_nuts.jl.

# Bit-level equality (distinguishes 0.0 / -0.0 and NaN payloads that `==` hides).
mlc_same_bits(a::AbstractArray{Float64}, b::AbstractArray{Float64}) =
    size(a) == size(b) && reinterpret(UInt64, vec(Array(a))) == reinterpret(UInt64, vec(Array(b)))
mlc_same_bits(a::Float64, b::Float64) = reinterpret(UInt64, a) === reinterpret(UInt64, b)

# Analytic-backend-supported models (so `cache.backend_cache` is populated and the
# compaction gate turns on): a scalar normal and a two-parameter location/log-scale.
@tea static function mlc_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

@tea static function mlc_two_param()
    mu ~ normal(0.0, 1.0)
    log_sigma ~ normal(0.0, 0.5)
    {:y} ~ normal(mu, exp(log_sigma))
    return mu
end

# Build a sampler-style reject-on gradient cache and matching model target.
function mlc_cache_and_target(model, constraints, num_params, num_chains; seed=0)
    rng = MersenneTwister(seed)
    params = randn(rng, num_params, num_chains)
    cache = UncertainTea.BatchedLogjointGradientCache(
        model, params, (), constraints; reject_invalid_parameters=true,
    )
    target = UncertainTea.BatchedModelDensityTarget(cache)
    return cache, target, params
end

@testset "mlc_gate_active_fraction" begin
    cache, _, _ = mlc_cache_and_target(mlc_conjugate_gauss, choicemap((:y, 0.3)), 1, 64)
    @test cache.backend_cache !== nothing            # analytic tier available
    @test cache.thread_plan === nothing              # fused gauss stays serial
    # Above the 0.5 active-fraction threshold (including the all-active common
    # case): the full-width path stays in charge.
    @test !UncertainTea._batched_lane_compaction_beneficial(cache, 64)
    @test !UncertainTea._batched_lane_compaction_beneficial(cache, 40)  # 62.5%
    @test !UncertainTea._batched_lane_compaction_beneficial(cache, 32)  # exactly 50%
    # Below it: compact.
    @test UncertainTea._batched_lane_compaction_beneficial(cache, 31)
    @test UncertainTea._batched_lane_compaction_beneficial(cache, 1)
    # Degenerate: no active lane never compacts.
    @test !UncertainTea._batched_lane_compaction_beneficial(cache, 0)
end

# Compaction gather/scatter must equal the full-width analytic gradient bit-for-bit
# on the active lanes, evaluate exactly k columns, and leave inactive lanes alone.
function mlc_check_bitwise(model, constraints, num_params, num_chains, active; seed=0)
    cache, target, params = mlc_cache_and_target(model, constraints, num_params, num_chains; seed=seed)
    active_count = count(active)

    # full-width reference through the 4-arg (unmasked) target method
    ref_values = fill(NaN, num_chains)
    ref_grad = fill(NaN, num_params, num_chains)
    UncertainTea.batched_target_logdensity_and_gradient!(ref_values, ref_grad, target, params)

    # compacted path: seed the destinations with a sentinel so an untouched
    # inactive lane is detectable.
    got_values = fill(-99.0, num_chains)
    got_grad = fill(-99.0, num_params, num_chains)
    ok = UncertainTea._batched_compact_logjoint_and_gradient!(
        got_values, got_grad, cache, params, active, active_count,
    )
    @test ok
    # the analytic backend was invoked over EXACTLY count(active) columns
    @test cache.backend_cache.workspace.batched_environment[].batch_size == active_count

    for chain_index = 1:num_chains
        if active[chain_index]
            @test mlc_same_bits(got_values[chain_index], ref_values[chain_index])
            @test mlc_same_bits(view(got_grad, :, chain_index), view(ref_grad, :, chain_index))
        else
            # inactive lanes are don't-cares: left at the sentinel, never written
            @test got_values[chain_index] == -99.0
            @test all(==(-99.0), view(got_grad, :, chain_index))
        end
    end
end

# Per-column (vectorized) constraints must NOT compact: the compact call reuses the
# cache's fixed length-C constraints vector, which cannot be sliced to k columns
# without rebuilding the cache. The gate keeps them full width, and the active-aware
# target method still returns the correct full-width values on the active lanes.
@testset "mlc_per_column_constraints_stay_full_width" begin
    num_chains = 6
    per_column_constraints = [choicemap((:y, 0.1 * chain)) for chain = 1:num_chains]
    rng = MersenneTwister(31)
    params = randn(rng, 1, num_chains)
    cache = UncertainTea.BatchedLogjointGradientCache(
        mlc_conjugate_gauss, params, (), per_column_constraints; reject_invalid_parameters=true,
    )
    target = UncertainTea.BatchedModelDensityTarget(cache)
    # even a single active lane stays full width when constraints are per-column
    @test !UncertainTea._batched_lane_compaction_beneficial(cache, 1)

    ref_values = fill(NaN, num_chains)
    ref_grad = fill(NaN, 1, num_chains)
    UncertainTea.batched_target_logdensity_and_gradient!(ref_values, ref_grad, target, params)

    active = falses(num_chains)
    active[[2, 5]] .= true
    got_values = fill(NaN, num_chains)
    got_grad = fill(NaN, 1, num_chains)
    UncertainTea.batched_target_logdensity_and_gradient!(
        got_values, got_grad, target, params, active, count(active),
    )
    for chain_index in findall(active)
        @test mlc_same_bits(got_values[chain_index], ref_values[chain_index])
        @test mlc_same_bits(view(got_grad, :, chain_index), view(ref_grad, :, chain_index))
    end
end

@testset "mlc_compact_matches_full_width_bitwise" begin
    # scalar gauss, sparse active set
    active64 = falses(64)
    active64[[2, 5, 9, 30, 31, 60]] .= true
    mlc_check_bitwise(mlc_conjugate_gauss, choicemap((:y, 0.3)), 1, 64, active64; seed=11)

    # two-parameter model, a different sparse mask
    active48 = falses(48)
    active48[[1, 7, 8, 20, 47]] .= true
    mlc_check_bitwise(mlc_two_param, choicemap((:y, 0.7)), 2, 48, active48; seed=12)

    # single active lane (extreme compaction)
    active_one = falses(32)
    active_one[17] = true
    mlc_check_bitwise(mlc_conjugate_gauss, choicemap((:y, 0.3)), 1, 32, active_one; seed=13)
end

# The active-aware target method must agree bit-for-bit with the full-width method
# on the active lanes whether it took the compact branch (sparse active) or the
# full-width branch (dense active), for both scalar and per-chain callers.
@testset "mlc_target_method_active_lanes_bitwise" begin
    num_params, num_chains = 2, 40
    cache, target, params =
        mlc_cache_and_target(mlc_two_param, choicemap((:y, 0.7)), num_params, num_chains; seed=21)

    ref_values = fill(NaN, num_chains)
    ref_grad = fill(NaN, num_params, num_chains)
    UncertainTea.batched_target_logdensity_and_gradient!(ref_values, ref_grad, target, params)

    # sparse (compacts) and dense (full width) both match the reference on active lanes
    sparse = falses(num_chains)
    sparse[[3, 4, 10, 25]] .= true
    dense = trues(num_chains)
    for active in (sparse, dense)
        active_count = count(active)
        got_values = fill(NaN, num_chains)
        got_grad = fill(NaN, num_params, num_chains)
        UncertainTea.batched_target_logdensity_and_gradient!(
            got_values, got_grad, target, params, active, active_count,
        )
        for chain_index = 1:num_chains
            active[chain_index] || continue
            @test mlc_same_bits(got_values[chain_index], ref_values[chain_index])
            @test mlc_same_bits(view(got_grad, :, chain_index), view(ref_grad, :, chain_index))
        end
    end
end

# End-to-end smoke: many chains (so the active fraction genuinely drops below the
# threshold mid-trajectory and the compact branch runs) still produce finite,
# deterministic draws through the masked strategy.
@testset "mlc_masked_nuts_end_to_end" begin
    constraints = choicemap((:y, 0.3))
    run() = batched_nuts(
        mlc_conjugate_gauss, (), constraints;
        num_chains=64, num_samples=120, num_warmup=80,
        tree_strategy=:masked, rng=MersenneTwister(487),
    )
    first_run = run()
    second_run = run()
    @test all(isfinite, posterior_array(first_run))
    @test all(<(1.2), rhat(first_run))
    for chain_index in eachindex(first_run.chains)
        @test first_run.chains[chain_index].unconstrained_samples ==
              second_run.chains[chain_index].unconstrained_samples
    end
end

# issue #160: device sync-leaf lane compaction.
#
# The Tier-1 (sync-per-leaf) masked NUTS leaf runs its gradient over ALL C columns
# even once most chains have finished/diverged. Lane compaction gathers the active
# columns, evaluates the gradient over just those `k` columns, and scatters the
# results back to the active lanes. The gradient of column c depends ONLY on
# `params[:, c]`, the SHARED observations, and its own private slot scratch, so
# gather -> gradient -> scatter is a pure permutation over independent columns: the
# active lanes receive BITWISE-identical values (the safety argument behind keeping
# the CPU()-Float64 device-vs-host bitwise oracle in device_masked_nuts.jl).
#
# These tests pin, at the kernel/helper level, the pieces that the oracle can only
# check end to end: (1) the k-column gradient is BITWISE equal to the full-width
# gradient on the gathered columns and untouched beyond k; (2) gather/scatter is a
# faithful permutation that leaves inactive lanes untouched; (3) the active-fraction
# gate compacts only below 50% and never for the all-active leaf; (4) per-column
# observations/arguments make compaction ineligible (fall back to full width); and
# (5) end to end the sync path actually engages compaction, saves gradient columns,
# and stays deterministic. All run on KernelAbstractions.CPU() at Float64.

using KernelAbstractions
using KernelAbstractions: CPU

const dnutsc = UncertainTea

# Small (untiled) shared-observation gauss: eligible for compaction.
@tea static function dnutsc_gauss(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# Two-parameter model reused from the oracle to drive deep trees end to end.
@tea static function dnutsc_two_param()
    mu ~ normal(0.0, 1.0)
    log_sigma ~ normal(0.0, 0.5)
    {:y} ~ normal(mu, exp(log_sigma))
    return mu
end

@testset "dnutsc_compact_gradient_bitwise_on_active_lanes" begin
    n = 4
    C = 6
    ys = [0.4, -0.7, 1.1, 0.2]
    cm = choicemap((:y => i, ys[i]) for i = 1:n)
    ws = DeviceBatchedWorkspace(dnutsc_gauss, C; args=(n,), constraints=cm)
    @test dnutsc._device_nuts_compaction_eligible(ws)   # shared obs + shared args
    be = ws.backend
    P = ws.parameter_count

    params = randn(MersenneTwister(99), P, C)
    # Full-width reference; this also uploads `params` into `ws.params_device`.
    v_ref, g_ref = device_batched_logjoint_gradient!(ws, params)

    # Gather three active columns, launch the gradient over just those k, and compare
    # the compact result to the full-width reference BITWISE (==, not approx).
    active_cols = [2, 4, 5]
    k = length(active_cols)
    index = fill!(KernelAbstractions.allocate(be, Int32, C), Int32(0))
    idx_host = zeros(Int32, C)
    idx_host[1:k] .= Int32.(active_cols)
    copyto!(index, idx_host)
    compact_params = fill!(KernelAbstractions.allocate(be, Float64, P, C), 0.0)
    compact_grad = fill!(KernelAbstractions.allocate(be, Float64, P, C), NaN)
    compact_tot = fill!(KernelAbstractions.allocate(be, Float64, C), NaN)

    dnutsc._device_nuts_gather_columns!(be)(compact_params, ws.params_device, index; ndrange=(P, k))
    dnutsc._device_launch_gradient_compact!(ws, compact_params, compact_grad, compact_tot, k)
    KernelAbstractions.synchronize(be)

    cg = Array(compact_grad)
    ct = Array(compact_tot)
    for (slot, c) in enumerate(active_cols)
        @test cg[:, slot] == g_ref[:, c]   # BITWISE gradient equality on the gathered lane
        @test ct[slot] == v_ref[c]         # BITWISE logjoint equality on the gathered lane
    end
    # Columns beyond k are never launched, so the sentinel survives.
    @test all(isnan, cg[:, (k+1):C])
    @test all(isnan, ct[(k+1):C])

    # Scatter the compact result back to the active lanes; inactive lanes untouched.
    dest_grad = fill!(KernelAbstractions.allocate(be, Float64, P, C), -999.0)
    dest_tot = fill!(KernelAbstractions.allocate(be, Float64, C), -999.0)
    dnutsc._device_nuts_scatter_gradient!(be)(dest_grad, dest_tot, compact_grad, compact_tot, index; ndrange=(P, k))
    KernelAbstractions.synchronize(be)
    dg = Array(dest_grad)
    dt = Array(dest_tot)
    for (slot, c) in enumerate(active_cols)
        @test dg[:, c] == cg[:, slot]
        @test dt[c] == ct[slot]
        @test dg[:, c] == g_ref[:, c]   # end-to-end: active lane matches full width
        @test dt[c] == v_ref[c]
    end
    inactive = setdiff(1:C, active_cols)
    @test all(==(-999.0), dg[:, inactive])   # inactive lanes left untouched (don't-cares)
    @test all(==(-999.0), dt[inactive])
end

@testset "dnutsc_active_fraction_gate" begin
    C = 6
    dws = dnutsc.DeviceNUTSWorkspace(dnutsc_two_param, C, 5; constraints=choicemap((:y, 0.7)))
    function mask(cols)
        v = fill(false, C)
        for c in cols
            v[c] = true
        end
        return v
    end
    # All active / exactly 50% / none -> full width (returns 0).
    @test dnutsc._device_nuts_build_compaction!(dws, mask(1:C)) == 0
    @test dnutsc._device_nuts_build_compaction!(dws, mask([1, 2, 3])) == 0
    @test dnutsc._device_nuts_build_compaction!(dws, mask(Int[])) == 0
    # Below 50% -> compact; the gather index records the active columns in order.
    @test dnutsc._device_nuts_build_compaction!(dws, mask([2, 5])) == 2
    @test dws.compact_index_host[1:2] == Int32[2, 5]
    @test Array(dws.compact_index)[1:2] == Int32[2, 5]
    # A single active lane still compacts (k = 1).
    @test dnutsc._device_nuts_build_compaction!(dws, mask([4])) == 1
    @test dws.compact_index_host[1] == Int32(4)
end

@testset "dnutsc_compaction_ineligible_per_column" begin
    n = 4
    C = 3
    ys = [0.4, -0.7, 1.1, 0.2]
    cm = choicemap((:y => i, ys[i]) for i = 1:n)
    # Per-chain (batched) arguments keep C observation columns AND make `args` a
    # Vector, so the shared-broadcast precondition fails -> ineligible (full width).
    ws = DeviceBatchedWorkspace(dnutsc_gauss, C; args=[(n,) for _ = 1:C], constraints=cm)
    @test size(ws.observed_device, 2) == C
    @test !dnutsc._device_nuts_compaction_eligible(ws)
end

@testset "dnutsc_sync_leaf_compaction_engages_and_saves" begin
    C = 6
    kwargs = (
        num_chains=C, num_samples=200, num_warmup=0, step_size=0.05,
        adapt_step_size=false, adapt_mass_matrix=false, tree_strategy=:masked,
        device_sync_per_leaf=true, per_chain_adaptation=false,
    )
    dnutsc._device_nuts_reset_compaction_stats!()
    r1 = batched_nuts(dnutsc_two_param, (), choicemap((:y, 0.7)); backend=CPU(), rng=MersenneTwister(7), kwargs...)
    leaves = dnutsc._DEVICE_NUTS_GRADIENT_LEAVES[]
    compacted = dnutsc._DEVICE_NUTS_COMPACTED_LEAVES[]
    columns = dnutsc._DEVICE_NUTS_GRADIENT_COLUMNS[]

    @test leaves > 0
    @test compacted > 0                 # the < 50% gate actually engaged
    @test columns < leaves * C          # strictly fewer gradient columns than full width
    # Every compacted leaf ran 1..(C-1)/2 = 1..2 columns; every other leaf ran C.
    @test columns <= (leaves - compacted) * C + compacted * 2
    @test columns >= (leaves - compacted) * C + compacted * 1

    # Determinism: an identical seed reproduces the draws bitwise (compaction adds no
    # nondeterminism), and the sync-path result stays finite/sane.
    r2 = batched_nuts(dnutsc_two_param, (), choicemap((:y, 0.7)); backend=CPU(), rng=MersenneTwister(7), kwargs...)
    @test posterior_array(r1) == posterior_array(r2)
    @test all(isfinite, posterior_array(r1))
end

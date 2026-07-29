# Observation-parallel tiled device gradient (issue #153).
#
# The serial gradient kernel (`gradient_kernel.jl`) launches one thread per
# `(parameter, chain)` and each thread re-walks the WHOLE plan, including a loop
# that scans all N observations sequentially: O(P*N) serial work per chain. For
# observation-heavy models that under-fills the GPU (P*C threads only) and makes
# the observation scan the bottleneck.
#
# When the plan has a single loop whose body is a pure additive reduction over
# independent iterations (the gauss / logistic / GLM shapes), the observation loop
# is lowered to a TILED kernel over `(parameter, tile, chain)` that computes the
# partial logjoint value + P derivative channels for a contiguous tile of
# iterations, plus a per-chain reduction kernel that folds the tiles into the
# prelude (prior) contribution. This raises the launched-thread count to
# `P * ntiles * C` and turns the O(P*N) serial scan into O(P*N/ntiles) parallel
# work + a deterministic reduction.
#
# NUMERICS: the tile reduction reassociates the observation sum versus the serial
# scan, so results are STATISTICALLY (not bitwise) equal to the untiled path. The
# reduction order is deterministic (sequential tile order within a chain), so a
# given workspace is reproducible run to run. Tiling only engages for loops with
# at least `DEVICE_GRADIENT_TILE_MIN_OBS` iterations, so small models (every device
# parity/oracle test) keep the untiled serial scan and stay bitwise identical.

# ---- tileability detection (over the lowered device plan) ----------------------

# Slots the body (a non-loop step sequence) reads; `nothing` if the body itself
# contains a loop (nested loops are not tiled).
function _device_body_reads(body_steps)
    reads = BitSet()
    for step in body_steps
        step isa DeviceLoopStep && return nothing
        _device_step_expr_reads!(reads, step)
    end
    return reads
end

# Binding slots the body materializes (a body-written slot that is read anywhere
# would break per-iteration independence).
function _device_body_written(body_steps)
    written = BitSet()
    for step in body_steps
        step isa DeviceLoopStep && return nothing
        if _device_step_writes_binding(step) && step.binding_slot > Int32(0)
            push!(written, Int(step.binding_slot))
        end
    end
    return written
end

# Returns `(prelude, body, loop_id)` when the plan has exactly one top-level loop
# whose body is a tileable additive reduction; `nothing` otherwise. Tileable means:
#   * a single loop in the whole plan (`loop_count == 1`, one top-level loop step),
#   * the body reads no per-iteration index (its iterator slot), and
#   * every slot the body writes is dead (read nowhere), so each iteration's
#     contribution depends only on the pre-loop slot state and its own observation
#     rows -- exactly the condition under which a tile of iterations can be summed
#     independently and reduced.
function _device_detect_tileable_loop(plan::DeviceExecutionPlan)
    plan.loop_count == Int32(1) || return nothing
    loop_idx = 0
    for (i, step) in enumerate(plan.steps)
        if step isa DeviceLoopStep
            loop_idx == 0 || return nothing
            loop_idx = i
        end
    end
    loop_idx == 0 && return nothing

    loop = plan.steps[loop_idx]
    body_reads = _device_body_reads(loop.body)
    isnothing(body_reads) && return nothing
    loop.iterator_slot > Int32(0) && Int(loop.iterator_slot) in body_reads && return nothing

    body_written = _device_body_written(loop.body)
    isnothing(body_written) && return nothing
    reads_by_loop = Dict{Int32,BitSet}()
    _device_collect_expr_reads!(reads_by_loop, plan.steps, Int32(0))
    all_reads = BitSet()
    for (_, r) in reads_by_loop
        union!(all_reads, r)
    end
    isempty(intersect(body_written, all_reads)) || return nothing

    prelude = tuple((plan.steps[i] for i in eachindex(plan.steps) if i != loop_idx)...)
    return (prelude=prelude, body=loop.body, loop_id=loop.loop_id)
end

# Only the binomial step reads the exact-integer observation mirror; when the plan
# has none, the `observed_int` upload is dropped (issue #153).
_device_plan_has_count_family(plan::DeviceExecutionPlan) = _device_steps_have_count(plan.steps)
function _device_steps_have_count(steps)
    for step in steps
        step isa DeviceBinomialChoiceStep && return true
        step isa DeviceLoopStep && _device_steps_have_count(step.body) && return true
    end
    return false
end

# ---- tile sizing ---------------------------------------------------------------

# Below this many loop iterations the serial scan is kept: the launch/reduction
# overhead of three kernels is not worth it and tiny models stay bitwise identical
# to the untiled path.
const DEVICE_GRADIENT_TILE_MIN_OBS = 256
# Target tile count per chain; the iterations-per-tile is chosen so `ntiles` is at
# most this, bounding the partial-sum scratch while keeping occupancy high.
const DEVICE_GRADIENT_TILE_TARGET = 256

function _device_tile_sizing(n_obs::Integer)
    n = Int(n_obs)
    iters_per_tile = max(1, cld(n, DEVICE_GRADIENT_TILE_TARGET))
    ntiles = cld(n, iters_per_tile)
    return Int32(iters_per_tile), Int32(ntiles)
end

# ---- descriptor ----------------------------------------------------------------

struct DeviceTiledGradient{T,PP<:DeviceExecutionPlan{T},BP<:DeviceExecutionPlan{T}}
    # everything except the tileable loop; a `_device_gradient_kernel!` launch
    # over this writes the prior value/gradient AND materializes the pre-loop
    # bindings the body reads
    prelude_plan::PP
    # the loop body as a stand-alone step tuple, scored per tile
    body_plan::BP
    loop_id::Int32
    base_cursor::Int32     # observed rows emitted before the loop (0-based)
    stride::Int32          # observed rows one body iteration consumes
    n_obs::Int32           # loop trip count (static for the workspace)
    iters_per_tile::Int32
    ntiles::Int32
end

function _build_device_tiled_gradient(plan::DeviceExecutionPlan{T}, tileable, bundle) where {T}
    n_obs = bundle.trip_counts[Int(tileable.loop_id)]
    iters_per_tile, ntiles = _device_tile_sizing(n_obs)
    prelude_plan = DeviceExecutionPlan{T}(tileable.prelude, plan.slot_count, Int32(0))
    body_plan = DeviceExecutionPlan{T}(tileable.body, plan.slot_count, plan.loop_count)
    return DeviceTiledGradient(
        prelude_plan,
        body_plan,
        tileable.loop_id,
        bundle.tile_base_cursor,
        bundle.tile_stride,
        n_obs,
        iters_per_tile,
        ntiles,
    )
end

# ---- kernels -------------------------------------------------------------------

# One thread per `(parameter, tile, chain)`: sum the loop body's logjoint value and
# the `pidx` derivative channel over the tile's contiguous iterations. Iteration
# `t` (0-based) reads observation rows starting at 1-based cursor
# `base_cursor + 1 + t*stride`. Prelude bindings live in `slots` (written by the
# prelude kernel); the body may write its own dead binding slots -- benign, never
# read -- so no synchronization within the tile is needed.
@kernel function _device_tiled_gradient_partials_kernel!(
    partial_totals,
    partial_grads,
    body_plan,
    @Const(params),
    @Const(observed),
    @Const(observed_int),
    slots,
    @Const(trip_counts),
    @Const(loop_starts),
    base_cursor::Int32,
    stride::Int32,
    n_obs::Int32,
    iters_per_tile::Int32,
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    tile = idx[2]
    b = idx[3]
    TD = eltype(slots)
    t0 = Int32(tile - 1) * iters_per_tile
    t1 = min(t0 + iters_per_tile, n_obs)
    total = zero(TD)
    t = t0
    while t < t1
        cur = base_cursor + Int32(1) + t * stride
        contribution, _ = _device_grad_score_steps(
            body_plan.steps, slots, params, observed, observed_int, trip_counts, loop_starts, pidx, b, cur,
        )
        total += contribution
        t += Int32(1)
    end
    @inbounds partial_grads[pidx, tile, b] = _device_dual_deriv(total)
    if pidx == 1
        @inbounds partial_totals[tile, b] = _device_dual_value(total)
    end
end

# Fold the per-tile partials into the prelude value/gradient already resident in
# `totals`/`gradients`. Sequential tile order keeps a deterministic reduction.
@kernel function _device_tiled_gradient_reduce_kernel!(
    gradients,
    totals,
    @Const(partial_grads),
    @Const(partial_totals),
    ntiles::Int32,
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    g = @inbounds gradients[pidx, b]
    tile = Int32(1)
    while tile <= ntiles
        g += @inbounds partial_grads[pidx, tile, b]
        tile += Int32(1)
    end
    @inbounds gradients[pidx, b] = g
    if pidx == 1
        v = @inbounds totals[b]
        tile = Int32(1)
        while tile <= ntiles
            v += @inbounds partial_totals[tile, b]
            tile += Int32(1)
        end
        @inbounds totals[b] = v
    end
end

# ---- launch --------------------------------------------------------------------

# Launch the fused device gradient against `inner.params_device` in place (no
# upload, no download, no synchronize): the working position is already resident.
# Uses the tiled path when the workspace flagged a tileable loop, else the serial
# `(P, C)` scan. Kernels launched on one backend execute in order, so the three
# tiled kernels need no interior synchronize; the caller synchronizes once.
function _device_launch_gradient!(inner::DeviceBatchedWorkspace)
    tg = inner.tiled_gradient
    if tg === nothing
        _device_gradient_kernel!(inner.backend)(
            inner.totals_device,
            inner.gradients_device,
            inner.plan,
            inner.params_device,
            inner.observed_device,
            inner.observed_int_device,
            inner.grad_slots_device,
            inner.trip_counts_device,
            inner.loop_starts_device;
            ndrange=(inner.parameter_count, inner.batch_size),
        )
    else
        _device_launch_tiled_gradient!(inner, tg)
    end
    return nothing
end

function _device_launch_tiled_gradient!(inner::DeviceBatchedWorkspace, tg::DeviceTiledGradient)
    be = inner.backend
    P = inner.parameter_count
    C = inner.batch_size
    # prelude: prior value/gradient into the final buffers + pre-loop bindings into
    # the gradient slot scratch the tiled body reads
    _device_gradient_kernel!(be)(
        inner.totals_device,
        inner.gradients_device,
        tg.prelude_plan,
        inner.params_device,
        inner.observed_device,
        inner.observed_int_device,
        inner.grad_slots_device,
        inner.trip_counts_device,
        inner.loop_starts_device;
        ndrange=(P, C),
    )
    # observation-parallel body partials
    _device_tiled_gradient_partials_kernel!(be)(
        inner.tile_partial_totals,
        inner.tile_partial_grads,
        tg.body_plan,
        inner.params_device,
        inner.observed_device,
        inner.observed_int_device,
        inner.grad_slots_device,
        inner.trip_counts_device,
        inner.loop_starts_device,
        tg.base_cursor,
        tg.stride,
        tg.n_obs,
        tg.iters_per_tile;
        ndrange=(P, Int(tg.ntiles), C),
    )
    # fold the tiles into the prelude value/gradient
    _device_tiled_gradient_reduce_kernel!(be)(
        inner.gradients_device,
        inner.totals_device,
        inner.tile_partial_grads,
        inner.tile_partial_totals,
        tg.ntiles;
        ndrange=(P, C),
    )
    return nothing
end

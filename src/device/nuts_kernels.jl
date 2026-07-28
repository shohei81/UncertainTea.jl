# Device-resident masked batched NUTS.
#
# `batched_nuts(...; tree_strategy=:masked, backend=...)` runs the mask-based
# iterative-doubling trajectory (masked_doubling.jl) device-resident: all the
# per-leaf P x C arrays -- positions, momenta, gradients, and the dyadic U-turn
# checkpoints -- stay on the device across the whole doubling trajectory, and
# only O(num_chains) vectors (bit masks + per-chain energies/logjoints) cross the
# host<->device bus per leaf step and per merge. The host keeps the RNG draws (in
# exactly the CPU masked path's order) and the O(C) scalar bookkeeping (log
# weights, energies, accept stats, selection/turning/divergence/active masks,
# tree depths, integration steps).
#
# Only the doubling ROUND loop is on the device. The per-iteration trajectory
# initialization (`_initialize_batched_nuts_continuations!`) and finalization
# (`_finalize_batched_nuts_proposals!`) reuse the host code verbatim -- they run
# once per outer iteration, so their P x C transfers (one upload of the
# continuation frontier before the rounds, one download of the accepted proposal
# after) do not violate the O(C)-per-round budget; they mirror the once-per-
# iteration position/gradient download the device HMC loop already does during
# warmup.
#
# Results are statistically equivalent to the host masked path, and -- with
# adaptation OFF at a fixed step size -- numerically identical to it on the CPU()
# reference backend at Float64 (the RNG draw order and reduction order are
# preserved; the only residual is the fused device gradient's ~1e-16 disagreement
# with the host gradient cache, which flips no accept decision). With step-size
# adaptation, dual averaging amplifies that 1e-16 difference, so the adaptive path
# is only statistically equivalent. `test/uncertaintea/core/device_masked_nuts.jl` checks both.
# Everything is `T`-generic (diagonal mass only), so the CPU reference backend and
# a GPU backend (e.g. Metal at Float32) share one path.

# ---- kernels -------------------------------------------------------------------
# The Hamiltonian, final-step validity, and column-accept kernels are shared with
# device HMC (`hmc_kernels.jl`); only the direction-aware leaf integrator, the
# masked column copies, the checkpoint store, and the two U-turn reductions are
# new here.

# Direction-aware momentum kick: p += sign[b] * factor * grad, over active chains.
@kernel function _device_nuts_kick!(p, @Const(grad), @Const(active), @Const(sign), factor)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(active[b]) != 0x00
        @inbounds p[pidx, b] += sign[b] * factor * grad[pidx, b]
    end
end

# Direction-aware position drift, over active chains. The arithmetic must MATCH the
# host integrator's association bit-for-bit -- `signed_step * (inverse_mass * p)`
# with `signed_step = direction * step` -- because floating-point multiply is not
# associative and a 1-ulp energy difference amplifies through dual-averaging step
# adaptation into a macroscopic trajectory divergence (see device_masked_nuts.jl's tight oracle).
@kernel function _device_nuts_drift!(q, @Const(p), @Const(inverse_mass), @Const(active), @Const(sign), step)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(active[b]) != 0x00
        signed = sign[b] * step
        @inbounds q[pidx, b] += signed * (inverse_mass[pidx] * p[pidx, b])
    end
end

# Per-chain-step variants (issue #137): the step size is a C-length device vector
# `step[b]` instead of a scalar. `scale` folds the leapfrog half-factor into the
# kick (0.5 for a half-kick). The shared `inverse_mass` P-vector is unchanged, and
# the `signed_step * (inverse_mass * p)` association is preserved per chain. These
# power ONLY the pooled-mass / per-chain-step device path; the scalar kernels above
# still drive the bitwise-oracle-pinned shared path untouched.
@kernel function _device_nuts_kick_perchain!(p, @Const(grad), @Const(active), @Const(sign), @Const(step), scale)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(active[b]) != 0x00
        @inbounds p[pidx, b] += sign[b] * (scale * step[b]) * grad[pidx, b]
    end
end

@kernel function _device_nuts_drift_perchain!(q, @Const(p), @Const(inverse_mass), @Const(active), @Const(sign), @Const(step))
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(active[b]) != 0x00
        signed = sign[b] * step[b]
        @inbounds q[pidx, b] += signed * (inverse_mass[pidx] * p[pidx, b])
    end
end

# Masked column copy of a (position, momentum, gradient) triple: for each chain
# whose UInt8 mask is set, copy the whole column from source to destination.
@kernel function _device_nuts_copy_columns!(
    dest_position, dest_momentum, dest_gradient,
    @Const(src_position), @Const(src_momentum), @Const(src_gradient), @Const(mask),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(mask[b]) != 0x00
        @inbounds dest_position[pidx, b] = src_position[pidx, b]
        @inbounds dest_momentum[pidx, b] = src_momentum[pidx, b]
        @inbounds dest_gradient[pidx, b] = src_gradient[pidx, b]
    end
end

# Unconditional full column copy of a (position, momentum, gradient) triple.
@kernel function _device_nuts_copy_columns_all!(
    dest_position, dest_momentum, dest_gradient,
    @Const(src_position), @Const(src_momentum), @Const(src_gradient),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    @inbounds dest_position[pidx, b] = src_position[pidx, b]
    @inbounds dest_momentum[pidx, b] = src_momentum[pidx, b]
    @inbounds dest_gradient[pidx, b] = src_gradient[pidx, b]
end

# Store the current leaf (position, momentum) into checkpoint slot `slot` for each
# masked chain. `checkpoint` is laid out `parameter_count x (max_tree_depth+1) x C`.
@kernel function _device_nuts_store_checkpoint!(
    checkpoint_position, checkpoint_momentum,
    @Const(current_position), @Const(current_momentum), @Const(mask), slot::Int,
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(mask[b]) != 0x00
        @inbounds checkpoint_position[pidx, slot, b] = current_position[pidx, b]
        @inbounds checkpoint_momentum[pidx, slot, b] = current_momentum[pidx, b]
    end
end

# Odd-leaf dyadic U-turn test for one dyadic block ending at the current leaf: for
# each masked chain, compare the block's start checkpoint (slot `slot`) against the
# current endpoint. `sign[b] > 0` orients (checkpoint -> current); otherwise the
# arguments swap. Velocities are metric-aware (M^{-1} p, diagonal `inverse_mass`),
# mirroring the host `_is_turning`/`_turning_velocity_dot`. ORs a turn into
# `turning` (never clears it), so the host can fold multiple blocks by launching
# this once per block over a zeroed `turning`.
@kernel function _device_nuts_dyadic_turning!(
    turning,
    @Const(checkpoint_position), @Const(checkpoint_momentum),
    @Const(current_position), @Const(current_momentum),
    @Const(mask), @Const(sign), @Const(inverse_mass), slot::Int, num_params::Int,
)
    b = @index(Global)
    if @inbounds(mask[b]) != 0x00
        left_dot = zero(eltype(current_position))
        right_dot = zero(eltype(current_position))
        forward = @inbounds(sign[b]) > 0
        for pidx = 1:num_params
            cp = @inbounds checkpoint_position[pidx, slot, b]
            cm = @inbounds checkpoint_momentum[pidx, slot, b]
            qp = @inbounds current_position[pidx, b]
            qm = @inbounds current_momentum[pidx, b]
            im = @inbounds inverse_mass[pidx]
            # forward: left=checkpoint, right=current; backward: swap.
            if forward
                delta = qp - cp
                left_dot += delta * im * cm
                right_dot += delta * im * qm
            else
                delta = cp - qp
                left_dot += delta * im * qm
                right_dot += delta * im * cm
            end
        end
        if left_dot <= 0 || right_dot <= 0
            @inbounds turning[b] = 0x01
        end
    end
end

# Whole-trajectory (merge-level) U-turn over the continuation frontier: for each
# active chain, delta = right - left, turn if delta.M^{-1}left_mom <= 0 ||
# delta.M^{-1}right_mom <= 0 (metric-aware, mirrors host _batched_is_turning!).
@kernel function _device_nuts_frontier_turning!(
    turning,
    @Const(left_position), @Const(right_position),
    @Const(left_momentum), @Const(right_momentum),
    @Const(active), @Const(inverse_mass), num_params::Int,
)
    b = @index(Global)
    if @inbounds(active[b]) != 0x00
        left_dot = zero(eltype(left_position))
        right_dot = zero(eltype(left_position))
        for pidx = 1:num_params
            delta = @inbounds(right_position[pidx, b]) - @inbounds(left_position[pidx, b])
            im = @inbounds inverse_mass[pidx]
            left_dot += delta * im * @inbounds(left_momentum[pidx, b])
            right_dot += delta * im * @inbounds(right_momentum[pidx, b])
        end
        @inbounds turning[b] = (left_dot <= 0 || right_dot <= 0) ? 0x01 : 0x00
    else
        @inbounds turning[b] = 0x00
    end
end

# Per-chain kinetic energy: kinetic[b] = 0.5 * sum_p (momentum^2 * inverse_mass).
@kernel function _device_nuts_kinetic!(kinetic, @Const(momentum), @Const(inverse_mass), num_params::Int)
    b = @index(Global)
    acc = zero(eltype(kinetic))
    for pidx = 1:num_params
        m = @inbounds momentum[pidx, b]
        acc += m * m * @inbounds(inverse_mass[pidx])
    end
    @inbounds kinetic[b] = acc / 2
end

# Per-chain "did the proposal move off the current state" flag, computed in the
# backend's OWN precision. This mirrors the host `_batched_positions_moved!`
# (exact column inequality) but compares the device proposal against a device copy
# of the current position, so a no-move proposal is NOT spuriously flagged as moved
# by a lower-precision (e.g. Float32) host<->device round-trip.
@kernel function _device_nuts_moved!(moved, @Const(proposal_position), @Const(current_position), num_params::Int)
    b = @index(Global)
    is_moved = 0x00
    for pidx = 1:num_params
        if @inbounds(proposal_position[pidx, b]) != @inbounds(current_position[pidx, b])
            is_moved = 0x01
        end
    end
    @inbounds moved[b] = is_moved
end

# ---- device NUTS workspace -----------------------------------------------------

# Device buffers for the masked doubling round loop. Wraps a `DeviceBatchedWorkspace`
# whose `params_device`/`gradients_device`/`totals_device` are the leaf `tree_next`
# position/gradient/logjoint scratch (as device HMC uses them). Diagonal mass only.
mutable struct DeviceNUTSWorkspace{T,B<:KernelAbstractions.Backend}
    inner::DeviceBatchedWorkspace{T}
    backend::B
    num_params::Int
    num_chains::Int
    max_tree_depth::Int
    inverse_mass::Any        # P
    sign::Any                # C   per-chain direction (+/-1) as T
    step::Any                # C   per-chain step size (pooled-mass/per-chain-step mode)
    working_momentum::Any    # P x C   leaf tree_next momentum (p)
    # subtree buffers
    tree_current_position::Any
    tree_current_momentum::Any
    tree_current_gradient::Any
    tree_left_position::Any
    tree_left_momentum::Any
    tree_left_gradient::Any
    tree_right_position::Any
    tree_right_momentum::Any
    tree_right_gradient::Any
    tree_proposal_position::Any
    tree_proposal_momentum::Any
    tree_proposal_gradient::Any
    # continuation frontier buffers
    left_position::Any
    left_momentum::Any
    left_gradient::Any
    right_position::Any
    right_momentum::Any
    right_gradient::Any
    proposal_position::Any
    proposal_momentum::Any
    proposal_gradient::Any
    # dyadic checkpoints: P x (max_tree_depth+1) x C
    checkpoint_position::Any
    checkpoint_momentum::Any
    # per-chain device scratch (C)
    valid::Any               # UInt8
    active::Any              # UInt8
    turning::Any             # UInt8
    proposed_energy::Any     # T
    kinetic::Any             # T
    mask_a::Any              # UInt8 generic mask upload buffer
    mask_b::Any              # UInt8
    mask_c::Any              # UInt8
    # host staging (avoid per-call allocation)
    host_u8::Vector{UInt8}
    host_energy::Vector{T}
    sign_host::Vector{T}
    step_host::Vector{T}
    inverse_mass_host::Vector{T}
    kinetic_host::Vector{Float64}
    advanced_scratch::Vector{Bool}
    checkpoint_scratch::Vector{Bool}
    # Movement detection (fix: precision-robust `accepted_step`).
    current_position::Any    # P x C  device copy of the current (pre-trajectory) position
    moved::Any               # C UInt8  per-chain moved flag (device)
    # Reusable P x C host staging buffer for the once-per-iteration frontier
    # upload / accepted-proposal download, so those transfers do not allocate.
    host_mat::Matrix{T}
    # Reusable P x C Float64 buffer holding the SHARED diagonal mass broadcast into
    # per-chain columns, for the pooled-mass / per-chain-step host init (whose
    # per-chain leapfrog overload requires a per-chain mass matrix). The device
    # rounds still consume the shared `inverse_mass` P-vector.
    inverse_mass_cols::Matrix{Float64}
end

function DeviceNUTSWorkspace(
    model::TeaModel,
    num_chains::Integer,
    max_tree_depth::Integer;
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=Float64,
    args=(),
    constraints=choicemap(),
)
    inner = DeviceBatchedWorkspace(
        model, num_chains; backend=backend, precision=precision, args=args, constraints=constraints,
    )
    _device_ensure_gradient_buffers!(inner)
    T = precision
    P = inner.parameter_count
    C = inner.batch_size
    D = Int(max_tree_depth)
    # Zero-initialize every P x C device buffer: a chain that is never active in any
    # round (e.g. its initial one-step trajectory diverged/turned while others
    # continue) never has its `tree_current_*` column written, yet the unmasked leaf
    # gradient runs over all columns -- so an uninitialized column would feed garbage
    # (and potentially NaN/FP noise) into the gradient. Zeros are a valid finite
    # UNCONSTRAINED position, so those ignored lanes stay finite and harmless.
    mat() = fill!(KernelAbstractions.allocate(backend, T, P, C), zero(T))
    vecT() = KernelAbstractions.allocate(backend, T, C)
    vecU8() = KernelAbstractions.allocate(backend, UInt8, C)
    ckpt() = fill!(KernelAbstractions.allocate(backend, T, P, max(D + 1, 1), C), zero(T))
    return DeviceNUTSWorkspace{T,typeof(backend)}(
        inner, backend, P, C, D,
        fill!(KernelAbstractions.allocate(backend, T, P), zero(T)),
        vecT(),
        vecT(),
        mat(),
        mat(), mat(), mat(),
        mat(), mat(), mat(),
        mat(), mat(), mat(),
        mat(), mat(), mat(),
        mat(), mat(), mat(),
        mat(), mat(), mat(),
        mat(), mat(), mat(),
        ckpt(), ckpt(),
        vecU8(), vecU8(), vecU8(),
        vecT(), vecT(),
        vecU8(), vecU8(), vecU8(),
        Vector{UInt8}(undef, C),
        Vector{T}(undef, C),
        Vector{T}(undef, C),
        Vector{T}(undef, C),
        Vector{T}(undef, P),
        Vector{Float64}(undef, C),
        Vector{Bool}(undef, C),
        Vector{Bool}(undef, C),
        mat(),
        vecU8(),
        Matrix{T}(undef, P, C),
        Matrix{Float64}(undef, P, C),
    )
end

# ---- host<->device transfer helpers -------------------------------------------

_upload_mask!(dev, host_bits::AbstractVector{Bool}, stage::Vector{UInt8}) = begin
    @inbounds for i in eachindex(host_bits)
        stage[i] = host_bits[i] ? 0x01 : 0x00
    end
    copyto!(dev, stage)
    return dev
end

_download_bits!(host_bits::AbstractVector{Bool}, dev, stage::Vector{UInt8}) = begin
    copyto!(stage, dev)
    @inbounds for i in eachindex(host_bits)
        host_bits[i] = stage[i] != 0x00
    end
    return host_bits
end

_download_bits_or!(host_bits::AbstractVector{Bool}, dev, stage::Vector{UInt8}) = begin
    copyto!(stage, dev)
    @inbounds for i in eachindex(host_bits)
        host_bits[i] |= stage[i] != 0x00
    end
    return host_bits
end

_download_reals!(host::AbstractVector{Float64}, dev, stage::Vector) = begin
    copyto!(stage, dev)
    @inbounds for i in eachindex(host)
        host[i] = Float64(stage[i])
    end
    return host
end

# Upload a host P x C Float64 matrix into a device T matrix (converts precision)
# through a caller-owned `stage::Matrix{T}` buffer so the transfer does not allocate.
_upload_matrix!(dev, host::AbstractMatrix{Float64}, stage::Matrix) = begin
    @inbounds for i in eachindex(host, stage)
        stage[i] = host[i]
    end
    copyto!(dev, stage)
    return dev
end
# Download a device T matrix into a host P x C Float64 matrix through `stage`.
_download_matrix!(host::AbstractMatrix{Float64}, dev, stage::Matrix) = begin
    copyto!(stage, dev)
    @inbounds for i in eachindex(host, stage)
        host[i] = Float64(stage[i])
    end
    return host
end

# ---- device leaf leapfrog ------------------------------------------------------

# One masked leapfrog leaf from `tree_current` in each chain's `sign` direction,
# leaving the leaf in (params_device, working_momentum, gradients_device, totals_device)
# and refreshing `valid`/`proposed_energy`. Mirrors `batched_leapfrog_step_to!` for a
# single step (initial half-kick, drift, gradient, closing half-kick; no flip).
# Downloads only `proposed_energy` (C) + `valid` (C).
function _device_nuts_leaf!(dws::DeviceNUTSWorkspace{T}, ws, step_size::Real) where {T}
    be = dws.backend
    P = dws.num_params
    C = dws.num_chains
    inner = dws.inner
    q = inner.params_device
    grad = inner.gradients_device
    logj = inner.totals_device
    p = dws.working_momentum
    h = convert(T, step_size)
    half = convert(T, step_size / 2)

    _upload_mask!(dws.active, ws.subtree_active, dws.host_u8)
    copyto!(dws.valid, dws.active)
    _device_nuts_copy_columns_all!(be)(
        q,
        p,
        grad,
        dws.tree_current_position,
        dws.tree_current_momentum,
        dws.tree_current_gradient;
        ndrange=(P, C),
    )
    _device_nuts_kick!(be)(p, dws.tree_current_gradient, dws.active, dws.sign, half; ndrange=(P, C))
    _device_nuts_drift!(be)(q, p, dws.inverse_mass, dws.active, dws.sign, h; ndrange=(P, C))
    _device_launch_gradient!(inner)
    _device_hmc_validity_update_final!(be)(dws.valid, grad, logj, P; ndrange=C)
    _device_nuts_kick!(be)(p, grad, dws.valid, dws.sign, half; ndrange=(P, C))
    _device_hmc_hamiltonian!(be)(dws.proposed_energy, p, dws.inverse_mass, logj, P; ndrange=C)
    KernelAbstractions.synchronize(be)

    _download_bits!(ws.control.step_valid, dws.valid, dws.host_u8)
    _download_reals!(ws.subtree_proposed_energy, dws.proposed_energy, dws.host_energy)
    _download_reals!(ws.proposed_logjoint, logj, dws.host_energy)
    return dws
end

# Per-chain-step leaf (issue #137): identical to `_device_nuts_leaf!` above except
# the half-kick / drift use the per-chain `dws.step` device vector instead of a
# scalar. Selected by passing `nothing` as the step to the doubling round. The
# shared `dws.inverse_mass` P-vector is unchanged.
function _device_nuts_leaf!(dws::DeviceNUTSWorkspace{T}, ws, ::Nothing) where {T}
    be = dws.backend
    P = dws.num_params
    C = dws.num_chains
    inner = dws.inner
    q = inner.params_device
    grad = inner.gradients_device
    logj = inner.totals_device
    p = dws.working_momentum
    half = convert(T, 0.5)

    _upload_mask!(dws.active, ws.subtree_active, dws.host_u8)
    copyto!(dws.valid, dws.active)
    _device_nuts_copy_columns_all!(be)(
        q,
        p,
        grad,
        dws.tree_current_position,
        dws.tree_current_momentum,
        dws.tree_current_gradient;
        ndrange=(P, C),
    )
    _device_nuts_kick_perchain!(be)(p, dws.tree_current_gradient, dws.active, dws.sign, dws.step, half; ndrange=(P, C))
    _device_nuts_drift_perchain!(be)(q, p, dws.inverse_mass, dws.active, dws.sign, dws.step; ndrange=(P, C))
    _device_launch_gradient!(inner)
    _device_hmc_validity_update_final!(be)(dws.valid, grad, logj, P; ndrange=C)
    _device_nuts_kick_perchain!(be)(p, grad, dws.valid, dws.sign, dws.step, half; ndrange=(P, C))
    _device_hmc_hamiltonian!(be)(dws.proposed_energy, p, dws.inverse_mass, logj, P; ndrange=C)
    KernelAbstractions.synchronize(be)

    _download_bits!(ws.control.step_valid, dws.valid, dws.host_u8)
    _download_reals!(ws.subtree_proposed_energy, dws.proposed_energy, dws.host_energy)
    _download_reals!(ws.proposed_logjoint, logj, dws.host_energy)
    return dws
end

# ---- device round-loop stages (mirror the host masked cohort) ------------------

# Mirror `_initialize_batched_nuts_subtree_states!`: seed tree_current/left/right/
# proposal from the continuation frontier chosen per chain by direction.
function _device_initialize_subtree_states!(dws::DeviceNUTSWorkspace{T}, ws, active::AbstractVector{Bool}) where {T}
    be = dws.backend
    P = dws.num_params
    C = dws.num_chains
    fill!(ws.subtree_copy_left, false)
    fill!(ws.subtree_copy_right, false)
    fill!(ws.subtree_select_proposal, false)
    @inbounds for c in eachindex(active)
        active[c] || continue
        if ws.control.step_direction[c] < 0
            ws.subtree_copy_left[c] = true
            start_logjoint = ws.left_logjoint[c]
        else
            ws.subtree_copy_right[c] = true
            start_logjoint = ws.right_logjoint[c]
        end
        ws.tree_current_logjoint[c] = start_logjoint
        ws.tree_left_logjoint[c] = start_logjoint
        ws.tree_right_logjoint[c] = start_logjoint
        ws.tree_proposal_logjoint[c] = start_logjoint
    end
    _upload_mask!(dws.mask_a, ws.subtree_copy_left, dws.host_u8)
    _upload_mask!(dws.mask_b, ws.subtree_copy_right, dws.host_u8)
    # tree_current / tree_left / tree_right / tree_proposal <- left (copy_left) or right (copy_right)
    for (dp, dm, dg) in (
        (dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient),
        (dws.tree_left_position, dws.tree_left_momentum, dws.tree_left_gradient),
        (dws.tree_right_position, dws.tree_right_momentum, dws.tree_right_gradient),
        (dws.tree_proposal_position, dws.tree_proposal_momentum, dws.tree_proposal_gradient),
    )
        _device_nuts_copy_columns!(be)(
            dp,
            dm,
            dg,
            dws.left_position,
            dws.left_momentum,
            dws.left_gradient,
            dws.mask_a;
            ndrange=(P, C),
        )
        _device_nuts_copy_columns!(be)(
            dp,
            dm,
            dg,
            dws.right_position,
            dws.right_momentum,
            dws.right_gradient,
            dws.mask_b;
            ndrange=(P, C),
        )
    end
    KernelAbstractions.synchronize(be)
    return dws
end

# Mirror `_advance_batched_nuts_subtree_cohort!`: host leaf-advance arithmetic +
# device accept-copy / checkpoint / dyadic turning / frontier scatter. Returns
# whether any chain is still expanding.
function _device_advance_cohort_impl!(dws::DeviceNUTSWorkspace{T}, ws, max_delta_energy::Float64, rng::AbstractRNG) where {T}
    be = dws.backend
    P = dws.num_params
    C = dws.num_chains
    fill!(ws.subtree_copy_left, false)
    fill!(ws.subtree_copy_right, false)
    fill!(ws.subtree_select_proposal, false)
    # ws.subtree_turning is NOT cleared here: it is reset once per round by
    # _reset_batched_nuts_subtree_scratch! and stays sticky across leaf steps
    # (mirrors _advance_batched_nuts_subtree_cohort!) so the merge gate can
    # discard a subtree that U-turned on any earlier leaf.
    advanced = dws.advanced_scratch
    checkpoint = dws.checkpoint_scratch
    fill!(advanced, false)
    fill!(checkpoint, false)
    leaf_index = -1
    @inbounds for c = 1:C
        ws.subtree_active[c] || continue
        if !ws.control.step_valid[c]
            ws.subtree_divergent[c] = true
            ws.subtree_active[c] = false
            continue
        end
        ws.tree_current_logjoint[c] = ws.proposed_logjoint[c]
        advanced[c] = true
        ws.subtree_integration_steps[c] += 1
        if ws.control.step_direction[c] < 0
            ws.subtree_copy_left[c] = true
            ws.tree_left_logjoint[c] = ws.tree_current_logjoint[c]
        else
            ws.subtree_copy_right[c] = true
            ws.tree_right_logjoint[c] = ws.tree_current_logjoint[c]
        end
        leaf = _advance_tree_leaf(
            ws.subtree_proposed_energy[c],
            ws.current_energy[c],
            max_delta_energy,
            ws.subtree_log_weight[c],
            rng,
        )
        ws.subtree_delta_energy[c] = leaf.delta_energy
        if leaf.divergent
            ws.subtree_divergent[c] = true
            ws.subtree_active[c] = false
            continue
        end
        ws.subtree_accept_prob[c] = leaf.accept_prob
        ws.subtree_accept_stat_sum[c] += leaf.accept_prob
        ws.subtree_accept_stat_count[c] += 1
        ws.subtree_candidate_log_weight[c] = leaf.candidate_log_weight
        ws.subtree_combined_log_weight[c] = leaf.combined_log_weight
        if leaf.select_proposal
            ws.subtree_select_proposal[c] = true
            ws.tree_proposal_logjoint[c] = ws.tree_current_logjoint[c]
            ws.subtree_proposal_energy[c] = ws.subtree_proposed_energy[c]
            ws.subtree_proposal_energy_error[c] = leaf.delta_energy
        end
        ws.subtree_log_weight[c] = leaf.combined_log_weight
        leaf_index = ws.subtree_integration_steps[c] - 1
        checkpoint[c] = true
    end

    # device: accept-copy tree_current <- tree_next for advanced chains.
    _upload_mask!(dws.mask_a, advanced, dws.host_u8)
    _device_nuts_copy_columns!(be)(
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.inner.params_device, dws.working_momentum, dws.inner.gradients_device, dws.mask_a; ndrange=(P, C),
    )

    # device: checkpoint store (even leaf) or dyadic turning (odd leaf).
    if leaf_index >= 0
        _upload_mask!(dws.mask_b, checkpoint, dws.host_u8)
        if iseven(leaf_index)
            slot = count_ones(leaf_index) + 1
            _device_nuts_store_checkpoint!(be)(
                dws.checkpoint_position, dws.checkpoint_momentum,
                dws.tree_current_position, dws.tree_current_momentum, dws.mask_b, slot; ndrange=(P, C),
            )
        else
            fill!(dws.turning, 0x00)
            for k = 1:trailing_ones(leaf_index)
                block_start = leaf_index - (1 << k) + 1
                slot = count_ones(block_start) + 1
                _device_nuts_dyadic_turning!(be)(
                    dws.turning, dws.checkpoint_position, dws.checkpoint_momentum,
                    dws.tree_current_position, dws.tree_current_momentum, dws.mask_b, dws.sign,
                    dws.inverse_mass, slot, P; ndrange=C,
                )
            end
            KernelAbstractions.synchronize(be)
            _download_bits_or!(ws.subtree_turning, dws.turning, dws.host_u8)
        end
    end

    # device: scatter tree_left/right/proposal <- tree_current.
    _upload_mask!(dws.mask_a, ws.subtree_copy_left, dws.host_u8)
    _upload_mask!(dws.mask_b, ws.subtree_copy_right, dws.host_u8)
    _upload_mask!(dws.mask_c, ws.subtree_select_proposal, dws.host_u8)
    _device_nuts_copy_columns!(be)(
        dws.tree_left_position,
        dws.tree_left_momentum,
        dws.tree_left_gradient,
        dws.tree_current_position,
        dws.tree_current_momentum,
        dws.tree_current_gradient,
        dws.mask_a;
        ndrange=(P, C),
    )
    _device_nuts_copy_columns!(be)(
        dws.tree_right_position,
        dws.tree_right_momentum,
        dws.tree_right_gradient,
        dws.tree_current_position,
        dws.tree_current_momentum,
        dws.tree_current_gradient,
        dws.mask_b;
        ndrange=(P, C),
    )
    _device_nuts_copy_columns!(be)(
        dws.tree_proposal_position,
        dws.tree_proposal_momentum,
        dws.tree_proposal_gradient,
        dws.tree_current_position,
        dws.tree_current_momentum,
        dws.tree_current_gradient,
        dws.mask_c;
        ndrange=(P, C),
    )
    KernelAbstractions.synchronize(be)

    # fold turning into subtree_active.
    any_active = false
    @inbounds for c = 1:C
        ws.subtree_active[c] = ws.subtree_active[c] && !ws.subtree_turning[c]
        any_active |= ws.subtree_active[c]
    end
    return any_active
end

# Mirror `_merge_batched_nuts_subtree_cohort!` (+ continuation-frontier merge).
function _device_merge_cohort!(dws::DeviceNUTSWorkspace{T}, ws, rng::AbstractRNG) where {T}
    be = dws.backend
    P = dws.num_params
    C = dws.num_chains

    @inbounds for c = 1:C
        ws.subtree_active[c] && continue
        ws.continuation_select_proposal[c] = false
    end

    # merge continuation frontiers: left <- tree_left (copy_left), right <- tree_right (copy_right)
    fill!(ws.subtree_copy_left, false)
    fill!(ws.subtree_copy_right, false)
    @inbounds for c = 1:C
        ws.subtree_active[c] || continue
        if ws.control.step_direction[c] < 0
            ws.subtree_copy_left[c] = true
            ws.left_logjoint[c] = ws.tree_left_logjoint[c]
        else
            ws.subtree_copy_right[c] = true
            ws.right_logjoint[c] = ws.tree_right_logjoint[c]
        end
    end
    _upload_mask!(dws.mask_a, ws.subtree_copy_left, dws.host_u8)
    _upload_mask!(dws.mask_b, ws.subtree_copy_right, dws.host_u8)
    _device_nuts_copy_columns!(be)(
        dws.left_position,
        dws.left_momentum,
        dws.left_gradient,
        dws.tree_left_position,
        dws.tree_left_momentum,
        dws.tree_left_gradient,
        dws.mask_a;
        ndrange=(P, C),
    )
    _device_nuts_copy_columns!(be)(
        dws.right_position,
        dws.right_momentum,
        dws.right_gradient,
        dws.tree_right_position,
        dws.tree_right_momentum,
        dws.tree_right_gradient,
        dws.mask_b;
        ndrange=(P, C),
    )

    # merge-level whole-trajectory turning.
    _upload_mask!(dws.mask_c, ws.subtree_active, dws.host_u8)
    _device_nuts_frontier_turning!(be)(
        dws.turning,
        dws.left_position,
        dws.right_position,
        dws.left_momentum,
        dws.right_momentum,
        dws.mask_c,
        dws.inverse_mass,
        P;
        ndrange=C,
    )
    # kinetic of the tree proposal momentum (for selected chains' proposal energy).
    _device_nuts_kinetic!(be)(dws.kinetic, dws.tree_proposal_momentum, dws.inverse_mass, P; ndrange=C)
    KernelAbstractions.synchronize(be)
    _download_bits!(ws.subtree_merged_turning, dws.turning, dws.host_u8)
    _download_reals!(dws.kinetic_host, dws.kinetic, dws.host_energy)

    @inbounds for c = 1:C
        ws.subtree_active[c] || continue
        merge = _merge_subtree_stats(ws.continuation_log_weight[c], ws.subtree_log_weight[c], rng)
        ws.continuation_select_proposal[c] = merge.select_proposal
        ws.continuation_candidate_log_weight[c] = merge.candidate_log_weight
        ws.continuation_combined_log_weight[c] = merge.combined_log_weight
        if merge.select_proposal
            proposal_energy = dws.kinetic_host[c] - ws.tree_proposal_logjoint[c]
            ws.subtree_proposal_energy[c] = proposal_energy
            ws.subtree_proposal_energy_error[c] = proposal_energy - ws.current_energy[c]
        end
        _merge_batched_subtree_summary!(ws, c)
    end

    # device: final continuation proposal <- tree proposal for selected chains.
    _upload_mask!(dws.mask_a, ws.continuation_select_proposal, dws.host_u8)
    _device_nuts_copy_columns!(be)(
        dws.proposal_position,
        dws.proposal_momentum,
        dws.proposal_gradient,
        dws.tree_proposal_position,
        dws.tree_proposal_momentum,
        dws.tree_proposal_gradient,
        dws.mask_a;
        ndrange=(P, C),
    )
    KernelAbstractions.synchronize(be)
    @inbounds for c = 1:C
        ws.continuation_select_proposal[c] || continue
        ws.proposed_logjoint[c] = ws.continuation_proposal_logjoint[c]
    end
    return dws
end

# Mirror `_masked_nuts_doubling_round!`.
function _device_masked_nuts_doubling_round!(
    dws::DeviceNUTSWorkspace{T},
    ws,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    step_size,
    rng::AbstractRNG,
) where {T}
    _reset_batched_nuts_subtree_scratch!(ws)
    _update_batched_nuts_continuation_active!(ws, max_tree_depth) || return false
    round_active = ws.control.scheduler.continuation_active
    round_depth = 0
    @inbounds for c in eachindex(round_active)
        round_active[c] || continue
        round_depth = max(round_depth, ws.control.tree_depths[c])
    end
    copyto!(ws.subtree_active, round_active)
    copyto!(ws.control.scheduler.subtree_started, round_active)
    @inbounds for c in eachindex(ws.control.step_direction)
        ws.control.step_direction[c] = _sample_nuts_direction(rng)
        dws.sign_host[c] = convert(T, ws.control.step_direction[c])
    end
    copyto!(dws.sign, dws.sign_host)
    _device_initialize_subtree_states!(dws, ws, ws.subtree_active)

    any_expanding = true
    for _ = 1:(1<<round_depth)
        any_expanding || break
        _device_nuts_leaf!(dws, ws, step_size)
        any_expanding = _device_advance_cohort_impl!(dws, ws, max_delta_energy, rng)
    end

    fill!(ws.subtree_active, false)
    any_merging = false
    @inbounds for c in eachindex(round_active)
        round_active[c] || continue
        ws.control.tree_depths[c] += 1
        if ws.subtree_integration_steps[c] == 0
            ws.control.divergent_step[c] = ws.subtree_divergent[c]
        elseif ws.subtree_turning[c] || ws.subtree_divergent[c]
            # Invalid subtree: canonical NUTS discards the whole doubling
            # (mirrors _masked_nuts_doubling_round!).
            _discard_invalid_batched_subtree!(ws, c)
        else
            ws.subtree_active[c] = true
            any_merging = true
        end
    end
    if any_merging
        _device_merge_cohort!(dws, ws, rng)
    end
    return true
end

# ---- device masked-NUTS proposal generator ------------------------------------

# Fills the host `BatchedNUTSWorkspace` (ws) proposal outputs for one iteration,
# device-resident. Reuses the host init/finalize verbatim; only the doubling
# rounds run on the device.
function _device_batched_nuts_proposals_masked!(
    dws::DeviceNUTSWorkspace{T},
    ws::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    current_logjoint::AbstractVector{Float64},
    current_gradient::AbstractMatrix{Float64},
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    rng::AbstractRNG,
) where {T}
    # init on host (one host gradient + RNG draws in the CPU masked path order).
    # `step_size` may be a scalar (shared adaptation) or a C-length per-chain vector
    # (pooled-mass / per-chain-step mode, issue #137); the host init/first-step and
    # the device round loop both index it per chain when it is a vector. With a
    # per-chain step the host per-chain leapfrog overload requires a per-chain mass
    # matrix, so broadcast the SHARED diagonal into columns for the init (the device
    # rounds still consume the shared `inverse_mass` P-vector uploaded below).
    if step_size isa AbstractVector
        @inbounds for c in axes(dws.inverse_mass_cols, 2)
            for pidx in axes(dws.inverse_mass_cols, 1)
                dws.inverse_mass_cols[pidx, c] = inverse_mass_matrix[pidx]
            end
        end
        init_mass = dws.inverse_mass_cols
    else
        init_mass = inverse_mass_matrix
    end
    _initialize_batched_nuts_continuations!(
        ws, model, position, current_logjoint, current_gradient,
        init_mass, args, constraints, step_size, max_delta_energy, rng,
    )

    # upload the continuation frontier + diagonal mass to the device.
    dws.inverse_mass_host .= convert.(T, inverse_mass_matrix)
    copyto!(dws.inverse_mass, dws.inverse_mass_host)
    # Per-chain step: stage the C-length step vector onto the device and signal the
    # round loop (via `round_step === nothing`) to run the per-chain-step leaf.
    round_step = step_size
    if step_size isa AbstractVector
        @inbounds for c in eachindex(step_size)
            dws.step_host[c] = convert(T, step_size[c])
        end
        copyto!(dws.step, dws.step_host)
        round_step = nothing
    end
    # A device-precision copy of the current position, for precision-robust movement
    # detection after the rounds (see the accepted_step override below).
    _upload_matrix!(dws.current_position, position, dws.host_mat)
    _upload_matrix!(dws.left_position, ws.left_position, dws.host_mat)
    _upload_matrix!(dws.left_momentum, ws.left_momentum, dws.host_mat)
    _upload_matrix!(dws.left_gradient, ws.left_gradient, dws.host_mat)
    _upload_matrix!(dws.right_position, ws.right_position, dws.host_mat)
    _upload_matrix!(dws.right_momentum, ws.right_momentum, dws.host_mat)
    _upload_matrix!(dws.right_gradient, ws.right_gradient, dws.host_mat)
    _upload_matrix!(dws.proposal_position, ws.proposal_position, dws.host_mat)
    _upload_matrix!(dws.proposal_momentum, ws.proposal_momentum, dws.host_mat)
    _upload_matrix!(dws.proposal_gradient, ws.proposal_gradient, dws.host_mat)

    while _device_masked_nuts_doubling_round!(dws, ws, max_tree_depth, max_delta_energy, round_step, rng)
    end

    # download the accepted continuation proposal for host finalize + recording.
    _download_matrix!(ws.proposal_position, dws.proposal_position, dws.host_mat)
    _download_matrix!(ws.proposal_momentum, dws.proposal_momentum, dws.host_mat)
    _download_matrix!(ws.proposal_gradient, dws.proposal_gradient, dws.host_mat)

    _finalize_batched_nuts_proposals!(ws, position)

    # Override `accepted_step` with a device-precision movement check. `_finalize`
    # (via `_batched_positions_moved!`) compares the DOWNLOADED proposal against the
    # Float64 host position; on a lower-precision backend the upload/download round
    # trip perturbs a genuine no-move proposal into a spurious "moved", corrupting the
    # accept diagnostic (and, downstream, letting the rounded copy overwrite the host
    # position). Comparing on-device -- proposal vs a device copy of the current
    # position, both in backend precision -- matches the host semantics exactly at
    # Float64 and stays correct at Float32.
    _device_nuts_moved!(dws.backend)(
        dws.moved, dws.proposal_position, dws.current_position, dws.num_params; ndrange=dws.num_chains,
    )
    KernelAbstractions.synchronize(dws.backend)
    _download_bits!(ws.control.accepted_step, dws.moved, dws.host_u8)
    return ws
end

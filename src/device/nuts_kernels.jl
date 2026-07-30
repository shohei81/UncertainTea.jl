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
# Two round-loop strategies (issue #152):
#   * `sync_per_leaf=true`  -- the Tier-1 order-preserving path: the per-leaf
#     accept/select bookkeeping stays on the host, so the RNG draw order and the
#     reduction order match the host masked path exactly and the CPU()-Float64
#     device-vs-host BITWISE oracle holds. One host round-trip per leaf.
#   * `sync_per_leaf=false` (DEFAULT, Tier 2) -- the async path: the round's RNG is
#     pre-drawn and uploaded once, the per-leaf accept/select/divergence/log-weight
#     and the dyadic-U-turn fold run device-side (`_device_nuts_advance!` +
#     `_device_masked_nuts_doubling_round_async!`), and the host reads the subtree
#     state back in ONE batch per doubling ROUND instead of per leaf. This departs
#     from the CPU RNG draw order, so it is only STATISTICALLY (not bitwise)
#     equivalent to the host masked path -- validated by SBC + posterior-moment
#     agreement + divergence-rate, not by the bitwise oracle.
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

# The direction-aware leaf integrator (kick + drift, shared step and per-chain
# step) and the full-column copy are FUSED into the leaf micro-kernels below
# (`_device_nuts_leaf_pre[_perchain]!` / `_device_nuts_leaf_post[_perchain]!`,
# issue #152) -- the standalone kick/drift/copy-all kernels are no longer launched.

# Masked column copy of a (position, momentum, gradient) triple: for each chain
# whose UInt8 mask is set, copy the whole column from source to destination. Still
# used for the single-mask accept-copy and final-proposal copy; the multi-mask
# scatters are fused (see `_device_nuts_scatter3!` / `_device_nuts_merge_copy!`).
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

# ---- lane-compaction gather / scatter (issue #160) -----------------------------
# The masked leaf gradient runs over ALL C columns even once most chains have
# finished/diverged (measured waste 60.8% at C=64, 68.3% at C=256). The gradient of
# column c depends ONLY on `params[:, c]` (columns fully independent), so once the
# active fraction drops below 50% the sync leaf gathers the active columns into a
# compact buffer, evaluates the gradient over just those `k` columns, and scatters
# the results back to the active lanes. See `_device_nuts_leaf_gradient!`.

# Gather: copy the `slot`-th active column (source column `index[slot]`) of `src`
# into the front of `dest`. One thread per (parameter, slot); `index[1:k]` is the
# host-built active->original map. Reads the whole P-row column so the gathered
# lanes score exactly as they would at their original lane.
@kernel function _device_nuts_gather_columns!(dest, @Const(src), @Const(index))
    idx = @index(Global, NTuple)
    pidx = idx[1]
    slot = idx[2]
    c = @inbounds index[slot]
    @inbounds dest[pidx, slot] = src[pidx, c]
end

# Scatter: write the compact gradient columns + per-column logjoint totals back to
# the active lanes `index[slot]`. Inactive lanes are intentionally left untouched --
# they are downstream don't-cares (the post-gradient leaf gates every read of
# grad/logjoint behind `active`/`valid`), so a stale inactive entry is equivalent to
# the full-width path overwriting it with a never-read value.
@kernel function _device_nuts_scatter_gradient!(
    grad, totals, @Const(compact_grad), @Const(compact_totals), @Const(index),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    slot = idx[2]
    c = @inbounds index[slot]
    @inbounds grad[pidx, c] = compact_grad[pidx, slot]
    if pidx == 1
        @inbounds totals[c] = compact_totals[slot]
    end
end

# ---- fused leaf micro-kernels (issue #152 Tier 1) ------------------------------
# These fuse the sequences of per-element micro-kernels in `_device_nuts_leaf!`
# and the round stages into single launches WITHOUT changing the order of any
# floating-point op. Every arithmetic expression is copied verbatim from the
# component kernels above (same association `signed * (inverse_mass * p)`, same
# reduction order over `pidx`), so the CPU()-Float64 device-vs-host bitwise oracle
# is preserved.

# Pre-gradient leaf: full column copy (tree_current -> working q/p/grad) fused with
# the initial half-kick and drift over active chains. Replaces
# copy_columns_all + kick + drift (3 launches -> 1). Element-wise (P x C); each
# thread owns one (pidx, b) so the kicked `pv` it drifts with is its own register.
@kernel function _device_nuts_leaf_pre!(
    q, p, grad,
    @Const(src_position), @Const(src_momentum), @Const(src_gradient),
    @Const(inverse_mass), @Const(active), @Const(sign), step, half,
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    qv = @inbounds src_position[pidx, b]
    pv = @inbounds src_momentum[pidx, b]
    gv = @inbounds src_gradient[pidx, b]
    if @inbounds(active[b]) != 0x00
        s = @inbounds sign[b]
        pv += s * half * gv
        signed = s * step
        qv += signed * (@inbounds(inverse_mass[pidx]) * pv)
    end
    @inbounds q[pidx, b] = qv
    @inbounds p[pidx, b] = pv
    @inbounds grad[pidx, b] = gv
end

# Per-chain-step pre-gradient leaf (issue #137): `half_scale` folds the leapfrog
# half factor into the per-chain step kick (0.5), mirroring
# `_device_nuts_kick_perchain!` + `_device_nuts_drift_perchain!`.
@kernel function _device_nuts_leaf_pre_perchain!(
    q, p, grad,
    @Const(src_position), @Const(src_momentum), @Const(src_gradient),
    @Const(inverse_mass), @Const(active), @Const(sign), @Const(step), half_scale,
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    qv = @inbounds src_position[pidx, b]
    pv = @inbounds src_momentum[pidx, b]
    gv = @inbounds src_gradient[pidx, b]
    if @inbounds(active[b]) != 0x00
        s = @inbounds sign[b]
        sb = @inbounds step[b]
        pv += s * (half_scale * sb) * gv
        signed = s * sb
        qv += signed * (@inbounds(inverse_mass[pidx]) * pv)
    end
    @inbounds q[pidx, b] = qv
    @inbounds p[pidx, b] = pv
    @inbounds grad[pidx, b] = gv
end

# Post-gradient leaf: final-step validity fold, closing half-kick over valid
# chains, and the proposed Hamiltonian, all in one per-chain (ndrange = C) launch.
# Replaces validity_update_final + kick + hamiltonian (3 launches -> 1). Seeds
# `valid` from `active` directly (so the leaf no longer needs a device valid<-active
# copy). The closing kick and the kinetic reduction run in the same per-element
# order as the component kernels.
@kernel function _device_nuts_leaf_post!(
    valid, proposed_energy, p,
    @Const(grad), @Const(logjoint), @Const(active), @Const(sign), @Const(inverse_mass),
    half, num_params::Int,
)
    b = @index(Global)
    if @inbounds(active[b]) != 0x00
        ok = isfinite(@inbounds logjoint[b])
        for pidx = 1:num_params
            ok &= isfinite(@inbounds grad[pidx, b])
        end
        if ok
            s = @inbounds sign[b]
            for pidx = 1:num_params
                @inbounds p[pidx, b] += s * half * grad[pidx, b]
            end
        end
        @inbounds valid[b] = ok ? 0x01 : 0x00
    else
        @inbounds valid[b] = 0x00
    end
    kinetic = zero(eltype(proposed_energy))
    for pidx = 1:num_params
        m = @inbounds p[pidx, b]
        kinetic += m * m * @inbounds(inverse_mass[pidx])
    end
    @inbounds proposed_energy[b] = kinetic / 2 - @inbounds(logjoint[b])
end

# Per-chain-step post-gradient leaf (issue #137).
@kernel function _device_nuts_leaf_post_perchain!(
    valid, proposed_energy, p,
    @Const(grad), @Const(logjoint), @Const(active), @Const(sign), @Const(step), @Const(inverse_mass),
    half_scale, num_params::Int,
)
    b = @index(Global)
    if @inbounds(active[b]) != 0x00
        ok = isfinite(@inbounds logjoint[b])
        for pidx = 1:num_params
            ok &= isfinite(@inbounds grad[pidx, b])
        end
        if ok
            s = @inbounds sign[b]
            sb = @inbounds step[b]
            for pidx = 1:num_params
                @inbounds p[pidx, b] += s * (half_scale * sb) * grad[pidx, b]
            end
        end
        @inbounds valid[b] = ok ? 0x01 : 0x00
    else
        @inbounds valid[b] = 0x00
    end
    kinetic = zero(eltype(proposed_energy))
    for pidx = 1:num_params
        m = @inbounds p[pidx, b]
        kinetic += m * m * @inbounds(inverse_mass[pidx])
    end
    @inbounds proposed_energy[b] = kinetic / 2 - @inbounds(logjoint[b])
end

# Fused subtree-state seed (mirror `_device_initialize_subtree_states!`): each of
# the four subtree endpoints (current/left/right/proposal) receives the SAME chosen
# frontier column -- the left frontier for `mask[b,1]` (copy_left) chains, the right
# frontier for `mask[b,2]` (copy_right) chains. copy_left/copy_right are mutually
# exclusive by construction (direction sign), so the `elseif` matches the two
# independent masked copies exactly. Replaces 8 masked-copy launches with 1.
@kernel function _device_nuts_init_states!(
    cur_p, cur_m, cur_g, tl_p, tl_m, tl_g, tr_p, tr_m, tr_g, pr_p, pr_m, pr_g,
    @Const(l_p), @Const(l_m), @Const(l_g), @Const(r_p), @Const(r_m), @Const(r_g), @Const(mask),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(mask[b, 1]) != 0x00
        vp = @inbounds l_p[pidx, b]
        vm = @inbounds l_m[pidx, b]
        vg = @inbounds l_g[pidx, b]
        @inbounds cur_p[pidx, b] = vp
        @inbounds cur_m[pidx, b] = vm
        @inbounds cur_g[pidx, b] = vg
        @inbounds tl_p[pidx, b] = vp
        @inbounds tl_m[pidx, b] = vm
        @inbounds tl_g[pidx, b] = vg
        @inbounds tr_p[pidx, b] = vp
        @inbounds tr_m[pidx, b] = vm
        @inbounds tr_g[pidx, b] = vg
        @inbounds pr_p[pidx, b] = vp
        @inbounds pr_m[pidx, b] = vm
        @inbounds pr_g[pidx, b] = vg
    elseif @inbounds(mask[b, 2]) != 0x00
        vp = @inbounds r_p[pidx, b]
        vm = @inbounds r_m[pidx, b]
        vg = @inbounds r_g[pidx, b]
        @inbounds cur_p[pidx, b] = vp
        @inbounds cur_m[pidx, b] = vm
        @inbounds cur_g[pidx, b] = vg
        @inbounds tl_p[pidx, b] = vp
        @inbounds tl_m[pidx, b] = vm
        @inbounds tl_g[pidx, b] = vg
        @inbounds tr_p[pidx, b] = vp
        @inbounds tr_m[pidx, b] = vm
        @inbounds tr_g[pidx, b] = vg
        @inbounds pr_p[pidx, b] = vp
        @inbounds pr_m[pidx, b] = vm
        @inbounds pr_g[pidx, b] = vg
    end
end

# Fused cohort scatter (mirror the three masked copies in
# `_device_advance_cohort_impl!`): tree_left/right/proposal <- tree_current under
# copy_left (mask[b,1]) / copy_right (mask[b,2]) / select_proposal (mask[b,3]). The
# three destinations are distinct buffers, so the independent `if`s cannot collide.
# Replaces 3 masked-copy launches with 1.
@kernel function _device_nuts_scatter3!(
    dl_p, dl_m, dl_g, dr_p, dr_m, dr_g, dp_p, dp_m, dp_g,
    @Const(s_p), @Const(s_m), @Const(s_g), @Const(mask),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    sp = @inbounds s_p[pidx, b]
    sm = @inbounds s_m[pidx, b]
    sg = @inbounds s_g[pidx, b]
    if @inbounds(mask[b, 1]) != 0x00
        @inbounds dl_p[pidx, b] = sp
        @inbounds dl_m[pidx, b] = sm
        @inbounds dl_g[pidx, b] = sg
    end
    if @inbounds(mask[b, 2]) != 0x00
        @inbounds dr_p[pidx, b] = sp
        @inbounds dr_m[pidx, b] = sm
        @inbounds dr_g[pidx, b] = sg
    end
    if @inbounds(mask[b, 3]) != 0x00
        @inbounds dp_p[pidx, b] = sp
        @inbounds dp_m[pidx, b] = sm
        @inbounds dp_g[pidx, b] = sg
    end
end

# Fused continuation-frontier merge copy (mirror the two masked copies in
# `_device_merge_cohort!`): left <- tree_left (copy_left, mask[b,1]) and
# right <- tree_right (copy_right, mask[b,2]). Distinct source/destination pairs,
# mutually exclusive masks. Replaces 2 masked-copy launches with 1.
@kernel function _device_nuts_merge_copy!(
    l_p, l_m, l_g, r_p, r_m, r_g,
    @Const(tl_p), @Const(tl_m), @Const(tl_g), @Const(tr_p), @Const(tr_m), @Const(tr_g), @Const(mask),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    if @inbounds(mask[b, 1]) != 0x00
        @inbounds l_p[pidx, b] = tl_p[pidx, b]
        @inbounds l_m[pidx, b] = tl_m[pidx, b]
        @inbounds l_g[pidx, b] = tl_g[pidx, b]
    end
    if @inbounds(mask[b, 2]) != 0x00
        @inbounds r_p[pidx, b] = tr_p[pidx, b]
        @inbounds r_m[pidx, b] = tr_m[pidx, b]
        @inbounds r_g[pidx, b] = tr_g[pidx, b]
    end
end

# Fused merge reductions: the whole-trajectory frontier U-turn test and the tree
# proposal kinetic energy, both per-chain (ndrange = C). The two reductions read
# disjoint inputs and write disjoint outputs, and each keeps its component kernel's
# `pidx` reduction order. Replaces frontier_turning + kinetic (2 launches -> 1).
@kernel function _device_nuts_frontier_turning_kinetic!(
    turning, kinetic,
    @Const(left_position), @Const(right_position), @Const(left_momentum), @Const(right_momentum),
    @Const(proposal_momentum), @Const(active), @Const(inverse_mass), num_params::Int,
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
    acc = zero(eltype(kinetic))
    for pidx = 1:num_params
        m = @inbounds proposal_momentum[pidx, b]
        acc += m * m * @inbounds(inverse_mass[pidx])
    end
    @inbounds kinetic[b] = acc / 2
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

# The merge-level whole-trajectory U-turn and the tree-proposal kinetic energy are
# FUSED into `_device_nuts_frontier_turning_kinetic!` (issue #152); the standalone
# frontier-turning and kinetic kernels are no longer launched.

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

# ---- issue #152 Tier 2: device-side per-leaf accept/select ---------------------
# Per-chain (ndrange = C) transcription of the host `_advance_batched_nuts_subtree_cohort!`
# scalar loop body (nuts/kernel.jl lines 16-64): step-invalid + delta-energy
# divergence, accept-prob + accept-stat accumulation, the multinomial progressive
# proposal selection (logaddexp + a PRE-DRAWN uniform, `leaf_uniform[leaf_row, b]`),
# and the per-leaf frontier/proposal-copy decision masks. Runs on the device with no
# host round-trip, so the whole round's leaves enqueue back-to-back and the host reads
# the subtree state once at round end. `divergent`/`turning` are sticky across the
# round (init 0 once); a chain that turned on an earlier odd leaf is folded out of
# `active` at the top here (and by `_device_nuts_fold_turning!` right after the dyadic
# test). This departs from the CPU RNG draw order -> only statistically equivalent.
@kernel function _device_nuts_advance!(
    active, divergent, turning, log_weight,
    integration_steps, accept_stat_sum, accept_stat_count,
    tree_current_logjoint, tree_left_logjoint, tree_right_logjoint,
    tree_proposal_logjoint, proposal_energy, proposal_energy_error,
    copy_left, copy_right, select_proposal, advanced, checkpoint,
    @Const(valid), @Const(proposed_energy), @Const(logjoint), @Const(current_energy),
    @Const(sign), @Const(leaf_uniform), leaf_offset::Int, max_delta_energy,
)
    b = @index(Global)
    T = eltype(log_weight)
    # Clear the per-leaf decision masks (they gate the P x C copies that follow).
    @inbounds copy_left[b] = 0x00
    @inbounds copy_right[b] = 0x00
    @inbounds select_proposal[b] = 0x00
    @inbounds advanced[b] = 0x00
    @inbounds checkpoint[b] = 0x00
    # Fold any earlier U-turn into active (sticky). KernelAbstractions forbids `return`,
    # so inactive lanes fall through the nested branches as no-ops.
    if @inbounds(turning[b]) != 0x00
        @inbounds active[b] = 0x00
    end
    if @inbounds(active[b]) != 0x00
        if @inbounds(valid[b]) == 0x00
            # Step-invalid divergence: divergent + inactive WITHOUT counting a step.
            @inbounds divergent[b] = 0x01
            @inbounds active[b] = 0x00
        else
            # Advance (current <- next); the step count + the frontier copy decision
            # are made BEFORE the delta-energy check, exactly as the host does.
            lj = @inbounds logjoint[b]
            @inbounds tree_current_logjoint[b] = lj
            @inbounds integration_steps[b] += Int32(1)
            @inbounds advanced[b] = 0x01
            if @inbounds(sign[b]) < zero(T)
                @inbounds copy_left[b] = 0x01
                @inbounds tree_left_logjoint[b] = lj
            else
                @inbounds copy_right[b] = 0x01
                @inbounds tree_right_logjoint[b] = lj
            end
            delta = @inbounds(proposed_energy[b]) - @inbounds(current_energy[b])
            if !isfinite(delta) || delta > max_delta_energy
                # Delta-energy divergence (still counts the step above).
                @inbounds divergent[b] = 0x01
                @inbounds active[b] = 0x00
            else
                accept_prob = min(one(T), exp(min(zero(T), -delta)))
                @inbounds accept_stat_sum[b] += accept_prob
                @inbounds accept_stat_count[b] += Int32(1)
                candidate = -@inbounds(proposed_energy[b])
                lw = @inbounds log_weight[b]
                ninf = T(-Inf)
                # combined = logaddexp(lw, candidate) (device _logaddexp transcription).
                combined = if lw == ninf
                    candidate
                elseif candidate == ninf
                    lw
                else
                    hi = max(lw, candidate)
                    hi + log1p(exp(min(lw, candidate) - hi))
                end
                u = @inbounds leaf_uniform[leaf_offset+b]
                if !isfinite(lw) || log(u) < candidate - combined
                    @inbounds select_proposal[b] = 0x01
                    @inbounds tree_proposal_logjoint[b] = lj
                    @inbounds proposal_energy[b] = @inbounds(proposed_energy[b])
                    @inbounds proposal_energy_error[b] = delta
                end
                @inbounds log_weight[b] = combined
                @inbounds checkpoint[b] = 0x01
            end
        end
    end
end

# Fold the sticky U-turn flag into the active mask (active &= !turning); enqueued
# after the odd-leaf dyadic test so the next leaf's pre-gradient kernel already sees
# the chain dropped out.
@kernel function _device_nuts_fold_turning!(active, @Const(turning))
    b = @index(Global)
    if @inbounds(turning[b]) != 0x00
        @inbounds active[b] = 0x00
    end
end

# Cohort scatter with three SEPARATE per-chain mask vectors (async path): the fused
# `_device_nuts_scatter3!` above reads a packed C x 3 matrix, but the async advance
# kernel writes three independent UInt8 mask vectors, so this variant consumes them
# directly (no mask repack). tree_left/right/proposal <- tree_current.
@kernel function _device_nuts_scatter3_v!(
    dl_p, dl_m, dl_g, dr_p, dr_m, dr_g, dp_p, dp_m, dp_g,
    @Const(s_p), @Const(s_m), @Const(s_g),
    @Const(mask_left), @Const(mask_right), @Const(mask_sel),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    sp = @inbounds s_p[pidx, b]
    sm = @inbounds s_m[pidx, b]
    sg = @inbounds s_g[pidx, b]
    if @inbounds(mask_left[b]) != 0x00
        @inbounds dl_p[pidx, b] = sp
        @inbounds dl_m[pidx, b] = sm
        @inbounds dl_g[pidx, b] = sg
    end
    if @inbounds(mask_right[b]) != 0x00
        @inbounds dr_p[pidx, b] = sp
        @inbounds dr_m[pidx, b] = sm
        @inbounds dr_g[pidx, b] = sg
    end
    if @inbounds(mask_sel[b]) != 0x00
        @inbounds dp_p[pidx, b] = sp
        @inbounds dp_m[pidx, b] = sm
        @inbounds dp_g[pidx, b] = sg
    end
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
    # Batched mask upload buffer (issue #152): a C x 3 UInt8 device matrix holding up
    # to three per-chain masks packed for the fused init / scatter / merge-copy
    # kernels, so several small mask uploads collapse into one copyto! per stage.
    mask_batch::Any          # C x 3 UInt8
    # host staging (avoid per-call allocation)
    host_u8::Vector{UInt8}
    host_mask_batch::Matrix{UInt8}   # C x 3
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
    # ---- issue #152 Tier 2: async round + device-side accept/select ------------
    # When `sync_per_leaf` is true the round loop runs the Tier-1 order-preserving
    # path (one host round-trip per leaf; keeps the CPU()-Float64 bitwise oracle).
    # When false (default) it runs the async path: the round's RNG is pre-drawn and
    # uploaded once, the per-leaf accept/select/divergence/log-weight/dyadic-U-turn
    # bookkeeping runs on the device, and the host reads one status batch per round
    # instead of per leaf. The async path departs from the CPU RNG draw order, so it
    # is only STATISTICALLY (not bitwise) equivalent to the host masked path.
    sync_per_leaf::Bool
    # Device-resident per-chain (C) subtree state the async advance kernel owns
    # (mirrors the host `ws.subtree_*` scalars during a round; downloaded once at
    # round end for the host merge).
    d_subtree_active::Any        # UInt8
    d_subtree_divergent::Any     # UInt8
    d_subtree_turning::Any       # UInt8 (sticky across the round)
    d_log_weight::Any            # T
    d_integration_steps::Any     # Int32
    d_accept_stat_sum::Any       # T
    d_accept_stat_count::Any     # Int32
    d_current_energy::Any        # T   reference energy (uploaded once per draw)
    d_tree_current_logjoint::Any # T
    d_tree_left_logjoint::Any    # T
    d_tree_right_logjoint::Any   # T
    d_tree_proposal_logjoint::Any # T
    d_proposal_energy::Any       # T   subtree_proposal_energy
    d_proposal_energy_error::Any # T
    d_copy_left::Any             # UInt8
    d_copy_right::Any            # UInt8
    d_select_proposal::Any       # UInt8
    d_advanced::Any              # UInt8
    d_checkpoint::Any            # UInt8
    # Pre-drawn round RNG: a FLAT (1<<max_tree_depth)*C vector of uniforms laid out
    # leaf-major (leaf i's chain-b uniform at index i*C + b), so a round of `nleaves`
    # leaves uploads only the contiguous prefix `1:nleaves*C`.
    d_leaf_uniform::Any          # T
    host_leaf_uniform::Vector{T} # host staging (rand! fills the used prefix)
    # Host staging for the round-end integer downloads.
    host_i32::Vector{Int32}
    # ---- issue #160: sync-leaf lane-compaction scratch -------------------------
    # The Tier-1 (sync-per-leaf) masked leaf gathers its active columns into
    # `compact_params[:, 1:k]`, runs the gradient over those k columns into
    # `compact_gradient[:, 1:k]` / `compact_totals[1:k]`, and scatters back to the
    # active lanes `compact_index[slot]`. All are pre-sized to the full batch width
    # so the gather/scatter never allocates per leaf; the all-active leaf keeps the
    # unchanged full-width path (zero added work in the common early-round case).
    # `compact_index_host` builds the active->original map host-side (the sync path
    # already reads `ws.subtree_active` on the host every leaf).
    compact_params::Any        # P x C  (T)
    compact_gradient::Any      # P x C  (T)
    compact_totals::Any        # C      (T)
    compact_index::Any         # C      (Int32) device gather/scatter index
    compact_index_host::Vector{Int32}  # C host staging
end

function DeviceNUTSWorkspace(
    model::TeaModel,
    num_chains::Integer,
    max_tree_depth::Integer;
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=Float64,
    args=(),
    constraints=choicemap(),
    sync_per_leaf::Bool=false,
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
    vecI32() = fill!(KernelAbstractions.allocate(backend, Int32, C), Int32(0))
    ckpt() = fill!(KernelAbstractions.allocate(backend, T, P, max(D + 1, 1), C), zero(T))
    max_leaves = 1 << max(D, 0)
    leaf_uniform = fill!(KernelAbstractions.allocate(backend, T, max_leaves * C), zero(T))
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
        fill!(KernelAbstractions.allocate(backend, UInt8, C, 3), 0x00),
        Vector{UInt8}(undef, C),
        Matrix{UInt8}(undef, C, 3),
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
        # ---- issue #152 Tier 2 buffers -----------------------------------------
        sync_per_leaf,
        vecU8(), vecU8(), vecU8(),          # d_subtree_active/divergent/turning
        vecT(),                              # d_log_weight
        vecI32(),                            # d_integration_steps
        vecT(),                              # d_accept_stat_sum
        vecI32(),                            # d_accept_stat_count
        vecT(),                              # d_current_energy
        vecT(), vecT(), vecT(), vecT(),      # d_tree_current/left/right/proposal_logjoint
        vecT(), vecT(),                      # d_proposal_energy / _error
        vecU8(), vecU8(), vecU8(), vecU8(), vecU8(), # copy_left/right/select/advanced/checkpoint
        leaf_uniform,                        # d_leaf_uniform
        Vector{T}(undef, max_leaves * C),    # host_leaf_uniform
        Vector{Int32}(undef, C),             # host_i32
        # ---- issue #160 sync-leaf compaction scratch ---------------------------
        mat(),                               # compact_params (P x C)
        mat(),                               # compact_gradient (P x C)
        vecT(),                              # compact_totals (C)
        vecI32(),                            # compact_index (C, zero-init)
        zeros(Int32, C),                     # compact_index_host
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

# Batched mask upload (issue #152): pack several C-length host Bool masks into the
# columns of a C x 3 UInt8 staging matrix and push them to the device in ONE
# copyto!. `cols` is a tuple of host Bool vectors (1..3 of them); columns of
# `stage`/`dev` beyond `length(cols)` are left as-is (the consuming kernel never
# reads them). Replaces N separate `_upload_mask!` transfers with one.
_upload_masks!(dev, cols::Tuple, stage::Matrix{UInt8}) = begin
    @inbounds for (k, bits) in enumerate(cols)
        for i in eachindex(bits)
            stage[i, k] = bits[i] ? 0x01 : 0x00
        end
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

# Download a device Int32 vector into a host Int vector through `stage` (issue #152
# Tier 2 round-end batch).
_download_ints!(host::AbstractVector{<:Integer}, dev, stage::Vector{Int32}) = begin
    copyto!(stage, dev)
    @inbounds for i in eachindex(host)
        host[i] = Int(stage[i])
    end
    return host
end

# Upload a host Float64 C-vector into a device T vector (converts precision) through
# a caller-owned `stage::Vector{T}` so the transfer does not allocate.
_upload_reals!(dev, host::AbstractVector{Float64}, stage::Vector) = begin
    @inbounds for i in eachindex(host, stage)
        stage[i] = host[i]
    end
    copyto!(dev, stage)
    return dev
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

# ---- sync-leaf lane compaction (issue #160) ------------------------------------
#
# WHY. The masked doubling trajectory runs a FULL-WIDTH batched gradient at every
# leapfrog leaf even once most chains have finished/diverged. Mirroring the host
# lane compaction (commit e52c21c, PR #198), the Tier-1 (sync-per-leaf) device leaf
# gathers the active columns, evaluates the gradient over just those `k` columns, and
# scatters the results back to the active lanes.
#
# WHY IT IS BITWISE SAFE. The gradient of column c reads ONLY `params[:, c]`, the
# SHARED observations, and its own private slot-scratch column -- columns are fully
# independent -- so gather -> gradient -> scatter is a pure permutation over
# independent columns: the active lanes receive BITWISE-identical logjoint/gradient
# values. The inactive lanes are downstream don't-cares (the post-gradient leaf gates
# every read behind `active`/`valid`), so leaving their gradient/total entries stale
# equals the full-width path overwriting them with never-read values. No RNG lives in
# the gradient, so the masked-doubling draw order is untouched and the CPU()-Float64
# device-vs-host BITWISE oracle (`dnuts_device_vs_host_masked_exact`) still holds.
#
# SCOPE. Only the Tier-1 SYNC leaf (`_device_nuts_leaf!`) is compacted: it reads the
# active mask (`ws.subtree_active`) on the host every leaf, so the active count `k`
# and the gather index are already available host-side with no extra sync. The Tier-2
# async leaf (`_device_nuts_leaf_async!`) keeps a device-resident active mask with NO
# per-leaf host round-trip, so compacting it needs a device-side stream compaction
# (prefix sum) to obtain `k`; that is a documented follow-up and stays full width.
# The tiled observation-parallel gradient (issue #153) is likewise left full width
# (see `_device_launch_gradient_compact!`); the eligibility gate below excludes both
# per-column observations/arguments and the tiled path, falling back to full width.

# Compact once the active fraction falls below this. Above it the gather/scatter +
# narrower-launch overhead would eat the saved lane work, so the full-width path stays
# in charge (and the all-active leaf -- the common early-round case -- is completely
# unchanged, keeping its zero-added-work guarantee).
const _DEVICE_NUTS_COMPACTION_ACTIVE_FRACTION = 0.5

# Diagnostic counters (issue #160): total gradient columns actually launched across
# sync-leaf calls, how many sync leaves ran, and how many took the compact path.
# Tests reset and read these to assert the compact leaf evaluated EXACTLY
# count(active) columns and that the gate engaged. Not consulted by production logic.
const _DEVICE_NUTS_GRADIENT_COLUMNS = Ref(0)
const _DEVICE_NUTS_GRADIENT_LEAVES = Ref(0)
const _DEVICE_NUTS_COMPACTED_LEAVES = Ref(0)

function _device_nuts_reset_compaction_stats!()
    _DEVICE_NUTS_GRADIENT_COLUMNS[] = 0
    _DEVICE_NUTS_GRADIENT_LEAVES[] = 0
    _DEVICE_NUTS_COMPACTED_LEAVES[] = 0
    return nothing
end

# Diagnostic opt-out (issue #160): flip to `false` to force every sync leaf back onto
# the full-width gradient regardless of the active fraction. Defaults to `true`. Only
# an A/B benchmark or a regression bisect should touch it; production always compacts.
const _DEVICE_NUTS_COMPACTION_ENABLED = Ref(true)

# Compaction is bitwise-safe only when EVERY per-column gradient input other than the
# params column is shared across columns: SHARED observations (a single broadcast
# column, `size(observed, 2) == 1`, so `_device_obs_col` returns 1 for any lane) and
# SHARED model arguments (`args isa Tuple`, staged identically into every slot column
# by `_device_stage_gradient_arguments!`). Per-column observations/constraints/args
# would make column `slot` score a different chain's conditioning than original column
# `index[slot]`, so those keep the full width. The tiled path (issue #153) is excluded
# too -- it stays full width for now.
function _device_nuts_compaction_eligible(inner::DeviceBatchedWorkspace)
    _DEVICE_NUTS_COMPACTION_ENABLED[] || return false
    inner.tiled_gradient === nothing || return false
    size(inner.observed_device, 2) == 1 || return false
    size(inner.observed_int_device, 2) == 1 || return false
    inner.args isa Tuple || return false
    return true
end

# Build the host->device gather index for the active columns and return `k` (0 => run
# the unchanged full-width gradient). Gated to `1 <= k` and active fraction below
# `_DEVICE_NUTS_COMPACTION_ACTIVE_FRACTION` (`2k < C` implies `k < C`). The whole
# length-C index is uploaded in one copy; the tail `k+1:C` is stale but never read (the
# gather/scatter launch only touches slots `1:k`).
function _device_nuts_build_compaction!(dws::DeviceNUTSWorkspace, active_host::AbstractVector{Bool})
    _device_nuts_compaction_eligible(dws.inner) || return 0
    C = dws.num_chains
    idx = dws.compact_index_host
    k = 0
    @inbounds for c = 1:C
        if active_host[c]
            k += 1
            idx[k] = Int32(c)
        end
    end
    (k >= 1 && 2 * k < C) || return 0
    copyto!(dws.compact_index, idx)
    return k
end

# Run the sync-leaf gradient, compacted when beneficial. On the compact path: gather
# the k active columns of `params_device`, evaluate the gradient over k columns, and
# scatter gradient/logjoint back to the active lanes. On the full-width path this is
# exactly the unchanged `_device_launch_gradient!`. Kernels on one backend run in
# submission order, so gather -> gradient -> scatter needs no interior synchronize
# (the leaf synchronizes once after the post-gradient kernel, unchanged). Returns `k`
# (0 on the full-width path).
function _device_nuts_leaf_gradient!(dws::DeviceNUTSWorkspace, ws)
    inner = dws.inner
    k = _device_nuts_build_compaction!(dws, ws.subtree_active)
    _DEVICE_NUTS_GRADIENT_LEAVES[] += 1
    if k == 0
        _device_launch_gradient!(inner)
        _DEVICE_NUTS_GRADIENT_COLUMNS[] += inner.batch_size
        return 0
    end
    be = dws.backend
    P = dws.num_params
    _device_nuts_gather_columns!(be)(
        dws.compact_params, inner.params_device, dws.compact_index; ndrange=(P, k),
    )
    _device_launch_gradient_compact!(inner, dws.compact_params, dws.compact_gradient, dws.compact_totals, k)
    _device_nuts_scatter_gradient!(be)(
        inner.gradients_device, inner.totals_device,
        dws.compact_gradient, dws.compact_totals, dws.compact_index; ndrange=(P, k),
    )
    _DEVICE_NUTS_COMPACTED_LEAVES[] += 1
    _DEVICE_NUTS_GRADIENT_COLUMNS[] += k
    return k
end

# ---- device leaf leapfrog ------------------------------------------------------

# One masked leapfrog leaf from `tree_current` in each chain's `sign` direction,
# leaving the leaf in (params_device, working_momentum, gradients_device, totals_device)
# and refreshing `valid`/`proposed_energy`. Mirrors `batched_leapfrog_step_to!` for a
# single step (initial half-kick, drift, gradient, closing half-kick; no flip).
# Downloads only `proposed_energy` (C) + `valid` (C). Uploads the active mask (C) and,
# on the lane-compaction path (issue #160, active fraction < 50%), the length-C gather
# index (C Int32); both stay within the O(C)-per-leaf transfer budget.
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

    # One mask upload (active); the fused post-gradient kernel seeds `valid` from
    # `active` on-device, so the old device valid<-active copy is gone.
    _upload_mask!(dws.active, ws.subtree_active, dws.host_u8)
    # Fused pre-gradient leaf: copy tree_current -> (q, p, grad) + initial half-kick
    # + drift, in one launch (was copy_columns_all + kick + drift).
    _device_nuts_leaf_pre!(be)(
        q, p, grad,
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.inverse_mass, dws.active, dws.sign, h, half; ndrange=(P, C),
    )
    # Gradient over the active columns only when most chains have finished (issue
    # #160); the full-width path is unchanged when all/most lanes are still live.
    _device_nuts_leaf_gradient!(dws, ws)
    # Fused post-gradient leaf: final-step validity + closing half-kick + proposed
    # Hamiltonian, in one launch (was validity_update_final + kick + hamiltonian).
    _device_nuts_leaf_post!(be)(
        dws.valid, dws.proposed_energy, p,
        grad, logj, dws.active, dws.sign, dws.inverse_mass, half, P; ndrange=C,
    )
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
    _device_nuts_leaf_pre_perchain!(be)(
        q, p, grad,
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.inverse_mass, dws.active, dws.sign, dws.step, half; ndrange=(P, C),
    )
    # Lane compaction (issue #160); see the shared-step overload above.
    _device_nuts_leaf_gradient!(dws, ws)
    _device_nuts_leaf_post_perchain!(be)(
        dws.valid, dws.proposed_energy, p,
        grad, logj, dws.active, dws.sign, dws.step, dws.inverse_mass, half, P; ndrange=C,
    )
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
    # One batched upload of both direction masks, one fused seed kernel:
    # tree_current / tree_left / tree_right / tree_proposal <- left (copy_left,
    # mask col 1) or right (copy_right, mask col 2). Was 2 uploads + 8 launches.
    _upload_masks!(dws.mask_batch, (ws.subtree_copy_left, ws.subtree_copy_right), dws.host_mask_batch)
    _device_nuts_init_states!(be)(
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.tree_left_position, dws.tree_left_momentum, dws.tree_left_gradient,
        dws.tree_right_position, dws.tree_right_momentum, dws.tree_right_gradient,
        dws.tree_proposal_position, dws.tree_proposal_momentum, dws.tree_proposal_gradient,
        dws.left_position, dws.left_momentum, dws.left_gradient,
        dws.right_position, dws.right_momentum, dws.right_gradient,
        dws.mask_batch; ndrange=(P, C),
    )
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

    # device: scatter tree_left/right/proposal <- tree_current, one batched upload of
    # the three masks (copy_left/copy_right/select_proposal) + one fused kernel.
    _upload_masks!(
        dws.mask_batch,
        (ws.subtree_copy_left, ws.subtree_copy_right, ws.subtree_select_proposal),
        dws.host_mask_batch,
    )
    _device_nuts_scatter3!(be)(
        dws.tree_left_position, dws.tree_left_momentum, dws.tree_left_gradient,
        dws.tree_right_position, dws.tree_right_momentum, dws.tree_right_gradient,
        dws.tree_proposal_position, dws.tree_proposal_momentum, dws.tree_proposal_gradient,
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.mask_batch; ndrange=(P, C),
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
    # One batched upload of both direction masks + one fused merge-copy kernel:
    # left <- tree_left (copy_left, col 1), right <- tree_right (copy_right, col 2).
    _upload_masks!(dws.mask_batch, (ws.subtree_copy_left, ws.subtree_copy_right), dws.host_mask_batch)
    _device_nuts_merge_copy!(be)(
        dws.left_position, dws.left_momentum, dws.left_gradient,
        dws.right_position, dws.right_momentum, dws.right_gradient,
        dws.tree_left_position, dws.tree_left_momentum, dws.tree_left_gradient,
        dws.tree_right_position, dws.tree_right_momentum, dws.tree_right_gradient,
        dws.mask_batch; ndrange=(P, C),
    )

    # merge-level whole-trajectory turning + tree-proposal kinetic, fused (both are
    # per-chain reductions with disjoint inputs/outputs).
    _upload_mask!(dws.mask_c, ws.subtree_active, dws.host_u8)
    _device_nuts_frontier_turning_kinetic!(be)(
        dws.turning, dws.kinetic,
        dws.left_position, dws.right_position, dws.left_momentum, dws.right_momentum,
        dws.tree_proposal_momentum, dws.mask_c, dws.inverse_mass, P; ndrange=C,
    )
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

# ---- issue #152 Tier 2: async round loop ---------------------------------------

# Async leaf (no host round-trip): fused pre-gradient kernel + gradient + fused
# post-gradient kernel, all enqueued against the DEVICE-resident active mask
# (`dws.d_subtree_active`). Unlike `_device_nuts_leaf!` it neither synchronizes nor
# downloads -- the device advance kernel reads `valid`/`proposed_energy`/`totals`
# straight off the device. Scalar shared step.
function _device_nuts_leaf_async!(dws::DeviceNUTSWorkspace{T}, ws, step_size::Real) where {T}
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
    _device_nuts_leaf_pre!(be)(
        q, p, grad,
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.inverse_mass, dws.d_subtree_active, dws.sign, h, half; ndrange=(P, C),
    )
    _device_launch_gradient!(inner)
    _device_nuts_leaf_post!(be)(
        dws.valid, dws.proposed_energy, p,
        grad, logj, dws.d_subtree_active, dws.sign, dws.inverse_mass, half, P; ndrange=C,
    )
    return dws
end

# Per-chain-step async leaf (issue #137).
function _device_nuts_leaf_async!(dws::DeviceNUTSWorkspace{T}, ws, ::Nothing) where {T}
    be = dws.backend
    P = dws.num_params
    C = dws.num_chains
    inner = dws.inner
    q = inner.params_device
    grad = inner.gradients_device
    logj = inner.totals_device
    p = dws.working_momentum
    half = convert(T, 0.5)
    _device_nuts_leaf_pre_perchain!(be)(
        q, p, grad,
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.inverse_mass, dws.d_subtree_active, dws.sign, dws.step, half; ndrange=(P, C),
    )
    _device_launch_gradient!(inner)
    _device_nuts_leaf_post_perchain!(be)(
        dws.valid, dws.proposed_energy, p,
        grad, logj, dws.d_subtree_active, dws.sign, dws.step, dws.inverse_mass, half, P; ndrange=C,
    )
    return dws
end

# Async per-leaf cohort advance: the device accept/select kernel + the P x C copies
# it drives (accept-copy, dyadic checkpoint/turning, frontier scatter). No host
# round-trip. `leaf_index` is the 0-based leaf within the round (the host loop
# counter); the checkpoint/turning schedule is a pure function of it, so no download
# is needed to drive it.
function _device_advance_cohort_async!(
    dws::DeviceNUTSWorkspace{T}, ws, max_delta_energy::Float64, leaf_index::Int,
) where {T}
    be = dws.backend
    P = dws.num_params
    C = dws.num_chains
    _device_nuts_advance!(be)(
        dws.d_subtree_active, dws.d_subtree_divergent, dws.d_subtree_turning, dws.d_log_weight,
        dws.d_integration_steps, dws.d_accept_stat_sum, dws.d_accept_stat_count,
        dws.d_tree_current_logjoint, dws.d_tree_left_logjoint, dws.d_tree_right_logjoint,
        dws.d_tree_proposal_logjoint, dws.d_proposal_energy, dws.d_proposal_energy_error,
        dws.d_copy_left, dws.d_copy_right, dws.d_select_proposal, dws.d_advanced, dws.d_checkpoint,
        dws.valid, dws.proposed_energy, dws.inner.totals_device, dws.d_current_energy,
        dws.sign, dws.d_leaf_uniform, leaf_index * C, convert(T, max_delta_energy); ndrange=C,
    )
    # accept-copy: tree_current <- (q, p, grad) for advanced chains.
    _device_nuts_copy_columns!(be)(
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.inner.params_device, dws.working_momentum, dws.inner.gradients_device, dws.d_advanced; ndrange=(P, C),
    )
    # even leaf: store checkpoint; odd leaf: dyadic U-turn fold + active fold.
    if iseven(leaf_index)
        slot = count_ones(leaf_index) + 1
        _device_nuts_store_checkpoint!(be)(
            dws.checkpoint_position, dws.checkpoint_momentum,
            dws.tree_current_position, dws.tree_current_momentum, dws.d_checkpoint, slot; ndrange=(P, C),
        )
    else
        for k = 1:trailing_ones(leaf_index)
            block_start = leaf_index - (1 << k) + 1
            slot = count_ones(block_start) + 1
            _device_nuts_dyadic_turning!(be)(
                dws.d_subtree_turning, dws.checkpoint_position, dws.checkpoint_momentum,
                dws.tree_current_position, dws.tree_current_momentum, dws.d_checkpoint, dws.sign,
                dws.inverse_mass, slot, P; ndrange=C,
            )
        end
        _device_nuts_fold_turning!(be)(dws.d_subtree_active, dws.d_subtree_turning; ndrange=C)
    end
    # frontier / proposal scatter: tree_left/right/proposal <- tree_current.
    _device_nuts_scatter3_v!(be)(
        dws.tree_left_position, dws.tree_left_momentum, dws.tree_left_gradient,
        dws.tree_right_position, dws.tree_right_momentum, dws.tree_right_gradient,
        dws.tree_proposal_position, dws.tree_proposal_momentum, dws.tree_proposal_gradient,
        dws.tree_current_position, dws.tree_current_momentum, dws.tree_current_gradient,
        dws.d_copy_left, dws.d_copy_right, dws.d_select_proposal; ndrange=(P, C),
    )
    return dws
end

# Async doubling round (issue #152 Tier 2): pre-draw the round's RNG once, run the
# leaves device-resident with device-side accept/select, then read the subtree state
# back in ONE round-end batch and reuse the host merge / continuation update verbatim.
function _device_masked_nuts_doubling_round_async!(
    dws::DeviceNUTSWorkspace{T},
    ws,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    step_size,
    rng::AbstractRNG,
) where {T}
    be = dws.backend
    C = dws.num_chains
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

    # Initialize the device-resident per-chain subtree state for the round (mirrors
    # the host `_reset_batched_nuts_subtree_scratch!` values the device now owns).
    _upload_mask!(dws.d_subtree_active, round_active, dws.host_u8)
    fill!(dws.d_subtree_divergent, 0x00)
    fill!(dws.d_subtree_turning, 0x00)
    fill!(dws.d_log_weight, convert(T, -Inf))
    fill!(dws.d_integration_steps, Int32(0))
    fill!(dws.d_accept_stat_sum, zero(T))
    fill!(dws.d_accept_stat_count, Int32(0))
    fill!(dws.d_proposal_energy, convert(T, Inf))
    fill!(dws.d_proposal_energy_error, convert(T, Inf))
    _upload_reals!(dws.d_current_energy, ws.current_energy, dws.host_energy)
    _upload_reals!(dws.d_tree_current_logjoint, ws.tree_current_logjoint, dws.host_energy)
    _upload_reals!(dws.d_tree_left_logjoint, ws.tree_left_logjoint, dws.host_energy)
    _upload_reals!(dws.d_tree_right_logjoint, ws.tree_right_logjoint, dws.host_energy)
    _upload_reals!(dws.d_tree_proposal_logjoint, ws.tree_proposal_logjoint, dws.host_energy)

    # Pre-draw the round's leaf uniforms (nleaves x C, leaf-major) and upload the
    # contiguous prefix in one copy. Departs from the CPU RNG draw order.
    nleaves = 1 << round_depth
    used = nleaves * C
    @inbounds rand!(rng, view(dws.host_leaf_uniform, 1:used))
    # Offset-form copy of the contiguous prefix (view + GPU copyto! would scalar-index).
    copyto!(dws.d_leaf_uniform, 1, dws.host_leaf_uniform, 1, used)

    # Enqueue every leaf of the round with NO host sync in between.
    for leaf_index = 0:(nleaves-1)
        _device_nuts_leaf_async!(dws, ws, step_size)
        _device_advance_cohort_async!(dws, ws, max_delta_energy, leaf_index)
    end

    # ONE round-end sync + batched download of the subtree state the host merge reads.
    KernelAbstractions.synchronize(be)
    _download_ints!(ws.subtree_integration_steps, dws.d_integration_steps, dws.host_i32)
    _download_ints!(ws.subtree_accept_stat_count, dws.d_accept_stat_count, dws.host_i32)
    _download_reals!(ws.subtree_accept_stat_sum, dws.d_accept_stat_sum, dws.host_energy)
    _download_reals!(ws.subtree_log_weight, dws.d_log_weight, dws.host_energy)
    _download_bits!(ws.subtree_divergent, dws.d_subtree_divergent, dws.host_u8)
    _download_bits!(ws.subtree_turning, dws.d_subtree_turning, dws.host_u8)
    _download_reals!(ws.subtree_proposal_energy, dws.d_proposal_energy, dws.host_energy)
    _download_reals!(ws.subtree_proposal_energy_error, dws.d_proposal_energy_error, dws.host_energy)
    _download_reals!(ws.tree_proposal_logjoint, dws.d_tree_proposal_logjoint, dws.host_energy)
    _download_reals!(ws.tree_left_logjoint, dws.d_tree_left_logjoint, dws.host_energy)
    _download_reals!(ws.tree_right_logjoint, dws.d_tree_right_logjoint, dws.host_energy)

    # Round-end host bookkeeping + merge -- identical to the sync-per-leaf path.
    fill!(ws.subtree_active, false)
    any_merging = false
    @inbounds for c in eachindex(round_active)
        round_active[c] || continue
        ws.control.tree_depths[c] += 1
        if ws.subtree_integration_steps[c] == 0
            ws.control.divergent_step[c] = ws.subtree_divergent[c]
        elseif ws.subtree_turning[c] || ws.subtree_divergent[c]
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

    if dws.sync_per_leaf
        # Tier-1 order-preserving path (one host round-trip per leaf; keeps the
        # CPU()-Float64 bitwise oracle, `dnuts_device_vs_host_masked_exact`).
        while _device_masked_nuts_doubling_round!(dws, ws, max_tree_depth, max_delta_energy, round_step, rng)
        end
    else
        # Tier-2 async path (pre-drawn round RNG + device-side accept/select; one
        # status batch per round). Statistically equivalent to the host masked path.
        while _device_masked_nuts_doubling_round_async!(dws, ws, max_tree_depth, max_delta_energy, round_step, rng)
        end
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

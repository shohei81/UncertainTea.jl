# Mask-based iterative doubling for batched NUTS (NumPyro-style lockstep).
#
# Unlike the hybrid path (depth-cohort scheduler plus the per-chain scalar
# recursion tail in tree_dynamics.jl), every chain runs the same doubling round
# at the same time: round r expands a subtree of 2^r leaves through repeated
# masked single-leaf steps, then all started chains merge. A chain that
# diverges or turns mid-subtree drops out of the expand mask but stays in the
# round for the merge; a chain whose continuation terminates (turning,
# divergence, or max depth) leaves the active set between rounds. Every
# leapfrog step is one full-width batched gradient call -- masked lanes waste
# gradient work by design; that is exactly the shape device gradients need.
# All active chains share the same tree depth (== round index), so no chain
# ever falls out of lockstep and the scalar tail is never entered.
#
# The per-leaf and per-merge math is the existing cohort machinery
# (_advance_batched_nuts_subtree_cohort!, _merge_batched_nuts_subtree_cohort!,
# and the tree_expand.jl helpers they call); this file only replaces the
# depth-cohort scheduler with a single all-active-chains round loop. The
# kernel access structs are built once per phase with the block masks aliasing
# the live workspace arrays, so the expand mask shrinks in place.
#
# RNG order (deterministic under a seed):
#   1. initialization: shared with the hybrid path (momentum for all chains,
#      one direction per chain, one first-step proposal draw per valid chain,
#      chain-major).
#   2. per round: one direction draw per chain (all chains, chain-major),
#      whether or not the chain is still active.
#   3. per leaf step: one proposal-selection draw per chain still in the
#      expand mask, in chain order (skipped while the running subtree log
#      weight is -Inf, exactly as in _advance_tree_leaf).
#   4. per merge: one draw per merging chain in chain order, consumed only
#      when that chain's subtree is valid (no internal U-turn or divergence --
#      invalid subtrees are discarded whole and never merge) and its log
#      weight is finite; the draw decides the biased progressive swap
#      P = min(1, w_subtree / w_continuation) (_merge_subtree_stats).

function _batched_nuts_proposals_masked!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    current_logjoint::AbstractVector{Float64},
    current_gradient::AbstractMatrix{Float64},
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    _initialize_batched_nuts_continuations!(
        workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        divergence_threshold,
        rng,
    )
    while _masked_nuts_doubling_round!(
        workspace,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_tree_depth,
        divergence_threshold,
        rng,
    )
    end
    _finalize_batched_nuts_proposals!(workspace, position)
    return workspace
end

function _masked_nuts_doubling_round!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    _reset_batched_nuts_subtree_scratch!(workspace)
    _update_batched_nuts_continuation_active!(workspace, max_tree_depth) || return false
    round_active = workspace.control.scheduler.continuation_active
    round_depth = 0
    for chain_index in eachindex(round_active)
        round_active[chain_index] || continue
        round_depth = max(round_depth, workspace.control.tree_depths[chain_index])
    end
    copyto!(workspace.subtree_active, round_active)
    copyto!(workspace.control.scheduler.subtree_started, round_active)
    for chain_index in eachindex(workspace.control.step_direction)
        workspace.control.step_direction[chain_index] = _sample_nuts_direction(rng)
    end
    _initialize_batched_nuts_subtree_states!(workspace, workspace.subtree_active)

    expand_access = _masked_nuts_expand_access(workspace, round_depth)
    any_expanding = true
    for _ = 1:(1<<round_depth)
        any_expanding || break
        _batched_nuts_kernel_leapfrog!(
            workspace,
            expand_access,
            model,
            inverse_mass_matrix,
            args,
            constraints,
            step_size,
        )
        _batched_nuts_kernel_hamiltonian!(expand_access, inverse_mass_matrix)
        any_expanding = _advance_batched_nuts_subtree_cohort!(
            workspace,
            expand_access,
            inverse_mass_matrix,
            divergence_threshold,
            rng,
        )
    end

    fill!(workspace.subtree_active, false)
    any_merging = false
    for chain_index in eachindex(round_active)
        round_active[chain_index] || continue
        workspace.control.tree_depths[chain_index] += 1
        if workspace.subtree_integration_steps[chain_index] == 0
            workspace.control.divergent_step[chain_index] =
                workspace.subtree_divergent[chain_index]
        elseif workspace.subtree_turning[chain_index] ||
               workspace.subtree_divergent[chain_index]
            # Invalid subtree: canonical NUTS discards the whole doubling.
            _discard_invalid_batched_subtree!(workspace, chain_index)
        else
            workspace.subtree_active[chain_index] = true
            any_merging = true
        end
    end
    if any_merging
        merge_access = _masked_nuts_merge_access(workspace, round_depth)
        _merge_batched_nuts_subtree_cohort!(
            workspace,
            merge_access,
            inverse_mass_matrix,
            rng,
        )
    end
    return true
end

# Access builders for the masked path. Unlike the emission scheduler, the
# control-block masks alias the live workspace arrays (no per-step rebuild).

function _masked_nuts_expand_access(
    workspace::BatchedNUTSWorkspace,
    round_depth::Int,
)
    ir = BatchedNUTSExpandIR(
        round_depth,
        count(workspace.subtree_active),
        1 << round_depth,
        workspace.subtree_active,
        workspace.control.step_direction,
    )
    block = BatchedNUTSExpandControlBlock(
        ir,
        workspace.subtree_active,
        workspace.control.step_direction,
    )
    descriptor = _batched_nuts_step_descriptor(workspace, block)
    state = _batched_nuts_step_state(workspace, descriptor)
    frame = _batched_nuts_kernel_frame(workspace, state)
    return _batched_nuts_kernel_access(workspace, frame)
end

function _masked_nuts_merge_access(
    workspace::BatchedNUTSWorkspace,
    round_depth::Int,
)
    ir = BatchedNUTSMergeIR(
        round_depth,
        count(workspace.subtree_active),
        workspace.control.scheduler.subtree_started,
        workspace.subtree_active,
    )
    block = BatchedNUTSMergeControlBlock(
        ir,
        workspace.control.scheduler.subtree_started,
        workspace.subtree_active,
    )
    descriptor = _batched_nuts_step_descriptor(workspace, block)
    state = _batched_nuts_step_state(workspace, descriptor)
    frame = _batched_nuts_kernel_frame(workspace, state)
    return _batched_nuts_kernel_access(workspace, frame)
end

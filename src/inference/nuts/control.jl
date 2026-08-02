struct NUTSSubtreeSummary
    log_weight::Float64
    accept_stat_sum::Float64
    accept_stat_count::Int
    integration_steps::Int
    turning::Bool
    divergent::Bool
end

function _reset_nuts_subtree_summary!(
    summary::NUTSSubtreeMetadataState,
)
    summary.log_weight = -Inf
    summary.accept_stat_sum = 0.0
    summary.accept_stat_count = 0
    summary.integration_steps = 0
    summary.proposed_energy = Inf
    summary.delta_energy = Inf
    summary.proposal_energy = Inf
    summary.proposal_energy_error = Inf
    summary.accept_prob = 0.0
    summary.candidate_log_weight = -Inf
    summary.combined_log_weight = -Inf
    summary.turning = false
    summary.divergent = false
    return summary
end

function _nuts_subtree_summary(
    summary::NUTSSubtreeMetadataState,
)
    return NUTSSubtreeSummary(
        summary.log_weight,
        summary.accept_stat_sum,
        summary.accept_stat_count,
        summary.integration_steps,
        summary.turning,
        summary.divergent,
    )
end

function _copyto_nuts_state!(destination::NUTSState, source::NUTSState)
    copyto!(destination.position, source.position)
    copyto!(destination.momentum, source.momentum)
    destination.logjoint = source.logjoint
    copyto!(destination.gradient, source.gradient)
    return destination
end

function _copy_masked_columns!(
    destination::AbstractMatrix,
    source::AbstractMatrix,
    mask::AbstractVector{Bool},
)
    size(destination) == size(source) ||
        throw(
            DimensionMismatch("expected masked column copy inputs of matching size, got $(size(destination)) and $(size(source))"),
        )
    length(mask) == size(destination, 2) ||
        throw(DimensionMismatch("expected masked column copy mask of length $(size(destination, 2)), got $(length(mask))"))
    for chain_index in eachindex(mask)
        mask[chain_index] || continue
        copyto!(view(destination, :, chain_index), view(source, :, chain_index))
    end
    return destination
end

function _copy_masked_nuts_buffers!(
    destination_position::AbstractMatrix,
    destination_momentum::AbstractMatrix,
    destination_gradient::AbstractMatrix,
    source_position::AbstractMatrix,
    source_momentum::AbstractMatrix,
    source_gradient::AbstractMatrix,
    mask::AbstractVector{Bool},
)
    _copy_masked_columns!(destination_position, source_position, mask)
    _copy_masked_columns!(destination_momentum, source_momentum, mask)
    _copy_masked_columns!(destination_gradient, source_gradient, mask)
    return destination_position, destination_momentum, destination_gradient
end

function _copy_masked_values!(
    destination::AbstractVector,
    source::AbstractVector,
    mask::AbstractVector{Bool},
)
    length(destination) == length(source) ||
        throw(
            DimensionMismatch(
                "expected masked value copy inputs of matching length, got $(length(destination)) and $(length(source))",
            ),
        )
    length(mask) == length(destination) ||
        throw(DimensionMismatch("expected masked value copy mask of length $(length(destination)), got $(length(mask))"))
    for index in eachindex(mask)
        mask[index] || continue
        destination[index] = source[index]
    end
    return destination
end

function _single_chain_mask!(
    mask::AbstractVector{Bool},
    chain_index::Int,
)
    1 <= chain_index <= length(mask) ||
        throw(BoundsError(mask, chain_index))
    fill!(mask, false)
    mask[chain_index] = true
    return mask
end

function _sync_batched_tree_logjoint!(
    workspace::BatchedNUTSWorkspace,
    mask::AbstractVector{Bool},
)
    for chain_index in eachindex(mask)
        mask[chain_index] || continue
        tree_workspace = workspace.column_tree_workspaces[chain_index]
        tree_workspace.current.logjoint = workspace.tree_current_logjoint[chain_index]
        tree_workspace.next.logjoint = workspace.proposed_logjoint[chain_index]
        tree_workspace.left.logjoint = workspace.tree_left_logjoint[chain_index]
        tree_workspace.right.logjoint = workspace.tree_right_logjoint[chain_index]
        tree_workspace.proposal.logjoint = workspace.tree_proposal_logjoint[chain_index]
    end
    return workspace
end

function _sync_batched_continuation_logjoint!(
    workspace::BatchedNUTSWorkspace,
    mask::AbstractVector{Bool},
)
    for chain_index in eachindex(mask)
        mask[chain_index] || continue
        continuation = workspace.column_continuation_states[chain_index]
        continuation.left.logjoint = workspace.left_logjoint[chain_index]
        continuation.right.logjoint = workspace.right_logjoint[chain_index]
        continuation.proposal.logjoint = workspace.continuation_proposal_logjoint[chain_index]
    end
    return workspace
end

function _copy_single_batched_continuation_frontier_from_tree!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
    direction::Int,
)
    if direction < 0
        mask = _single_chain_mask!(workspace.subtree_copy_left, chain_index)
        fill!(workspace.subtree_copy_right, false)
        _copy_masked_nuts_buffers!(
            workspace.left_position,
            workspace.left_momentum,
            workspace.left_gradient,
            workspace.tree_left_position,
            workspace.tree_left_momentum,
            workspace.tree_left_gradient,
            mask,
        )
        _copy_masked_values!(workspace.left_logjoint, workspace.tree_left_logjoint, mask)
        _sync_batched_continuation_logjoint!(workspace, mask)
    else
        mask = _single_chain_mask!(workspace.subtree_copy_right, chain_index)
        fill!(workspace.subtree_copy_left, false)
        _copy_masked_nuts_buffers!(
            workspace.right_position,
            workspace.right_momentum,
            workspace.right_gradient,
            workspace.tree_right_position,
            workspace.tree_right_momentum,
            workspace.tree_right_gradient,
            mask,
        )
        _copy_masked_values!(workspace.right_logjoint, workspace.tree_right_logjoint, mask)
        _sync_batched_continuation_logjoint!(workspace, mask)
    end
    return workspace
end

function _copy_single_batched_continuation_proposal_from_tree!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
)
    mask = _single_chain_mask!(workspace.subtree_select_proposal, chain_index)
    _copy_masked_nuts_buffers!(
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        mask,
    )
    _copy_masked_values!(
        workspace.continuation_proposal_logjoint,
        workspace.tree_proposal_logjoint,
        mask,
    )
    _sync_batched_continuation_logjoint!(workspace, mask)
    return workspace
end

function _update_single_batched_continuation_turning!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
    inverse_mass_matrix::Union{AbstractVector,AbstractMatrix},
)
    active = _single_chain_mask!(workspace.subtree_active, chain_index)
    _batched_is_turning!(
        workspace.subtree_merged_turning,
        workspace.left_position,
        workspace.right_position,
        workspace.left_momentum,
        workspace.right_momentum,
        inverse_mass_matrix,
        active,
    )
    workspace.subtree_active[chain_index] = false
    return workspace
end

function _reset_batched_nuts_subtree_scratch!(
    workspace::BatchedNUTSWorkspace,
)
    fill!(workspace.subtree_log_weight, -Inf)
    fill!(workspace.subtree_accept_stat_sum, 0.0)
    fill!(workspace.subtree_accept_stat_count, 0)
    fill!(workspace.subtree_integration_steps, 0)
    fill!(workspace.subtree_proposed_energy, Inf)
    fill!(workspace.subtree_delta_energy, Inf)
    fill!(workspace.subtree_proposal_energy, Inf)
    fill!(workspace.subtree_proposal_energy_error, Inf)
    fill!(workspace.subtree_accept_prob, 0.0)
    fill!(workspace.subtree_candidate_log_weight, -Inf)
    fill!(workspace.subtree_combined_log_weight, -Inf)
    fill!(workspace.subtree_turning, false)
    fill!(workspace.subtree_merged_turning, false)
    fill!(workspace.subtree_divergent, false)
    fill!(workspace.subtree_active, false)
    fill!(workspace.control.scheduler.subtree_started, false)
    fill!(workspace.subtree_copy_left, false)
    fill!(workspace.subtree_copy_right, false)
    fill!(workspace.subtree_select_proposal, false)
    workspace.control.scheduler.active_depth = 0
    workspace.control.scheduler.active_depth_count = 0
    workspace.control.scheduler.phase = NUTSSchedulerIdle
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _update_batched_nuts_continuation_active!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
)
    any_active = false
    for chain_index in eachindex(workspace.control.tree_depths)
        workspace.control.scheduler.continuation_active[chain_index] = _nuts_continuation_active(
            workspace.control.tree_depths[chain_index],
            max_tree_depth,
            workspace.control.divergent_step[chain_index],
            workspace.control.continuation_turning[chain_index],
        )
        any_active |= workspace.control.scheduler.continuation_active[chain_index]
    end
    return any_active
end

function _select_batched_nuts_active_depth!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
)
    workspace.control.scheduler.active_depth = 0
    workspace.control.scheduler.active_depth_count = 0
    for depth = 1:(max_tree_depth-1)
        depth_count = 0
        for chain_index in eachindex(workspace.control.tree_depths)
            workspace.control.tree_depths[chain_index] == depth || continue
            workspace.control.scheduler.continuation_active[chain_index] || continue
            depth_count += 1
        end
        if depth_count > workspace.control.scheduler.active_depth_count
            workspace.control.scheduler.active_depth = depth
            workspace.control.scheduler.active_depth_count = depth_count
        end
    end
    return workspace.control.scheduler.active_depth_count > 0
end

function _batched_nuts_active_depth(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
)
    _update_batched_nuts_continuation_active!(workspace, max_tree_depth)
    _select_batched_nuts_active_depth!(workspace, max_tree_depth)
    return workspace.control.scheduler.active_depth, workspace.control.scheduler.active_depth_count
end

function _activate_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
)
    fill!(workspace.subtree_active, false)
    any_active = false
    for chain_index in eachindex(workspace.control.tree_depths)
        workspace.control.tree_depths[chain_index] == workspace.control.scheduler.active_depth || continue
        workspace.control.scheduler.continuation_active[chain_index] || continue
        workspace.subtree_active[chain_index] = true
        any_active = true
    end
    return any_active
end

function _prepare_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
    rng::AbstractRNG,
)
    _reset_batched_nuts_subtree_scratch!(workspace)
    _update_batched_nuts_continuation_active!(workspace, max_tree_depth) || return 0
    _select_batched_nuts_active_depth!(
        workspace,
        max_tree_depth,
    ) || return 0
    _activate_batched_nuts_subtree_cohort!(workspace) || return 0
    _sample_batched_nuts_directions!(workspace.control.step_direction, rng, workspace.subtree_active)
    _initialize_batched_nuts_subtree_states!(workspace, workspace.subtree_active)
    return workspace.control.scheduler.active_depth
end

function _begin_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    max_tree_depth::Int,
    rng::AbstractRNG,
)
    active_depth = _prepare_batched_nuts_subtree_cohort!(
        workspace,
        max_tree_depth,
        rng,
    )
    if active_depth > 0
        workspace.control.scheduler.remaining_steps = 1 << active_depth
        workspace.control.scheduler.phase = NUTSSchedulerExpand
        return true
    end
    workspace.control.scheduler.phase = NUTSSchedulerDone
    workspace.control.scheduler.remaining_steps = 0
    return false
end

function _batched_nuts_merge_masks(
    workspace::BatchedNUTSWorkspace,
)
    num_chains = length(workspace.subtree_active)
    started_chains = falses(num_chains)
    merge_active = falses(num_chains)
    for chain_index in eachindex(workspace.control.tree_depths)
        workspace.control.tree_depths[chain_index] ==
        workspace.control.scheduler.active_depth || continue
        started =
            workspace.subtree_integration_steps[chain_index] > 0 ||
            workspace.subtree_divergent[chain_index]
        started_chains[chain_index] = started
        # An invalid subtree (internal U-turn or divergence) never merges:
        # canonical multinomial NUTS discards the entire final doubling.
        merge_active[chain_index] =
            started &&
            workspace.subtree_integration_steps[chain_index] > 0 &&
            !workspace.subtree_turning[chain_index] &&
            !workspace.subtree_divergent[chain_index]
    end
    return started_chains, merge_active
end

function _batched_nuts_control_ir(
    workspace::BatchedNUTSWorkspace,
)
    scheduler = workspace.control.scheduler
    if scheduler.phase == NUTSSchedulerIdle
        return BatchedNUTSIdleIR()
    elseif scheduler.phase == NUTSSchedulerExpand
        return BatchedNUTSExpandIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            scheduler.remaining_steps,
            copy(workspace.subtree_active),
            copy(workspace.control.step_direction),
        )
    elseif scheduler.phase == NUTSSchedulerMerge
        started_chains, merge_active = _batched_nuts_merge_masks(workspace)
        return BatchedNUTSMergeIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            started_chains,
            merge_active,
        )
    end
    return BatchedNUTSDoneIR()
end

function _batched_nuts_control_block(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_control_block(_batched_nuts_control_ir(workspace))
end

function _batched_nuts_control_block(
    ir::BatchedNUTSIdleIR,
)
    return BatchedNUTSIdleControlBlock(ir)
end

function _batched_nuts_control_block(
    ir::BatchedNUTSExpandIR,
)
    return BatchedNUTSExpandControlBlock(
        ir,
        copy(ir.active_chains),
        copy(ir.step_direction),
    )
end

function _batched_nuts_control_block(
    ir::BatchedNUTSMergeIR,
)
    return BatchedNUTSMergeControlBlock(
        ir,
        copy(ir.started_chains),
        copy(ir.merge_active),
    )
end

function _batched_nuts_control_block(
    ir::BatchedNUTSDoneIR,
)
    return BatchedNUTSDoneControlBlock(ir)
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_step_descriptor(
        workspace,
        _batched_nuts_control_block(workspace),
    )
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSIdleControlBlock,
)
    return BatchedNUTSIdleStepDescriptor(block)
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSExpandControlBlock,
)
    return BatchedNUTSExpandStepDescriptor(
        block,
        workspace.subtree_copy_left,
        workspace.subtree_copy_right,
        workspace.subtree_select_proposal,
        workspace.subtree_turning,
    )
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSMergeControlBlock,
)
    return BatchedNUTSMergeStepDescriptor(
        block,
        workspace.continuation_select_proposal,
        workspace.subtree_merged_turning,
    )
end

function _batched_nuts_step_descriptor(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSDoneControlBlock,
)
    return BatchedNUTSDoneStepDescriptor(block)
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_step_state(
        workspace,
        _batched_nuts_step_descriptor(workspace),
    )
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSIdleStepDescriptor,
)
    return BatchedNUTSIdleStepState(descriptor)
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSExpandStepDescriptor,
)
    return BatchedNUTSExpandStepState(
        descriptor,
        workspace.subtree_log_weight,
        workspace.subtree_proposed_energy,
        workspace.subtree_delta_energy,
        workspace.subtree_proposal_energy,
        workspace.subtree_proposal_energy_error,
        workspace.subtree_accept_prob,
        workspace.subtree_candidate_log_weight,
        workspace.subtree_combined_log_weight,
    )
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSMergeStepDescriptor,
)
    return BatchedNUTSMergeStepState(
        descriptor,
        workspace.subtree_proposal_energy,
        workspace.subtree_proposal_energy_error,
        workspace.continuation_candidate_log_weight,
        workspace.continuation_combined_log_weight,
    )
end

function _batched_nuts_step_state(
    workspace::BatchedNUTSWorkspace,
    descriptor::BatchedNUTSDoneStepDescriptor,
)
    return BatchedNUTSDoneStepState(descriptor)
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_kernel_frame(
        workspace,
        _batched_nuts_step_state(workspace),
    )
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSIdleStepState,
)
    return BatchedNUTSIdleKernelFrame(state)
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSExpandStepState,
)
    return BatchedNUTSExpandKernelFrame(
        state,
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        workspace.tree_current_logjoint,
        workspace.tree_next_position,
        workspace.tree_next_momentum,
        workspace.tree_next_gradient,
        workspace.proposed_logjoint,
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.tree_left_logjoint,
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.tree_right_logjoint,
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.tree_proposal_logjoint,
        workspace.current_energy,
    )
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSMergeStepState,
)
    return BatchedNUTSMergeKernelFrame(
        state,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.left_logjoint,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.right_logjoint,
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.tree_proposal_logjoint,
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.proposed_logjoint,
        workspace.continuation_proposal_logjoint,
        workspace.continuation_log_weight,
        workspace.current_energy,
    )
end

function _batched_nuts_kernel_frame(
    workspace::BatchedNUTSWorkspace,
    state::BatchedNUTSDoneStepState,
)
    return BatchedNUTSDoneKernelFrame(state)
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
)
    return _batched_nuts_kernel_program(
        workspace,
        _batched_nuts_kernel_access(workspace),
    )
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSIdleKernelFrame,
)
    return _batched_nuts_kernel_program(
        workspace,
        _batched_nuts_kernel_access(workspace, frame),
    )
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSExpandKernelFrame,
)
    return _batched_nuts_kernel_program(
        workspace,
        _batched_nuts_kernel_access(workspace, frame),
    )
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSMergeKernelFrame,
)
    return _batched_nuts_kernel_program(
        workspace,
        _batched_nuts_kernel_access(workspace, frame),
    )
end

function _batched_nuts_kernel_program(
    workspace::BatchedNUTSWorkspace,
    frame::BatchedNUTSDoneKernelFrame,
)
    return _batched_nuts_kernel_program(
        workspace,
        _batched_nuts_kernel_access(workspace, frame),
    )
end

_batched_nuts_kernel_ops(program::AbstractBatchedNUTSKernelProgram) = program.ops
_batched_nuts_kernel_steps(::BatchedNUTSIdleKernelProgram) = _BATCHED_NUTS_IDLE_KERNEL_STEPS
_batched_nuts_kernel_steps(::BatchedNUTSExpandKernelProgram) = _BATCHED_NUTS_EXPAND_KERNEL_STEPS
_batched_nuts_kernel_steps(::BatchedNUTSMergeKernelProgram) = _BATCHED_NUTS_MERGE_KERNEL_STEPS
_batched_nuts_kernel_steps(::BatchedNUTSDoneKernelProgram) = _BATCHED_NUTS_DONE_KERNEL_STEPS

_batched_nuts_kernel_returns(::BatchedNUTSIdleKernelProgram) = false
_batched_nuts_kernel_returns(::BatchedNUTSExpandKernelProgram) = true
_batched_nuts_kernel_returns(::BatchedNUTSMergeKernelProgram) = true
_batched_nuts_kernel_returns(::BatchedNUTSDoneKernelProgram) = false

_batched_nuts_frame_control_block(frame::BatchedNUTSIdleKernelFrame) = frame.state.descriptor.block
_batched_nuts_frame_control_block(frame::BatchedNUTSExpandKernelFrame) = frame.state.descriptor.block
_batched_nuts_frame_control_block(frame::BatchedNUTSMergeKernelFrame) = frame.state.descriptor.block
_batched_nuts_frame_control_block(frame::BatchedNUTSDoneKernelFrame) = frame.state.descriptor.block

function _batched_nuts_kernel_execution_state()
    return BatchedNUTSKernelExecutionState(false)
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ::BatchedNUTSIdleIR,
)
    workspace.control.scheduler.phase = NUTSSchedulerIdle
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSIdleControlBlock,
)
    return _load_batched_nuts_control_ir!(workspace, block.ir)
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ir::BatchedNUTSExpandIR,
)
    workspace.control.scheduler.active_depth = ir.active_depth
    workspace.control.scheduler.active_depth_count = ir.active_depth_count
    workspace.control.scheduler.phase = NUTSSchedulerExpand
    workspace.control.scheduler.remaining_steps = ir.remaining_steps
    copyto!(workspace.subtree_active, ir.active_chains)
    copyto!(workspace.control.step_direction, ir.step_direction)
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSExpandControlBlock,
)
    _load_batched_nuts_control_ir!(workspace, block.ir)
    copyto!(workspace.subtree_active, block.active_chains)
    copyto!(workspace.control.step_direction, block.step_direction)
    return workspace
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ir::BatchedNUTSMergeIR,
)
    workspace.control.scheduler.active_depth = ir.active_depth
    workspace.control.scheduler.active_depth_count = ir.active_depth_count
    workspace.control.scheduler.phase = NUTSSchedulerMerge
    workspace.control.scheduler.remaining_steps = 0
    copyto!(workspace.control.scheduler.subtree_started, ir.started_chains)
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSMergeControlBlock,
)
    _load_batched_nuts_control_ir!(workspace, block.ir)
    copyto!(workspace.control.scheduler.subtree_started, block.started_chains)
    return workspace
end

function _load_batched_nuts_control_ir!(
    workspace::BatchedNUTSWorkspace,
    ::BatchedNUTSDoneIR,
)
    workspace.control.scheduler.phase = NUTSSchedulerDone
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _load_batched_nuts_control_block!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSDoneControlBlock,
)
    return _load_batched_nuts_control_ir!(workspace, block.ir)
end

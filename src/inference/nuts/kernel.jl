function _advance_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    inverse_mass_matrix,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    any_active = false
    fill!(access.copy_left, false)
    fill!(access.copy_right, false)
    fill!(access.select_proposal, false)
    # access.turning (workspace.subtree_turning) is NOT cleared here: it is
    # reset once per subtree by _reset_batched_nuts_subtree_scratch! and must
    # stay sticky across leaf steps so the merge gate can discard a subtree
    # that U-turned on any earlier leaf.
    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] || continue
        tree_workspace = workspace.column_tree_workspaces[chain_index]

        if !workspace.control.step_valid[chain_index]
            workspace.subtree_divergent[chain_index] = true
            workspace.subtree_active[chain_index] = false
            continue
        end

        access.current_logjoint[chain_index] = access.proposed_logjoint[chain_index]
        tree_workspace.next.logjoint = access.proposed_logjoint[chain_index]
        _copyto_nuts_state!(tree_workspace.current, tree_workspace.next)
        workspace.subtree_integration_steps[chain_index] += 1

        if workspace.control.step_direction[chain_index] < 0
            access.copy_left[chain_index] = true
            access.left_logjoint[chain_index] = access.current_logjoint[chain_index]
        else
            access.copy_right[chain_index] = true
            access.right_logjoint[chain_index] = access.current_logjoint[chain_index]
        end

        leaf = _advance_tree_leaf(
            access.proposed_energy[chain_index],
            access.current_energy[chain_index],
            divergence_threshold,
            access.log_weight[chain_index],
            rng,
        )
        access.delta_energy[chain_index] = leaf.delta_energy
        if leaf.divergent
            workspace.subtree_divergent[chain_index] = true
            workspace.subtree_active[chain_index] = false
            continue
        end

        access.accept_prob[chain_index] = leaf.accept_prob
        workspace.subtree_accept_stat_sum[chain_index] += leaf.accept_prob
        workspace.subtree_accept_stat_count[chain_index] += 1
        access.candidate_log_weight[chain_index] = leaf.candidate_log_weight
        access.combined_log_weight[chain_index] = leaf.combined_log_weight
        if leaf.select_proposal
            access.select_proposal[chain_index] = true
            access.proposal_logjoint[chain_index] = access.current_logjoint[chain_index]
            access.proposal_energy[chain_index] = access.proposed_energy[chain_index]
            access.proposal_energy_error[chain_index] = leaf.delta_energy
        end
        access.log_weight[chain_index] = leaf.combined_log_weight

        # Generalized (dyadic) U-turn: the new leaf lives in tree_workspace.current
        # (just copied from next). Even leaves store a checkpoint; odd leaves test
        # every dyadic block ending here against its start checkpoint.
        leaf_index = workspace.subtree_integration_steps[chain_index] - 1
        if iseven(leaf_index)
            _store_tree_checkpoint!(
                tree_workspace.checkpoint_positions,
                tree_workspace.checkpoint_momenta,
                leaf_index,
                tree_workspace.current.position,
                tree_workspace.current.momentum,
            )
        elseif _dyadic_turning(
            tree_workspace.checkpoint_positions,
            tree_workspace.checkpoint_momenta,
            leaf_index,
            tree_workspace.current.position,
            tree_workspace.current.momentum,
            workspace.control.step_direction[chain_index],
            _chain_inverse_mass(inverse_mass_matrix, chain_index),
        )
            access.turning[chain_index] = true
        end
    end

    _copy_masked_nuts_buffers!(
        access.left_position,
        access.left_momentum,
        access.left_gradient,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        access.copy_left,
    )
    _copy_masked_nuts_buffers!(
        access.right_position,
        access.right_momentum,
        access.right_gradient,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        access.copy_right,
    )
    _copy_masked_nuts_buffers!(
        access.proposal_position,
        access.proposal_momentum,
        access.proposal_gradient,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        access.select_proposal,
    )
    _copy_masked_values!(
        access.proposal_logjoint,
        access.current_logjoint,
        access.select_proposal,
    )
    _sync_batched_tree_logjoint!(
        workspace,
        workspace.subtree_active .|
        access.copy_left .|
        access.copy_right .|
        access.select_proposal,
    )

    # Subtree-level turning is now decided per-particle by the dyadic checkpoint
    # scheme inside the loop above (access.turning); the merge-level whole-
    # trajectory U-turn check lives in _merge_batched_nuts_subtree_cohort!.
    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] =
            workspace.subtree_active[chain_index] && !access.turning[chain_index]
        any_active |= workspace.subtree_active[chain_index]
    end
    return any_active
end

function _activate_batched_nuts_subtree_merge_cohort!(
    workspace::BatchedNUTSWorkspace,
    block::BatchedNUTSMergeControlBlock,
)
    copyto!(workspace.control.scheduler.subtree_started, block.started_chains)
    fill!(workspace.subtree_active, false)
    any_started = false
    for chain_index in eachindex(workspace.control.tree_depths)
        block.started_chains[chain_index] || continue

        workspace.control.tree_depths[chain_index] += 1
        if !block.merge_active[chain_index]
            if workspace.subtree_integration_steps[chain_index] == 0
                workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
            else
                # Invalid subtree with completed leaves: discard the doubling
                # entirely, keeping only accounting and termination flags.
                _discard_invalid_batched_subtree!(workspace, chain_index)
            end
            continue
        end

        workspace.subtree_active[chain_index] = true
        any_started = true
    end
    return any_started
end

function _merge_batched_nuts_subtree_cohort!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSMergeKernelAccess,
    inverse_mass_matrix,
    rng::AbstractRNG,
)
    # The select mask persists across merges; clear the lanes that are not in
    # this merge so the masked copies below cannot replay a stale selection
    # (on a chain's first transition that would copy an uninitialized
    # tree-proposal column into the live proposal buffers).
    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] && continue
        access.select_proposal[chain_index] = false
    end
    _merge_batched_nuts_continuation_frontiers!(workspace, workspace.subtree_active)
    _batched_is_turning!(
        access.merged_turning,
        access.left_position,
        access.right_position,
        access.left_momentum,
        access.right_momentum,
        inverse_mass_matrix,
        workspace.subtree_active,
    )

    for chain_index in eachindex(workspace.subtree_active)
        workspace.subtree_active[chain_index] || continue
        merge = _merge_subtree_stats(
            access.continuation_log_weight[chain_index],
            workspace.subtree_log_weight[chain_index],
            rng,
        )
        access.select_proposal[chain_index] = merge.select_proposal
        access.candidate_log_weight[chain_index] = merge.candidate_log_weight
        access.combined_log_weight[chain_index] = merge.combined_log_weight
        if merge.select_proposal
            access.proposal_energy[chain_index] = _hamiltonian(
                access.tree_proposal_logjoint[chain_index],
                view(access.tree_proposal_momentum, :, chain_index),
                _chain_inverse_mass(inverse_mass_matrix, chain_index),
            )
            access.proposal_energy_error[chain_index] =
                access.proposal_energy[chain_index] -
                access.current_energy[chain_index]
        end
        _merge_batched_subtree_summary!(workspace, chain_index)
    end

    _copy_masked_nuts_buffers!(
        access.proposal_position,
        access.proposal_momentum,
        access.proposal_gradient,
        access.tree_proposal_position,
        access.tree_proposal_momentum,
        access.tree_proposal_gradient,
        access.select_proposal,
    )
    _copy_masked_values!(
        access.proposed_logjoint,
        access.continuation_proposal_logjoint,
        access.select_proposal,
    )
    _sync_batched_continuation_logjoint!(
        workspace,
        workspace.subtree_active .| access.select_proposal,
    )
    return workspace
end

# The kernel step sequence for the current scheduler phase (expand/merge/
# idle/done) runs directly: build the access view over the live workspace
# buffers, pick the phase program, and execute its steps in order.
function _step_batched_nuts_subtree_scheduler!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    access = _batched_nuts_kernel_access(workspace)
    program = _batched_nuts_kernel_program(workspace, access)
    execution = _batched_nuts_kernel_execution_state()
    for step in _batched_nuts_kernel_steps(program)
        _execute_batched_nuts_kernel_step!(
            workspace,
            access,
            step,
            execution,
            model,
            inverse_mass_matrix,
            args,
            constraints,
            step_size,
            divergence_threshold,
            rng,
        )
    end
    return _batched_nuts_kernel_returns(program)
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::AbstractBatchedNUTSKernelAccess,
    ::BatchedNUTSReloadControlStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_reload_control!(workspace, access)
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    ::BatchedNUTSLeapfrogStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_leapfrog!(
        workspace,
        access,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    ::BatchedNUTSHamiltonianStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_hamiltonian!(access, inverse_mass_matrix)
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    ::BatchedNUTSAdvanceStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    execution.any_active = _advance_batched_nuts_subtree_cohort!(
        workspace,
        access,
        inverse_mass_matrix,
        divergence_threshold,
        rng,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSMergeKernelAccess,
    ::BatchedNUTSActivateMergeStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    execution.any_active = _activate_batched_nuts_subtree_merge_cohort!(
        workspace,
        access.block,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSMergeKernelAccess,
    ::BatchedNUTSMergeStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    execution.any_active || return nothing
    _merge_batched_nuts_subtree_cohort!(
        workspace,
        access,
        inverse_mass_matrix,
        rng,
    )
    return nothing
end

function _execute_batched_nuts_kernel_step!(
    workspace::BatchedNUTSWorkspace,
    access::AbstractBatchedNUTSKernelAccess,
    ::BatchedNUTSTransitionPhaseStep,
    execution::BatchedNUTSKernelExecutionState,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    _batched_nuts_kernel_transition_phase!(workspace, access, execution)
    return nothing
end

function _batched_nuts_kernel_reload_control!(
    workspace::BatchedNUTSWorkspace,
    access::AbstractBatchedNUTSKernelAccess,
)
    _load_batched_nuts_control_block!(workspace, _batched_nuts_access_control_block(access))
    return workspace
end

function _batched_nuts_kernel_leapfrog!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    model::TeaModel,
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
)
    _batched_nuts_leapfrog_step_to!(
        workspace,
        model,
        access.next_position,
        access.next_momentum,
        access.next_gradient,
        access.proposed_logjoint,
        access.current_position,
        access.current_momentum,
        access.current_gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        access.block.step_direction,
        access.block.active_chains,
    )
    return access
end

function _batched_nuts_kernel_hamiltonian!(
    access::BatchedNUTSExpandKernelAccess,
    inverse_mass_matrix,
)
    _batched_hamiltonian!(
        access.proposed_energy,
        access.proposed_logjoint,
        access.next_momentum,
        inverse_mass_matrix,
    )
    return access
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSExpandKernelAccess,
    execution::BatchedNUTSKernelExecutionState,
)
    workspace.control.scheduler.remaining_steps -= 1
    if !execution.any_active || workspace.control.scheduler.remaining_steps == 0
        workspace.control.scheduler.phase = NUTSSchedulerMerge
    end
    return workspace
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    access::BatchedNUTSMergeKernelAccess,
    execution::BatchedNUTSKernelExecutionState,
)
    workspace.control.scheduler.phase = NUTSSchedulerDone
    workspace.control.scheduler.remaining_steps = 0
    return workspace
end

function _batched_nuts_kernel_transition_phase!(
    workspace::BatchedNUTSWorkspace,
    access::Union{BatchedNUTSIdleKernelAccess,BatchedNUTSDoneKernelAccess},
    execution::BatchedNUTSKernelExecutionState,
)
    return workspace
end

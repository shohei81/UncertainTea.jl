function _build_nuts_subtree(
    subtree_workspace::NUTSSubtreeWorkspace,
    target::AbstractDensityTarget,
    start_state::NUTSState,
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    step_size::Float64,
    direction::Int,
    depth::Int,
    initial_hamiltonian::Float64,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    current = subtree_workspace.current
    next = subtree_workspace.next
    left = subtree_workspace.left
    right = subtree_workspace.right
    proposal = subtree_workspace.proposal
    summary = subtree_workspace.summary
    checkpoint_positions = subtree_workspace.checkpoint_positions
    checkpoint_momenta = subtree_workspace.checkpoint_momenta
    _copyto_nuts_state!(current, start_state)
    _copyto_nuts_state!(left, start_state)
    _copyto_nuts_state!(right, start_state)
    _copyto_nuts_state!(proposal, start_state)
    _reset_nuts_subtree_summary!(summary)

    for _ = 1:(1<<depth)
        if !leapfrog_step!(next, target, current, inverse_mass_matrix, direction * step_size)
            summary.divergent = true
            break
        end
        current, next = next, current
        summary.integration_steps += 1
        if direction < 0
            _copyto_nuts_state!(left, current)
        else
            _copyto_nuts_state!(right, current)
        end

        summary.proposed_energy = _hamiltonian(
            current.logjoint,
            current.momentum,
            inverse_mass_matrix,
        )
        summary.delta_energy = summary.proposed_energy - initial_hamiltonian
        if !isfinite(summary.delta_energy) || summary.delta_energy > divergence_threshold
            summary.divergent = true
            break
        end

        summary.accept_prob = min(1.0, exp(min(0.0, -summary.delta_energy)))
        summary.accept_stat_sum += summary.accept_prob
        summary.accept_stat_count += 1
        summary.candidate_log_weight = -summary.proposed_energy
        summary.combined_log_weight = _logaddexp(
            summary.log_weight,
            summary.candidate_log_weight,
        )
        if !isfinite(summary.log_weight) || log(rand(rng)) <
                                            summary.candidate_log_weight - summary.combined_log_weight
            _copyto_nuts_state!(proposal, current)
            summary.proposal_energy = summary.proposed_energy
            summary.proposal_energy_error = summary.delta_energy
        end
        summary.log_weight = summary.combined_log_weight

        leaf_index = summary.integration_steps - 1
        if iseven(leaf_index)
            _store_tree_checkpoint!(
                checkpoint_positions,
                checkpoint_momenta,
                leaf_index,
                current.position,
                current.momentum,
            )
        elseif _dyadic_turning(
            checkpoint_positions,
            checkpoint_momenta,
            leaf_index,
            current.position,
            current.momentum,
            direction,
            inverse_mass_matrix,
        )
            summary.turning = true
            break
        end
    end

    return _nuts_subtree_summary(summary)
end

function _continue_nuts_proposal!(
    continuation::NUTSContinuationState,
    target::AbstractDensityTarget,
    initial_hamiltonian::Float64,
    tree_workspace::NUTSSubtreeWorkspace,
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    while _nuts_continuation_active(
        continuation.tree_depth,
        max_tree_depth,
        continuation.divergent,
        continuation.turning,
    )
        direction = _sample_nuts_direction(rng)
        subtree = _build_nuts_subtree(
            tree_workspace,
            target,
            _nuts_subtree_start_state(continuation, direction),
            inverse_mass_matrix,
            step_size,
            direction,
            continuation.tree_depth,
            initial_hamiltonian,
            divergence_threshold,
            rng,
        )
        continuation.tree_depth += 1

        if subtree.integration_steps == 0
            continuation.divergent = subtree.divergent
            break
        end

        if subtree.turning || subtree.divergent
            # Canonical multinomial NUTS (Stan base_nuts) discards the entire
            # final doubling when the new subtree is invalid (internal U-turn
            # or divergence): no weight merge, no proposal swap, no frontier
            # update. Only the leapfrog/accept-stat accounting carries over.
            continuation.integration_steps += subtree.integration_steps
            continuation.accept_stat_sum += subtree.accept_stat_sum
            continuation.accept_stat_count += subtree.accept_stat_count
            continuation.turning = subtree.turning
            continuation.divergent = subtree.divergent
            break
        end

        _copy_nuts_continuation_frontier_from_tree!(
            continuation,
            tree_workspace,
            direction,
        )

        combined_log_weight = continuation.log_weight
        if isfinite(tree_workspace.summary.log_weight)
            combined_log_weight = _logaddexp(
                continuation.log_weight,
                tree_workspace.summary.log_weight,
            )
            # Biased progressive sampling at the doubling merge (Stan
            # base_nuts; Betancourt 2017 App. A.3.2): P(swap) =
            # min(1, w_subtree / w_continuation). See _merge_subtree_stats.
            if log(rand(rng)) < tree_workspace.summary.log_weight - continuation.log_weight
                _copy_nuts_continuation_proposal_from_tree!(
                    continuation,
                    tree_workspace,
                )
            end
        end

        _merge_nuts_subtree_summary!(
            continuation,
            tree_workspace,
            combined_log_weight,
        )
        _merge_nuts_continuation_turning!(
            continuation,
            tree_workspace.summary.turning,
            inverse_mass_matrix,
        )
    end
    return continuation
end

function _continue_batched_nuts_proposal!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
    target::AbstractDensityTarget,
    initial_hamiltonian::Float64,
    tree_workspace::NUTSSubtreeWorkspace,
    inverse_mass_matrix::Vector{Float64},
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    continuation = workspace.column_continuation_states[chain_index]
    while _nuts_continuation_active(
        workspace.control.tree_depths[chain_index],
        max_tree_depth,
        workspace.control.divergent_step[chain_index],
        workspace.control.continuation_turning[chain_index],
    )
        direction = _sample_nuts_direction(rng)
        subtree = _build_nuts_subtree(
            tree_workspace,
            target,
            _nuts_subtree_start_state(continuation, direction),
            inverse_mass_matrix,
            step_size,
            direction,
            workspace.control.tree_depths[chain_index],
            initial_hamiltonian,
            divergence_threshold,
            rng,
        )
        workspace.control.tree_depths[chain_index] += 1
        workspace.subtree_log_weight[chain_index] = subtree.log_weight
        workspace.subtree_accept_stat_sum[chain_index] = subtree.accept_stat_sum
        workspace.subtree_accept_stat_count[chain_index] = subtree.accept_stat_count
        workspace.subtree_integration_steps[chain_index] = subtree.integration_steps
        workspace.subtree_turning[chain_index] = subtree.turning
        workspace.subtree_divergent[chain_index] = subtree.divergent
        workspace.subtree_merged_turning[chain_index] = false
        workspace.subtree_proposal_energy[chain_index] = Inf
        workspace.subtree_proposal_energy_error[chain_index] = Inf

        if subtree.integration_steps == 0
            workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
            break
        end

        if subtree.turning || subtree.divergent
            # Invalid final doubling: discard it entirely (see the scalar
            # continuation above), keeping only the accounting and flags.
            _discard_invalid_batched_subtree!(workspace, chain_index)
            break
        end

        _copy_single_batched_continuation_frontier_from_tree!(
            workspace,
            chain_index,
            direction,
        )

        workspace.continuation_select_proposal[chain_index] = false
        workspace.continuation_candidate_log_weight[chain_index] = -Inf
        workspace.continuation_combined_log_weight[chain_index] =
            workspace.continuation_log_weight[chain_index]
        if isfinite(workspace.subtree_log_weight[chain_index])
            workspace.subtree_proposal_energy[chain_index] = _hamiltonian(
                tree_workspace.proposal.logjoint,
                tree_workspace.proposal.momentum,
                inverse_mass_matrix,
            )
            workspace.subtree_proposal_energy_error[chain_index] =
                workspace.subtree_proposal_energy[chain_index] - initial_hamiltonian
            workspace.continuation_candidate_log_weight[chain_index] =
                workspace.subtree_log_weight[chain_index]
            workspace.continuation_combined_log_weight[chain_index] = _logaddexp(
                workspace.continuation_log_weight[chain_index],
                workspace.continuation_candidate_log_weight[chain_index],
            )
            # Biased progressive sampling at the doubling merge (Stan
            # base_nuts; Betancourt 2017 App. A.3.2). See _merge_subtree_stats.
            workspace.continuation_select_proposal[chain_index] =
                log(rand(rng)) <
                workspace.continuation_candidate_log_weight[chain_index] -
                workspace.continuation_log_weight[chain_index]
            workspace.tree_proposal_logjoint[chain_index] = tree_workspace.proposal.logjoint
            if workspace.continuation_select_proposal[chain_index]
                _copy_single_batched_continuation_proposal_from_tree!(
                    workspace,
                    chain_index,
                )
            end
        end
        _update_single_batched_continuation_turning!(workspace, chain_index, inverse_mass_matrix)
        _merge_batched_subtree_summary!(workspace, chain_index)
    end
    return workspace
end

function _continue_batched_nuts_batched_subtree!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    max_tree_depth > 1 || return false
    _begin_batched_nuts_subtree_scheduler!(
        workspace,
        max_tree_depth,
        rng,
    ) || return false
    while _step_batched_nuts_subtree_scheduler!(
        workspace,
        model,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        divergence_threshold,
        rng,
    )
    end

    return true
end

# Workspace-reusing NUTS transition: `tree_workspace` and `continuation` are
# scratch state fully re-initialized on entry (the first-step init resets every
# continuation field and each subtree build resets the tree scratch), so hot
# sampler loops allocate them once and pass them to every iteration. The
# returned proposal aliases `continuation.proposal`; callers that reuse the
# continuation must copy the accepted position/gradient values out instead of
# retaining the buffers across iterations.
function _nuts_proposal(
    tree_workspace::NUTSSubtreeWorkspace,
    continuation::NUTSContinuationState,
    target::AbstractDensityTarget,
    position::AbstractVector{Float64},
    current_logjoint::Float64,
    current_gradient::AbstractVector{Float64},
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    initial_momentum = _sample_momentum(rng, inverse_mass_matrix)
    initial_state = _load_nuts_state!(
        tree_workspace.current,
        position,
        initial_momentum,
        current_logjoint,
        current_gradient,
    )
    initial_hamiltonian = _hamiltonian(initial_state.logjoint, initial_state.momentum, inverse_mass_matrix)
    direction = _sample_nuts_direction(rng)
    first_step_valid = leapfrog_step!(
        tree_workspace.next,
        target,
        initial_state,
        inverse_mass_matrix,
        direction * step_size,
    )
    _initialize_nuts_first_trajectory!(
        continuation,
        tree_workspace,
        first_step_valid,
        direction,
        initial_hamiltonian,
        inverse_mass_matrix,
        divergence_threshold,
        rng,
    )
    _continue_nuts_proposal!(
        continuation,
        target,
        initial_hamiltonian,
        tree_workspace,
        inverse_mass_matrix,
        step_size,
        max_tree_depth,
        divergence_threshold,
        rng,
    )
    accept_stat, proposed_energy, energy_error, moved = _nuts_proposal_summary(
        continuation,
        position,
    )
    return continuation.proposal,
    accept_stat,
    continuation.tree_depth,
    continuation.integration_steps,
    proposed_energy,
    energy_error,
    continuation.divergent,
    moved
end

# Allocating convenience wrapper: builds fresh per-call workspaces. Hot loops
# (single-chain NUTS, Gibbs) hoist the workspaces and call the method above.
function _nuts_proposal(
    target::AbstractDensityTarget,
    position::AbstractVector{Float64},
    current_logjoint::Float64,
    current_gradient::AbstractVector{Float64},
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    return _nuts_proposal(
        NUTSSubtreeWorkspace(length(position), max_tree_depth),
        NUTSContinuationState(length(position)),
        target,
        position,
        current_logjoint,
        current_gradient,
        inverse_mass_matrix,
        step_size,
        max_tree_depth,
        divergence_threshold,
        rng,
    )
end

function _batched_nuts_proposals!(
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
    num_chains = size(position, 2)
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
    while _continue_batched_nuts_batched_subtree!(
        workspace,
        model,
        position,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        max_tree_depth,
        divergence_threshold,
        rng,
    )
    end
    for chain_index = 1:num_chains
        target = ModelDensityTarget(
            model,
            _batched_args(args, chain_index),
            _batched_constraints(constraints, chain_index),
            workspace.column_gradient_caches[chain_index],
        )
        _continue_batched_nuts_proposal!(
            workspace,
            chain_index,
            target,
            workspace.current_energy[chain_index],
            workspace.column_tree_workspaces[chain_index],
            _chain_inverse_mass_vector(inverse_mass_matrix, chain_index),
            _chain_step_size(step_size, chain_index),
            max_tree_depth,
            divergence_threshold,
            rng,
        )
    end
    _finalize_batched_nuts_proposals!(workspace, position)
    return workspace
end

function _batched_nuts_state(
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    logjoint::AbstractVector{Float64},
    gradient::AbstractMatrix{Float64},
    chain_index::Int,
)
    return NUTSState(
        view(position, :, chain_index),
        view(momentum, :, chain_index),
        logjoint[chain_index],
        view(gradient, :, chain_index),
    )
end

# Per-iteration buffer reuse (issue #142): `sqrt.(Float64.(imm))` and
# `trues(num_chains)` used to be allocated on every draw; they now fill the
# workspace buffers added in data_types.jl. Elementwise `sqrt(Float64(x))`
# matches the old broadcast bitwise.
function _batched_nuts_sqrt_inverse_mass!(
    workspace::BatchedNUTSWorkspace,
    inverse_mass_matrix::AbstractVector,
)
    buffer = workspace.sqrt_inverse_mass
    @inbounds for index in eachindex(buffer, inverse_mass_matrix)
        buffer[index] = sqrt(Float64(inverse_mass_matrix[index]))
    end
    return buffer
end

# Per-chain overload: one sqrt inverse-mass column per chain.
function _batched_nuts_sqrt_inverse_mass!(
    workspace::BatchedNUTSWorkspace,
    inverse_mass_matrices::AbstractMatrix,
)
    buffer = workspace.sqrt_inverse_mass_columns
    @inbounds for index in eachindex(buffer, inverse_mass_matrices)
        buffer[index] = sqrt(Float64(inverse_mass_matrices[index]))
    end
    return buffer
end

function _all_chains_active!(workspace::BatchedNUTSWorkspace)
    fill!(workspace.all_chains_active, true)
    return workspace.all_chains_active
end

function _initialize_batched_nuts_continuations!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    current_logjoint::AbstractVector{Float64},
    current_gradient::AbstractMatrix{Float64},
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    num_chains = size(position, 2)
    _sample_batched_momentum!(
        workspace.current_momentum,
        rng,
        _batched_nuts_sqrt_inverse_mass!(workspace, inverse_mass_matrix),
    )
    _batched_hamiltonian!(workspace.current_energy, current_logjoint, workspace.current_momentum, inverse_mass_matrix)
    fill!(workspace.control.accepted_step, true)
    _sample_batched_nuts_directions!(workspace.control.step_direction, rng, workspace.control.accepted_step)
    _batched_nuts_leapfrog_step!(
        workspace,
        model,
        position,
        workspace.current_momentum,
        current_gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        workspace.control.step_direction,
        workspace.control.accepted_step,
    )
    _batched_hamiltonian!(workspace.proposed_energy, workspace.proposed_logjoint, workspace.proposal_momentum, inverse_mass_matrix)

    fill!(workspace.continuation_log_weight, -Inf)
    fill!(workspace.continuation_accept_stat_sum, 0.0)
    fill!(workspace.continuation_accept_stat_count, 0)
    fill!(workspace.accept_prob, 0.0)
    fill!(workspace.control.accepted_step, false)
    fill!(workspace.control.divergent_step, false)
    fill!(workspace.control.continuation_turning, false)
    fill!(workspace.control.tree_depths, 1)
    fill!(workspace.control.integration_steps, 0)
    copyto!(workspace.tree_current_logjoint, current_logjoint)
    copyto!(workspace.left_logjoint, current_logjoint)
    copyto!(workspace.right_logjoint, current_logjoint)
    copyto!(workspace.continuation_proposal_logjoint, current_logjoint)
    copyto!(workspace.continuation_proposed_energy, workspace.current_energy)
    fill!(workspace.continuation_delta_energy, 0.0)
    _load_batched_nuts_first_states!(
        workspace,
        position,
        current_logjoint,
        current_gradient,
        _all_chains_active!(workspace),
    )

    for chain_index = 1:num_chains
        tree_workspace = workspace.column_tree_workspaces[chain_index]
        moved, divergent = _initialize_batched_nuts_first_step!(
            workspace,
            chain_index,
            tree_workspace.current,
            tree_workspace.next,
            workspace.control.step_valid[chain_index],
            workspace.control.step_direction[chain_index],
            workspace.current_energy[chain_index],
            _chain_inverse_mass(inverse_mass_matrix, chain_index),
            divergence_threshold,
            rng,
        )
        workspace.control.divergent_step[chain_index] = divergent
        workspace.control.accepted_step[chain_index] = moved
    end
    fill!(workspace.subtree_active, false)
    for chain_index = 1:num_chains
        workspace.subtree_active[chain_index] =
            workspace.control.divergent_step[chain_index] || !workspace.control.accepted_step[chain_index]
    end
    _copy_masked_nuts_buffers!(
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        position,
        workspace.current_momentum,
        current_gradient,
        workspace.subtree_active,
    )
    _copy_masked_values!(workspace.proposed_logjoint, current_logjoint, workspace.subtree_active)
    _copy_masked_values!(workspace.continuation_proposal_logjoint, current_logjoint, workspace.subtree_active)
    return workspace
end

function _batched_nuts_leapfrog_step!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    gradient::AbstractMatrix{Float64},
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    return _batched_nuts_leapfrog_step_to!(
        workspace,
        model,
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.proposed_logjoint,
        position,
        momentum,
        gradient,
        inverse_mass_matrix,
        args,
        constraints,
        step_size,
        direction,
        active,
    )
end

function _batched_nuts_leapfrog_step_to!(
    workspace::BatchedNUTSWorkspace,
    model::TeaModel,
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    gradient::AbstractMatrix{Float64},
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    target = BatchedModelDensityTarget(workspace.gradient_cache)
    return batched_leapfrog_step_to!(
        destination_position,
        destination_momentum,
        destination_gradient,
        destination_logjoint,
        workspace.control.step_valid,
        position,
        momentum,
        gradient,
        target,
        inverse_mass_matrix,
        step_size,
        direction,
        active,
    )
end

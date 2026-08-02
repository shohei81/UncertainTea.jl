function _load_batched_nuts_first_states!(
    workspace::BatchedNUTSWorkspace,
    position::AbstractMatrix,
    current_logjoint::AbstractVector,
    current_gradient::AbstractMatrix,
    active::AbstractVector{Bool},
)
    length(active) == size(position, 2) ||
        throw(DimensionMismatch("expected first-state active mask of length $(size(position, 2)), got $(length(active))"))
    _copy_masked_nuts_buffers!(
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        position,
        workspace.current_momentum,
        current_gradient,
        active,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_next_position,
        workspace.tree_next_momentum,
        workspace.tree_next_gradient,
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        active,
    )
    _copy_masked_values!(workspace.tree_current_logjoint, current_logjoint, active)
    _sync_batched_tree_logjoint!(workspace, active)
    return workspace
end

function _load_nuts_state!(
    destination::NUTSState,
    position::AbstractVector,
    momentum::AbstractVector,
    logjoint::Real,
    gradient::AbstractVector,
)
    copyto!(destination.position, position)
    copyto!(destination.momentum, momentum)
    destination.logjoint = Float64(logjoint)
    copyto!(destination.gradient, gradient)
    return destination
end

function _initialize_nuts_continuation!(
    continuation::NUTSContinuationState,
    left::NUTSState,
    right::NUTSState,
    proposal::NUTSState,
    proposal_energy::Float64,
    proposal_energy_error::Float64,
    log_weight::Float64,
    accept_stat_sum::Float64,
    accept_stat_count::Int,
    integration_steps::Int,
    tree_depth::Int,
    turning::Bool,
    divergent::Bool,
)
    _copyto_nuts_state!(continuation.left, left)
    _copyto_nuts_state!(continuation.right, right)
    _copyto_nuts_state!(continuation.proposal, proposal)
    continuation.proposal_energy = proposal_energy
    continuation.proposal_energy_error = proposal_energy_error
    continuation.log_weight = log_weight
    continuation.accept_stat_sum = accept_stat_sum
    continuation.accept_stat_count = accept_stat_count
    continuation.integration_steps = integration_steps
    continuation.tree_depth = tree_depth
    continuation.turning = turning
    continuation.divergent = divergent
    return continuation
end

function _initialize_nuts_first_step!(
    continuation::NUTSContinuationState,
    current_state::NUTSState,
    proposed_state::NUTSState,
    valid::Bool,
    direction::Int,
    initial_hamiltonian::Float64,
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    _initialize_nuts_continuation!(
        continuation,
        current_state,
        current_state,
        current_state,
        initial_hamiltonian,
        0.0,
        -initial_hamiltonian,
        0.0,
        0,
        0,
        1,
        false,
        false,
    )

    valid || return (false, true)

    if direction < 0
        _copyto_nuts_state!(continuation.left, proposed_state)
    else
        _copyto_nuts_state!(continuation.right, proposed_state)
    end

    proposed_hamiltonian = _hamiltonian(proposed_state.logjoint, proposed_state.momentum, inverse_mass_matrix)
    delta_energy = proposed_hamiltonian - initial_hamiltonian
    continuation.integration_steps = 1
    if !isfinite(delta_energy) || delta_energy > divergence_threshold
        continuation.divergent = true
        return (false, true)
    end

    continuation.accept_stat_sum = min(1.0, exp(min(0.0, -delta_energy)))
    continuation.accept_stat_count = 1
    candidate_log_weight = -proposed_hamiltonian
    combined_log_weight = _logaddexp(continuation.log_weight, candidate_log_weight)
    # Unbiased multinomial between the initial point and the first leaf: this
    # builds the initial two-point trajectory, so it stays unbiased like all
    # within-trajectory selection; only the doubling merges in the expansion
    # loop use the Stan-style biased swap (_merge_subtree_stats).
    moved = log(rand(rng)) < candidate_log_weight - combined_log_weight
    if moved
        _copyto_nuts_state!(continuation.proposal, proposed_state)
        continuation.proposal_energy = proposed_hamiltonian
        continuation.proposal_energy_error = delta_energy
    end
    continuation.log_weight = combined_log_weight
    continuation.turning = _is_turning(
        continuation.left.position,
        continuation.right.position,
        continuation.left.momentum,
        continuation.right.momentum,
        inverse_mass_matrix,
    )
    return (moved, false)
end

function _initialize_nuts_first_trajectory!(
    continuation::NUTSContinuationState,
    tree_workspace::NUTSSubtreeWorkspace,
    valid::Bool,
    direction::Int,
    initial_hamiltonian::Float64,
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    return _initialize_nuts_first_step!(
        continuation,
        tree_workspace.current,
        tree_workspace.next,
        valid,
        direction,
        initial_hamiltonian,
        inverse_mass_matrix,
        divergence_threshold,
        rng,
    )
end

function _initialize_batched_nuts_first_step!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
    current_state::NUTSState,
    proposed_state::NUTSState,
    valid::Bool,
    direction::Int,
    initial_hamiltonian::Float64,
    inverse_mass_matrix::AbstractVector{Float64},
    divergence_threshold::Float64,
    rng::AbstractRNG,
)
    continuation = workspace.column_continuation_states[chain_index]
    _copyto_nuts_state!(continuation.left, current_state)
    _copyto_nuts_state!(continuation.right, current_state)
    _copyto_nuts_state!(continuation.proposal, current_state)
    continuation.proposal_energy = initial_hamiltonian
    continuation.proposal_energy_error = 0.0
    workspace.left_logjoint[chain_index] = current_state.logjoint
    workspace.right_logjoint[chain_index] = current_state.logjoint
    workspace.continuation_proposal_logjoint[chain_index] = current_state.logjoint
    workspace.continuation_log_weight[chain_index] = -initial_hamiltonian
    workspace.continuation_accept_stat_sum[chain_index] = 0.0
    workspace.continuation_accept_stat_count[chain_index] = 0
    workspace.continuation_proposed_energy[chain_index] = initial_hamiltonian
    workspace.continuation_delta_energy[chain_index] = 0.0
    workspace.continuation_accept_prob[chain_index] = 0.0
    workspace.continuation_candidate_log_weight[chain_index] = -Inf
    workspace.continuation_combined_log_weight[chain_index] = -Inf
    workspace.continuation_select_proposal[chain_index] = false
    workspace.control.integration_steps[chain_index] = 0
    workspace.control.tree_depths[chain_index] = 1
    workspace.control.continuation_turning[chain_index] = false
    workspace.control.divergent_step[chain_index] = false

    valid || begin
        workspace.control.divergent_step[chain_index] = true
        return (false, true)
    end

    if direction < 0
        _copyto_nuts_state!(continuation.left, proposed_state)
        workspace.left_logjoint[chain_index] = proposed_state.logjoint
    else
        _copyto_nuts_state!(continuation.right, proposed_state)
        workspace.right_logjoint[chain_index] = proposed_state.logjoint
    end

    workspace.continuation_proposed_energy[chain_index] =
        _hamiltonian(proposed_state.logjoint, proposed_state.momentum, inverse_mass_matrix)
    delta_energy = workspace.continuation_proposed_energy[chain_index] - initial_hamiltonian
    workspace.continuation_delta_energy[chain_index] = delta_energy
    workspace.control.integration_steps[chain_index] = 1
    if !isfinite(delta_energy) || delta_energy > divergence_threshold
        workspace.control.divergent_step[chain_index] = true
        return (false, true)
    end

    workspace.continuation_accept_prob[chain_index] = min(1.0, exp(min(0.0, -delta_energy)))
    workspace.continuation_accept_stat_sum[chain_index] = workspace.continuation_accept_prob[chain_index]
    workspace.continuation_accept_stat_count[chain_index] = 1
    workspace.continuation_candidate_log_weight[chain_index] = -workspace.continuation_proposed_energy[chain_index]
    workspace.continuation_combined_log_weight[chain_index] = _logaddexp(
        workspace.continuation_log_weight[chain_index],
        workspace.continuation_candidate_log_weight[chain_index],
    )
    # Unbiased multinomial between the initial point and the first leaf (see
    # _initialize_nuts_first_step!); doubling merges bias instead.
    workspace.continuation_select_proposal[chain_index] =
        log(rand(rng)) <
        workspace.continuation_candidate_log_weight[chain_index] -
        workspace.continuation_combined_log_weight[chain_index]
    moved = workspace.continuation_select_proposal[chain_index]
    if moved
        _copyto_nuts_state!(continuation.proposal, proposed_state)
        continuation.proposal_energy = workspace.continuation_proposed_energy[chain_index]
        continuation.proposal_energy_error = delta_energy
        workspace.continuation_proposal_logjoint[chain_index] = proposed_state.logjoint
    end
    workspace.continuation_log_weight[chain_index] = workspace.continuation_combined_log_weight[chain_index]
    workspace.control.continuation_turning[chain_index] = _is_turning(
        continuation.left.position,
        continuation.right.position,
        continuation.left.momentum,
        continuation.right.momentum,
        inverse_mass_matrix,
    )
    sync_mask = falses(length(workspace.left_logjoint))
    sync_mask[chain_index] = true
    _sync_batched_continuation_logjoint!(workspace, sync_mask)
    return (moved, false)
end

function NUTSSubtreeWorkspace(num_params::Int, max_tree_depth::Int)
    state() = NUTSState(zeros(num_params), zeros(num_params), 0.0, zeros(num_params))
    summary = NUTSSubtreeMetadataState(-Inf, 0.0, 0, 0, Inf, Inf, Inf, Inf, 0.0, -Inf, -Inf, false, false)
    checkpoint_columns = max(max_tree_depth + 1, 1)
    checkpoint_positions = zeros(num_params, checkpoint_columns)
    checkpoint_momenta = zeros(num_params, checkpoint_columns)
    return NUTSSubtreeWorkspace(
        state(),
        state(),
        state(),
        state(),
        state(),
        summary,
        checkpoint_positions,
        checkpoint_momenta,
    )
end

function NUTSContinuationState(num_params::Int)
    state() = NUTSState(zeros(num_params), zeros(num_params), 0.0, zeros(num_params))
    return NUTSContinuationState(state(), state(), state(), Inf, Inf, -Inf, 0.0, 0, 0, 0, false, false)
end

function _logaddexp(x::Float64, y::Float64)
    if x == -Inf
        return y
    elseif y == -Inf
        return x
    end
    high = max(x, y)
    return high + log1p(exp(min(x, y) - high))
end

# Identity-metric U-turn criterion (kept for unit tests / reference); all
# production call sites use the metric-aware variant below.
function _is_turning(
    left_position::AbstractVector,
    right_position::AbstractVector,
    left_momentum::AbstractVector,
    right_momentum::AbstractVector,
)
    delta = right_position .- left_position
    return dot(delta, left_momentum) <= 0 || dot(delta, right_momentum) <= 0
end

# Metric-aware U-turn criterion: the trajectory velocity is M^{-1} p (see
# _mass_drift), so the turn test projects the frontier momenta through the
# inverse mass matrix instead of using the raw momenta.
function _is_turning(
    left_position::AbstractVector,
    right_position::AbstractVector,
    left_momentum::AbstractVector,
    right_momentum::AbstractVector,
    inverse_mass_matrix::Union{AbstractVector,MassMetric},
)
    left_dot = _turning_velocity_dot(left_position, right_position, left_momentum, inverse_mass_matrix)
    left_dot <= 0 && return true
    right_dot = _turning_velocity_dot(left_position, right_position, right_momentum, inverse_mass_matrix)
    return right_dot <= 0
end

function _turning_velocity_dot(
    left_position::AbstractVector,
    right_position::AbstractVector,
    momentum::AbstractVector,
    inverse_mass_matrix::AbstractVector,
)
    acc = 0.0
    @inbounds for index in eachindex(left_position, right_position, momentum, inverse_mass_matrix)
        acc +=
            (right_position[index] - left_position[index]) *
            inverse_mass_matrix[index] *
            momentum[index]
    end
    return acc
end

function _turning_velocity_dot(
    left_position::AbstractVector,
    right_position::AbstractVector,
    momentum::AbstractVector,
    metric::MassMetric,
)
    delta = right_position .- left_position
    return dot(delta, _mass_drift(metric, momentum))
end

# Metric-aware batched U-turn check: velocities are M^{-1} p per chain.
# `inverse_mass_matrix` is a shared diagonal Vector (P) or a per-chain diagonal
# Matrix (P x C); see _batched_mass_value.
function _batched_is_turning!(
    destination::AbstractVector{Bool},
    left_position::AbstractMatrix,
    right_position::AbstractMatrix,
    left_momentum::AbstractMatrix,
    right_momentum::AbstractMatrix,
    inverse_mass_matrix::Union{AbstractVector,AbstractMatrix},
    active::AbstractVector{Bool},
)
    num_chains = size(left_position, 2)
    size(left_position) == size(right_position) == size(left_momentum) == size(right_momentum) ||
        throw(DimensionMismatch("batched turning check requires matching position and momentum matrices"))
    length(destination) == num_chains ||
        throw(DimensionMismatch("expected turning destination of length $num_chains, got $(length(destination))"))
    length(active) == num_chains ||
        throw(DimensionMismatch("expected active mask of length $num_chains, got $(length(active))"))

    for chain_index = 1:num_chains
        if !active[chain_index]
            destination[chain_index] = false
            continue
        end
        left_dot = 0.0
        right_dot = 0.0
        for parameter_index in axes(left_position, 1)
            delta = right_position[parameter_index, chain_index] - left_position[parameter_index, chain_index]
            inverse_mass = _batched_mass_value(inverse_mass_matrix, parameter_index, chain_index)
            left_dot += delta * inverse_mass * left_momentum[parameter_index, chain_index]
            right_dot += delta * inverse_mass * right_momentum[parameter_index, chain_index]
        end
        destination[chain_index] = left_dot <= 0 || right_dot <= 0
    end
    return destination
end

_batched_mass_value(inverse_mass_matrix::AbstractVector, parameter_index::Int, chain_index::Int) =
    inverse_mass_matrix[parameter_index]
_batched_mass_value(inverse_mass_matrices::AbstractMatrix, parameter_index::Int, chain_index::Int) =
    inverse_mass_matrices[parameter_index, chain_index]

function _merge_batched_nuts_continuation_frontiers!(
    workspace::BatchedNUTSWorkspace,
    active::AbstractVector{Bool},
)
    length(active) == size(workspace.left_position, 2) ||
        throw(
            DimensionMismatch(
                "expected continuation-frontier active mask of length $(size(workspace.left_position, 2)), got $(length(active))",
            ),
        )
    fill!(workspace.subtree_copy_left, false)
    fill!(workspace.subtree_copy_right, false)
    for chain_index in eachindex(active)
        active[chain_index] || continue
        if workspace.control.step_direction[chain_index] < 0
            workspace.subtree_copy_left[chain_index] = true
            workspace.left_logjoint[chain_index] = workspace.tree_left_logjoint[chain_index]
        else
            workspace.subtree_copy_right[chain_index] = true
            workspace.right_logjoint[chain_index] = workspace.tree_right_logjoint[chain_index]
        end
    end
    _copy_masked_nuts_buffers!(
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.subtree_copy_right,
    )
    _sync_batched_continuation_logjoint!(workspace, active)
    return workspace
end

function _initialize_batched_nuts_subtree_states!(
    workspace::BatchedNUTSWorkspace,
    active::AbstractVector{Bool},
)
    length(active) == size(workspace.left_position, 2) ||
        throw(
            DimensionMismatch(
                "expected subtree-state active mask of length $(size(workspace.left_position, 2)), got $(length(active))",
            ),
        )
    fill!(workspace.subtree_copy_left, false)
    fill!(workspace.subtree_copy_right, false)
    fill!(workspace.subtree_select_proposal, false)
    for chain_index in eachindex(active)
        active[chain_index] || continue
        if workspace.control.step_direction[chain_index] < 0
            workspace.subtree_copy_left[chain_index] = true
            start_logjoint = workspace.left_logjoint[chain_index]
        else
            workspace.subtree_copy_right[chain_index] = true
            start_logjoint = workspace.right_logjoint[chain_index]
        end
        workspace.tree_current_logjoint[chain_index] = start_logjoint
        workspace.tree_left_logjoint[chain_index] = start_logjoint
        workspace.tree_right_logjoint[chain_index] = start_logjoint
        workspace.tree_proposal_logjoint[chain_index] = start_logjoint
    end
    _copy_masked_nuts_buffers!(
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_current_position,
        workspace.tree_current_momentum,
        workspace.tree_current_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_left_position,
        workspace.tree_left_momentum,
        workspace.tree_left_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_right_position,
        workspace.tree_right_momentum,
        workspace.tree_right_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.left_position,
        workspace.left_momentum,
        workspace.left_gradient,
        workspace.subtree_copy_left,
    )
    _copy_masked_nuts_buffers!(
        workspace.tree_proposal_position,
        workspace.tree_proposal_momentum,
        workspace.tree_proposal_gradient,
        workspace.right_position,
        workspace.right_momentum,
        workspace.right_gradient,
        workspace.subtree_copy_right,
    )
    _copy_masked_values!(workspace.tree_current_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_current_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _copy_masked_values!(workspace.tree_left_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_left_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _copy_masked_values!(workspace.tree_right_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_right_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _copy_masked_values!(workspace.tree_proposal_logjoint, workspace.left_logjoint, workspace.subtree_copy_left)
    _copy_masked_values!(workspace.tree_proposal_logjoint, workspace.right_logjoint, workspace.subtree_copy_right)
    _sync_batched_tree_logjoint!(workspace, active)
    return workspace
end

function _merge_batched_subtree_summary!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
)
    continuation = workspace.column_continuation_states[chain_index]
    workspace.control.integration_steps[chain_index] += workspace.subtree_integration_steps[chain_index]
    workspace.continuation_accept_stat_sum[chain_index] += workspace.subtree_accept_stat_sum[chain_index]
    workspace.continuation_accept_stat_count[chain_index] += workspace.subtree_accept_stat_count[chain_index]
    if isfinite(workspace.subtree_log_weight[chain_index])
        if workspace.continuation_select_proposal[chain_index]
            continuation.proposal_energy = workspace.subtree_proposal_energy[chain_index]
            continuation.proposal_energy_error =
                workspace.subtree_proposal_energy_error[chain_index]
            workspace.continuation_proposal_logjoint[chain_index] =
                workspace.tree_proposal_logjoint[chain_index]
            workspace.continuation_proposed_energy[chain_index] =
                continuation.proposal_energy
            workspace.continuation_delta_energy[chain_index] =
                continuation.proposal_energy_error
        end
        workspace.continuation_log_weight[chain_index] = workspace.continuation_combined_log_weight[chain_index]
    end

    workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
    workspace.control.continuation_turning[chain_index] =
        workspace.subtree_turning[chain_index] ||
        workspace.subtree_merged_turning[chain_index]
    return workspace
end

# Canonical multinomial NUTS discards an invalid final doubling (internal
# U-turn or divergence) entirely: the subtree's weight, proposal, and frontier
# never merge into the continuation. Only the leapfrog / accept-stat accounting
# and the termination flags carry over so the trajectory stops and diagnostics
# stay complete.
function _discard_invalid_batched_subtree!(
    workspace::BatchedNUTSWorkspace,
    chain_index::Int,
)
    workspace.control.integration_steps[chain_index] +=
        workspace.subtree_integration_steps[chain_index]
    workspace.continuation_accept_stat_sum[chain_index] +=
        workspace.subtree_accept_stat_sum[chain_index]
    workspace.continuation_accept_stat_count[chain_index] +=
        workspace.subtree_accept_stat_count[chain_index]
    workspace.control.continuation_turning[chain_index] = workspace.subtree_turning[chain_index]
    workspace.control.divergent_step[chain_index] = workspace.subtree_divergent[chain_index]
    return workspace
end

function _copy_nuts_continuation_frontier_from_tree!(
    continuation::NUTSContinuationState,
    subtree_workspace::NUTSSubtreeWorkspace,
    direction::Int,
)
    if direction < 0
        _copyto_nuts_state!(continuation.left, subtree_workspace.left)
    else
        _copyto_nuts_state!(continuation.right, subtree_workspace.right)
    end
    return continuation
end

function _copy_nuts_continuation_proposal_from_tree!(
    continuation::NUTSContinuationState,
    subtree_workspace::NUTSSubtreeWorkspace,
)
    _copyto_nuts_state!(continuation.proposal, subtree_workspace.proposal)
    continuation.proposal_energy = subtree_workspace.summary.proposal_energy
    continuation.proposal_energy_error = subtree_workspace.summary.proposal_energy_error
    return continuation
end

function _merge_nuts_continuation_turning!(
    continuation::NUTSContinuationState,
    subtree_turning::Bool,
    inverse_mass_matrix::Union{AbstractVector,MassMetric},
)
    continuation.turning =
        subtree_turning || _is_turning(
            continuation.left.position,
            continuation.right.position,
            continuation.left.momentum,
            continuation.right.momentum,
            inverse_mass_matrix,
        )
    return continuation
end

function _merge_nuts_subtree_summary!(
    continuation::NUTSContinuationState,
    subtree_workspace::NUTSSubtreeWorkspace,
    combined_log_weight::Float64,
)
    summary = subtree_workspace.summary
    continuation.integration_steps += summary.integration_steps
    continuation.accept_stat_sum += summary.accept_stat_sum
    continuation.accept_stat_count += summary.accept_stat_count
    if isfinite(summary.log_weight)
        continuation.log_weight = combined_log_weight
    end

    continuation.divergent = summary.divergent
    return continuation
end

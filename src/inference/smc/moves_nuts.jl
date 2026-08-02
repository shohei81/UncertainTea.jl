function _single_gaussian_logdensity(
    position::AbstractVector,
    location::AbstractVector,
    log_scale::AbstractVector,
)
    length(position) == length(location) == length(log_scale) ||
        throw(DimensionMismatch("expected Gaussian single-particle vectors of matching length"))
    value = -0.5 * length(position) * log(2.0 * pi)
    for parameter_index in eachindex(position, location, log_scale)
        scale = exp(log_scale[parameter_index])
        noise = (position[parameter_index] - location[parameter_index]) / scale
        value -= log_scale[parameter_index] + 0.5 * noise * noise
    end
    return value
end

function _single_gaussian_gradient!(
    destination::AbstractVector,
    position::AbstractVector,
    location::AbstractVector,
    log_scale::AbstractVector,
)
    length(destination) == length(position) == length(location) == length(log_scale) ||
        throw(DimensionMismatch("expected Gaussian single-particle vectors of matching length"))
    for parameter_index in eachindex(destination, position, location, log_scale)
        destination[parameter_index] =
            -(position[parameter_index] - location[parameter_index]) * exp(-2.0 * log_scale[parameter_index])
    end
    return destination
end

function _sample_batched_nuts_directions!(
    destination::AbstractVector{Int},
    rng::AbstractRNG,
)
    for index in eachindex(destination)
        destination[index] = _sample_nuts_direction(rng)
    end
    return destination
end

function _tempered_target_value_and_gradient!(
    gradient_destination::AbstractVector,
    proposal_gradient::AbstractVector,
    model::TeaModel,
    model_gradient_cache::LogjointGradientCache,
    position::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
)
    model_logjoint = logjoint_unconstrained(model, position, args, constraints)
    model_gradient = _logjoint_gradient!(model_gradient_cache, position)
    proposal_logdensity = _single_gaussian_logdensity(position, proposal_location, proposal_log_scale)
    _single_gaussian_gradient!(proposal_gradient, position, proposal_location, proposal_log_scale)
    one_minus_beta = 1.0 - beta
    for parameter_index in eachindex(gradient_destination, model_gradient, proposal_gradient)
        gradient_destination[parameter_index] =
            beta * model_gradient[parameter_index] + one_minus_beta * proposal_gradient[parameter_index]
    end
    tempered_value = beta * model_logjoint + one_minus_beta * proposal_logdensity
    return tempered_value, model_logjoint, proposal_logdensity
end

function _batched_tempered_nuts_leapfrog_step_to!(
    destination_position::AbstractMatrix,
    destination_momentum::AbstractMatrix,
    destination_gradient::AbstractMatrix,
    destination_logjoint::AbstractVector,
    valid::AbstractVector{Bool},
    start_position::AbstractMatrix,
    start_momentum::AbstractMatrix,
    start_gradient::AbstractMatrix,
    logjoint_values::AbstractVector,
    logjoint_gradient::AbstractMatrix,
    logproposal_values::AbstractVector,
    logproposal_gradient::AbstractMatrix,
    proposal_noise::AbstractMatrix,
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    inverse_mass_matrix::AbstractVector,
    step_size::Float64,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    num_particles = size(start_position, 2)
    target = BatchedTemperedDensityTarget(
        model,
        args,
        constraints,
        cache,
        proposal_location,
        proposal_log_scale,
        logjoint_values,
        logjoint_gradient,
        logproposal_values,
        logproposal_gradient,
        proposal_noise,
        Vector{Float64}(undef, num_particles),
        beta,
    )
    return batched_leapfrog_step_to!(
        destination_position,
        destination_momentum,
        destination_gradient,
        destination_logjoint,
        valid,
        start_position,
        start_momentum,
        start_gradient,
        target,
        inverse_mass_matrix,
        step_size,
        direction,
        active,
    )
end

function _copy_nuts_column_to_state!(
    destination::NUTSState,
    position::AbstractMatrix,
    momentum::AbstractMatrix,
    logjoint::AbstractVector,
    gradient::AbstractMatrix,
    index::Int,
)
    return _load_nuts_state!(
        destination,
        view(position, :, index),
        view(momentum, :, index),
        logjoint[index],
        view(gradient, :, index),
    )
end

function _initialize_tempered_nuts_depth_cohort!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    cohort_depth::Int,
    rng::AbstractRNG,
)
    cohort_workspace = workspace.cohort
    control = workspace.control
    num_particles = length(continuations)
    active = control.active
    subtree_active = control.subtree_active
    directions = control.directions
    current_position = cohort_workspace.current_position
    current_momentum = cohort_workspace.current_momentum
    current_gradient = cohort_workspace.current_gradient
    current_logjoint = cohort_workspace.current_logjoint
    next_position = cohort_workspace.next_position
    next_momentum = cohort_workspace.next_momentum
    next_gradient = cohort_workspace.next_gradient
    next_logjoint = cohort_workspace.next_logjoint
    left_position = cohort_workspace.left_position
    left_momentum = cohort_workspace.left_momentum
    left_gradient = cohort_workspace.left_gradient
    left_logjoint = cohort_workspace.left_logjoint
    right_position = cohort_workspace.right_position
    right_momentum = cohort_workspace.right_momentum
    right_gradient = cohort_workspace.right_gradient
    right_logjoint = cohort_workspace.right_logjoint
    proposal_position = cohort_workspace.proposal_position
    proposal_momentum = cohort_workspace.proposal_momentum
    proposal_gradient = cohort_workspace.proposal_gradient
    proposal_logjoint = cohort_workspace.proposal_logjoint
    any(active) || return workspace
    workspace.scheduler.active_depth == cohort_depth ||
        throw(ArgumentError("cohort depth mismatch: expected $(workspace.scheduler.active_depth), got $cohort_depth"))
    _sample_batched_nuts_directions!(directions, rng)
    _reset_tempered_nuts_cohort_statistics!(workspace)

    for particle_index = 1:num_particles
        active[particle_index] || continue
        start_state = _nuts_subtree_start_state(continuations[particle_index], directions[particle_index])
        copyto!(view(current_position, :, particle_index), start_state.position)
        copyto!(view(current_momentum, :, particle_index), start_state.momentum)
        copyto!(view(current_gradient, :, particle_index), start_state.gradient)
        current_logjoint[particle_index] = start_state.logjoint
        copyto!(view(left_position, :, particle_index), start_state.position)
        copyto!(view(left_momentum, :, particle_index), start_state.momentum)
        copyto!(view(left_gradient, :, particle_index), start_state.gradient)
        left_logjoint[particle_index] = start_state.logjoint
        copyto!(view(right_position, :, particle_index), start_state.position)
        copyto!(view(right_momentum, :, particle_index), start_state.momentum)
        copyto!(view(right_gradient, :, particle_index), start_state.gradient)
        right_logjoint[particle_index] = start_state.logjoint
        copyto!(view(proposal_position, :, particle_index), start_state.position)
        copyto!(view(proposal_momentum, :, particle_index), start_state.momentum)
        copyto!(view(proposal_gradient, :, particle_index), start_state.gradient)
        proposal_logjoint[particle_index] = start_state.logjoint
    end

    return workspace
end

function _expand_tempered_nuts_depth_cohort!(
    workspace::TemperedNUTSMoveWorkspace,
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    divergence_threshold::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    cohort_workspace = workspace.cohort
    control = workspace.control
    num_particles = size(cohort_workspace.current_position, 2)
    subtree_active = control.subtree_active
    directions = control.directions
    current_position = cohort_workspace.current_position
    current_momentum = cohort_workspace.current_momentum
    current_gradient = cohort_workspace.current_gradient
    current_logjoint = cohort_workspace.current_logjoint
    next_position = cohort_workspace.next_position
    next_momentum = cohort_workspace.next_momentum
    next_gradient = cohort_workspace.next_gradient
    next_logjoint = cohort_workspace.next_logjoint
    left_position = cohort_workspace.left_position
    left_momentum = cohort_workspace.left_momentum
    left_gradient = cohort_workspace.left_gradient
    left_logjoint = cohort_workspace.left_logjoint
    right_position = cohort_workspace.right_position
    right_momentum = cohort_workspace.right_momentum
    right_gradient = cohort_workspace.right_gradient
    right_logjoint = cohort_workspace.right_logjoint
    proposal_position = cohort_workspace.proposal_position
    proposal_momentum = cohort_workspace.proposal_momentum
    proposal_gradient = cohort_workspace.proposal_gradient
    proposal_logjoint = cohort_workspace.proposal_logjoint
    logjoint_values = cohort_workspace.logjoint_values
    logjoint_gradient = cohort_workspace.logjoint_gradient
    logproposal_values = cohort_workspace.logproposal_values
    logproposal_gradient = cohort_workspace.logproposal_gradient
    proposal_noise = cohort_workspace.proposal_noise
    valid = control.valid
    subtree_log_weight = cohort_workspace.subtree_log_weight
    subtree_accept_stat_sum = cohort_workspace.subtree_accept_stat_sum
    subtree_accept_stat_count = cohort_workspace.subtree_accept_stat_count
    subtree_integration_steps = cohort_workspace.subtree_integration_steps
    subtree_proposal_energy = cohort_workspace.subtree_proposal_energy
    subtree_proposal_energy_error = cohort_workspace.subtree_proposal_energy_error
    subtree_turning = cohort_workspace.subtree_turning
    subtree_divergent = cohort_workspace.subtree_divergent
    checkpoint_positions = cohort_workspace.checkpoint_positions
    checkpoint_momenta = cohort_workspace.checkpoint_momenta

    for _ = 1:workspace.scheduler.remaining_steps
        any(subtree_active) || break
        _batched_tempered_nuts_leapfrog_step_to!(
            next_position,
            next_momentum,
            next_gradient,
            next_logjoint,
            valid,
            current_position,
            current_momentum,
            current_gradient,
            logjoint_values,
            logjoint_gradient,
            logproposal_values,
            logproposal_gradient,
            proposal_noise,
            model,
            cache,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            inverse_mass_matrix,
            step_size,
            directions,
            subtree_active,
        )

        for particle_index = 1:num_particles
            subtree_active[particle_index] || continue
            if !valid[particle_index]
                subtree_divergent[particle_index] = true
                subtree_active[particle_index] = false
                continue
            end
            subtree_integration_steps[particle_index] += 1
            if directions[particle_index] < 0
                copyto!(view(left_position, :, particle_index), view(next_position, :, particle_index))
                copyto!(view(left_momentum, :, particle_index), view(next_momentum, :, particle_index))
                copyto!(view(left_gradient, :, particle_index), view(next_gradient, :, particle_index))
                left_logjoint[particle_index] = next_logjoint[particle_index]
            else
                copyto!(view(right_position, :, particle_index), view(next_position, :, particle_index))
                copyto!(view(right_momentum, :, particle_index), view(next_momentum, :, particle_index))
                copyto!(view(right_gradient, :, particle_index), view(next_gradient, :, particle_index))
                right_logjoint[particle_index] = next_logjoint[particle_index]
            end
            proposed_energy = _hamiltonian(
                next_logjoint[particle_index],
                view(next_momentum, :, particle_index),
                inverse_mass_matrix,
            )
            leaf = _advance_tree_leaf(
                proposed_energy,
                initial_hamiltonian[particle_index],
                divergence_threshold,
                subtree_log_weight[particle_index],
                rng,
            )
            if leaf.divergent
                subtree_divergent[particle_index] = true
                subtree_active[particle_index] = false
                continue
            end
            subtree_accept_stat_sum[particle_index] += leaf.accept_prob
            subtree_accept_stat_count[particle_index] += 1
            if leaf.select_proposal
                copyto!(view(proposal_position, :, particle_index), view(next_position, :, particle_index))
                copyto!(view(proposal_momentum, :, particle_index), view(next_momentum, :, particle_index))
                copyto!(view(proposal_gradient, :, particle_index), view(next_gradient, :, particle_index))
                proposal_logjoint[particle_index] = next_logjoint[particle_index]
                subtree_proposal_energy[particle_index] = proposed_energy
                subtree_proposal_energy_error[particle_index] = leaf.delta_energy
            end
            subtree_log_weight[particle_index] = leaf.combined_log_weight

            # Generalized (dyadic) U-turn: the new leaf lives in next_position /
            # next_momentum. Even leaves store a checkpoint; odd leaves test every
            # dyadic block ending here against its start checkpoint.
            leaf_index = subtree_integration_steps[particle_index] - 1
            if iseven(leaf_index)
                _store_tree_checkpoint!(
                    view(checkpoint_positions, :, :, particle_index),
                    view(checkpoint_momenta, :, :, particle_index),
                    leaf_index,
                    view(next_position, :, particle_index),
                    view(next_momentum, :, particle_index),
                )
            else
                subtree_turning[particle_index] = _dyadic_turning(
                    view(checkpoint_positions, :, :, particle_index),
                    view(checkpoint_momenta, :, :, particle_index),
                    leaf_index,
                    view(next_position, :, particle_index),
                    view(next_momentum, :, particle_index),
                    directions[particle_index],
                    inverse_mass_matrix,
                )
            end
            subtree_active[particle_index] = !subtree_turning[particle_index]
        end

        current_position, next_position = next_position, current_position
        current_momentum, next_momentum = next_momentum, current_momentum
        current_gradient, next_gradient = next_gradient, current_gradient
        current_logjoint, next_logjoint = next_logjoint, current_logjoint
    end

    return workspace
end

function _merge_tempered_nuts_depth_cohort!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    cohort_workspace = workspace.cohort
    active = workspace.control.active
    directions = workspace.control.directions
    num_particles = length(continuations)
    left_position = cohort_workspace.left_position
    left_momentum = cohort_workspace.left_momentum
    left_gradient = cohort_workspace.left_gradient
    left_logjoint = cohort_workspace.left_logjoint
    right_position = cohort_workspace.right_position
    right_momentum = cohort_workspace.right_momentum
    right_gradient = cohort_workspace.right_gradient
    right_logjoint = cohort_workspace.right_logjoint
    proposal_position = cohort_workspace.proposal_position
    proposal_momentum = cohort_workspace.proposal_momentum
    proposal_gradient = cohort_workspace.proposal_gradient
    proposal_logjoint = cohort_workspace.proposal_logjoint
    subtree_log_weight = cohort_workspace.subtree_log_weight
    subtree_accept_stat_sum = cohort_workspace.subtree_accept_stat_sum
    subtree_accept_stat_count = cohort_workspace.subtree_accept_stat_count
    subtree_integration_steps = cohort_workspace.subtree_integration_steps
    subtree_proposal_energy = cohort_workspace.subtree_proposal_energy
    subtree_proposal_energy_error = cohort_workspace.subtree_proposal_energy_error
    subtree_turning = cohort_workspace.subtree_turning
    subtree_divergent = cohort_workspace.subtree_divergent

    for particle_index = 1:num_particles
        active[particle_index] || continue
        continuation = continuations[particle_index]
        continuation.tree_depth += 1
        if subtree_integration_steps[particle_index] == 0
            continuation.divergent = subtree_divergent[particle_index]
            continue
        end
        if subtree_turning[particle_index] || subtree_divergent[particle_index]
            # Canonical multinomial NUTS discards an invalid final doubling
            # (internal U-turn or divergence) entirely: no weight merge, no
            # proposal swap, no frontier update. Only accounting and the
            # termination flags carry over.
            continuation.integration_steps += subtree_integration_steps[particle_index]
            continuation.accept_stat_sum += subtree_accept_stat_sum[particle_index]
            continuation.accept_stat_count += subtree_accept_stat_count[particle_index]
            continuation.turning = subtree_turning[particle_index]
            continuation.divergent = subtree_divergent[particle_index]
            continue
        end
        if directions[particle_index] < 0
            _copy_nuts_column_to_state!(
                continuation.left,
                left_position,
                left_momentum,
                left_logjoint,
                left_gradient,
                particle_index,
            )
        else
            _copy_nuts_column_to_state!(
                continuation.right,
                right_position,
                right_momentum,
                right_logjoint,
                right_gradient,
                particle_index,
            )
        end
        merge = _merge_subtree_stats(
            continuation.log_weight,
            subtree_log_weight[particle_index],
            rng,
        )
        if merge.select_proposal
            _copy_nuts_column_to_state!(
                continuation.proposal,
                proposal_position,
                proposal_momentum,
                proposal_logjoint,
                proposal_gradient,
                particle_index,
            )
            continuation.proposal_energy = subtree_proposal_energy[particle_index]
            continuation.proposal_energy_error = subtree_proposal_energy_error[particle_index]
        end
        continuation.integration_steps += subtree_integration_steps[particle_index]
        continuation.accept_stat_sum += subtree_accept_stat_sum[particle_index]
        continuation.accept_stat_count += subtree_accept_stat_count[particle_index]
        if isfinite(subtree_log_weight[particle_index])
            continuation.log_weight = merge.combined_log_weight
        end
        continuation.divergent = subtree_divergent[particle_index]
        _merge_nuts_continuation_turning!(
            continuation,
            subtree_turning[particle_index],
            inverse_mass_matrix,
        )
    end

    return continuations
end

function _batched_nuts_move!(
    workspace::TemperedNUTSMoveWorkspace,
    particles::AbstractMatrix,
    logjoint_values::AbstractVector,
    logproposal_values::AbstractVector,
    log_ratio::AbstractVector,
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    _validate_tempered_nuts_move_workspace!(workspace, particles, inverse_mass_matrix)
    num_particles = workspace.num_particles
    parameter_total = workspace.parameter_total
    cache = workspace.cache
    momentum = workspace.momentum
    proposal_particles = workspace.proposal_particles
    proposal_momentum = workspace.proposal_momentum
    current_logjoint_gradient = workspace.current_logjoint_gradient
    current_logproposal_gradient = workspace.current_logproposal_gradient
    current_tempered_gradient = workspace.current_tempered_gradient
    current_tempered_values = workspace.current_tempered_values
    proposal_logjoint_values = workspace.proposal_logjoint_values
    proposal_logproposal_values = workspace.proposal_logproposal_values
    proposal_tempered_values = workspace.proposal_tempered_values
    proposal_logjoint_gradient = workspace.proposal_logjoint_gradient
    proposal_logproposal_gradient = workspace.proposal_logproposal_gradient
    proposal_tempered_gradient = workspace.proposal_tempered_gradient
    proposal_noise = workspace.proposal_noise
    current_hamiltonian = workspace.current_hamiltonian
    directions = workspace.directions
    valid = workspace.valid
    inverse_mass = workspace.inverse_mass
    tree_workspaces = workspace.tree_workspaces
    continuations = workspace.continuations
    total_accept_stat = 0.0

    _batched_tempered_target!(
        current_tempered_values,
        current_tempered_gradient,
        logjoint_values,
        current_logjoint_gradient,
        logproposal_values,
        current_logproposal_gradient,
        proposal_noise,
        model,
        cache,
        particles,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
    )
    _sample_batched_momentum!(momentum, rng, workspace.sqrt_inverse_mass)
    _sample_batched_nuts_directions!(directions, rng)
    copyto!(proposal_particles, particles)
    copyto!(proposal_momentum, momentum)
    fill!(valid, true)

    for particle_index in axes(particles, 2), parameter_index in axes(particles, 1)
        signed_step_size = directions[particle_index] * step_size
        proposal_momentum[parameter_index, particle_index] +=
            (signed_step_size / 2) * current_tempered_gradient[parameter_index, particle_index]
        proposal_particles[parameter_index, particle_index] +=
            signed_step_size * inverse_mass[parameter_index] * proposal_momentum[parameter_index, particle_index]
    end

    _batched_tempered_target!(
        proposal_tempered_values,
        proposal_tempered_gradient,
        proposal_logjoint_values,
        proposal_logjoint_gradient,
        proposal_logproposal_values,
        proposal_logproposal_gradient,
        proposal_noise,
        model,
        cache,
        proposal_particles,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
    )

    for particle_index = 1:num_particles
        if !isfinite(proposal_tempered_values[particle_index]) ||
           !isfinite(proposal_logjoint_values[particle_index]) ||
           !all(isfinite, view(proposal_tempered_gradient, :, particle_index))
            valid[particle_index] = false
            continue
        end
        signed_step_size = directions[particle_index] * step_size
        for parameter_index = 1:parameter_total
            proposal_momentum[parameter_index, particle_index] +=
                (signed_step_size / 2) * proposal_tempered_gradient[parameter_index, particle_index]
        end
    end

    _batched_hamiltonian!(current_hamiltonian, current_tempered_values, momentum, inverse_mass)

    for particle_index = 1:num_particles
        tree_workspace = tree_workspaces[particle_index]
        continuation = continuations[particle_index]
        _load_nuts_state!(
            tree_workspace.current,
            view(particles, :, particle_index),
            view(momentum, :, particle_index),
            current_tempered_values[particle_index],
            view(current_tempered_gradient, :, particle_index),
        )
        _load_nuts_state!(
            tree_workspace.next,
            view(proposal_particles, :, particle_index),
            view(proposal_momentum, :, particle_index),
            proposal_tempered_values[particle_index],
            view(proposal_tempered_gradient, :, particle_index),
        )
        _initialize_nuts_first_trajectory!(
            continuation,
            tree_workspace,
            valid[particle_index],
            directions[particle_index],
            current_hamiltonian[particle_index],
            inverse_mass,
            divergence_threshold,
            rng,
        )
    end

    if max_tree_depth > 1
        _continue_batched_tempered_nuts_cohorts!(
            workspace,
            continuations,
            model,
            cache,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            step_size,
            max_tree_depth,
            divergence_threshold,
            current_hamiltonian,
            inverse_mass,
            rng,
        )
    end

    for particle_index = 1:num_particles
        current_position = view(particles, :, particle_index)
        continuation = continuations[particle_index]
        proposal = continuation.proposal
        accept_stat, _, _, moved = _nuts_proposal_summary(continuation, current_position)
        total_accept_stat += accept_stat
        if moved
            copyto!(view(particles, :, particle_index), proposal.position)
            if max_tree_depth == 1 && valid[particle_index]
                logjoint_values[particle_index] = proposal_logjoint_values[particle_index]
                logproposal_values[particle_index] = proposal_logproposal_values[particle_index]
            else
                logjoint_values[particle_index] = logjoint_unconstrained(model, proposal.position, args, constraints)
                logproposal_values[particle_index] = _single_gaussian_logdensity(
                    proposal.position,
                    proposal_location,
                    proposal_log_scale,
                )
            end
            log_ratio[particle_index] = logjoint_values[particle_index] - logproposal_values[particle_index]
        end
    end

    return total_accept_stat / num_particles
end

function _batched_nuts_move!(
    particles::AbstractMatrix,
    logjoint_values::AbstractVector,
    logproposal_values::AbstractVector,
    log_ratio::AbstractVector,
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    workspace = TemperedNUTSMoveWorkspace(model, particles, args, constraints, max_tree_depth)
    return _batched_nuts_move!(
        workspace,
        particles,
        logjoint_values,
        logproposal_values,
        log_ratio,
        model,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
        step_size,
        max_tree_depth,
        divergence_threshold,
        inverse_mass_matrix,
        rng,
    )
end

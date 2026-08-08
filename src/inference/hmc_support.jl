function _validate_hmc_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("HMC requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("HMC requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("HMC requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("HMC requires step_size > 0"))
    num_leapfrog_steps > 0 || throw(ArgumentError("HMC requires num_leapfrog_steps > 0"))
    0 < target_accept < 1 || throw(ArgumentError("HMC requires 0 < target_accept < 1"))
    divergence_threshold > 0 || throw(ArgumentError("HMC requires divergence_threshold > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("HMC requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("HMC requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_nuts_arguments(
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    max_tree_depth::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    num_params > 0 || throw(ArgumentError("NUTS requires at least one parameterized latent choice"))
    num_samples > 0 || throw(ArgumentError("NUTS requires num_samples > 0"))
    num_warmup >= 0 || throw(ArgumentError("NUTS requires num_warmup >= 0"))
    step_size > 0 || throw(ArgumentError("NUTS requires step_size > 0"))
    max_tree_depth > 0 || throw(ArgumentError("NUTS requires max_tree_depth > 0"))
    0 < target_accept < 1 || throw(ArgumentError("NUTS requires 0 < target_accept < 1"))
    divergence_threshold > 0 || throw(ArgumentError("NUTS requires divergence_threshold > 0"))
    mass_matrix_regularization > 0 || throw(ArgumentError("NUTS requires mass_matrix_regularization > 0"))
    mass_matrix_min_samples > 0 || throw(ArgumentError("NUTS requires mass_matrix_min_samples > 0"))
    return nothing
end

function _validate_hmc_chains_arguments(num_chains::Int, sampler_name::AbstractString="HMC")
    num_chains > 0 || throw(ArgumentError("$sampler_name requires num_chains > 0"))
    return nothing
end

function _validate_batched_hmc_arguments(
    num_chains::Int,
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    args,
    constraints,
)
    _validate_hmc_chains_arguments(num_chains)
    _validate_hmc_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )
    _validate_batched_args(args, num_chains)
    _validate_batched_constraints(constraints, num_chains)
    return nothing
end

function _validate_batched_nuts_arguments(
    num_chains::Int,
    num_params::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    max_tree_depth::Int,
    target_accept::Real,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    args,
    constraints,
)
    _validate_hmc_chains_arguments(num_chains, "NUTS")
    _validate_nuts_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )
    _validate_batched_args(args, num_chains)
    _validate_batched_constraints(constraints, num_chains)
    return nothing
end

function _initial_hmc_position(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    initial_params,
    rng::AbstractRNG,
)
    resolved = _resolve_signature_plan(model, constraints, args)
    return _initial_hmc_position(model, resolved, args, constraints, initial_params, rng)
end

# Resolved-plan core: the batched initial-position loop resolves the signature
# plan once per batch (issue #156) and threads it through here, so per chain
# neither the conditioning signature nor the transform re-derives the plan.
function _initial_hmc_position(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    args::Tuple,
    constraints::ChoiceMap,
    initial_params,
    rng::AbstractRNG,
)
    layout = resolved.plan.parameter_layout
    if isnothing(initial_params)
        # A prior draw of the SIGNATURE latents (not the syntactic default
        # layout): observations are read from the constraints, so the initial
        # unconstrained vector must match the conditioned latent set (#95 PR-6).
        constrained = _signature_initial_parameters(model, args, resolved, constraints; rng=rng)
        return _transform_to_unconstrained(model, resolved, constrained, args, constraints)
    end

    expected = parametercount(layout)
    constrained_expected = parametervaluecount(layout)
    if length(initial_params) == expected
        return Float64[value for value in initial_params]
    elseif length(initial_params) == constrained_expected
        return _transform_to_unconstrained(model, resolved, Float64[value for value in initial_params], args, constraints)
    end
    throw(
        DimensionMismatch(
            "expected $expected unconstrained or $constrained_expected constrained initial parameters, got $(length(initial_params))",
        ),
    )
end

# Signature-aware reconstruction of one stored draw's constrained column (#95):
# the latent set follows the chain's conditioning signature, so the constrained
# value is read through the signature transform (observations resolved from the
# chain's constraints), not the syntactic default `parameterlayout(model)`. Used
# by the batched HMC/NUTS samplers and their device counterparts.
function _write_signature_constrained_sample!(
    constrained_samples::AbstractArray,
    model::TeaModel,
    position_column::AbstractVector,
    sample_index::Int,
    chain_args::Tuple,
    chain_constraints::ChoiceMap,
    chain_index::Int,
)
    copyto!(
        view(constrained_samples, :, sample_index, chain_index),
        transform_to_constrained(model, collect(position_column), chain_args, chain_constraints),
    )
    return constrained_samples
end

function _sample_momentum(rng::AbstractRNG, inverse_mass_matrix::AbstractVector)
    return randn(rng, length(inverse_mass_matrix)) ./ sqrt.(inverse_mass_matrix)
end

function _initial_batched_hmc_positions(
    model::TeaModel,
    args,
    constraints,
    initial_params,
    rng::AbstractRNG,
    num_params::Int,
    constrained_num_params::Int,
    num_chains::Int;
    init_strategy::Symbol=:prior,
)
    batch_args = _validate_batched_args(model, args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    # One signature resolution for the whole batch: every column of a batch
    # shares one conditioning signature (see `_representative_constraints`), so
    # re-deriving it per chain is pure setup overhead (issue #156).
    resolved = _resolve_signature_plan_representative(model, batch_constraints, batch_args)
    positions = Matrix{Float64}(undef, num_params, num_chains)
    _fill_batched_initial_positions!(
        positions,
        1:num_chains,
        model,
        resolved,
        batch_args,
        batch_constraints,
        initial_params,
        init_strategy,
        rng,
        num_params,
        constrained_num_params,
        num_chains,
    )
    return positions
end

# Whether the requested initialization can be re-drawn to recover from a
# non-finite starting point (issue #162): prior/uniform draws and Pathfinder
# resampling all vary with the RNG, so a fresh seed gives a new position. An
# explicit `initial_params` array is a fixed value -- re-drawing would just
# reproduce it -- so those never retry.
_init_is_redrawable(initial_params) =
    isnothing(initial_params) || initial_params isa PathfinderResult

# Fill the given `columns` of `positions` with fresh initial draws. Used both
# for the first full draw (columns = 1:num_chains) and for the retry loop's
# re-draw of the non-finite columns (issue #162). Draw order is chain-major with
# a per-column MersenneTwister, matching the original loop so the :prior full
# draw is bitwise unchanged.
function _fill_batched_initial_positions!(
    positions::AbstractMatrix,
    columns,
    model::TeaModel,
    resolved,
    batch_args,
    batch_constraints,
    initial_params,
    init_strategy::Symbol,
    rng::AbstractRNG,
    num_params::Int,
    constrained_num_params::Int,
    num_chains::Int,
)
    seeds = rand(rng, UInt, length(columns))
    for (k, chain_index) in enumerate(columns)
        chain_initial_params = _chain_initial_params(
            initial_params, chain_index, num_params, constrained_num_params, num_chains,
        )
        chain_rng = MersenneTwister(seeds[k])
        positions[:, chain_index] = _draw_batched_initial_position_column(
            model,
            resolved,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            chain_initial_params,
            init_strategy,
            num_params,
            chain_rng,
        )
    end
    return positions
end

# One column's initial unconstrained position. `init_strategy=:uniform` draws each
# unconstrained coordinate from Uniform(-2, 2) -- the Stan/NumPyro default
# protocol -- but only when no explicit/Pathfinder initialization is supplied for the
# chain; otherwise the requested initialization (prior draw, explicit params, or
# Pathfinder draw) is honored.
function _draw_batched_initial_position_column(
    model::TeaModel,
    resolved,
    chain_args,
    chain_constraints,
    chain_initial_params,
    init_strategy::Symbol,
    num_params::Int,
    chain_rng::AbstractRNG,
)
    if init_strategy === :uniform && isnothing(chain_initial_params)
        column = Vector{Float64}(undef, num_params)
        @inbounds for i = 1:num_params
            column[i] = 4.0 * rand(chain_rng) - 2.0
        end
        return column
    end
    return _initial_hmc_position(
        model, resolved, chain_args, chain_constraints, chain_initial_params, chain_rng,
    )
end

# Re-draw the given non-finite `columns` in place, recomputing the batch
# signature plan (retries are rare, so this setup cost is amortized against the
# gradient re-evaluation). Issue #162.
function _redraw_batched_initial_positions!(
    positions::AbstractMatrix,
    columns,
    model::TeaModel,
    batch_args,
    batch_constraints,
    initial_params,
    init_strategy::Symbol,
    rng::AbstractRNG,
    num_params::Int,
    constrained_num_params::Int,
    num_chains::Int,
)
    resolved = _resolve_signature_plan_representative(model, batch_constraints, batch_args)
    return _fill_batched_initial_positions!(
        positions,
        columns,
        model,
        resolved,
        batch_args,
        batch_constraints,
        initial_params,
        init_strategy,
        rng,
        num_params,
        constrained_num_params,
        num_chains,
    )
end

# Columns whose initial unconstrained logjoint or gradient is non-finite.
function _nonfinite_init_columns(
    logjoint::AbstractVector, gradient::AbstractMatrix,
)
    bad = Int[]
    for chain_index in eachindex(logjoint)
        if !isfinite(logjoint[chain_index]) ||
           !all(isfinite, view(gradient, :, chain_index))
            push!(bad, chain_index)
        end
    end
    return bad
end

function _sample_batched_momentum(
    rng::AbstractRNG,
    inverse_mass_matrix::AbstractVector,
    num_chains::Int,
)
    momentum = Matrix{Float64}(undef, length(inverse_mass_matrix), num_chains)
    _sample_batched_momentum!(momentum, rng, sqrt.(Float64.(inverse_mass_matrix)))
    return momentum
end

function _sample_batched_momentum!(
    destination::AbstractMatrix,
    rng::AbstractRNG,
    sqrt_inverse_mass_matrix::AbstractVector,
)
    size(destination, 1) == length(sqrt_inverse_mass_matrix) ||
        throw(
            DimensionMismatch(
                "expected momentum matrix with $(length(sqrt_inverse_mass_matrix)) rows, got $(size(destination, 1))",
            ),
        )

    for chain_index in axes(destination, 2)
        for parameter_index in eachindex(sqrt_inverse_mass_matrix)
            destination[parameter_index, chain_index] =
                randn(rng) / sqrt_inverse_mass_matrix[parameter_index]
        end
    end
    return destination
end

# Per-chain overload: each destination column is filled from its own sqrt inverse
# mass column. The (chain-major) draw order matches the scalar overload, so with
# identical columns it is bitwise identical; used only by the per-chain path.
function _sample_batched_momentum!(
    destination::AbstractMatrix,
    rng::AbstractRNG,
    sqrt_inverse_mass_matrices::AbstractMatrix,
)
    size(destination) == size(sqrt_inverse_mass_matrices) ||
        throw(DimensionMismatch("expected momentum matrix of size $(size(sqrt_inverse_mass_matrices)), got $(size(destination))"))

    for chain_index in axes(destination, 2)
        for parameter_index in axes(destination, 1)
            destination[parameter_index, chain_index] =
                randn(rng) / sqrt_inverse_mass_matrices[parameter_index, chain_index]
        end
    end
    return destination
end

# Column selector: shared mode threads a single inverse-mass Vector; per-chain
# mode threads a Matrix whose columns are the per-chain inverse-mass diagonals.
_chain_inverse_mass(inverse_mass_matrix::AbstractVector, chain_index::Int) = inverse_mass_matrix
_chain_inverse_mass(inverse_mass_matrices::AbstractMatrix, chain_index::Int) =
    view(inverse_mass_matrices, :, chain_index)

# Materialized column selectors for the per-chain continuation path, which reuses
# the single-chain subtree builder requiring a concrete Vector / scalar. Shared
# mode returns the same objects it is handed, so it stays bitwise identical.
_chain_step_size(step_size::Real, chain_index::Int) = Float64(step_size)
_chain_step_size(step_sizes::AbstractVector, chain_index::Int) = Float64(step_sizes[chain_index])
_chain_inverse_mass_vector(inverse_mass_matrix::Vector{Float64}, chain_index::Int) = inverse_mass_matrix
_chain_inverse_mass_vector(inverse_mass_matrices::AbstractMatrix, chain_index::Int) =
    collect(Float64, view(inverse_mass_matrices, :, chain_index))

function _update_sqrt_inverse_mass_matrix!(
    destination::AbstractVector,
    inverse_mass_matrix::AbstractVector,
)
    length(destination) == length(inverse_mass_matrix) ||
        throw(
            DimensionMismatch("expected inverse mass matrix of length $(length(destination)), got $(length(inverse_mass_matrix))"),
        )
    for index in eachindex(destination, inverse_mass_matrix)
        destination[index] = sqrt(Float64(inverse_mass_matrix[index]))
    end
    return destination
end

function _batched_kinetic_energy(
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    energy = Vector{Float64}(undef, size(momentum, 2))
    return _batched_kinetic_energy!(energy, momentum, inverse_mass_matrix)
end

function _batched_kinetic_energy!(
    destination::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    size(momentum, 1) == length(inverse_mass_matrix) ||
        throw(DimensionMismatch("expected momentum matrix with $(length(inverse_mass_matrix)) rows, got $(size(momentum, 1))"))
    size(momentum, 2) == length(destination) ||
        throw(DimensionMismatch("expected kinetic-energy destination of length $(size(momentum, 2)), got $(length(destination))"))

    for chain_index in axes(momentum, 2)
        kinetic_energy = 0.0
        for parameter_index in eachindex(inverse_mass_matrix)
            momentum_value = momentum[parameter_index, chain_index]
            kinetic_energy += momentum_value^2 * inverse_mass_matrix[parameter_index]
        end
        destination[chain_index] = kinetic_energy / 2
    end
    return destination
end

# Per-chain overload: each chain's kinetic energy uses its own inverse-mass column.
function _batched_kinetic_energy!(
    destination::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrices::AbstractMatrix,
)
    size(momentum) == size(inverse_mass_matrices) ||
        throw(DimensionMismatch("expected inverse-mass matrix of size $(size(momentum)), got $(size(inverse_mass_matrices))"))
    size(momentum, 2) == length(destination) ||
        throw(DimensionMismatch("expected kinetic-energy destination of length $(size(momentum, 2)), got $(length(destination))"))

    for chain_index in axes(momentum, 2)
        kinetic_energy = 0.0
        for parameter_index in axes(momentum, 1)
            momentum_value = momentum[parameter_index, chain_index]
            kinetic_energy += momentum_value^2 * inverse_mass_matrices[parameter_index, chain_index]
        end
        destination[chain_index] = kinetic_energy / 2
    end
    return destination
end

function _batched_hamiltonian(
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    hamiltonian = Vector{Float64}(undef, length(logjoint_values))
    return _batched_hamiltonian!(hamiltonian, logjoint_values, momentum, inverse_mass_matrix)
end

# Per-chain overload of the Hamiltonian assembly.
function _batched_hamiltonian!(
    destination::AbstractVector,
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrices::AbstractMatrix,
)
    length(logjoint_values) == length(destination) ||
        throw(DimensionMismatch("expected hamiltonian inputs of length $(length(destination)), got $(length(logjoint_values))"))

    _batched_kinetic_energy!(destination, momentum, inverse_mass_matrices)
    for chain_index in eachindex(destination)
        destination[chain_index] -= Float64(logjoint_values[chain_index])
    end
    return destination
end

function _batched_hamiltonian!(
    destination::AbstractVector,
    logjoint_values::AbstractVector,
    momentum::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    length(logjoint_values) == length(destination) ||
        throw(DimensionMismatch("expected hamiltonian inputs of length $(length(destination)), got $(length(logjoint_values))"))

    _batched_kinetic_energy!(destination, momentum, inverse_mass_matrix)
    for chain_index in eachindex(destination)
        destination[chain_index] -= Float64(logjoint_values[chain_index])
    end
    return destination
end

function _batched_acceptance_probability(log_accept_ratio::AbstractVector)
    probabilities = Vector{Float64}(undef, length(log_accept_ratio))
    return _batched_acceptance_probability!(probabilities, log_accept_ratio)
end

function _batched_acceptance_probability!(
    destination::AbstractVector,
    log_accept_ratio::AbstractVector,
)
    length(destination) == length(log_accept_ratio) ||
        throw(
            DimensionMismatch(
                "expected acceptance-probability destination of length $(length(log_accept_ratio)), got $(length(destination))",
            ),
        )

    for index in eachindex(log_accept_ratio)
        destination[index] = _acceptance_probability(log_accept_ratio[index])
    end
    return destination
end

function _mean_acceptance_stats!(
    destination::AbstractVector,
    accept_sum::AbstractVector,
    accept_count::AbstractVector{Int},
)
    length(destination) == length(accept_sum) == length(accept_count) ||
        throw(
            DimensionMismatch(
                "expected acceptance-stat inputs of matching length, got $(length(destination)), $(length(accept_sum)), and $(length(accept_count))",
            ),
        )
    for index in eachindex(destination)
        destination[index] = accept_count[index] == 0 ? 0.0 : accept_sum[index] / accept_count[index]
    end
    return destination
end

function _mean_acceptance_stat(
    accept_sum::Real,
    accept_count::Integer,
)
    return accept_count == 0 ? 0.0 : Float64(accept_sum) / accept_count
end

function _position_moved(
    proposal_position::AbstractVector,
    current_position::AbstractVector,
)
    length(proposal_position) == length(current_position) ||
        throw(
            DimensionMismatch(
                "expected moved-position inputs of matching length, got $(length(proposal_position)) and $(length(current_position))",
            ),
        )
    for index in eachindex(proposal_position, current_position)
        proposal_position[index] == current_position[index] || return true
    end
    return false
end

function _nuts_proposal_summary(
    continuation::NUTSContinuationState,
    current_position::AbstractVector,
)
    proposed_energy = continuation.proposal_energy
    energy_error = continuation.proposal_energy_error
    accept_stat = _mean_acceptance_stat(
        continuation.accept_stat_sum,
        continuation.accept_stat_count,
    )
    moved = _position_moved(continuation.proposal.position, current_position)
    return accept_stat, proposed_energy, energy_error, moved
end

function _batched_positions_moved!(
    destination::AbstractVector{Bool},
    proposal_position::AbstractMatrix,
    current_position::AbstractMatrix,
)
    size(proposal_position) == size(current_position) ||
        throw(
            DimensionMismatch(
                "expected moved-position inputs of matching size, got $(size(proposal_position)) and $(size(current_position))",
            ),
        )
    length(destination) == size(proposal_position, 2) ||
        throw(
            DimensionMismatch(
                "expected moved-position destination of length $(size(proposal_position, 2)), got $(length(destination))",
            ),
        )
    for chain_index in eachindex(destination)
        moved = false
        for parameter_index in axes(proposal_position, 1)
            if proposal_position[parameter_index, chain_index] != current_position[parameter_index, chain_index]
                moved = true
                break
            end
        end
        destination[chain_index] = moved
    end
    return destination
end

function _finalize_batched_nuts_proposals!(
    workspace::BatchedNUTSWorkspace,
    position::AbstractMatrix,
)
    copyto!(workspace.proposed_logjoint, workspace.continuation_proposal_logjoint)
    copyto!(workspace.proposed_energy, workspace.continuation_proposed_energy)
    copyto!(workspace.energy_error, workspace.continuation_delta_energy)
    _mean_acceptance_stats!(
        workspace.accept_prob,
        workspace.continuation_accept_stat_sum,
        workspace.continuation_accept_stat_count,
    )
    _batched_positions_moved!(workspace.control.accepted_step, workspace.proposal_position, position)
    return workspace
end

function _mean_batched_adaptation_probability(
    accept_prob::AbstractVector,
    divergent::AbstractVector,
)
    length(accept_prob) == length(divergent) ||
        throw(
            DimensionMismatch(
                "expected acceptance and divergence vectors of matching length, got $(length(accept_prob)) and $(length(divergent))",
            ),
        )
    isempty(accept_prob) && return 0.0

    total = 0.0
    for index in eachindex(accept_prob, divergent)
        total += divergent[index] ? 0.0 : accept_prob[index]
    end
    return total / length(accept_prob)
end

function _sample_nuts_direction(rng::AbstractRNG)
    return rand(rng, Bool) ? 1 : -1
end

function _sample_batched_nuts_directions!(
    destination::AbstractVector{Int},
    rng::AbstractRNG,
    active::AbstractVector{Bool},
)
    length(destination) == length(active) ||
        throw(DimensionMismatch("expected NUTS direction destination of length $(length(active)), got $(length(destination))"))
    for index in eachindex(destination, active)
        active[index] || continue
        destination[index] = _sample_nuts_direction(rng)
    end
    return destination
end

function _nuts_continuation_active(
    tree_depth::Integer,
    max_tree_depth::Integer,
    divergent::Bool,
    turning::Bool,
)
    return tree_depth < max_tree_depth && !divergent && !turning
end

function _nuts_subtree_start_state(
    continuation::NUTSContinuationState,
    direction::Int,
)
    return direction < 0 ? continuation.left : continuation.right
end

function _batched_leapfrog!(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_gradient::Matrix{Float64},
    inverse_mass_matrix::Vector{Float64},
    args,
    constraints,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    target = BatchedModelDensityTarget(workspace.gradient_cache)
    return batched_leapfrog_trajectory!(
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.proposed_logjoint,
        workspace.valid,
        position,
        workspace.momentum,
        current_gradient,
        target,
        inverse_mass_matrix,
        step_size,
        num_leapfrog_steps,
    )
end

# Per-chain overload: per-chain step sizes + inverse-mass columns.
function _batched_leapfrog!(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    position::Matrix{Float64},
    current_gradient::Matrix{Float64},
    inverse_mass_matrices::AbstractMatrix,
    args,
    constraints,
    step_sizes::AbstractVector{Float64},
    num_leapfrog_steps::Int,
)
    target = BatchedModelDensityTarget(workspace.gradient_cache)
    return batched_leapfrog_trajectory!(
        workspace.proposal_position,
        workspace.proposal_momentum,
        workspace.proposal_gradient,
        workspace.proposed_logjoint,
        workspace.valid,
        position,
        workspace.momentum,
        current_gradient,
        target,
        inverse_mass_matrices,
        step_sizes,
        num_leapfrog_steps,
    )
end

function _leapfrog(
    model::TeaModel,
    position::Vector{Float64},
    momentum::Vector{Float64},
    gradient_cache::LogjointGradientCache,
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    args::Tuple,
    constraints::ChoiceMap,
    step_size::Float64,
    num_leapfrog_steps::Int,
)
    target = ModelDensityTarget(model, args, constraints, gradient_cache)
    return leapfrog_trajectory(target, position, momentum, inverse_mass_matrix, step_size, num_leapfrog_steps)
end

function _kinetic_energy(momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return sum((momentum .^ 2) .* inverse_mass_matrix) / 2
end

function _hamiltonian(logjoint_value::Float64, momentum::AbstractVector, inverse_mass_matrix::AbstractVector)
    return -logjoint_value + _kinetic_energy(momentum, inverse_mass_matrix)
end

# Allocation-free integrator bookkeeping (issue #142). The batched leapfrog
# updates below used to be written as per-chain broadcast slices, e.g.
# `p[:, c] .+= h .* g[:, c]`: every RHS `getindex` slice copies, ~4 small
# allocations per chain per leapfrog step (measured 1.45 ms + 246 KB per step
# at 512 chains). The explicit column loops keep the identical elementwise
# arithmetic and evaluation order, so seeded trajectories stay bitwise
# unchanged.

function _column_all_finite(values::AbstractMatrix, chain_index::Int)
    @inbounds for row in axes(values, 1)
        isfinite(values[row, chain_index]) || return false
    end
    return true
end

# destination[:, c] .+= scale .* source[:, c]
function _add_scaled_column!(
    destination::AbstractMatrix{Float64},
    source::AbstractMatrix{Float64},
    chain_index::Int,
    scale::Float64,
)
    @inbounds for row in axes(destination, 1)
        destination[row, chain_index] += scale * source[row, chain_index]
    end
    return destination
end

# q[:, c] .+= step_size .* (inverse_mass_matrix .* p[:, c])
function _add_mass_drift_column!(
    q::AbstractMatrix{Float64},
    p::AbstractMatrix{Float64},
    inverse_mass_matrix::AbstractVector,
    chain_index::Int,
    step_size::Float64,
)
    @inbounds for row in axes(q, 1)
        q[row, chain_index] += step_size * (inverse_mass_matrix[row] * p[row, chain_index])
    end
    return q
end

# Per-chain overload: q[:, c] .+= step_size .* (inverse_mass_matrices[:, c] .* p[:, c])
function _add_mass_drift_column!(
    q::AbstractMatrix{Float64},
    p::AbstractMatrix{Float64},
    inverse_mass_matrices::AbstractMatrix,
    chain_index::Int,
    step_size::Float64,
)
    @inbounds for row in axes(q, 1)
        q[row, chain_index] += step_size * (inverse_mass_matrices[row, chain_index] * p[row, chain_index])
    end
    return q
end

# p[:, c] .*= -1
function _negate_column!(p::AbstractMatrix{Float64}, chain_index::Int)
    @inbounds for row in axes(p, 1)
        p[row, chain_index] *= -1
    end
    return p
end

# Single-chain drift q .+= step_size .* M^{-1} p. The diagonal-metric loop
# avoids materializing `inverse_mass_matrix .* p` (`_mass_drift` allocates a
# fresh vector per call); the dense MassMetric solve keeps the existing path.
function _apply_mass_drift!(
    q::AbstractVector{Float64},
    p::AbstractVector{Float64},
    inverse_mass_matrix::AbstractVector,
    step_size::Float64,
)
    @inbounds for index in eachindex(q, p, inverse_mass_matrix)
        q[index] += step_size * (inverse_mass_matrix[index] * p[index])
    end
    return q
end

function _apply_mass_drift!(
    q::AbstractVector{Float64},
    p::AbstractVector{Float64},
    metric::MassMetric,
    step_size::Float64,
)
    q .+= step_size .* _mass_drift(metric, p)
    return q
end

function leapfrog_step!(
    destination::NUTSState,
    target::AbstractDensityTarget,
    state::NUTSState,
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    step_size::Float64,
)
    q = destination.position
    p = destination.momentum
    gradient = destination.gradient
    copyto!(q, state.position)
    copyto!(p, state.momentum)
    p .+= (step_size / 2) .* state.gradient
    _apply_mass_drift!(q, p, inverse_mass_matrix, step_size)
    value, proposed_gradient = target_logdensity_and_gradient!(target, q)
    isfinite(value) || return false
    all(isfinite, proposed_gradient) || return false
    copyto!(gradient, proposed_gradient)
    p .+= (step_size / 2) .* gradient
    destination.logjoint = value
    return true
end

function leapfrog_trajectory(
    target::AbstractDensityTarget,
    position::AbstractVector{Float64},
    momentum::AbstractVector{Float64},
    inverse_mass_matrix::Union{Vector{Float64},MassMetric},
    step_size::Float64,
    num_steps::Int,
)
    q = copy(position)
    p = copy(momentum)

    gradient = target_gradient!(target, q)
    all(isfinite, gradient) || return nothing
    p .+= (step_size / 2) .* gradient

    for leapfrog_step = 1:num_steps
        _apply_mass_drift!(q, p, inverse_mass_matrix, step_size)
        gradient = target_gradient!(target, q)
        all(isfinite, gradient) || return nothing

        if leapfrog_step < num_steps
            p .+= step_size .* gradient
        end
    end

    p .+= (step_size / 2) .* gradient
    p .*= -1

    value = target_logdensity(target, q)
    isfinite(value) || return nothing
    return q, p, value
end

function batched_leapfrog_trajectory!(
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    valid::AbstractVector{Bool},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    current_gradient::AbstractMatrix{Float64},
    target::AbstractBatchedDensityTarget,
    inverse_mass_matrix::AbstractVector,
    step_size::Float64,
    num_steps::Int,
)
    q = destination_position
    p = destination_momentum
    num_chains = size(q, 2)
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(current_gradient) == size(position) ||
        throw(DimensionMismatch("expected current gradient workspace of size $(size(position)), got $(size(current_gradient))"))
    size(destination_gradient) == size(position) ||
        throw(
            DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(destination_gradient))"),
        )

    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index = 1:num_chains
        if !_column_all_finite(gradient, chain_index)
            valid[chain_index] = false
        else
            _add_scaled_column!(p, gradient, chain_index, step_size / 2)
        end
    end

    for leapfrog_step = 1:num_steps
        for chain_index = 1:num_chains
            valid[chain_index] || continue
            _add_mass_drift_column!(q, p, inverse_mass_matrix, chain_index, step_size)
        end

        if leapfrog_step < num_steps
            gradient = batched_target_gradient!(destination_gradient, target, q)
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !_column_all_finite(gradient, chain_index)
                    valid[chain_index] = false
                else
                    _add_scaled_column!(p, gradient, chain_index, step_size)
                end
            end
        else
            proposed_logjoint, _ = batched_target_logdensity_and_gradient!(
                destination_logjoint,
                destination_gradient,
                target,
                q,
            )
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !_column_all_finite(destination_gradient, chain_index) ||
                   !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index = 1:num_chains
        valid[chain_index] || continue
        _add_scaled_column!(p, destination_gradient, chain_index, step_size / 2)
        _negate_column!(p, chain_index)
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

# Per-chain overload: each chain integrates with its own step size (`step_sizes`)
# and its own inverse-mass column (`inverse_mass_matrices[:, chain]`). Used only by
# the per-chain-adaptation path; the scalar overload above is untouched so the
# shared-adaptation path stays bitwise identical.
function batched_leapfrog_trajectory!(
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    valid::AbstractVector{Bool},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    current_gradient::AbstractMatrix{Float64},
    target::AbstractBatchedDensityTarget,
    inverse_mass_matrices::AbstractMatrix,
    step_sizes::AbstractVector{Float64},
    num_steps::Int,
)
    q = destination_position
    p = destination_momentum
    num_chains = size(q, 2)
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(current_gradient) == size(position) ||
        throw(DimensionMismatch("expected current gradient workspace of size $(size(position)), got $(size(current_gradient))"))
    size(destination_gradient) == size(position) ||
        throw(
            DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(destination_gradient))"),
        )
    size(inverse_mass_matrices, 2) == num_chains ||
        throw(DimensionMismatch("expected $num_chains inverse-mass columns, got $(size(inverse_mass_matrices, 2))"))
    length(step_sizes) == num_chains ||
        throw(DimensionMismatch("expected $num_chains step sizes, got $(length(step_sizes))"))

    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, true)
    gradient = current_gradient

    for chain_index = 1:num_chains
        if !_column_all_finite(gradient, chain_index)
            valid[chain_index] = false
        else
            step_size = step_sizes[chain_index]
            _add_scaled_column!(p, gradient, chain_index, step_size / 2)
        end
    end

    for leapfrog_step = 1:num_steps
        for chain_index = 1:num_chains
            valid[chain_index] || continue
            step_size = step_sizes[chain_index]
            _add_mass_drift_column!(q, p, inverse_mass_matrices, chain_index, step_size)
        end

        if leapfrog_step < num_steps
            gradient = batched_target_gradient!(destination_gradient, target, q)
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !_column_all_finite(gradient, chain_index)
                    valid[chain_index] = false
                else
                    step_size = step_sizes[chain_index]
                    _add_scaled_column!(p, gradient, chain_index, step_size)
                end
            end
        else
            proposed_logjoint, _ = batched_target_logdensity_and_gradient!(
                destination_logjoint,
                destination_gradient,
                target,
                q,
            )
            for chain_index = 1:num_chains
                valid[chain_index] || continue
                if !_column_all_finite(destination_gradient, chain_index) ||
                   !isfinite(proposed_logjoint[chain_index])
                    valid[chain_index] = false
                end
            end
        end
    end

    for chain_index = 1:num_chains
        valid[chain_index] || continue
        step_size = step_sizes[chain_index]
        _add_scaled_column!(p, destination_gradient, chain_index, step_size / 2)
        _negate_column!(p, chain_index)
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

function batched_leapfrog_step_to!(
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    valid::AbstractVector{Bool},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    gradient::AbstractMatrix{Float64},
    target::AbstractBatchedDensityTarget,
    inverse_mass_matrix::AbstractVector,
    step_size::Float64,
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    q = destination_position
    p = destination_momentum
    num_chains = size(position, 2)
    size(position) == size(momentum) == size(gradient) ||
        throw(DimensionMismatch("batched leapfrog step requires matching position, momentum, and gradient matrices"))
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(destination_gradient) == size(position) ||
        throw(
            DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(destination_gradient))"),
        )
    length(destination_logjoint) == num_chains == length(valid) ||
        throw(DimensionMismatch("expected batched leapfrog vectors of length $num_chains"))
    length(direction) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched leapfrog directions, got $(length(direction))"))
    length(active) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched leapfrog activity flags, got $(length(active))"))

    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, false)
    # Count the live lanes while doing the momentum half-step / drift so lane
    # compaction (issue #160) can skip the finished/diverged columns in the
    # gradient below without a second pass over `active`.
    active_count = 0
    for chain_index = 1:num_chains
        active[chain_index] || continue
        active_count += 1
        valid[chain_index] = true
        signed_step = direction[chain_index] * step_size
        _add_scaled_column!(p, gradient, chain_index, signed_step / 2)
        _add_mass_drift_column!(q, p, inverse_mass_matrix, chain_index, signed_step)
    end

    proposed_logjoint, _ = batched_target_logdensity_and_gradient!(
        destination_logjoint,
        destination_gradient,
        target,
        q,
        active,
        active_count,
    )
    for chain_index = 1:num_chains
        valid[chain_index] || continue
        if !isfinite(proposed_logjoint[chain_index]) ||
           !_column_all_finite(destination_gradient, chain_index)
            valid[chain_index] = false
        else
            signed_step = direction[chain_index] * step_size
            _add_scaled_column!(p, destination_gradient, chain_index, signed_step / 2)
        end
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

# Per-chain overload: each chain steps with its own step size and inverse-mass
# column. Used only by the per-chain-adaptation path.
function batched_leapfrog_step_to!(
    destination_position::AbstractMatrix{Float64},
    destination_momentum::AbstractMatrix{Float64},
    destination_gradient::AbstractMatrix{Float64},
    destination_logjoint::AbstractVector{Float64},
    valid::AbstractVector{Bool},
    position::AbstractMatrix{Float64},
    momentum::AbstractMatrix{Float64},
    gradient::AbstractMatrix{Float64},
    target::AbstractBatchedDensityTarget,
    inverse_mass_matrices::AbstractMatrix,
    step_sizes::AbstractVector{Float64},
    direction::AbstractVector{Int},
    active::AbstractVector{Bool},
)
    q = destination_position
    p = destination_momentum
    num_chains = size(position, 2)
    size(position) == size(momentum) == size(gradient) ||
        throw(DimensionMismatch("batched leapfrog step requires matching position, momentum, and gradient matrices"))
    size(q) == size(position) ||
        throw(DimensionMismatch("expected proposal position workspace of size $(size(position)), got $(size(q))"))
    size(p) == size(position) ||
        throw(DimensionMismatch("expected proposal momentum workspace of size $(size(position)), got $(size(p))"))
    size(destination_gradient) == size(position) ||
        throw(
            DimensionMismatch("expected proposal gradient workspace of size $(size(position)), got $(size(destination_gradient))"),
        )
    length(destination_logjoint) == num_chains == length(valid) ||
        throw(DimensionMismatch("expected batched leapfrog vectors of length $num_chains"))
    length(direction) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched leapfrog directions, got $(length(direction))"))
    length(active) == num_chains ||
        throw(DimensionMismatch("expected $num_chains batched leapfrog activity flags, got $(length(active))"))
    size(inverse_mass_matrices, 2) == num_chains ||
        throw(DimensionMismatch("expected $num_chains inverse-mass columns, got $(size(inverse_mass_matrices, 2))"))
    length(step_sizes) == num_chains ||
        throw(DimensionMismatch("expected $num_chains step sizes, got $(length(step_sizes))"))

    copyto!(q, position)
    copyto!(p, momentum)
    fill!(valid, false)
    # Count the live lanes for lane compaction (issue #160); see the shared-step
    # overload above.
    active_count = 0
    for chain_index = 1:num_chains
        active[chain_index] || continue
        active_count += 1
        valid[chain_index] = true
        signed_step = direction[chain_index] * step_sizes[chain_index]
        _add_scaled_column!(p, gradient, chain_index, signed_step / 2)
        _add_mass_drift_column!(q, p, inverse_mass_matrices, chain_index, signed_step)
    end

    proposed_logjoint, _ = batched_target_logdensity_and_gradient!(
        destination_logjoint,
        destination_gradient,
        target,
        q,
        active,
        active_count,
    )
    for chain_index = 1:num_chains
        valid[chain_index] || continue
        if !isfinite(proposed_logjoint[chain_index]) ||
           !_column_all_finite(destination_gradient, chain_index)
            valid[chain_index] = false
        else
            signed_step = direction[chain_index] * step_sizes[chain_index]
            _add_scaled_column!(p, destination_gradient, chain_index, signed_step / 2)
        end
    end

    return q, p, destination_logjoint, destination_gradient, valid
end

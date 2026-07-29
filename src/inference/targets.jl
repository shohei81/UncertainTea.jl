abstract type AbstractDensityTarget end
abstract type AbstractBatchedDensityTarget end

struct ModelDensityTarget{M<:TeaModel,A<:Tuple,C<:ChoiceMap,G<:LogjointGradientCache} <: AbstractDensityTarget
    model::M
    args::A
    constraints::C
    gradient_cache::G
end

function target_logdensity(target::ModelDensityTarget, position::AbstractVector)
    # sampler-owned evaluation: Stan-style reject semantics (issue #157) --
    # invalid distribution parameters at a trajectory position score -Inf (a
    # rejected proposal / divergence) instead of aborting the run.
    #
    # Evaluate through the observations the gradient cache already staged (issue
    # #188): the public `logjoint_unconstrained` re-stages all N observations on
    # every call (O(N) plan re-walk rebuilding `(:y,i)` addresses), which NUTS
    # runs at every Hamiltonian evaluation and which dominated the per-draw cost
    # once scoring became O(1) (#144/#189). The gradient cache holds the staged
    # observations (built reject-on for the samplers), so this reuses them with an
    # O(1) staleness check; draws stay bitwise identical (pure caching).
    return _logjoint_value_from_cache(target.gradient_cache, position)
end

function target_gradient!(target::ModelDensityTarget, position::AbstractVector)
    return _logjoint_gradient!(target.gradient_cache, position)
end

function target_logdensity_and_gradient!(target::ModelDensityTarget, position::AbstractVector)
    value = target_logdensity(target, position)
    isfinite(value) || return value, target.gradient_cache.buffer
    return value, target_gradient!(target, position)
end

mutable struct TemperedDensityTarget{M<:TeaModel,A<:Tuple,C<:ChoiceMap,G<:LogjointGradientCache} <: AbstractDensityTarget
    const model::M
    const args::A
    const constraints::C
    const gradient_cache::G
    const proposal_location::Vector{Float64}
    const proposal_log_scale::Vector{Float64}
    const proposal_gradient::Vector{Float64}
    const gradient_buffer::Vector{Float64}
    beta::Float64
end

function target_logdensity_and_gradient!(target::TemperedDensityTarget, position::AbstractVector)
    value, _, _ = _tempered_target_value_and_gradient!(
        target.gradient_buffer,
        target.proposal_gradient,
        target.model,
        target.gradient_cache,
        position,
        target.args,
        target.constraints,
        target.proposal_location,
        target.proposal_log_scale,
        target.beta,
    )
    return value, target.gradient_buffer
end

struct BatchedModelDensityTarget{G<:BatchedLogjointGradientCache} <: AbstractBatchedDensityTarget
    gradient_cache::G
end

# `gradient_destination` is scratch: implementations may return a target-owned
# buffer instead of writing into it. Callers must use the returned matrix.
function batched_target_gradient!(
    gradient_destination::AbstractMatrix,
    target::BatchedModelDensityTarget,
    positions::AbstractMatrix,
)
    return batched_logjoint_gradient_unconstrained!(target.gradient_cache, positions)
end

function batched_target_logdensity_and_gradient!(
    values_destination::AbstractVector,
    gradient_destination::AbstractMatrix,
    target::BatchedModelDensityTarget,
    positions::AbstractMatrix,
)
    values, gradient = _batched_logjoint_and_gradient_unconstrained!(
        values_destination,
        target.gradient_cache,
        positions,
    )
    gradient === gradient_destination || copyto!(gradient_destination, gradient)
    return values, gradient_destination
end

mutable struct BatchedTemperedDensityTarget{M<:TeaModel,A<:Tuple,C<:ChoiceMap,G<:BatchedLogjointGradientCache} <:
               AbstractBatchedDensityTarget
    const model::M
    const args::A
    const constraints::C
    const gradient_cache::G
    const proposal_location::Vector{Float64}
    const proposal_log_scale::Vector{Float64}
    const logjoint_values::Vector{Float64}
    const logjoint_gradient::Matrix{Float64}
    const logproposal_values::Vector{Float64}
    const logproposal_gradient::Matrix{Float64}
    const proposal_noise::Matrix{Float64}
    const tempered_values::Vector{Float64}
    beta::Float64
end

function BatchedTemperedDensityTarget(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    gradient_cache::BatchedLogjointGradientCache,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    parameter_total::Int,
    num_particles::Int,
)
    return BatchedTemperedDensityTarget(
        model,
        args,
        constraints,
        gradient_cache,
        collect(Float64, proposal_location),
        collect(Float64, proposal_log_scale),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        beta,
    )
end

function batched_target_gradient!(
    gradient_destination::AbstractMatrix,
    target::BatchedTemperedDensityTarget,
    positions::AbstractMatrix,
)
    _batched_tempered_target!(
        target.tempered_values,
        gradient_destination,
        target.logjoint_values,
        target.logjoint_gradient,
        target.logproposal_values,
        target.logproposal_gradient,
        target.proposal_noise,
        target.model,
        target.gradient_cache,
        positions,
        target.args,
        target.constraints,
        target.proposal_location,
        target.proposal_log_scale,
        target.beta,
    )
    return gradient_destination
end

function batched_target_logdensity_and_gradient!(
    values_destination::AbstractVector,
    gradient_destination::AbstractMatrix,
    target::BatchedTemperedDensityTarget,
    positions::AbstractMatrix,
)
    _batched_tempered_target!(
        values_destination,
        gradient_destination,
        target.logjoint_values,
        target.logjoint_gradient,
        target.logproposal_values,
        target.logproposal_gradient,
        target.proposal_noise,
        target.model,
        target.gradient_cache,
        positions,
        target.args,
        target.constraints,
        target.proposal_location,
        target.proposal_log_scale,
        target.beta,
    )
    return values_destination, gradient_destination
end

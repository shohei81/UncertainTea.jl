# --- affine-coupling normalizing-flow guide (issue #235) ----------------------
#
# A RealNVP-style stack of affine coupling layers on top of a mean-field
# Gaussian base. The base draws z0 = mu + exp(log_scale) .* eps (eps ~ N(0,I));
# each coupling layer masks half the coordinates ("passive"), leaves them fixed,
# and rescales/shifts the other half ("active") by linear conditioners of the
# passive part: z[active] = z[active] .* exp(s) + t, with s = S*z[passive] + b_s
# and t = T*z[passive] + b_t. Every layer has an analytic log|det| = sum(s), so
# the reparameterized ELBO stays exact; all flow parameters are packed into a
# single flat vector and are trained host-side. The scored particles are still
# just positions theta, so the device model-gradient kernels are UNCHANGED.
struct FlowGuide
    dim::Int
    num_layers::Int
    # Per-layer passive/active coordinate indices (fixed alternating masks).
    passive::Vector{Vector{Int}}
    active::Vector{Vector{Int}}
    # Flat parameter vector: [mu (dim); log_scale (dim); per layer
    # S (na*np), b_s (na), T (na*np), b_t (na)].
    params::Vector{Float64}
end

struct ADVIResult
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    location::Vector{Float64}
    log_scale::Vector{Float64}
    best_location::Vector{Float64}
    best_log_scale::Vector{Float64}
    elbo_history::Vector{Float64}
    gradient_norm_history::Vector{Float64}
    best_elbo::Float64
    num_particles::Int
    learning_rate::Float64
    gradient_backend::Symbol
    guide::Symbol
    # :fullrank -> the strict lower triangle of the Cholesky factor (the
    # diagonal lives in log_scale); :lowrank -> the d x k factor B of
    # Sigma = D^2 + B B'; nothing for :meanfield.
    scale_factor::Union{Nothing,Matrix{Float64}}
    best_scale_factor::Union{Nothing,Matrix{Float64}}
    # ADVI extensions (issue #235). `elbo` is the optimized objective
    # (:standard or :iwae); `iwae_samples` is the importance-sample group size
    # K used when `elbo === :iwae` (1 for :standard). `standard_elbo_history`
    # mirrors `elbo_history` but always records the plain single-sample ELBO so
    # the two bounds can be compared. `flow`/`best_flow` carry the fitted
    # affine-coupling normalizing-flow guide when `guide === :flow`.
    elbo::Symbol
    iwae_samples::Int
    standard_elbo_history::Vector{Float64}
    flow::Union{Nothing,FlowGuide}
    best_flow::Union{Nothing,FlowGuide}
end

# Mean-field compatibility constructor (also used by the device ADVI path).
function ADVIResult(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    location::Vector{Float64},
    log_scale::Vector{Float64},
    best_location::Vector{Float64},
    best_log_scale::Vector{Float64},
    elbo_history::Vector{Float64},
    gradient_norm_history::Vector{Float64},
    best_elbo::Float64,
    num_particles::Int,
    learning_rate::Float64,
    gradient_backend::Symbol,
)
    return ADVIResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        best_location,
        best_log_scale,
        elbo_history,
        gradient_norm_history,
        best_elbo,
        num_particles,
        learning_rate,
        gradient_backend,
        :meanfield,
        nothing,
        nothing,
        :standard,
        1,
        copy(elbo_history),
        nothing,
        nothing,
    )
end

function _advi_gradient_backend(cache::BatchedLogjointGradientCache)
    !isnothing(cache.backend_cache) && return :backend_native
    !isnothing(cache.flat_cache) && return :flat_forwarddiff
    return :column_forwarddiff
end

function _resolve_unconstrained_point(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    point,
    rng::AbstractRNG,
    name::AbstractString,
)
    if isnothing(point)
        return _initial_hmc_position(model, args, constraints, nothing, rng)
    end

    layout = _conditioned_parameter_layout(model, constraints)
    parameter_total = parametercount(layout)
    constrained_total = parametervaluecount(layout)
    if length(point) == parameter_total
        return Float64[value for value in point]
    elseif length(point) == constrained_total
        return transform_to_unconstrained(model, Float64[value for value in point], args, constraints)
    end

    throw(
        DimensionMismatch(
            "expected $name to have length $parameter_total in unconstrained space or $constrained_total in constrained space, got $(length(point))",
        ),
    )
end

function _resolve_scale_vector(name::AbstractString, scale, parameter_total::Int)
    if scale isa Real
        return fill(Float64(scale), parameter_total)
    end

    length(scale) == parameter_total ||
        throw(DimensionMismatch("expected $name to have length $parameter_total, got $(length(scale))"))
    return Float64[value for value in scale]
end

function _draw_gaussian_particles!(
    destination::AbstractMatrix,
    noise::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
    rng::AbstractRNG,
)
    size(destination) == size(noise) ||
        throw(DimensionMismatch("expected Gaussian particle destination and noise matrices to have matching shapes"))
    size(destination, 1) == length(location) == length(log_scale) ||
        throw(
            DimensionMismatch(
                "expected Gaussian particle matrices with $(length(location)) rows, got $(size(destination, 1))",
            ),
        )

    for parameter_index in eachindex(location, log_scale)
        scale = exp(log_scale[parameter_index])
        for particle_index in axes(destination, 2)
            epsilon = randn(rng)
            noise[parameter_index, particle_index] = epsilon
            destination[parameter_index, particle_index] =
                location[parameter_index] + scale * epsilon
        end
    end
    return destination
end

function _gaussian_entropy(log_scale::AbstractVector)
    entropy = 0.5 * length(log_scale) * (1.0 + log(2.0 * pi))
    for value in log_scale
        entropy += value
    end
    return entropy
end

function _gaussian_logdensity!(
    destination::AbstractVector,
    noise::AbstractMatrix,
    log_scale::AbstractVector,
)
    size(noise, 2) == length(destination) ||
        throw(
            DimensionMismatch(
                "expected Gaussian log-density destination of length $(size(noise, 2)), got $(length(destination))",
            ),
        )

    normalizer = 0.5 * size(noise, 1) * log(2.0 * pi)
    for value in log_scale
        normalizer += value
    end

    for particle_index in eachindex(destination)
        squared_norm = 0.0
        for parameter_index in axes(noise, 1)
            epsilon = noise[parameter_index, particle_index]
            squared_norm += epsilon * epsilon
        end
        destination[particle_index] = -normalizer - 0.5 * squared_norm
    end
    return destination
end

# --- structured guides (:fullrank / :lowrank) ---------------------------------
#
# Both guides keep the reparameterization estimator and the closed-form
# entropy of the mean-field path. :fullrank draws theta = mu + L eps with L
# lower triangular (diagonal exp(log_scale), strict lower triangle in
# `factor`), so the entropy is the same sum-of-log-diagonals. :lowrank draws
# theta = mu + D eps1 + B eps2 with Sigma = D^2 + B B'; its entropy needs
# logdet(Sigma), computed through the k x k matrix M = I + B' D^-2 B.

function _draw_fullrank_particles!(
    destination::AbstractMatrix,
    noise::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
    factor::AbstractMatrix,
    rng::AbstractRNG,
)
    Random.randn!(rng, noise)
    for particle_index in axes(destination, 2)
        for parameter_index in eachindex(location)
            value =
                location[parameter_index] +
                exp(log_scale[parameter_index]) * noise[parameter_index, particle_index]
            for lower_index = 1:(parameter_index-1)
                value += factor[parameter_index, lower_index] * noise[lower_index, particle_index]
            end
            destination[parameter_index, particle_index] = value
        end
    end
    return destination
end

function _draw_lowrank_particles!(
    destination::AbstractMatrix,
    noise::AbstractMatrix,
    rank_noise::AbstractMatrix,
    location::AbstractVector,
    log_scale::AbstractVector,
    factor::AbstractMatrix,
    rng::AbstractRNG,
)
    Random.randn!(rng, noise)
    Random.randn!(rng, rank_noise)
    for particle_index in axes(destination, 2)
        for parameter_index in eachindex(location)
            value =
                location[parameter_index] +
                exp(log_scale[parameter_index]) * noise[parameter_index, particle_index]
            for rank_index in axes(factor, 2)
                value += factor[parameter_index, rank_index] * rank_noise[rank_index, particle_index]
            end
            destination[parameter_index, particle_index] = value
        end
    end
    return destination
end

# Sigma^-1 B (via Woodbury: D^-2 B M^-1), the intermediate W = D^-2 B, and
# logdet(M) with M = I + B' D^-2 B, shared by the lowrank entropy value and
# its gradients.
function _lowrank_entropy_terms(log_scale::AbstractVector, factor::AbstractMatrix)
    W = similar(factor)
    for rank_index in axes(factor, 2), parameter_index in axes(factor, 1)
        W[parameter_index, rank_index] =
            factor[parameter_index, rank_index] * exp(-2.0 * log_scale[parameter_index])
    end
    M = Matrix{Float64}(I, size(factor, 2), size(factor, 2))
    mul!(M, transpose(factor), W, 1.0, 1.0)
    chol = cholesky(Symmetric(M))
    return W / chol, W, logdet(chol)
end

function _fullrank_entropy(log_scale::AbstractVector)
    return _gaussian_entropy(log_scale)
end

function _lowrank_entropy(log_scale::AbstractVector, factor::AbstractMatrix)
    _, _, logdet_M = _lowrank_entropy_terms(log_scale, factor)
    return _gaussian_entropy(log_scale) + 0.5 * logdet_M
end

function _accumulate_fullrank_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    factor_gradient::AbstractMatrix,
    gradients::AbstractMatrix,
    noise::AbstractMatrix,
    log_scale::AbstractVector,
    particle_valid::AbstractVector{Bool},
    num_valid_particles::Int,
)
    fill!(location_gradient, 0.0)
    fill!(log_scale_gradient, 0.0)
    fill!(factor_gradient, 0.0)
    for particle_index in axes(gradients, 2)
        particle_valid[particle_index] || continue
        for parameter_index in eachindex(location_gradient)
            gradient = gradients[parameter_index, particle_index]
            location_gradient[parameter_index] += gradient
            log_scale_gradient[parameter_index] += gradient * noise[parameter_index, particle_index]
            for lower_index = 1:(parameter_index-1)
                factor_gradient[parameter_index, lower_index] +=
                    gradient * noise[lower_index, particle_index]
            end
        end
    end
    for parameter_index in eachindex(location_gradient)
        location_gradient[parameter_index] /= num_valid_particles
        # reparameterization term plus the d/dlog_scale of the entropy (= 1)
        log_scale_gradient[parameter_index] =
            1.0 +
            exp(log_scale[parameter_index]) * log_scale_gradient[parameter_index] /
            num_valid_particles
    end
    factor_gradient ./= num_valid_particles
    return nothing
end

function _accumulate_lowrank_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    factor_gradient::AbstractMatrix,
    gradients::AbstractMatrix,
    noise::AbstractMatrix,
    rank_noise::AbstractMatrix,
    log_scale::AbstractVector,
    factor::AbstractMatrix,
    particle_valid::AbstractVector{Bool},
    num_valid_particles::Int,
)
    fill!(location_gradient, 0.0)
    fill!(log_scale_gradient, 0.0)
    fill!(factor_gradient, 0.0)
    for particle_index in axes(gradients, 2)
        particle_valid[particle_index] || continue
        for parameter_index in eachindex(location_gradient)
            gradient = gradients[parameter_index, particle_index]
            location_gradient[parameter_index] += gradient
            log_scale_gradient[parameter_index] += gradient * noise[parameter_index, particle_index]
            for rank_index in axes(factor, 2)
                factor_gradient[parameter_index, rank_index] +=
                    gradient * rank_noise[rank_index, particle_index]
            end
        end
    end
    sigma_inv_factor, W, _ = _lowrank_entropy_terms(log_scale, factor)
    for parameter_index in eachindex(location_gradient)
        location_gradient[parameter_index] /= num_valid_particles
        # (Sigma^-1)_ii = exp(-2w_i) - sum_l C[i,l] W[i,l]  (Woodbury diagonal)
        sigma_inv_diagonal = exp(-2.0 * log_scale[parameter_index])
        for rank_index in axes(factor, 2)
            sigma_inv_diagonal -=
                sigma_inv_factor[parameter_index, rank_index] * W[parameter_index, rank_index]
        end
        log_scale_gradient[parameter_index] =
            exp(2.0 * log_scale[parameter_index]) * sigma_inv_diagonal +
            exp(log_scale[parameter_index]) * log_scale_gradient[parameter_index] /
            num_valid_particles
    end
    for rank_index in axes(factor, 2), parameter_index in eachindex(location_gradient)
        factor_gradient[parameter_index, rank_index] =
            factor_gradient[parameter_index, rank_index] / num_valid_particles +
            sigma_inv_factor[parameter_index, rank_index]
    end
    return nothing
end

# Three-block variant of _clip_advi_gradients! for the structured guides.
function _clip_advi_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    factor_gradient::AbstractMatrix,
    gradient_clip::Real,
)
    gradient_norm = 0.0
    for value in location_gradient
        gradient_norm += value * value
    end
    for value in log_scale_gradient
        gradient_norm += value * value
    end
    for value in factor_gradient
        gradient_norm += value * value
    end
    gradient_norm = sqrt(gradient_norm)

    if isfinite(gradient_clip) && gradient_norm > gradient_clip
        scale = gradient_clip / gradient_norm
        location_gradient .*= scale
        log_scale_gradient .*= scale
        factor_gradient .*= scale
        return Float64(gradient_clip)
    end

    return gradient_norm
end

"""
    variational_covariance(result::ADVIResult; use_best=true) -> Matrix{Float64}

The covariance of the fitted variational approximation in unconstrained
space: diagonal for `:meanfield`, `L L'` for `:fullrank`, `D^2 + B B'` for
`:lowrank`. The `:flow` guide has no closed-form covariance, so it is estimated
from a fixed-seed Monte Carlo sample (deterministic given the fit).
"""
function variational_covariance(result::ADVIResult; use_best::Bool=true)
    log_scale = use_best ? result.best_log_scale : result.log_scale
    factor = use_best ? result.best_scale_factor : result.scale_factor
    if result.guide === :flow
        samples = variational_samples(
            result;
            num_samples=20000,
            space=:unconstrained,
            use_best=use_best,
            rng=MersenneTwister(0),
        )
        mean_column = sum(samples; dims=2) ./ size(samples, 2)
        centered = samples .- mean_column
        return centered * transpose(centered) ./ (size(samples, 2) - 1)
    elseif result.guide === :meanfield
        return Matrix(Diagonal(exp.(2.0 .* log_scale)))
    elseif result.guide === :fullrank
        cholesky_factor = copy(factor)
        for parameter_index in eachindex(log_scale)
            cholesky_factor[parameter_index, parameter_index] = exp(log_scale[parameter_index])
        end
        return cholesky_factor * transpose(cholesky_factor)
    end
    return Matrix(Diagonal(exp.(2.0 .* log_scale))) + factor * transpose(factor)
end

function _batched_transform_to_constrained!(
    destination::AbstractMatrix,
    model::TeaModel,
    params::AbstractMatrix,
)
    size(destination, 2) == size(params, 2) ||
        throw(
            DimensionMismatch(
                "expected constrained particle destination with $(size(params, 2)) columns, got $(size(destination, 2))",
            ),
        )

    for particle_index in axes(params, 2)
        _transform_to_constrained!(view(destination, :, particle_index), model, view(params, :, particle_index))
    end
    return destination
end

# Signature-aware column transform (#95 PR-6): the unconstrained columns hold
# the conditioned latent set, so reconstruction reads observations from the
# constraints rather than the syntactic default layout.
function _signature_batched_transform_to_constrained!(
    destination::AbstractMatrix,
    model::TeaModel,
    params::AbstractMatrix,
    args::Tuple,
    constraints::ChoiceMap,
)
    size(destination, 2) == size(params, 2) ||
        throw(
            DimensionMismatch(
                "expected constrained particle destination with $(size(params, 2)) columns, got $(size(destination, 2))",
            ),
        )

    for particle_index in axes(params, 2)
        destination[:, particle_index] =
            transform_to_constrained(model, collect(view(params, :, particle_index)), args, constraints)
    end
    return destination
end

function _clip_advi_gradients!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    gradient_clip::Real,
)
    gradient_norm = 0.0
    for value in location_gradient
        gradient_norm += value * value
    end
    for value in log_scale_gradient
        gradient_norm += value * value
    end
    gradient_norm = sqrt(gradient_norm)

    if isfinite(gradient_clip) && gradient_norm > gradient_clip
        scale = gradient_clip / gradient_norm
        for index in eachindex(location_gradient)
            location_gradient[index] *= scale
            log_scale_gradient[index] *= scale
        end
        return Float64(gradient_clip)
    end

    return gradient_norm
end

function _adam_ascent_step!(
    parameters::AbstractArray,
    first_moment::AbstractArray,
    second_moment::AbstractArray,
    gradient::AbstractArray,
    iteration::Int,
    learning_rate::Float64,
    beta1::Float64,
    beta2::Float64,
    epsilon::Float64,
)
    beta1_correction = 1.0 - beta1^iteration
    beta2_correction = 1.0 - beta2^iteration
    for index in eachindex(parameters, first_moment, second_moment, gradient)
        grad = gradient[index]
        first_moment[index] = beta1 * first_moment[index] + (1.0 - beta1) * grad
        second_moment[index] = beta2 * second_moment[index] + (1.0 - beta2) * grad * grad
        mhat = first_moment[index] / beta1_correction
        vhat = second_moment[index] / beta2_correction
        parameters[index] += learning_rate * mhat / (sqrt(vhat) + epsilon)
    end
    return parameters
end

function variational_mean(
    result::ADVIResult;
    space::Symbol=:constrained,
    use_best::Bool=true,
)
    if result.guide === :flow
        # The flow guide has no closed-form mean; estimate it from a fixed-seed
        # sample (deterministic given the fit).
        samples =
            variational_samples(result; num_samples=20000, space=space, use_best=use_best, rng=MersenneTwister(0))
        return vec(sum(samples; dims=2) ./ size(samples, 2))
    end
    parameters = use_best ? result.best_location : result.location
    if space === :unconstrained
        return copy(parameters)
    elseif space === :constrained
        return transform_to_constrained(result.model, parameters, result.args, result.constraints)
    end

    throw(ArgumentError("variational space must be :constrained or :unconstrained"))
end

function variational_samples(
    result::ADVIResult;
    num_samples::Int,
    space::Symbol=:constrained,
    use_best::Bool=true,
    rng::AbstractRNG=Random.default_rng(),
)
    num_samples > 0 || throw(ArgumentError("variational_samples requires num_samples > 0"))

    location = use_best ? result.best_location : result.location
    log_scale = use_best ? result.best_log_scale : result.log_scale
    factor = use_best ? result.best_scale_factor : result.scale_factor
    unconstrained = Matrix{Float64}(undef, length(location), num_samples)
    noise = similar(unconstrained)
    if result.guide === :flow
        flow = use_best ? result.best_flow : result.flow
        Random.randn!(rng, noise)
        for sample_index = 1:num_samples
            theta, _ = _flow_forward(flow, flow.params, view(noise, :, sample_index))
            for parameter_index = 1:flow.dim
                unconstrained[parameter_index, sample_index] = theta[parameter_index]
            end
        end
    elseif result.guide === :fullrank
        _draw_fullrank_particles!(unconstrained, noise, location, log_scale, factor, rng)
    elseif result.guide === :lowrank
        rank_noise = Matrix{Float64}(undef, size(factor, 2), num_samples)
        _draw_lowrank_particles!(unconstrained, noise, rank_noise, location, log_scale, factor, rng)
    else
        _draw_gaussian_particles!(unconstrained, noise, location, log_scale, rng)
    end

    if space === :unconstrained
        return unconstrained
    elseif space === :constrained
        layout = _conditioned_parameter_layout(result.model, result.constraints)
        constrained = Matrix{Float64}(undef, parametervaluecount(layout), num_samples)
        _signature_batched_transform_to_constrained!(
            constrained,
            result.model,
            unconstrained,
            result.args,
            result.constraints,
        )
        return constrained
    end

    throw(ArgumentError("variational space must be :constrained or :unconstrained"))
end

# --- flow-guide machinery (issue #235) ----------------------------------------

# Fixed alternating masks: coordinate i is "passive" in layer `layer` when
# (i-1)+(layer-1) is even, so consecutive layers swap the passive/active roles
# and every coordinate is transformed once at least two layers are stacked.
function _flow_masks(dim::Int, num_layers::Int)
    passive = Vector{Vector{Int}}(undef, num_layers)
    active = Vector{Vector{Int}}(undef, num_layers)
    for layer = 1:num_layers
        p = Int[]
        a = Int[]
        for i = 1:dim
            if iseven((i - 1) + (layer - 1))
                push!(p, i)
            else
                push!(a, i)
            end
        end
        passive[layer] = p
        active[layer] = a
    end
    return passive, active
end

function _flow_parameter_count(dim::Int, passive, active)
    total = 2 * dim
    for layer in eachindex(passive)
        na = length(active[layer])
        np = length(passive[layer])
        total += na * (2 * np + 2)
    end
    return total
end

# Build a flow guide initialized to the mean-field base: all coupling
# parameters start at zero so s = t = 0 and the flow is the identity map on top
# of z0 = mu + exp(log_scale) .* eps.
function _make_flow_guide(
    dim::Int,
    num_layers::Int,
    location::AbstractVector,
    log_scale::AbstractVector,
)
    dim >= 2 || throw(
        ArgumentError("batched_advi guide=:flow requires at least 2 latent dimensions, got $dim"),
    )
    num_layers >= 1 ||
        throw(ArgumentError("batched_advi requires num_flow_layers >= 1, got $num_layers"))
    passive, active = _flow_masks(dim, num_layers)
    params = zeros(Float64, _flow_parameter_count(dim, passive, active))
    for i = 1:dim
        params[i] = location[i]
        params[dim+i] = log_scale[i]
    end
    return FlowGuide(dim, num_layers, passive, active, params)
end

# Push a single base draw `eps` through the flow using parameter vector
# `params` (which may carry ForwardDiff duals). Returns the transformed
# position and the accumulated log|det (dtheta/deps)| = sum(log_scale) + sum(s).
function _flow_forward(guide::FlowGuide, params::AbstractVector{T}, eps::AbstractVector) where {T}
    dim = guide.dim
    z = Vector{T}(undef, dim)
    logdet = zero(T)
    for i = 1:dim
        z[i] = params[i] + exp(params[dim+i]) * eps[i]
        logdet += params[dim+i]
    end
    offset = 2 * dim
    for layer = 1:guide.num_layers
        A = guide.active[layer]
        P = guide.passive[layer]
        na = length(A)
        np = length(P)
        for j = 1:na
            s_j = zero(T)
            t_j = zero(T)
            for p = 1:np
                zp = z[P[p]]
                s_j += params[offset+(p-1)*na+j] * zp
                t_j += params[offset+na*np+na+(p-1)*na+j] * zp
            end
            s_j += params[offset+na*np+j]
            t_j += params[offset+2*na*np+na+j]
            a = A[j]
            z[a] = z[a] * exp(s_j) + t_j
            logdet += s_j
        end
        offset += na * (2 * np + 2)
    end
    return z, logdet
end

# --- importance-weighted (IWAE) reduction (issue #235) ------------------------
#
# The importance-weighted bound splits the `num_particles` draws into
# num_particles / K groups of K importance samples. Each group contributes a
# log-mean-exp of its log-weights log w_k = log p(theta_k) - log q(theta_k), and
# the K=1 case reduces exactly to the standard single-sample ELBO. The
# self-normalized weights returned (softmax within a group, scaled by 1/#groups
# so they sum to one across all particles) are the reparameterization-gradient
# coefficients: the IWAE gradient is sum_k wtilde_k * grad log w_k, which is the
# standard accumulation with 1/N replaced by wtilde_k.
function _iwae_weights_and_bound!(
    weights::AbstractVector,
    logw::AbstractVector,
    particle_valid::AbstractVector{Bool},
    num_particles::Int,
    group_size::Int,
)
    fill!(weights, 0.0)
    num_groups = num_particles ÷ group_size
    bound = 0.0
    groups_used = 0
    for g = 0:(num_groups-1)
        lo = g * group_size + 1
        hi = lo + group_size - 1
        max_logw = -Inf
        num_valid = 0
        for k = lo:hi
            if particle_valid[k]
                num_valid += 1
                max_logw = max(max_logw, logw[k])
            end
        end
        num_valid > 0 || continue
        sum_exp = 0.0
        for k = lo:hi
            particle_valid[k] || continue
            sum_exp += exp(logw[k] - max_logw)
        end
        log_sum_exp = max_logw + log(sum_exp)
        bound += log_sum_exp - log(num_valid)
        for k = lo:hi
            particle_valid[k] || continue
            weights[k] = exp(logw[k] - log_sum_exp)
        end
        groups_used += 1
    end
    groups_used > 0 ||
        throw(ArgumentError("batched_advi encountered only non-finite importance weights"))
    for k = 1:num_particles
        weights[k] /= groups_used
    end
    return bound / groups_used
end

# Weighted mean-field accumulation: identical to the standard reparameterization
# gradient but with the uniform 1/N particle weight replaced by `weights`
# (which sum to one). Used by the IWAE objective for the mean-field guide.
function _accumulate_meanfield_gradients_weighted!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    gradients::AbstractMatrix,
    noise::AbstractMatrix,
    log_scale::AbstractVector,
    weights::AbstractVector,
)
    fill!(location_gradient, 0.0)
    fill!(log_scale_gradient, 0.0)
    for particle_index in axes(gradients, 2)
        weight = weights[particle_index]
        weight == 0.0 && continue
        for parameter_index in eachindex(location_gradient)
            gradient = gradients[parameter_index, particle_index]
            location_gradient[parameter_index] += weight * gradient
            log_scale_gradient[parameter_index] +=
                weight * gradient * noise[parameter_index, particle_index]
        end
    end
    for parameter_index in eachindex(location_gradient)
        log_scale_gradient[parameter_index] =
            1.0 + exp(log_scale[parameter_index]) * log_scale_gradient[parameter_index]
    end
    return nothing
end

# Weighted full-rank accumulation: the IWAE analogue of
# `_accumulate_fullrank_gradients!`.
function _accumulate_fullrank_gradients_weighted!(
    location_gradient::AbstractVector,
    log_scale_gradient::AbstractVector,
    factor_gradient::AbstractMatrix,
    gradients::AbstractMatrix,
    noise::AbstractMatrix,
    log_scale::AbstractVector,
    weights::AbstractVector,
)
    fill!(location_gradient, 0.0)
    fill!(log_scale_gradient, 0.0)
    fill!(factor_gradient, 0.0)
    for particle_index in axes(gradients, 2)
        weight = weights[particle_index]
        weight == 0.0 && continue
        for parameter_index in eachindex(location_gradient)
            gradient = gradients[parameter_index, particle_index]
            location_gradient[parameter_index] += weight * gradient
            log_scale_gradient[parameter_index] +=
                weight * gradient * noise[parameter_index, particle_index]
            for lower_index = 1:(parameter_index-1)
                factor_gradient[parameter_index, lower_index] +=
                    weight * gradient * noise[lower_index, particle_index]
            end
        end
    end
    for parameter_index in eachindex(location_gradient)
        log_scale_gradient[parameter_index] =
            1.0 + exp(log_scale[parameter_index]) * log_scale_gradient[parameter_index]
    end
    return nothing
end

function _clip_flat_gradient!(gradient::AbstractVector, gradient_clip::Real)
    gradient_norm = 0.0
    for value in gradient
        gradient_norm += value * value
    end
    gradient_norm = sqrt(gradient_norm)
    if isfinite(gradient_clip) && gradient_norm > gradient_clip
        gradient .*= gradient_clip / gradient_norm
        return Float64(gradient_clip)
    end
    return gradient_norm
end

function batched_advi(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_steps::Int,
    num_particles::Int=32,
    learning_rate::Real=0.05,
    initial_params=nothing,
    initial_log_scale=-1.0,
    guide::Symbol=:meanfield,
    lowrank_rank::Int=1,
    num_flow_layers::Int=4,
    elbo::Symbol=:standard,
    iwae_samples::Int=0,
    beta1::Real=0.9,
    beta2::Real=0.999,
    adam_epsilon::Real=1e-8,
    gradient_clip::Real=Inf,
    callback=nothing,
    callback_every::Int=10,
    backend=nothing,
    precision=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    guide in (:meanfield, :fullrank, :lowrank, :flow) || throw(
        ArgumentError(
            "batched_advi guide must be :meanfield, :fullrank, :lowrank, or :flow, got $guide",
        ),
    )
    elbo in (:standard, :iwae) ||
        throw(ArgumentError("batched_advi elbo must be :standard or :iwae, got $elbo"))
    # The importance-weighted objective needs a closed-form per-particle guide
    # log-density; :lowrank's redundant reparameterization does not provide one,
    # so IWAE is supported for :meanfield, :fullrank, and :flow only.
    if elbo === :iwae
        guide in (:meanfield, :fullrank, :flow) || throw(
            ArgumentError(
                "batched_advi elbo=:iwae supports guide=:meanfield, :fullrank, or :flow, got $guide",
            ),
        )
    end
    if backend !== nothing
        guide === :meanfield ||
            throw(ArgumentError("batched_advi guide=$guide is CPU-only; the device path supports :meanfield"))
        elbo === :standard || throw(
            ArgumentError("batched_advi elbo=:iwae is CPU-only; the device path supports the standard ELBO"),
        )
    end
    if backend !== nothing
        # Device-resident ADVI inner loop. RNG stays host-side; results are
        # statistically equivalent to the CPU path (untouched when backend===nothing).
        backend isa KernelAbstractions.Backend ||
            throw(ArgumentError("batched_advi `backend` must be a KernelAbstractions.Backend or nothing, got $(typeof(backend))"))
        device_precision = precision === nothing ? default_device_precision(backend) : precision
        return _run_device_batched_advi(
            model, args, constraints;
            num_steps=num_steps,
            num_particles=num_particles,
            learning_rate=learning_rate,
            initial_params=initial_params,
            initial_log_scale=initial_log_scale,
            beta1=beta1,
            beta2=beta2,
            adam_epsilon=adam_epsilon,
            gradient_clip=gradient_clip,
            callback=callback,
            callback_every=callback_every,
            backend=backend,
            precision=device_precision,
            rng=rng,
        )
    end

    layout = _conditioned_parameter_layout(model, constraints)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_advi requires at least one parameterized latent choice"))
    num_steps > 0 || throw(ArgumentError("batched_advi requires num_steps > 0"))
    num_particles > 0 || throw(ArgumentError("batched_advi requires num_particles > 0"))
    learning_rate > 0 || throw(ArgumentError("batched_advi requires learning_rate > 0"))
    0 <= beta1 < 1 || throw(ArgumentError("batched_advi requires 0 <= beta1 < 1"))
    0 <= beta2 < 1 || throw(ArgumentError("batched_advi requires 0 <= beta2 < 1"))
    adam_epsilon > 0 || throw(ArgumentError("batched_advi requires adam_epsilon > 0"))
    gradient_clip > 0 || throw(ArgumentError("batched_advi requires gradient_clip > 0"))

    # IWAE importance-sample group size K: partition the particle batch into
    # num_particles / K groups, each of which forms one log-mean-exp bound. K=1
    # (the standard ELBO) needs no grouping.
    group_size = elbo === :iwae ? (iwae_samples > 0 ? iwae_samples : num_particles) : 1
    if elbo === :iwae
        1 <= group_size <= num_particles ||
            throw(ArgumentError("batched_advi requires 1 <= iwae_samples <= num_particles"))
        num_particles % group_size == 0 || throw(
            ArgumentError("batched_advi requires iwae_samples to divide num_particles evenly"),
        )
    end

    location = _resolve_unconstrained_point(model, args, constraints, initial_params, rng, "initial_params")
    log_scale = _resolve_scale_vector("initial_log_scale", initial_log_scale, parameter_total)

    if guide === :flow
        return _run_flow_advi(
            model,
            args,
            constraints,
            location,
            log_scale;
            num_steps=num_steps,
            num_particles=num_particles,
            num_flow_layers=num_flow_layers,
            learning_rate=Float64(learning_rate),
            elbo=elbo,
            group_size=group_size,
            beta1=Float64(beta1),
            beta2=Float64(beta2),
            adam_epsilon=Float64(adam_epsilon),
            gradient_clip=Float64(gradient_clip),
            callback=callback,
            callback_every=callback_every,
            rng=rng,
        )
    end
    particles = Matrix{Float64}(undef, parameter_total, num_particles)
    noise = similar(particles)
    values = Vector{Float64}(undef, num_particles)
    # Shape the gradient cache WITHOUT consuming the RNG: the cache only needs the
    # particle matrix shape and a finite representative point, not a real draw. The
    # first genuine draw happens inside the loop below, so a same-seed device run
    # (which has no pre-loop draw) sees an identical RNG stream (issue #108).
    particles .= location
    cache = BatchedLogjointGradientCache(model, particles, args, constraints)
    gradient_backend = _advi_gradient_backend(cache)

    # Structured-guide state: a zero factor makes both guides start at the
    # mean-field initialization.
    factor = nothing
    factor_gradient = nothing
    factor_m1 = nothing
    factor_m2 = nothing
    rank_noise = nothing
    if guide === :fullrank
        factor = zeros(Float64, parameter_total, parameter_total)
    elseif guide === :lowrank
        1 <= lowrank_rank <= parameter_total || throw(
            ArgumentError(
                "batched_advi requires 1 <= lowrank_rank <= $(parameter_total), got $lowrank_rank",
            ),
        )
        factor = zeros(Float64, parameter_total, lowrank_rank)
        rank_noise = Matrix{Float64}(undef, lowrank_rank, num_particles)
    end
    if !isnothing(factor)
        factor_gradient = zero(factor)
        factor_m1 = zero(factor)
        factor_m2 = zero(factor)
    end

    location_gradient = zeros(Float64, parameter_total)
    log_scale_gradient = zeros(Float64, parameter_total)
    location_m1 = zeros(Float64, parameter_total)
    location_m2 = zeros(Float64, parameter_total)
    log_scale_m1 = zeros(Float64, parameter_total)
    log_scale_m2 = zeros(Float64, parameter_total)
    elbo_history = Vector{Float64}(undef, num_steps)
    standard_elbo_history = Vector{Float64}(undef, num_steps)
    gradient_norm_history = Vector{Float64}(undef, num_steps)
    particle_valid = Vector{Bool}(undef, num_particles)
    # IWAE bookkeeping: per-particle log-weights and their self-normalized
    # reparameterization-gradient coefficients (unused on the standard path).
    logw = elbo === :iwae ? Vector{Float64}(undef, num_particles) : Float64[]
    iwae_weights = elbo === :iwae ? Vector{Float64}(undef, num_particles) : Float64[]
    best_location = copy(location)
    best_log_scale = copy(log_scale)
    best_factor = isnothing(factor) ? nothing : copy(factor)
    best_elbo = -Inf

    learning_rate_f64 = Float64(learning_rate)
    beta1_f64 = Float64(beta1)
    beta2_f64 = Float64(beta2)
    adam_epsilon_f64 = Float64(adam_epsilon)
    gradient_clip_f64 = Float64(gradient_clip)

    for iteration = 1:num_steps
        if guide === :fullrank
            _draw_fullrank_particles!(particles, noise, location, log_scale, factor, rng)
        elseif guide === :lowrank
            _draw_lowrank_particles!(particles, noise, rank_noise, location, log_scale, factor, rng)
        else
            _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
        end
        _batched_logjoint_and_gradient_unconstrained!(values, cache, particles)
        num_valid_particles = 0
        for particle_index = 1:num_particles
            particle_valid[particle_index] =
                isfinite(values[particle_index]) &&
                all(isfinite, view(cache.gradient_buffer, :, particle_index))
            num_valid_particles += particle_valid[particle_index]
        end
        num_valid_particles > 0 ||
            throw(ArgumentError("batched_advi encountered only non-finite unconstrained logjoint values or gradients"))
        num_valid_particles == num_particles ||
            @warn "batched_advi skipped particles with a non-finite logjoint or gradient" maxlog = 1

        # Closed-form standard ELBO (always recorded for comparison / :standard).
        entropy =
            guide === :lowrank ? _lowrank_entropy(log_scale, factor) : _gaussian_entropy(log_scale)
        elbo_total = 0.0
        for particle_index = 1:num_particles
            particle_valid[particle_index] || continue
            elbo_total += values[particle_index]
        end
        standard_elbo = elbo_total / num_valid_particles + entropy
        standard_elbo_history[iteration] = standard_elbo

        iwae_bound = 0.0
        if elbo === :iwae
            # log w_k = log p(theta_k) - log q(theta_k); the Gaussian guide
            # log-density is -normalizer - 0.5||eps||^2 (mean-field and full-rank
            # share the sum-of-log-diagonal normalizer).
            _gaussian_logdensity!(logw, noise, log_scale)
            for particle_index = 1:num_particles
                logw[particle_index] = values[particle_index] - logw[particle_index]
            end
            iwae_bound = _iwae_weights_and_bound!(
                iwae_weights,
                logw,
                particle_valid,
                num_particles,
                group_size,
            )
        end

        if elbo === :iwae && guide === :fullrank
            _accumulate_fullrank_gradients_weighted!(
                location_gradient,
                log_scale_gradient,
                factor_gradient,
                cache.gradient_buffer,
                noise,
                log_scale,
                iwae_weights,
            )
        elseif elbo === :iwae
            _accumulate_meanfield_gradients_weighted!(
                location_gradient,
                log_scale_gradient,
                cache.gradient_buffer,
                noise,
                log_scale,
                iwae_weights,
            )
        elseif guide === :fullrank
            _accumulate_fullrank_gradients!(
                location_gradient,
                log_scale_gradient,
                factor_gradient,
                cache.gradient_buffer,
                noise,
                log_scale,
                particle_valid,
                num_valid_particles,
            )
        elseif guide === :lowrank
            _accumulate_lowrank_gradients!(
                location_gradient,
                log_scale_gradient,
                factor_gradient,
                cache.gradient_buffer,
                noise,
                rank_noise,
                log_scale,
                factor,
                particle_valid,
                num_valid_particles,
            )
        else
            fill!(location_gradient, 0.0)
            fill!(log_scale_gradient, 1.0)
            for parameter_index = 1:parameter_total
                scale = exp(log_scale[parameter_index])
                mean_gradient = 0.0
                mean_scale_gradient = 0.0
                for particle_index = 1:num_particles
                    particle_valid[particle_index] || continue
                    target_gradient = cache.gradient_buffer[parameter_index, particle_index]
                    mean_gradient += target_gradient
                    mean_scale_gradient += target_gradient * scale * noise[parameter_index, particle_index]
                end
                location_gradient[parameter_index] = mean_gradient / num_valid_particles
                log_scale_gradient[parameter_index] += mean_scale_gradient / num_valid_particles
            end
        end

        gradient_norm_history[iteration] =
            isnothing(factor_gradient) ?
            _clip_advi_gradients!(location_gradient, log_scale_gradient, gradient_clip_f64) :
            _clip_advi_gradients!(
                location_gradient,
                log_scale_gradient,
                factor_gradient,
                gradient_clip_f64,
            )
        # The optimized objective is the IWAE bound when elbo=:iwae, else the
        # standard ELBO; `best_*` track the maximizer of that objective.
        objective = elbo === :iwae ? iwae_bound : standard_elbo
        elbo_history[iteration] = objective
        if objective > best_elbo
            best_elbo = objective
            copyto!(best_location, location)
            copyto!(best_log_scale, log_scale)
            isnothing(best_factor) || copyto!(best_factor, factor)
        end

        _adam_ascent_step!(
            location,
            location_m1,
            location_m2,
            location_gradient,
            iteration,
            learning_rate_f64,
            beta1_f64,
            beta2_f64,
            adam_epsilon_f64,
        )
        _adam_ascent_step!(
            log_scale,
            log_scale_m1,
            log_scale_m2,
            log_scale_gradient,
            iteration,
            learning_rate_f64,
            beta1_f64,
            beta2_f64,
            adam_epsilon_f64,
        )
        isnothing(factor) || _adam_ascent_step!(
            factor,
            factor_m1,
            factor_m2,
            factor_gradient,
            iteration,
            learning_rate_f64,
            beta1_f64,
            beta2_f64,
            adam_epsilon_f64,
        )
        isnothing(callback) || _invoke_progress_callback(
            callback, callback_every, :step, iteration, num_steps, NaN, 0)
    end

    return ADVIResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        best_location,
        best_log_scale,
        elbo_history,
        gradient_norm_history,
        best_elbo,
        num_particles,
        learning_rate_f64,
        gradient_backend,
        guide,
        factor,
        best_factor,
        elbo,
        group_size,
        standard_elbo_history,
        nothing,
        nothing,
    )
end

# Affine-coupling normalizing-flow ADVI (issue #235). The base draw and every
# coupling layer are reparameterized, so the reparameterization ELBO integrand
# per particle is  log p(theta) + 0.5||eps||^2 + (d/2)log(2pi) + logdet, whose
# expectation is E[log p] + H[q]. The flow parameters are trained by
# differentiating  sum_k weight_k (g_k . theta(eps_k, params) + logdet(eps_k,
# params))  through the flow with ForwardDiff, where g_k = grad_theta log
# p(theta_k) comes from the UNCHANGED batched model-gradient path and weight_k
# is 1/N (standard ELBO) or the self-normalized IWAE coefficient.
function _run_flow_advi(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    location::AbstractVector,
    log_scale::AbstractVector;
    num_steps::Int,
    num_particles::Int,
    num_flow_layers::Int,
    learning_rate::Float64,
    elbo::Symbol,
    group_size::Int,
    beta1::Float64,
    beta2::Float64,
    adam_epsilon::Float64,
    gradient_clip::Float64,
    callback,
    callback_every::Int,
    rng::AbstractRNG,
)
    dim = length(location)
    guide = _make_flow_guide(dim, num_flow_layers, location, log_scale)
    params = guide.params

    particles = Matrix{Float64}(undef, dim, num_particles)
    noise = similar(particles)
    values = Vector{Float64}(undef, num_particles)
    layer_logdet = Vector{Float64}(undef, num_particles)
    logw = Vector{Float64}(undef, num_particles)
    weights = Vector{Float64}(undef, num_particles)
    particle_valid = Vector{Bool}(undef, num_particles)

    # Shape the gradient cache without consuming the RNG (see the Gaussian path).
    particles .= location
    cache = BatchedLogjointGradientCache(model, particles, args, constraints)
    gradient_backend = _advi_gradient_backend(cache)

    gradient = zero(params)
    first_moment = zero(params)
    second_moment = zero(params)
    elbo_history = Vector{Float64}(undef, num_steps)
    standard_elbo_history = Vector{Float64}(undef, num_steps)
    gradient_norm_history = Vector{Float64}(undef, num_steps)
    best_params = copy(params)
    best_elbo = -Inf
    base_normalizer = 0.5 * dim * log(2.0 * pi)

    for iteration = 1:num_steps
        Random.randn!(rng, noise)
        for particle_index = 1:num_particles
            theta, logdet =
                _flow_forward(guide, params, view(noise, :, particle_index))
            for parameter_index = 1:dim
                particles[parameter_index, particle_index] = theta[parameter_index]
            end
            layer_logdet[particle_index] = logdet
        end
        _batched_logjoint_and_gradient_unconstrained!(values, cache, particles)

        num_valid_particles = 0
        for particle_index = 1:num_particles
            valid =
                isfinite(values[particle_index]) &&
                isfinite(layer_logdet[particle_index]) &&
                all(isfinite, view(cache.gradient_buffer, :, particle_index))
            particle_valid[particle_index] = valid
            num_valid_particles += valid
        end
        num_valid_particles > 0 ||
            throw(ArgumentError("batched_advi (flow) encountered only non-finite logjoint values or gradients"))

        # log w_k = log p(theta_k) - log q(theta_k), with
        # log q(theta_k) = -0.5||eps_k||^2 - (d/2)log(2pi) - logdet_k.
        for particle_index = 1:num_particles
            if particle_valid[particle_index]
                squared_norm = 0.0
                for parameter_index = 1:dim
                    epsilon = noise[parameter_index, particle_index]
                    squared_norm += epsilon * epsilon
                end
                logw[particle_index] =
                    values[particle_index] + 0.5 * squared_norm + base_normalizer +
                    layer_logdet[particle_index]
            else
                logw[particle_index] = -Inf
            end
        end

        # Sampled standard ELBO = mean of the per-particle log-weights.
        standard_total = 0.0
        for particle_index = 1:num_particles
            particle_valid[particle_index] || continue
            standard_total += logw[particle_index]
        end
        standard_elbo = standard_total / num_valid_particles
        standard_elbo_history[iteration] = standard_elbo

        objective = standard_elbo
        if elbo === :iwae
            objective =
                _iwae_weights_and_bound!(weights, logw, particle_valid, num_particles, group_size)
        else
            fill!(weights, 0.0)
            for particle_index = 1:num_particles
                particle_valid[particle_index] || continue
                weights[particle_index] = 1.0 / num_valid_particles
            end
        end
        elbo_history[iteration] = objective

        objective_fn =
            flow_params -> begin
                total = zero(eltype(flow_params))
                for particle_index = 1:num_particles
                    weight = weights[particle_index]
                    weight == 0.0 && continue
                    theta, logdet =
                        _flow_forward(guide, flow_params, view(noise, :, particle_index))
                    contribution = logdet
                    for parameter_index = 1:dim
                        contribution +=
                            cache.gradient_buffer[parameter_index, particle_index] *
                            theta[parameter_index]
                    end
                    total += weight * contribution
                end
                return total
            end
        ForwardDiff.gradient!(gradient, objective_fn, params)
        gradient_norm_history[iteration] = _clip_flat_gradient!(gradient, gradient_clip)

        if objective > best_elbo
            best_elbo = objective
            copyto!(best_params, params)
        end

        _adam_ascent_step!(
            params,
            first_moment,
            second_moment,
            gradient,
            iteration,
            learning_rate,
            beta1,
            beta2,
            adam_epsilon,
        )
        isnothing(callback) ||
            _invoke_progress_callback(callback, callback_every, :step, iteration, num_steps, NaN, 0)
    end

    best_flow = FlowGuide(dim, guide.num_layers, guide.passive, guide.active, copy(best_params))
    return ADVIResult(
        model,
        args,
        constraints,
        copy(params[1:dim]),
        copy(params[(dim+1):(2*dim)]),
        copy(best_params[1:dim]),
        copy(best_params[(dim+1):(2*dim)]),
        elbo_history,
        gradient_norm_history,
        best_elbo,
        num_particles,
        learning_rate,
        gradient_backend,
        :flow,
        nothing,
        nothing,
        elbo,
        group_size,
        standard_elbo_history,
        guide,
        best_flow,
    )
end

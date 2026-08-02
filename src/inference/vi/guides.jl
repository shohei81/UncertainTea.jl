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

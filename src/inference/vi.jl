include("vi/guides.jl")

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

    layout = _conditioned_parameter_layout(model, constraints, args)
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
        layout = _conditioned_parameter_layout(result.model, result.constraints, result.args)
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

"""
    batched_advi(model, args=(), constraints=choicemap(); num_iterations, num_particles=32, learning_rate=0.05, guide=:meanfield, elbo=:standard, kwargs...) -> ADVIResult

Automatic differentiation variational inference with `num_particles` particles
per Adam step, run for `num_iterations` steps. `guide` selects the variational
family: `:meanfield`, `:fullrank`, and `:lowrank` are Gaussian in unconstrained
space, while `:flow` stacks affine coupling layers (RealNVP-style) on a
mean-field base to capture nonlinear correlation and skew. `elbo=:standard`
(default) is the reparameterized ELBO; `elbo=:iwae` is the strictly tighter
importance-weighted bound with group size `iwae_samples` (available for
`:meanfield`, `:fullrank`, and `:flow`). Passing a KernelAbstractions `backend`
runs the mean-field/standard-ELBO path on the device.
"""
function batched_advi(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_iterations::Int,
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
    adtype::Symbol=:auto,
    rng::AbstractRNG=Random.default_rng(),
)
    adtype in (:auto, :forward, :reverse) ||
        throw(ArgumentError("batched_advi adtype must be :auto, :forward, or :reverse, got $(repr(adtype))"))
    !(adtype === :reverse && backend !== nothing) ||
        throw(ArgumentError("batched_advi adtype=:reverse is a host-only path and cannot be combined with a device `backend`"))
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
            num_iterations=num_iterations,
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

    layout = _conditioned_parameter_layout(model, constraints, args)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_advi requires at least one parameterized latent choice"))
    num_iterations > 0 || throw(ArgumentError("batched_advi requires num_iterations > 0"))
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
            num_iterations=num_iterations,
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
            adtype=adtype,
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
    cache = BatchedLogjointGradientCache(model, particles, args, constraints; adtype=adtype)
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
    elbo_history = Vector{Float64}(undef, num_iterations)
    standard_elbo_history = Vector{Float64}(undef, num_iterations)
    gradient_norm_history = Vector{Float64}(undef, num_iterations)
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

    for iteration = 1:num_iterations
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
            callback, callback_every, :step, iteration, num_iterations, NaN, 0)
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
    num_iterations::Int,
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
    adtype::Symbol=:auto,
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
    cache = BatchedLogjointGradientCache(model, particles, args, constraints; adtype=adtype)
    gradient_backend = _advi_gradient_backend(cache)

    gradient = zero(params)
    first_moment = zero(params)
    second_moment = zero(params)
    elbo_history = Vector{Float64}(undef, num_iterations)
    standard_elbo_history = Vector{Float64}(undef, num_iterations)
    gradient_norm_history = Vector{Float64}(undef, num_iterations)
    best_params = copy(params)
    best_elbo = -Inf
    base_normalizer = 0.5 * dim * log(2.0 * pi)

    for iteration = 1:num_iterations
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
            _invoke_progress_callback(callback, callback_every, :step, iteration, num_iterations, NaN, 0)
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

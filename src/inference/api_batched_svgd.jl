# Stein Variational Gradient Descent (Liu & Wang, NeurIPS 2016).
#
# SVGD is a DETERMINISTIC PARTICLE method that sits between variational
# inference and MCMC: N interacting particles follow a kernelized gradient flow
# toward the posterior. It is NOT an MCMC sampler -- the returned particles are
# an optimized particle approximation, they carry no chain/Rhat/ESS semantics,
# and no draw ordering is meaningful (see `SVGDResult` and the caveats below).
#
# Infrastructure reuse: every iteration is ONE batched unconstrained log-joint
# GRADIENT call over the P x N particle matrix -- exactly the call
# `batched_hmc`/`batched_advi` already make, so the analytic host gradient AND
# the device (`backend=`) gradient path come for free. The only SVGD-specific
# work is the N x N RBF kernel matrix + its gradient (a host O(N^2 * P)
# reduction) and the particle-update optimizer.
#
# Per-iteration update (particles in UNCONSTRAINED space):
#   phi(x_i) = (1/N) sum_j [ k(x_j, x_i) grad log p(x_j)
#                            + grad_{x_j} k(x_j, x_i) ]
# with the RBF kernel k(x, x') = exp(-||x - x'||^2 / (2 s)); the median-distance
# bandwidth heuristic sets s = 0.5 * median(pairwise ||x_i - x_j||^2) / log(N+1)
# (Liu & Wang), so grad_{x_j} k(x_j, x_i) = (1/s) k(x_j, x_i) (x_i - x_j). The
# step is taken by an Adam ascent (reusing `_adam_ascent_step!` from ADVI).

"""
    SVGDResult

The result of [`batched_svgd`](@ref): `num_particles` EQUAL-WEIGHT particles
approximating the posterior, in both unconstrained and constrained space.

These are OPTIMIZED PARTICLES from a deterministic variational/particle method,
NOT MCMC draws: there is no chain structure, no Rhat/ESS, and the column order
is arbitrary. Validate SVGD by posterior-moment recovery and particle spread,
not by MCMC convergence diagnostics.

Fields:
- `unconstrained_particles::Matrix{Float64}` -- `parameter_total x num_particles`.
- `constrained_particles::Matrix{Float64}` -- `constrained_total x num_particles`.
- `kernel_scale_history::Vector{Float64}` -- the RBF bandwidth `s` per iteration.
- `direction_norm_history::Vector{Float64}` -- the mean SVGD update-direction
  norm per iteration (a convergence diagnostic: it decays as the particles
  settle).
- `num_particles`, `num_iterations`, `learning_rate`, `gradient_backend`.

See also [`particle_mean`](@ref), [`particle_covariance`](@ref).
"""
struct SVGDResult
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_particles::Matrix{Float64}
    constrained_particles::Matrix{Float64}
    kernel_scale_history::Vector{Float64}
    direction_norm_history::Vector{Float64}
    num_particles::Int
    num_iterations::Int
    learning_rate::Float64
    gradient_backend::Symbol
end

numsamples(result::SVGDResult) = result.num_particles

function Base.show(io::IO, result::SVGDResult)
    print(
        io,
        "SVGDResult(model=",
        result.model.name,
        ", particles=",
        result.num_particles,
        ", iterations=",
        result.num_iterations,
        ", backend=",
        result.gradient_backend,
        ")",
    )
    return nothing
end

function _svgd_particle_matrix(result::SVGDResult, space::Symbol)
    if space === :constrained
        return result.constrained_particles
    elseif space === :unconstrained
        return result.unconstrained_particles
    end
    throw(ArgumentError("SVGD particle space must be :constrained or :unconstrained"))
end

"""
    particle_mean(result::SVGDResult; space=:constrained) -> Vector{Float64}

The particle-average posterior mean (equal weights) in the requested space.
"""
function particle_mean(result::SVGDResult; space::Symbol=:constrained)
    particles = _svgd_particle_matrix(result, space)
    dimension, count = size(particles)
    mean_vector = zeros(Float64, dimension)
    for particle_index = 1:count
        for row = 1:dimension
            mean_vector[row] += particles[row, particle_index]
        end
    end
    count > 0 && (mean_vector ./= count)
    return mean_vector
end

"""
    particle_covariance(result::SVGDResult; space=:constrained) -> Matrix{Float64}

The (unbiased, equal-weight) particle covariance in the requested space. With a
single particle this is a zero matrix.
"""
function particle_covariance(result::SVGDResult; space::Symbol=:constrained)
    particles = _svgd_particle_matrix(result, space)
    dimension, count = size(particles)
    covariance = zeros(Float64, dimension, dimension)
    count > 1 || return covariance
    mean_vector = particle_mean(result; space=space)
    for particle_index = 1:count
        for column = 1:dimension
            delta_column = particles[column, particle_index] - mean_vector[column]
            for row = 1:dimension
                covariance[row, column] +=
                    (particles[row, particle_index] - mean_vector[row]) * delta_column
            end
        end
    end
    covariance ./= (count - 1)
    return covariance
end

# Median of the STRICT UPPER TRIANGLE of the pairwise squared-distance matrix
# (the N(N-1)/2 distinct particle pairs; the zero diagonal is excluded). No
# `Statistics` dependency -- a local sort + midpoint.
function _svgd_median_upper_triangle(sqdist::AbstractMatrix, num_particles::Int)
    pair_count = div(num_particles * (num_particles - 1), 2)
    pair_count > 0 || return 0.0
    values = Vector{Float64}(undef, pair_count)
    index = 0
    for i = 1:num_particles
        for j = (i+1):num_particles
            index += 1
            values[index] = sqdist[i, j]
        end
    end
    sort!(values)
    if isodd(pair_count)
        return values[div(pair_count + 1, 2)]
    end
    upper = div(pair_count, 2)
    return 0.5 * (values[upper] + values[upper+1])
end

# Fill the N x N pairwise squared-distance matrix and return the RBF bandwidth
# `s` (kernel k = exp(-d^2 / (2 s))). `bandwidth=nothing` uses the median
# heuristic s = 0.5 * median(d^2) / log(N+1); a supplied positive value fixes s.
function _svgd_pairwise_and_scale!(
    sqdist::AbstractMatrix,
    particles::AbstractMatrix,
    bandwidth,
)
    parameter_total, num_particles = size(particles)
    for i = 1:num_particles
        sqdist[i, i] = 0.0
        for j = (i+1):num_particles
            accumulator = 0.0
            for row = 1:parameter_total
                delta = particles[row, i] - particles[row, j]
                accumulator += delta * delta
            end
            sqdist[i, j] = accumulator
            sqdist[j, i] = accumulator
        end
    end

    if bandwidth !== nothing
        return Float64(bandwidth)
    end

    median_sqdist = _svgd_median_upper_triangle(sqdist, num_particles)
    scale = 0.5 * median_sqdist / log(num_particles + 1)
    # Collapsed particles (all identical, or a single particle) give a zero
    # median -> a degenerate kernel. Fall back to a unit bandwidth so the run
    # stays finite; the repulsive term then spreads the particles back out.
    return scale > 0.0 ? scale : 1.0
end

# Accumulate the SVGD update direction phi (parameter_total x num_particles)
# into `direction` from the current particles, their log-density gradients, the
# pairwise squared distances, and the RBF bandwidth `s`. O(N^2 * P) host
# reduction. Returns the mean column norm of phi (a convergence diagnostic).
function _svgd_accumulate_direction!(
    direction::AbstractMatrix,
    particles::AbstractMatrix,
    gradient::AbstractMatrix,
    sqdist::AbstractMatrix,
    scale::Float64,
)
    parameter_total, num_particles = size(particles)
    inverse_scale = 1.0 / scale
    two_scale = 2.0 * scale
    fill!(direction, 0.0)
    for i = 1:num_particles
        for j = 1:num_particles
            kernel = exp(-sqdist[i, j] / two_scale)
            kernel_over_scale = kernel * inverse_scale
            for row = 1:parameter_total
                # attractive (kernel-smoothed score) + repulsive (kernel grad)
                direction[row, i] +=
                    kernel * gradient[row, j] +
                    kernel_over_scale * (particles[row, i] - particles[row, j])
            end
        end
    end

    inverse_count = 1.0 / num_particles
    direction_norm_accumulator = 0.0
    for i = 1:num_particles
        column_norm = 0.0
        for row = 1:parameter_total
            value = direction[row, i] * inverse_count
            direction[row, i] = value
            column_norm += value * value
        end
        direction_norm_accumulator += sqrt(column_norm)
    end
    return direction_norm_accumulator * inverse_count
end

function _svgd_guard_finite(gradient::AbstractMatrix)
    all(isfinite, gradient) || throw(
        ArgumentError(
            "batched_svgd encountered a non-finite unconstrained log-density gradient; " *
            "SVGD assumes a differentiable target at every particle (try a different " *
            "init_strategy, fewer particles, or a smaller learning_rate)",
        ),
    )
    return nothing
end

"""
    batched_svgd(model, args=(), constraints=choicemap(); num_particles, num_iterations, kwargs...) -> SVGDResult

Stein Variational Gradient Descent (Liu & Wang, 2016): evolve `num_particles`
interacting particles by a kernelized gradient flow toward the posterior and
return them as an [`SVGDResult`](@ref) -- an OPTIMIZED PARTICLE APPROXIMATION,
not MCMC draws.

Every iteration is one batched unconstrained log-joint gradient over the P x N
particle matrix (the analytic host gradient, or -- with `backend` set to a
`KernelAbstractions.Backend` -- the device gradient), plus an N x N RBF kernel
interaction on the host. The RBF bandwidth uses the median-distance heuristic by
default; pass a positive `bandwidth` to fix the kernel scale `s`. The particle
update is an Adam ascent.

Keyword arguments:
- `num_particles::Int=64`, `num_iterations::Int` (required).
- `learning_rate::Real=0.1` -- the Adam master step.
- `initial_params`, `init_strategy::Symbol=:prior` -- particle initialization, shared with
  the batched samplers (`_initial_batched_hmc_positions`).
- `bandwidth=nothing` -- `nothing` uses the median heuristic; a positive real
  fixes the RBF kernel scale `s`.
- `beta1`, `beta2`, `adam_epsilon` -- Adam hyperparameters.
- `backend`, `precision` -- run the per-iteration gradient device-resident.
- `callback`, `callback_every`, `rng`.

Validate SVGD by posterior-moment recovery and particle spread, NOT by Rhat/ESS.
Known caveats (non-goals): mode collapse / variance underestimation at small `N`
and in high dimensions, and bandwidth sensitivity. SVGD is a fast approximate-
posterior tool, not a NUTS replacement.

See also [`particle_mean`](@ref), [`particle_covariance`](@ref), and `batched_advi`.
"""
function batched_svgd(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_particles::Int=64,
    num_iterations::Int,
    learning_rate::Real=0.1,
    initial_params=nothing,
    init_strategy::Symbol=:prior,
    bandwidth=nothing,
    beta1::Real=0.9,
    beta2::Real=0.999,
    adam_epsilon::Real=1e-8,
    callback=nothing,
    callback_every::Int=10,
    backend=nothing,
    precision=nothing,
    adtype::Symbol=:auto,
    rng::AbstractRNG=Random.default_rng(),
)
    adtype in (:auto, :forward, :reverse) ||
        throw(ArgumentError("batched_svgd adtype must be :auto, :forward, or :reverse, got $(repr(adtype))"))
    !(adtype === :reverse && backend !== nothing) ||
        throw(ArgumentError("batched_svgd adtype=:reverse is a host-only path and cannot be combined with a device `backend`"))
    layout = _conditioned_parameter_layout(model, constraints, args)
    parameter_total = parametercount(layout)
    constrained_total = parametervaluecount(layout)
    parameter_total > 0 ||
        throw(ArgumentError("batched_svgd requires at least one parameterized latent choice"))
    num_particles > 0 || throw(ArgumentError("batched_svgd requires num_particles > 0"))
    num_iterations > 0 || throw(ArgumentError("batched_svgd requires num_iterations > 0"))
    learning_rate > 0 || throw(ArgumentError("batched_svgd requires learning_rate > 0"))
    0 <= beta1 < 1 || throw(ArgumentError("batched_svgd requires 0 <= beta1 < 1"))
    0 <= beta2 < 1 || throw(ArgumentError("batched_svgd requires 0 <= beta2 < 1"))
    adam_epsilon > 0 || throw(ArgumentError("batched_svgd requires adam_epsilon > 0"))
    if bandwidth !== nothing
        (bandwidth isa Real && bandwidth > 0) ||
            throw(ArgumentError("batched_svgd `bandwidth` must be nothing or a positive real, got $bandwidth"))
    end
    if backend !== nothing
        backend isa KernelAbstractions.Backend ||
            throw(ArgumentError("batched_svgd `backend` must be a KernelAbstractions.Backend or nothing, got $(typeof(backend))"))
    end

    # SVGD targets a SINGLE posterior, so args/constraints are shared across all
    # particles (constraints must be a ChoiceMap -- `_conditioned_parameter_layout`
    # above already requires it). `_complete_model_args` fills any default args.
    args isa Tuple || throw(ArgumentError("batched_svgd args must be a Tuple shared across particles"))
    shared_args = _complete_model_args(model, args)
    particles = _initial_batched_hmc_positions(
        model,
        shared_args,
        constraints,
        initial_params,
        rng,
        parameter_total,
        constrained_total,
        num_particles;
        init_strategy=init_strategy,
    )

    # Per-iteration gradient provider: host analytic/ForwardDiff gradient cache,
    # or a device-resident gradient. Both write a parameter_total x num_particles
    # unconstrained log-density gradient into `gradient_buffer`.
    gradient_buffer = Matrix{Float64}(undef, parameter_total, num_particles)
    if backend === nothing
        cache = BatchedLogjointGradientCache(model, particles, shared_args, constraints; adtype=adtype)
        gradient_backend = _advi_gradient_backend(cache)
        gradient_provider! = positions -> begin
            batched_logjoint_gradient_unconstrained!(cache, positions)
            copyto!(gradient_buffer, cache.gradient_buffer)
            return gradient_buffer
        end
    else
        device_precision = precision === nothing ? default_device_precision(backend) : precision
        inner = DeviceBatchedWorkspace(
            model, num_particles;
            backend=backend, precision=device_precision, args=shared_args, constraints=constraints,
        )
        _device_ensure_gradient_buffers!(inner)
        upload = Matrix{device_precision}(undef, parameter_total, num_particles)
        download = Matrix{device_precision}(undef, parameter_total, num_particles)
        gradient_backend = :device
        gradient_provider! = positions -> begin
            upload .= positions
            copyto!(inner.params_device, upload)
            _device_launch_gradient!(inner)
            KernelAbstractions.synchronize(backend)
            copyto!(download, inner.gradients_device)
            gradient_buffer .= download
            return gradient_buffer
        end
    end

    direction = Matrix{Float64}(undef, parameter_total, num_particles)
    first_moment = zeros(Float64, parameter_total, num_particles)
    second_moment = zeros(Float64, parameter_total, num_particles)
    sqdist = Matrix{Float64}(undef, num_particles, num_particles)
    kernel_scale_history = Vector{Float64}(undef, num_iterations)
    direction_norm_history = Vector{Float64}(undef, num_iterations)

    learning_rate_f64 = Float64(learning_rate)
    beta1_f64 = Float64(beta1)
    beta2_f64 = Float64(beta2)
    adam_epsilon_f64 = Float64(adam_epsilon)

    for iteration = 1:num_iterations
        gradient = gradient_provider!(particles)
        _svgd_guard_finite(gradient)
        scale = _svgd_pairwise_and_scale!(sqdist, particles, bandwidth)
        kernel_scale_history[iteration] = scale
        direction_norm_history[iteration] =
            _svgd_accumulate_direction!(direction, particles, gradient, sqdist, scale)
        _adam_ascent_step!(
            particles,
            first_moment,
            second_moment,
            direction,
            iteration,
            learning_rate_f64,
            beta1_f64,
            beta2_f64,
            adam_epsilon_f64,
        )
        isnothing(callback) || _invoke_progress_callback(
            callback, callback_every, :iteration, iteration, num_iterations, NaN, 0,
        )
    end

    constrained_particles = Matrix{Float64}(undef, constrained_total, num_particles)
    _signature_batched_transform_to_constrained!(
        constrained_particles, model, particles, shared_args, constraints,
    )

    return SVGDResult(
        model,
        shared_args,
        constraints,
        particles,
        constrained_particles,
        kernel_scale_history,
        direction_norm_history,
        num_particles,
        num_iterations,
        learning_rate_f64,
        gradient_backend,
    )
end

# Device-resident batched ADVI inner loop.
#
# RNG stays HOST-side: each step the host draws standard-normal noise and forms the
# reparameterized particles (`location .+ scale .* noise`) exactly as the CPU path
# does, then uploads them. The fused device gradient kernel scores the particles and
# a small reduction kernel produces the per-parameter mean location gradient and the
# noise-weighted scale-gradient accumulator on-device, so only two length-P vectors
# (plus the length-`num_particles` values) come back each step. Adam + ELBO
# bookkeeping stay on the host, matching `batched_advi`.
#
# Like the device gradient kernel, this assumes every particle is finite (the
# conjugate targets it is used on are well-behaved); it does not mask per-particle
# gradient validity the way the CPU reduction does.

# Per-parameter reduction over particles. `ndrange = parameter_total`.
#   location_gradient[p] = mean_j grad[p, j]
#   scale_gradient[p]    = mean_j grad[p, j] * scale[p] * noise[p, j]
@kernel function _device_advi_reduce!(
    location_gradient,
    scale_gradient,
    @Const(grad),
    @Const(noise),
    @Const(scale),
    num_particles::Int,
)
    p = @index(Global)
    mean_gradient = zero(eltype(location_gradient))
    mean_scale_gradient = zero(eltype(scale_gradient))
    s = @inbounds scale[p]
    for j = 1:num_particles
        g = @inbounds grad[p, j]
        mean_gradient += g
        mean_scale_gradient += g * s * @inbounds(noise[p, j])
    end
    inv_count = one(mean_gradient) / num_particles
    @inbounds location_gradient[p] = mean_gradient * inv_count
    @inbounds scale_gradient[p] = mean_scale_gradient * inv_count
end

# Device-resident ADVI. Mirrors the host `batched_advi` loop, but the per-step
# gradient scoring and the mean gradient reductions run on the device.
#
# SYNC POINTS per step (P = parameter_total, M = num_particles):
#   uploads   : particles (P x M), noise (P x M), scale (P)
#   downloads : values (M), location_gradient (P), scale_gradient (P)
function _run_device_batched_advi(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap;
    num_iterations::Int,
    num_particles::Int,
    learning_rate::Real,
    initial_params,
    initial_log_scale,
    beta1::Real,
    beta2::Real,
    adam_epsilon::Real,
    gradient_clip::Real,
    callback,
    callback_every::Int,
    backend::KernelAbstractions.Backend,
    precision::Type,
    rng::AbstractRNG,
)
    T = precision
    # Signature-aware sizing (#95): match the CPU batched_advi path, which sizes
    # from the conditioning signature rather than the syntactic default layout.
    layout = _batched_signature_layout(model, constraints)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_advi requires at least one parameterized latent choice"))
    num_iterations > 0 || throw(ArgumentError("batched_advi requires num_iterations > 0"))
    num_particles > 0 || throw(ArgumentError("batched_advi requires num_particles > 0"))
    learning_rate > 0 || throw(ArgumentError("batched_advi requires learning_rate > 0"))
    0 <= beta1 < 1 || throw(ArgumentError("batched_advi requires 0 <= beta1 < 1"))
    0 <= beta2 < 1 || throw(ArgumentError("batched_advi requires 0 <= beta2 < 1"))
    adam_epsilon > 0 || throw(ArgumentError("batched_advi requires adam_epsilon > 0"))
    gradient_clip > 0 || throw(ArgumentError("batched_advi requires gradient_clip > 0"))

    location = _resolve_unconstrained_point(model, args, constraints, initial_params, rng, "initial_params")
    log_scale = _resolve_scale_vector("initial_log_scale", initial_log_scale, parameter_total)

    # Device workspace (raises the unsupported-model ArgumentError pointing at the
    # lowering report) + the extra device buffers this loop needs.
    inner = DeviceBatchedWorkspace(
        model, num_particles; backend=backend, precision=precision, args=args, constraints=constraints,
    )
    _device_ensure_gradient_buffers!(inner)
    noise_device = KernelAbstractions.allocate(backend, T, parameter_total, num_particles)
    scale_device = KernelAbstractions.allocate(backend, T, parameter_total)
    location_gradient_device = KernelAbstractions.allocate(backend, T, parameter_total)
    scale_gradient_device = KernelAbstractions.allocate(backend, T, parameter_total)

    particles = Matrix{Float64}(undef, parameter_total, num_particles)
    noise = similar(particles)
    values = Vector{Float64}(undef, num_particles)
    particles_upload = Matrix{T}(undef, parameter_total, num_particles)
    noise_upload = Matrix{T}(undef, parameter_total, num_particles)
    scale_upload = Vector{T}(undef, parameter_total)
    values_download = Vector{T}(undef, num_particles)
    location_gradient_download = Vector{T}(undef, parameter_total)
    scale_gradient_download = Vector{T}(undef, parameter_total)

    location_gradient = zeros(Float64, parameter_total)
    log_scale_gradient = zeros(Float64, parameter_total)
    location_m1 = zeros(Float64, parameter_total)
    location_m2 = zeros(Float64, parameter_total)
    log_scale_m1 = zeros(Float64, parameter_total)
    log_scale_m2 = zeros(Float64, parameter_total)
    elbo_history = Vector{Float64}(undef, num_iterations)
    gradient_norm_history = Vector{Float64}(undef, num_iterations)
    best_location = copy(location)
    best_log_scale = copy(log_scale)
    best_elbo = -Inf

    learning_rate_f64 = Float64(learning_rate)
    beta1_f64 = Float64(beta1)
    beta2_f64 = Float64(beta2)
    adam_epsilon_f64 = Float64(adam_epsilon)
    gradient_clip_f64 = Float64(gradient_clip)

    for iteration = 1:num_iterations
        _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
        particles_upload .= particles
        noise_upload .= noise
        for parameter_index = 1:parameter_total
            scale_upload[parameter_index] = convert(T, exp(log_scale[parameter_index]))
        end
        copyto!(inner.params_device, particles_upload)
        copyto!(noise_device, noise_upload)
        copyto!(scale_device, scale_upload)

        _device_launch_gradient!(inner)
        _device_advi_reduce!(backend)(
            location_gradient_device,
            scale_gradient_device,
            inner.gradients_device,
            noise_device,
            scale_device,
            num_particles;
            ndrange=parameter_total,
        )
        KernelAbstractions.synchronize(backend)

        copyto!(values_download, inner.totals_device)
        copyto!(location_gradient_download, location_gradient_device)
        copyto!(scale_gradient_download, scale_gradient_device)
        for particle_index = 1:num_particles
            values[particle_index] = Float64(values_download[particle_index])
        end
        # Mirror the CPU path (src/inference/vi.jl): guard BOTH the objective
        # values and the (downloaded, reduced) gradients. The on-device reduction
        # averages over every particle, so a single non-finite particle gradient
        # poisons the mean; feeding that to Adam would silently corrupt the state
        # (issue #70). This host-side check throws, matching the CPU contract.
        (
            all(isfinite, values) &&
            all(isfinite, location_gradient_download) &&
            all(isfinite, scale_gradient_download)
        ) || throw(
            ArgumentError("batched_advi encountered only non-finite unconstrained logjoint values or gradients"),
        )

        for parameter_index = 1:parameter_total
            location_gradient[parameter_index] = Float64(location_gradient_download[parameter_index])
            log_scale_gradient[parameter_index] = 1.0 + Float64(scale_gradient_download[parameter_index])
        end

        gradient_norm_history[iteration] = _clip_advi_gradients!(
            location_gradient, log_scale_gradient, gradient_clip_f64,
        )
        elbo_total = 0.0
        for particle_index = 1:num_particles
            elbo_total += values[particle_index]
        end
        elbo = elbo_total / num_particles + _gaussian_entropy(log_scale)
        elbo_history[iteration] = elbo
        if elbo > best_elbo
            best_elbo = elbo
            copyto!(best_location, location)
            copyto!(best_log_scale, log_scale)
        end

        _adam_ascent_step!(
            location, location_m1, location_m2, location_gradient,
            iteration, learning_rate_f64, beta1_f64, beta2_f64, adam_epsilon_f64,
        )
        _adam_ascent_step!(
            log_scale, log_scale_m1, log_scale_m2, log_scale_gradient,
            iteration, learning_rate_f64, beta1_f64, beta2_f64, adam_epsilon_f64,
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
        :device,
    )
end

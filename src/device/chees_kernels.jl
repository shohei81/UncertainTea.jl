# ChEES-HMC device-resident inner loop (issue #161, increment 4). See
# docs/chees-hmc.md for the authoritative spec of the whole #161 effort and
# src/inference/api_batched_chees.jl for the CPU reference this file mirrors.
#
# This is the device analogue of the shared-mode CPU `batched_chees`: it reuses the
# device HMC machinery (`DeviceHMCWorkspace`, the residency-looped
# `_device_leapfrog_integrate!`, the device Hamiltonian/accept-column kernels) and
# swaps in the two things that make it ChEES rather than fixed-length HMC:
#
#   1. A per-iteration Halton-JITTERED trajectory length. We track the trajectory
#      TIME `T` as a HOST scalar (`trajectory_state.trajectory_length`), initialized
#      to `num_leapfrog_steps * ε`. Each iteration we compute the integer step count
#      `L_iter = max(1, ceil(jitter_val * T / ε))` from the SAME CPU helpers the host
#      path uses (`_chees_jitter_value` / `_chees_leapfrog_steps_from_jitter`, which
#      consume no RNG) and pass `L_iter` to `_device_leapfrog_integrate!`. The device
#      trajectory is already residency-looped with a SINGLE `synchronize` at its end,
#      so the "one sync per iteration" property is preserved regardless of `L_iter`.
#
#   2. Cross-chain ChEES trajectory-length adaptation, WARMUP ONLY. The estimator
#      needs, from the P x C proposal position + endpoint momentum, only the
#      acceptance-weighted proposal MEAN (per-parameter reduction over chains) and
#      per-chain whitened endpoint moments -- both reduced ON-DEVICE (issue #220,
#      `_device_chees_proposal_mean!` / `_device_chees_endpoint_terms!`), downloading
#      only O(P)+O(C) instead of two P x C matrices. The `initials` mean + whitened
#      start term are done host-side from the `position` mirror (always resident, no
#      transfer). The reduced scalars feed the SAME scalar state machine the CPU path
#      runs (`_chees_apply_trajectory_gradient!`, reused verbatim, not forked). The
#      step size dual-averages to the HARMONIC mean of the chains' accept
#      probabilities (`_chees_harmonic_mean_accept`) and the SHARED diagonal mass
#      adapts exactly as device HMC's shared mode.
#
# SYNC / TRANSFER POINTS (per iteration, num_chains = C, num_params = P):
#   uploads : inverse_mass (P), momentum (P x C), accept_mask (C UInt8), and during
#             ChEES warmup the accept-prob + divergence masks (C each)
#   downloads (always): valid (C), current/proposed Hamiltonian (C each), accepted
#             position (P x C), current_logjoint (C)
#   downloads (warmup iters only, for host adaptation + step-size re-search):
#             current_gradient (P x C) AND, when adapting the trajectory length, the
#             two per-chain ChEES moment vectors (C each) -- NO P x C ChEES download
#   downloads (sampling iters): NONE beyond the always-list -- there is NO ChEES
#             adaptation download once warmup ends, so sampling stays a pure device
#             loop with one sync per iteration, exactly like device HMC.
#
# Device constraint (same as device NUTS/HMC): the leapfrog kernels consume a single
# SHARED diagonal inverse-mass vector, so ChEES here uses a shared step + shared
# (pooled) diagonal mass. This is the CPU ChEES adaptation mode anyway (ChEES is an
# ensemble adaptation), so nothing is lost relative to the host `batched_chees`.
#
# The RNG stays HOST-side (momenta + accept/reject uniforms drawn on the host in the
# same shape as CPU `batched_chees`), so results are STATISTICALLY -- not bitwise --
# equivalent to the CPU path, exactly like `batched_hmc(...; backend=...)`. On the
# CPU() reference backend at Float64 the device and host ChEES recover the same
# posterior and converge `T` to the same sensible value.

# issue #220: on-device ChEES cross-chain reduction kernels. The trajectory-length
# estimator needs, from the P x C proposal position + endpoint momentum, only the
# acceptance-weighted proposal MEAN (a per-parameter reduction over chains) and
# three per-chain whitened scalars -- both reducible on-device to O(P)+O(C), so the
# warmup iteration no longer downloads two P x C matrices. A chain is excluded iff
# it diverged; on the device path a non-finite proposal already forces a non-finite
# Hamiltonian (hence divergence), and the initials are the previous accepted
# position (always finite), so the divergence mask reproduces the CPU estimator's
# separate finiteness guards, giving a bitwise-identical reduction at Float64.

# Pass 1: acceptance-weighted, divergence-masked proposal mean, per parameter
# (`prop_mean[p] = sum_c !div ? accept[c]*proposal[p,c] : 0 * inv_weight_sum`).
@kernel function _device_chees_proposal_mean!(
    prop_mean, @Const(proposals), @Const(accept), @Const(divmask), inv_weight_sum, num_chains::Int,
)
    p = @index(Global)
    s = zero(eltype(prop_mean))
    for c = 1:num_chains
        if @inbounds(divmask[c]) == 0x00
            s += @inbounds(accept[c]) * @inbounds(proposals[p, c])
        end
    end
    @inbounds prop_mean[p] = s * inv_weight_sum
end

# Pass 2: per-chain whitened endpoint moments -- squared endpoint distance and the
# endpoint/velocity inner product (the estimator's start term is done host-side
# from the always-resident `position` mirror). Diverged chains contribute zero.
@kernel function _device_chees_endpoint_terms!(
    sq_end, endpoint_vel, @Const(proposals), @Const(momentum), @Const(prop_mean),
    @Const(inverse_mass), @Const(divmask), num_params::Int,
)
    c = @index(Global)
    z = zero(eltype(sq_end))
    if @inbounds(divmask[c]) != 0x00
        @inbounds sq_end[c] = z
        @inbounds endpoint_vel[c] = z
    else
        se = z
        ev = z
        for p = 1:num_params
            sigma = @inbounds inverse_mass[p]
            inv_sqrt = one(sigma) / sqrt(sigma)
            we = (@inbounds(proposals[p, c]) - @inbounds(prop_mean[p])) * inv_sqrt
            wv = @inbounds(momentum[p, c]) * sigma * inv_sqrt
            se += we * we
            ev += we * (wv)
        end
        @inbounds sq_end[c] = se
        @inbounds endpoint_vel[c] = ev
    end
end

function _run_device_batched_chees(
    model::TeaModel,
    args,
    constraints;
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int,
    step_size::Real,
    num_leapfrog_steps::Int,
    jitter_amount::Real,
    adapt_trajectory_length::Bool,
    max_leapfrog_steps::Int,
    trajectory_decay_rate::Real,
    trajectory_learning_rate::Real,
    trajectory_adam_beta1::Real,
    trajectory_adam_beta2::Real,
    trajectory_adam_epsilon::Real,
    initial_params,
    init_strategy::Symbol,
    init_max_retries::Int,
    target_accept::Real,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    find_reasonable_step_size::Bool,
    divergence_threshold::Real,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    callback,
    callback_every::Int,
    backend::KernelAbstractions.Backend,
    precision::Type,
    rng::AbstractRNG,
    _trajectory_trace::Union{Nothing,AbstractVector{Float64}}=nothing,
)
    T = precision
    # Signature-aware sizing (#95): match the CPU batched_chees path and the
    # signature-aware device workspace, not the syntactic default layout.
    signature_layout = _batched_signature_layout(model, constraints)
    num_params = parametercount(signature_layout)
    constrained_num_params = parametervaluecount(signature_layout)
    _validate_batched_hmc_arguments(
        num_chains,
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
        args,
        constraints,
    )

    batch_args = _validate_batched_args(model, args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    position = _initial_batched_hmc_positions(
        model,
        batch_args,
        batch_constraints,
        initial_params,
        rng,
        num_params,
        constrained_num_params,
        num_chains;
        init_strategy=init_strategy,
    )

    # Build the device workspace first: this is where an unsupported model raises the
    # ArgumentError pointing back at `device_lowering_report`.
    ws = DeviceHMCWorkspace(
        model, num_chains; backend=backend, precision=precision, args=args, constraints=constraints,
    )

    # Host workspace: used only for the initial finite check, the reasonable
    # step-size search, and the window-end step-size re-search (all Float64).
    inverse_mass_matrix = ones(num_params)
    host_workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = host_workspace.current_gradient
    # Retry non-finite starting points (issue #162), mirroring the host batched_chees:
    # only re-drawable inits (prior/uniform) retry; a fixed `initial_params` array
    # reproduces the same value, so it fails fast.
    local gradient
    init_attempt = 0
    while true
        _, gradient = _batched_logjoint_and_gradient_unconstrained!(
            current_logjoint, host_workspace.gradient_cache, position,
        )
        copyto!(current_gradient, gradient)
        bad_columns = _nonfinite_init_columns(current_logjoint, current_gradient)
        isempty(bad_columns) && break
        if !_init_is_redrawable(initial_params) || init_attempt >= init_max_retries
            retried = _init_is_redrawable(initial_params) ? " after $init_max_retries re-draw(s)" : ""
            throw(
                ArgumentError(
                    "initial batched ChEES parameters produced a non-finite unconstrained logjoint or gradient in $(length(bad_columns)) of $num_chains chain(s)$retried; try init_strategy=:uniform or supply finite initial_params",
                ),
            )
        end
        init_attempt += 1
        _redraw_batched_initial_positions!(
            position,
            bad_columns,
            model,
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

    # Seed the device buffers with the initial state.
    copyto!(ws.position, convert(Array{T}, position))
    copyto!(ws.current_gradient, convert(Array{T}, current_gradient))
    copyto!(ws.current_logjoint, convert(Array{T}, current_logjoint))

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, constrained_num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    integration_steps_values = Matrix{Int}(undef, num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    chees_step_size = Float64(step_size)
    chees_target_accept = Float64(target_accept)
    chees_divergence_threshold = Float64(divergence_threshold)
    chees_jitter_amount = Float64(jitter_amount)

    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        chees_step_size = _find_reasonable_batched_step_size(
            host_workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            chees_step_size,
            chees_divergence_threshold,
            rng,
        )
    end
    driver = WarmupDriver(
        num_params,
        num_warmup,
        chees_step_size,
        chees_target_accept;
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_regularization=mass_matrix_regularization,
        mass_matrix_min_samples=mass_matrix_min_samples,
    )
    # The step-size re-search reads these host mirrors, which we refresh (download)
    # each warmup iteration; passing the same array objects keeps the search current.
    refind = BatchedStepSizeSearch(
        host_workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        batch_args,
        batch_constraints,
        chees_divergence_threshold,
        rng,
    )

    # Trajectory-length adaptation state, identical to the CPU path: `T` warm-starts
    # at `num_leapfrog_steps * ε` so with `jitter_amount = 0` and adaptation off the
    # first iteration integrates exactly `ceil(T/ε) = num_leapfrog_steps` steps.
    initial_trajectory_length = num_leapfrog_steps * chees_step_size
    trajectory_state = _ChEESTrajectoryState(
        initial_trajectory_length,
        log(initial_trajectory_length),
        _ChEESAdamState(
            trajectory_learning_rate,
            trajectory_adam_beta1,
            trajectory_adam_beta2,
            trajectory_adam_epsilon,
        ),
        1,
        Float64(trajectory_decay_rate),
        max_leapfrog_steps,
    )

    # Host staging + download buffers (allocated once).
    sqrt_inverse_mass = Vector{Float64}(undef, num_params)
    host_momentum = Matrix{Float64}(undef, num_params, num_chains)
    momentum_upload = Matrix{T}(undef, num_params, num_chains)
    inverse_mass_upload = Vector{T}(undef, num_params)
    host_valid = Vector{UInt8}(undef, num_chains)
    host_current_ham = Vector{T}(undef, num_chains)
    host_proposed_ham = Vector{T}(undef, num_chains)
    host_accept_mask = Vector{UInt8}(undef, num_chains)
    position_download = Matrix{T}(undef, num_params, num_chains)
    gradient_download = Matrix{T}(undef, num_params, num_chains)
    logjoint_download = Vector{T}(undef, num_chains)

    # ChEES warmup-only adaptation buffers. Allocated only when we will actually run
    # the trajectory update (warmup present AND adaptation on); otherwise zero-sized,
    # so the sampling-only / adaptation-off paths never download the proposal state.
    do_trajectory_adaptation = num_warmup > 0 && adapt_trajectory_length
    # `chees_initials` (the pre-trajectory position) stays host-resident -- it is the
    # `position` mirror, so its mean + whitened start term are computed host-side with
    # no transfer. The proposal position + endpoint momentum stay DEVICE-resident and
    # are reduced on-device (issue #220), so only O(P)+O(C) crosses the bus.
    chees_initials =
        do_trajectory_adaptation ? Matrix{Float64}(undef, num_params, num_chains) : Matrix{Float64}(undef, 0, 0)
    zmat_t() = Matrix{T}(undef, 0, 0)
    alloc_dev(dims...) =
        do_trajectory_adaptation ? KernelAbstractions.allocate(backend, T, dims...) :
        KernelAbstractions.allocate(backend, T, ntuple(_ -> 0, length(dims))...)
    chees_accept_dev = alloc_dev(num_chains)                       # C   uploaded accept probs
    chees_divmask_dev =
        do_trajectory_adaptation ? KernelAbstractions.allocate(backend, UInt8, num_chains) :
        KernelAbstractions.allocate(backend, UInt8, 0)             # C   1 iff diverged
    chees_prop_mean_dev = alloc_dev(num_params)                    # P   pass-1 output
    chees_sq_end_dev = alloc_dev(num_chains)                       # C   pass-2 output
    chees_ev_dev = alloc_dev(num_chains)                           # C   pass-2 output
    chees_accept_upload = do_trajectory_adaptation ? Vector{T}(undef, num_chains) : Vector{T}(undef, 0)
    chees_divmask_upload =
        do_trajectory_adaptation ? Vector{UInt8}(undef, num_chains) : Vector{UInt8}(undef, 0)
    chees_sq_end_host = do_trajectory_adaptation ? Vector{T}(undef, num_chains) : Vector{T}(undef, 0)
    chees_ev_host = do_trajectory_adaptation ? Vector{T}(undef, num_chains) : Vector{T}(undef, 0)
    chees_initials_mean = do_trajectory_adaptation ? Vector{Float64}(undef, num_params) : Vector{Float64}(undef, 0)

    accept_prob = Vector{Float64}(undef, num_chains)
    accepted_step = falses(num_chains)
    divergent_step = falses(num_chains)
    energy_error_vec = Vector{Float64}(undef, num_chains)
    current_ham_f64 = Vector{Float64}(undef, num_chains)
    proposed_ham_f64 = Vector{Float64}(undef, num_chains)
    mass_adaptation_weights = Vector{Float64}(undef, num_chains)

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        chees_step_size = driver.step_size
        inverse_mass_matrix = driver.inverse_mass_matrix
        report_step_size = chees_step_size
        inverse_mass_upload .= inverse_mass_matrix
        copyto!(ws.inverse_mass, inverse_mass_upload)

        # Per-iteration Halton-jittered step count `L_iter = max(1, ceil(jitter_val *
        # T / ε))`. `jitter_value` is reused in the ChEES gradient below (BlackJAX
        # feeds the SAME jitter draw to both). The Halton draw consumes no RNG, so the
        # momentum/accept draws stay in the same order as the host path.
        jitter_value = _chees_jitter_value(chees_jitter_amount, iteration)
        leapfrog_steps =
            _chees_leapfrog_steps_from_jitter(jitter_value, trajectory_state.trajectory_length, chees_step_size)

        _update_sqrt_inverse_mass_matrix!(sqrt_inverse_mass, inverse_mass_matrix)
        _sample_batched_momentum!(host_momentum, rng, sqrt_inverse_mass)
        momentum_upload .= host_momentum
        copyto!(ws.momentum, momentum_upload)

        # Snapshot the ChEES `initials` (pre-transition current positions) BEFORE the
        # trajectory. The host `position` mirror still holds the accepted position from
        # the previous iteration's end-of-loop download, which is exactly this
        # iteration's starting state.
        if iteration <= num_warmup && adapt_trajectory_length
            copyto!(chees_initials, position)
        end

        _device_hmc_hamiltonian!(ws.backend)(
            ws.current_hamiltonian, ws.momentum, ws.inverse_mass, ws.current_logjoint, num_params; ndrange=num_chains,
        )
        # Residency-looped device trajectory: ONE synchronize at its end regardless of
        # `leapfrog_steps`. This is the "one sync per iteration" property.
        _device_leapfrog_integrate!(ws, chees_step_size, leapfrog_steps)
        _device_hmc_hamiltonian!(ws.backend)(
            ws.proposed_hamiltonian, ws.working_momentum, ws.inverse_mass, ws.inner.totals_device, num_params; ndrange=num_chains,
        )
        KernelAbstractions.synchronize(ws.backend)

        # issue #220: the proposal position + endpoint momentum stay device-resident;
        # the ChEES cross-chain reduction runs on-device below (after accept/divergence
        # are known), so this iteration no longer downloads two P x C matrices. The
        # device final half-kick negates the endpoint momentum, matching the CPU
        # `_batched_leapfrog!` `_negate_column!`, so the whitened velocity term agrees.

        copyto!(host_valid, ws.valid)
        copyto!(host_current_ham, ws.current_hamiltonian)
        copyto!(host_proposed_ham, ws.proposed_hamiltonian)

        fill!(accepted_step, false)
        fill!(divergent_step, true)
        for chain_index = 1:num_chains
            if host_valid[chain_index] != 0x00
                current_ham = Float64(host_current_ham[chain_index])
                proposed_ham = Float64(host_proposed_ham[chain_index])
                current_ham_f64[chain_index] = current_ham
                proposed_ham_f64[chain_index] = proposed_ham
                log_accept_ratio = current_ham - proposed_ham
                energy_error_vec[chain_index] = proposed_ham - current_ham
                divergent_step[chain_index] =
                    !isfinite(energy_error_vec[chain_index]) ||
                    energy_error_vec[chain_index] > chees_divergence_threshold
                accept_prob[chain_index] = _acceptance_probability(log_accept_ratio)
                if log(rand(rng)) < min(0.0, log_accept_ratio)
                    accepted_step[chain_index] = true
                end
            else
                current_ham_f64[chain_index] = Float64(host_current_ham[chain_index])
                proposed_ham_f64[chain_index] = Inf
                energy_error_vec[chain_index] = Inf
                accept_prob[chain_index] = 0.0
            end
            host_accept_mask[chain_index] = accepted_step[chain_index] ? 0x01 : 0x00
        end

        copyto!(ws.accept_mask, host_accept_mask)
        _device_hmc_accept_columns!(ws.backend)(
            ws.position,
            ws.current_gradient,
            ws.current_logjoint,
            ws.inner.params_device,
            ws.inner.gradients_device,
            ws.inner.totals_device,
            ws.accept_mask;
            ndrange=(num_params, num_chains),
        )
        KernelAbstractions.synchronize(ws.backend)

        cumulative_divergences += count(divergent_step)

        # issue #220: on-device ChEES cross-chain reduction. Accept/divergence are now
        # known and the proposal position (`ws.inner.params_device`) + endpoint momentum
        # (`ws.working_momentum`) are still device-resident (untouched by the accept
        # column copy), so reduce them on-device BEFORE `warmup_update!` can re-search
        # and overwrite them. Only the two per-chain moment C-vectors come back; the
        # scalar state machine runs after `warmup_update!` with the final step size.
        chees_gradient_numerator = 0.0
        chees_acceptance_denominator = 0.0
        if iteration <= num_warmup && adapt_trajectory_length
            weight_sum = 0.0
            @inbounds for chain_index = 1:num_chains
                if divergent_step[chain_index]
                    chees_divmask_upload[chain_index] = 0x01
                    chees_accept_upload[chain_index] = zero(T)
                else
                    chees_divmask_upload[chain_index] = 0x00
                    chees_accept_upload[chain_index] = T(accept_prob[chain_index])
                    weight_sum += accept_prob[chain_index]
                end
            end
            copyto!(chees_accept_dev, chees_accept_upload)
            copyto!(chees_divmask_dev, chees_divmask_upload)
            inv_weight_sum = T(1.0 / (weight_sum + _CHEES_EPS))
            _device_chees_proposal_mean!(ws.backend)(
                chees_prop_mean_dev, ws.inner.params_device, chees_accept_dev, chees_divmask_dev,
                inv_weight_sum, num_chains; ndrange=num_params,
            )
            _device_chees_endpoint_terms!(ws.backend)(
                chees_sq_end_dev, chees_ev_dev, ws.inner.params_device, ws.working_momentum,
                chees_prop_mean_dev, ws.inverse_mass, chees_divmask_dev, num_params; ndrange=num_chains,
            )
            KernelAbstractions.synchronize(ws.backend)
            copyto!(chees_sq_end_host, chees_sq_end_dev)
            copyto!(chees_ev_host, chees_ev_dev)
            # Host: initials mean + whitened start term from the (always-resident,
            # always-finite) `position` mirror, then fold into the estimator's
            # numerator/denominator, matching the CPU `_chees_trajectory_update!` order.
            fill!(chees_initials_mean, 0.0)
            @inbounds for chain_index = 1:num_chains
                for parameter_index = 1:num_params
                    chees_initials_mean[parameter_index] += chees_initials[parameter_index, chain_index]
                end
            end
            inv_count = 1.0 / num_chains
            @inbounds for parameter_index = 1:num_params
                chees_initials_mean[parameter_index] *= inv_count
            end
            @inbounds for chain_index = 1:num_chains
                divergent_step[chain_index] && continue
                p_accept = accept_prob[chain_index]
                chees_acceptance_denominator += p_accept + _CHEES_EPS
                squared_start = 0.0
                for parameter_index = 1:num_params
                    sigma = inverse_mass_matrix[parameter_index]
                    inv_sqrt = 1.0 / sqrt(sigma)
                    whitened_start =
                        (chees_initials[parameter_index, chain_index] - chees_initials_mean[parameter_index]) * inv_sqrt
                    squared_start += whitened_start * whitened_start
                end
                chees_gradient_numerator +=
                    p_accept *
                    (Float64(chees_sq_end_host[chain_index]) - squared_start) *
                    Float64(chees_ev_host[chain_index])
            end
        end

        # Download the (accepted) position + logjoint for host bookkeeping. During
        # warmup also grab the gradient so the step-size re-search sees fresh state.
        copyto!(position_download, ws.position)
        copyto!(logjoint_download, ws.current_logjoint)
        for chain_index = 1:num_chains
            for parameter_index = 1:num_params
                position[parameter_index, chain_index] = Float64(position_download[parameter_index, chain_index])
            end
            current_logjoint[chain_index] = Float64(logjoint_download[chain_index])
        end
        if iteration <= num_warmup
            copyto!(gradient_download, ws.current_gradient)
            for chain_index = 1:num_chains
                for parameter_index = 1:num_params
                    current_gradient[parameter_index, chain_index] =
                        Float64(gradient_download[parameter_index, chain_index])
                end
            end
        end

        if iteration <= num_warmup
            _mass_adaptation_weights!(
                driver.variance_state,
                mass_adaptation_weights,
                accepted_step,
                accept_prob,
                divergent_step,
            )
            # Step-size dual averaging targets the HARMONIC mean of the chains'
            # acceptance probabilities (ChEES, not the arithmetic mean device HMC uses).
            accept_statistic = _chees_harmonic_mean_accept(accept_prob, divergent_step)
            warmup_update!(
                driver,
                iteration,
                accept_statistic,
                position,
                mass_adaptation_weights,
                refind,
            )
            # Cross-chain ChEES trajectory-length update, reusing the CPU helper
            # verbatim. Uses the (possibly just re-searched) `driver.step_size` for the
            # [ε, max·ε] clip, matching BlackJAX's use of `new_step_size` there.
            if adapt_trajectory_length
                # The cross-chain reduction (numerator/denominator) was done on-device
                # above; here run only the scalar ChEES state machine with the
                # (possibly just re-searched) `driver.step_size` (issue #220).
                _chees_apply_trajectory_gradient!(
                    trajectory_state,
                    chees_gradient_numerator,
                    chees_acceptance_denominator,
                    jitter_value,
                    driver.step_size,
                )
            end
            _trajectory_trace === nothing || push!(_trajectory_trace, trajectory_state.trajectory_length)
            if iteration == num_warmup
                warmup_finalize!(driver)
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, report_step_size, cumulative_divergences)
        end

        if iteration > num_warmup
            sample_index += 1
            for chain_index = 1:num_chains
                copyto!(view(unconstrained_samples, :, sample_index, chain_index), view(position, :, chain_index))
                _write_signature_constrained_sample!(
                    constrained_samples,
                    model,
                    view(position, :, chain_index),
                    sample_index,
                    _batched_args(batch_args, chain_index),
                    _batched_constraints(batch_constraints, chain_index),
                    chain_index,
                )
                logjoint_values[sample_index, chain_index] = current_logjoint[chain_index]
                acceptance_stats[sample_index, chain_index] = accept_prob[chain_index]
                energies[sample_index, chain_index] =
                    accepted_step[chain_index] ? proposed_ham_f64[chain_index] : current_ham_f64[chain_index]
                energy_errors[sample_index, chain_index] = energy_error_vec[chain_index]
                accepted[sample_index, chain_index] = accepted_step[chain_index]
                divergent[sample_index, chain_index] = divergent_step[chain_index]
                integration_steps_values[sample_index, chain_index] = leapfrog_steps
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, report_step_size, cumulative_divergences)
        end
    end

    mass_matrix = copy(driver.inverse_mass_matrix)
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
        chains[chain_index] = HMCChain(
            :chees,
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            unconstrained_samples[:, :, chain_index],
            constrained_samples[:, :, chain_index],
            vec(logjoint_values[:, chain_index]),
            vec(acceptance_stats[:, chain_index]),
            vec(energies[:, chain_index]),
            vec(energy_errors[:, chain_index]),
            vec(accepted[:, chain_index]),
            vec(divergent[:, chain_index]),
            driver.step_size,
            copy(mass_matrix),
            num_leapfrog_steps,
            0,
            zeros(Int, num_samples),
            vec(integration_steps_values[:, chain_index]),
            chees_target_accept,
            copy(driver.mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

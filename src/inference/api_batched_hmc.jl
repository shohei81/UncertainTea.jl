
function batched_hmc(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    per_chain_adaptation::Union{Nothing,Bool}=nothing,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    callback=nothing,
    callback_every::Int=10,
    backend=nothing,
    precision=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    # Per-chain adaptation is the default everywhere (issue #137). The HOST path uses
    # per-chain step + per-chain mass; the DEVICE path uses per-chain step + a SHARED
    # (pooled) diagonal mass, since the device leapfrog kernels consume a single
    # shared inverse-mass vector. An explicit user value is always respected.
    per_chain_adaptation = something(per_chain_adaptation, true)
    if backend !== nothing
        # Device-resident inner loop. RNG stays host-side, so results are statistically
        # equivalent to (not bitwise identical to) the CPU path. The CPU path below is
        # untouched when `backend === nothing`.
        backend isa KernelAbstractions.Backend ||
            throw(ArgumentError("batched_hmc `backend` must be a KernelAbstractions.Backend or nothing, got $(typeof(backend))"))
        device_precision = precision === nothing ? default_device_precision(backend) : precision
        return _run_device_batched_hmc(
            model, args, constraints;
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            initial_params=initial_params,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            per_chain_adaptation=per_chain_adaptation,
            find_reasonable_step_size=find_reasonable_step_size,
            divergence_threshold=divergence_threshold,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            callback=callback,
            callback_every=callback_every,
            backend=backend,
            precision=device_precision,
            rng=rng,
        )
    end

    # Signature-aware sizing (#95): the latent set (and thus per-chain position
    # rows and constrained result width) follows the conditioning signature, not
    # the syntactic default layout. The batched workspaces are already
    # signature-aware (PR-4), so this keeps the API buffers consistent with them.
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
        num_chains,
    )
    inverse_mass_matrix = ones(num_params)
    workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = workspace.current_gradient
    _, gradient = _batched_logjoint_and_gradient_unconstrained!(
        current_logjoint,
        workspace.gradient_cache,
        position,
    )
    copyto!(current_gradient, gradient)
    all(isfinite, current_logjoint) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained logjoint"))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial batched HMC parameters produced a non-finite unconstrained gradient"))

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, constrained_num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_target_accept = Float64(target_accept)
    hmc_divergence_threshold = Float64(divergence_threshold)

    if per_chain_adaptation
        return _batched_hmc_per_chain!(
            workspace,
            model,
            args,
            constraints,
            batch_args,
            batch_constraints,
            position,
            current_logjoint,
            current_gradient,
            unconstrained_samples,
            constrained_samples,
            logjoint_values,
            acceptance_stats,
            energies,
            energy_errors,
            accepted,
            divergent,
            num_params,
            num_chains,
            num_samples,
            num_warmup,
            total_iterations,
            hmc_step_size,
            hmc_target_accept,
            hmc_divergence_threshold,
            num_leapfrog_steps,
            adapt_step_size,
            adapt_mass_matrix,
            find_reasonable_step_size,
            mass_matrix_regularization,
            mass_matrix_min_samples,
            callback,
            callback_every,
            rng,
        )
    end

    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        hmc_step_size = _find_reasonable_batched_step_size(
            workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            hmc_step_size,
            hmc_divergence_threshold,
            rng,
        )
    end
    driver = WarmupDriver(
        num_params,
        num_warmup,
        hmc_step_size,
        hmc_target_accept;
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_regularization=mass_matrix_regularization,
        mass_matrix_min_samples=mass_matrix_min_samples,
    )
    refind = BatchedStepSizeSearch(
        workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        batch_args,
        batch_constraints,
        hmc_divergence_threshold,
        rng,
    )

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        hmc_step_size = driver.step_size
        inverse_mass_matrix = driver.inverse_mass_matrix
        _update_sqrt_inverse_mass_matrix!(workspace.sqrt_inverse_mass_matrix, inverse_mass_matrix)
        _sample_batched_momentum!(workspace.momentum, rng, workspace.sqrt_inverse_mass_matrix)
        proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            hmc_step_size,
            num_leapfrog_steps,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_matrix,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        energy_error = workspace.energy_error
        fill!(energy_error, Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)

        for chain_index = 1:num_chains
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error[chain_index] = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error[chain_index]) ||
                    energy_error[chain_index] > hmc_divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)
        accepted_step = workspace.accepted_step
        fill!(accepted_step, false)
        for chain_index = 1:num_chains
            if valid[chain_index] && log(rand(rng)) < min(0.0, log_accept_ratio[chain_index])
                copyto!(view(position, :, chain_index), view(proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = proposed_logjoint[chain_index]
                accepted_step[chain_index] = true
            end
        end

        cumulative_divergences += count(divergent_step)

        if iteration <= num_warmup
            _mass_adaptation_weights!(
                driver.variance_state,
                workspace.mass_adaptation_weights,
                accepted_step,
                accept_prob,
                divergent_step,
            )
            accept_statistic = _mean_batched_adaptation_probability(accept_prob, divergent_step)
            warmup_update!(
                driver,
                iteration,
                accept_statistic,
                position,
                workspace.mass_adaptation_weights,
                refind,
            )
            if iteration == num_warmup
                warmup_finalize!(driver)
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, hmc_step_size, cumulative_divergences)
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
                    accepted_step[chain_index] ? proposed_hamiltonian[chain_index] : current_hamiltonian[chain_index]
                energy_errors[sample_index, chain_index] = energy_error[chain_index]
                accepted[sample_index, chain_index] = accepted_step[chain_index]
                divergent[sample_index, chain_index] = divergent_step[chain_index]
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, hmc_step_size, cumulative_divergences)
        end
    end

    mass_matrix = copy(driver.inverse_mass_matrix)
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
        chains[chain_index] = HMCChain(
            :hmc,
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
            fill(num_leapfrog_steps, num_samples),
            hmc_target_accept,
            copy(driver.mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

# Per-chain warmup adaptation for batched HMC. Mirrors the shared-mode loop but
# each chain owns a WarmupDriver, an inverse-mass column, and its own step size.
# Callback contract: the reported `step_size` is the mean of the per-chain step
# sizes at the start of the iteration.
function _batched_hmc_per_chain!(
    workspace::BatchedHMCWorkspace,
    model::TeaModel,
    args,
    constraints,
    batch_args,
    batch_constraints,
    position::Matrix{Float64},
    current_logjoint::Vector{Float64},
    current_gradient::Matrix{Float64},
    unconstrained_samples::Array{Float64,3},
    constrained_samples::Array{Float64,3},
    logjoint_values::Matrix{Float64},
    acceptance_stats::Matrix{Float64},
    energies::Matrix{Float64},
    energy_errors::Matrix{Float64},
    accepted::BitMatrix,
    divergent::BitMatrix,
    num_params::Int,
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int,
    total_iterations::Int,
    hmc_step_size::Float64,
    hmc_target_accept::Float64,
    hmc_divergence_threshold::Float64,
    num_leapfrog_steps::Int,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    find_reasonable_step_size::Bool,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    callback,
    callback_every::Int,
    rng::AbstractRNG,
)
    inverse_mass_matrices = ones(num_params, num_chains)
    sqrt_inverse_mass_matrices = Matrix{Float64}(undef, num_params, num_chains)
    step_sizes = fill(hmc_step_size, num_chains)
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        step_sizes = _find_reasonable_batched_step_size_per_chain(
            workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrices,
            batch_args,
            batch_constraints,
            hmc_step_size,
            rng,
        )
    end

    drivers = [
        WarmupDriver(
            num_params,
            num_warmup,
            step_sizes[chain_index],
            hmc_target_accept;
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
        ) for chain_index = 1:num_chains
    ]
    # Per-chain re-search uses a single-chain reasonable step-size search on each
    # chain's own column gradient cache (chain-major RNG at warmup window ends).
    refinds = [
        ScalarStepSizeSearch(
            model,
            _logjoint_gradient_cache(
                model,
                collect(view(position, :, chain_index)),
                _batched_args(batch_args, chain_index),
                _batched_constraints(batch_constraints, chain_index);
                reject_invalid_parameters=true,
            ),
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            rng,
            collect(view(position, :, chain_index)),
            current_logjoint[chain_index],
        ) for chain_index = 1:num_chains
    ]

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        for chain_index = 1:num_chains
            step_sizes[chain_index] = drivers[chain_index].step_size
            @inbounds copyto!(
                view(inverse_mass_matrices, :, chain_index),
                drivers[chain_index].inverse_mass_matrix,
            )
        end
        mean_step_size = sum(step_sizes) / num_chains

        sqrt_inverse_mass_matrices .= sqrt.(inverse_mass_matrices)
        _sample_batched_momentum!(workspace.momentum, rng, sqrt_inverse_mass_matrices)
        proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrices,
            batch_args,
            batch_constraints,
            step_sizes,
            num_leapfrog_steps,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_matrices,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        energy_error = workspace.energy_error
        fill!(energy_error, Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)

        for chain_index = 1:num_chains
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    view(inverse_mass_matrices, :, chain_index),
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error[chain_index] = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error[chain_index]) ||
                    energy_error[chain_index] > hmc_divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)
        accepted_step = workspace.accepted_step
        fill!(accepted_step, false)
        for chain_index = 1:num_chains
            if valid[chain_index] && log(rand(rng)) < min(0.0, log_accept_ratio[chain_index])
                copyto!(view(position, :, chain_index), view(proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = proposed_logjoint[chain_index]
                accepted_step[chain_index] = true
            end
        end

        cumulative_divergences += count(divergent_step)

        if iteration <= num_warmup
            for chain_index = 1:num_chains
                mass_weight = _mass_adaptation_weight(
                    drivers[chain_index].variance_state,
                    accepted_step[chain_index],
                    accept_prob[chain_index],
                    divergent_step[chain_index],
                )
                accept_statistic = divergent_step[chain_index] ? 0.0 : accept_prob[chain_index]
                refinds[chain_index].position = collect(view(position, :, chain_index))
                refinds[chain_index].current_logjoint = current_logjoint[chain_index]
                warmup_update!(
                    drivers[chain_index],
                    iteration,
                    accept_statistic,
                    view(position, :, chain_index),
                    mass_weight,
                    refinds[chain_index],
                )
                if iteration == num_warmup
                    warmup_finalize!(drivers[chain_index])
                end
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, mean_step_size, cumulative_divergences)
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
                    accepted_step[chain_index] ? proposed_hamiltonian[chain_index] : current_hamiltonian[chain_index]
                energy_errors[sample_index, chain_index] = energy_error[chain_index]
                accepted[sample_index, chain_index] = accepted_step[chain_index]
                divergent[sample_index, chain_index] = divergent_step[chain_index]
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, mean_step_size, cumulative_divergences)
        end
    end

    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
        mass_matrix = copy(drivers[chain_index].inverse_mass_matrix)
        chains[chain_index] = HMCChain(
            :hmc,
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
            drivers[chain_index].step_size,
            mass_matrix,
            num_leapfrog_steps,
            0,
            zeros(Int, num_samples),
            fill(num_leapfrog_steps, num_samples),
            hmc_target_accept,
            copy(drivers[chain_index].mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

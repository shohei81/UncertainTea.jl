function batched_nuts(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    per_chain_adaptation::Union{Nothing,Bool}=nothing,
    find_reasonable_step_size::Bool=false,
    max_delta_energy::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    callback=nothing,
    callback_every::Int=10,
    tree_strategy::Symbol=:hybrid,
    backend=nothing,
    precision=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    tree_strategy in (:hybrid, :masked) ||
        throw(ArgumentError("batched_nuts tree_strategy must be :hybrid or :masked, got $(repr(tree_strategy))"))

    # Per-chain step-size adaptation is the DEFAULT everywhere (issue #137): with
    # shared adaptation, prior-draw initialization strands the chains whose initial
    # curvature the shared step size never fits (~5% divergence on the gauss
    # benchmark at every chain count), while per-chain warmup drivers recover all of
    # them (0% in the issue's experiment). The HOST path uses per-chain step + per-
    # chain mass; the DEVICE path uses per-chain step + a SHARED (pooled) diagonal
    # mass -- the device leapfrog kernels consume a single shared inverse_mass
    # vector, so a per-chain mass is not device-representable, but per-chain step is
    # the operative fix. An unset value resolves to per-chain on both; an explicit
    # user value is always respected.
    per_chain_adaptation = something(per_chain_adaptation, true)

    # Device-resident masked NUTS. When `backend` is given the masked doubling
    # trajectory runs device-resident (host-side RNG + O(num_chains) bookkeeping,
    # device-side leapfrog/gradient/tree ops); results are statistically -- and, on
    # the CPU() reference backend at Float64, numerically -- equivalent to the host
    # masked path. Only the masked strategy and shared (diagonal) adaptation apply.
    # Validate the cheap backend kwargs up front (fail fast), but defer the
    # expensive `DeviceNUTSWorkspace` construction (model lowering + large device
    # buffers) until after `_validate_batched_nuts_arguments` so a bad argument does
    # not lower/allocate before throwing.
    device_precision = nothing
    if backend !== nothing
        backend isa KernelAbstractions.Backend ||
            throw(ArgumentError("batched_nuts `backend` must be a KernelAbstractions.Backend or nothing, got $(typeof(backend))"))
        tree_strategy === :masked ||
            throw(ArgumentError("batched_nuts device backend requires tree_strategy=:masked, got $(repr(tree_strategy))"))
        # Per-chain adaptation IS supported on the device (issue #137): it routes to
        # the pooled-mass / per-chain-step driver (shared diagonal mass, per-chain
        # step). per_chain_adaptation=false still selects shared adaptation.
        device_precision = precision === nothing ? default_device_precision(backend) : precision
    end
    # Signature-aware sizing (#95): per-chain position rows and constrained
    # result width follow the conditioning signature, not the syntactic default
    # layout, matching the signature-aware batched workspaces (PR-4).
    signature_layout = _batched_signature_layout(model, constraints)
    num_params = parametercount(signature_layout)
    constrained_num_params = parametervaluecount(signature_layout)
    _validate_batched_nuts_arguments(
        num_chains,
        num_params,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        max_delta_energy,
        mass_matrix_regularization,
        mass_matrix_min_samples,
        args,
        constraints,
    )
    device_nuts_workspace =
        backend === nothing ? nothing :
        DeviceNUTSWorkspace(
            model, num_chains, max_tree_depth;
            backend=backend, precision=device_precision, args=args, constraints=constraints,
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
    workspace = BatchedNUTSWorkspace(model, position, batch_args, batch_constraints, max_tree_depth)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = workspace.current_gradient
    _, gradient = _batched_logjoint_and_gradient_unconstrained!(
        current_logjoint,
        workspace.gradient_cache,
        position,
    )
    copyto!(current_gradient, gradient)
    all(isfinite, current_logjoint) ||
        throw(ArgumentError("initial batched NUTS parameters produced a non-finite unconstrained logjoint"))
    all(isfinite, current_gradient) ||
        throw(ArgumentError("initial batched NUTS parameters produced a non-finite unconstrained gradient"))

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, constrained_num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    tree_depths = Matrix{Int}(undef, num_samples, num_chains)
    integration_steps_values = Matrix{Int}(undef, num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    nuts_step_size = Float64(step_size)
    nuts_target_accept = Float64(target_accept)
    nuts_max_delta_energy = Float64(max_delta_energy)

    if per_chain_adaptation && device_nuts_workspace !== nothing
        return _batched_nuts_device_per_chain!(
            device_nuts_workspace,
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
            tree_depths,
            integration_steps_values,
            num_params,
            num_chains,
            num_samples,
            num_warmup,
            total_iterations,
            nuts_step_size,
            nuts_target_accept,
            nuts_max_delta_energy,
            max_tree_depth,
            adapt_step_size,
            adapt_mass_matrix,
            find_reasonable_step_size,
            mass_matrix_regularization,
            mass_matrix_min_samples,
            callback,
            callback_every,
            rng,
        )
    elseif per_chain_adaptation
        return _batched_nuts_per_chain!(
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
            tree_depths,
            integration_steps_values,
            num_params,
            num_chains,
            num_samples,
            num_warmup,
            total_iterations,
            nuts_step_size,
            nuts_target_accept,
            nuts_max_delta_energy,
            max_tree_depth,
            adapt_step_size,
            adapt_mass_matrix,
            find_reasonable_step_size,
            mass_matrix_regularization,
            mass_matrix_min_samples,
            callback,
            callback_every,
            tree_strategy,
            rng,
        )
    end

    inverse_mass_matrix = ones(num_params)
    step_size_workspace = nothing
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        step_size_workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
        nuts_step_size = _find_reasonable_batched_step_size(
            step_size_workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            nuts_step_size,
            nuts_max_delta_energy,
            rng,
        )
    end
    driver = WarmupDriver(
        num_params,
        num_warmup,
        nuts_step_size,
        nuts_target_accept;
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_regularization=mass_matrix_regularization,
        mass_matrix_min_samples=mass_matrix_min_samples,
    )
    refind = BatchedStepSizeSearch(
        step_size_workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        batch_args,
        batch_constraints,
        nuts_max_delta_energy,
        rng,
    )

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        nuts_step_size = driver.step_size
        inverse_mass_matrix = driver.inverse_mass_matrix
        if tree_strategy === :masked && device_nuts_workspace !== nothing
            _device_batched_nuts_proposals_masked!(
                device_nuts_workspace,
                workspace,
                model,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrix,
                batch_args,
                batch_constraints,
                nuts_step_size,
                max_tree_depth,
                nuts_max_delta_energy,
                rng,
            )
        elseif tree_strategy === :masked
            _batched_nuts_proposals_masked!(
                workspace,
                model,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrix,
                batch_args,
                batch_constraints,
                nuts_step_size,
                max_tree_depth,
                nuts_max_delta_energy,
                rng,
            )
        else
            _batched_nuts_proposals!(
                workspace,
                model,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrix,
                batch_args,
                batch_constraints,
                nuts_step_size,
                max_tree_depth,
                nuts_max_delta_energy,
                rng,
            )
        end

        for chain_index = 1:num_chains
            if workspace.control.accepted_step[chain_index]
                copyto!(view(position, :, chain_index), view(workspace.proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(workspace.proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = workspace.proposed_logjoint[chain_index]
            end
        end

        cumulative_divergences += count(workspace.control.divergent_step)

        if iteration <= num_warmup
            @inbounds for chain_index = 1:num_chains
                workspace.mass_adaptation_weights[chain_index] = _mass_adaptation_weight(
                    driver.variance_state,
                    false,
                    workspace.accept_prob[chain_index],
                    workspace.control.divergent_step[chain_index],
                )
            end
            accept_statistic = _mean_batched_adaptation_probability(
                workspace.accept_prob,
                workspace.control.divergent_step,
            )
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
                callback, callback_every, :warmup, iteration, num_warmup, nuts_step_size, cumulative_divergences)
        else
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
                acceptance_stats[sample_index, chain_index] = workspace.accept_prob[chain_index]
                energies[sample_index, chain_index] = workspace.proposed_energy[chain_index]
                energy_errors[sample_index, chain_index] = workspace.energy_error[chain_index]
                accepted[sample_index, chain_index] = workspace.control.accepted_step[chain_index]
                divergent[sample_index, chain_index] = workspace.control.divergent_step[chain_index]
                tree_depths[sample_index, chain_index] = workspace.control.tree_depths[chain_index]
                integration_steps_values[sample_index, chain_index] = workspace.control.integration_steps[chain_index]
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, nuts_step_size, cumulative_divergences)
        end
    end

    mass_matrix = copy(driver.inverse_mass_matrix)
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
        chains[chain_index] = HMCChain(
            :nuts,
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
            0,
            max_tree_depth,
            vec(tree_depths[:, chain_index]),
            vec(integration_steps_values[:, chain_index]),
            nuts_target_accept,
            copy(driver.mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

# Per-chain warmup adaptation for batched NUTS. Each chain owns its own
# WarmupDriver (independent step size + diagonal inverse mass matrix). The batched
# tree machinery is threaded a per-chain step-size vector and an inverse-mass
# matrix (one column per chain); the shared-mode path is untouched. Dual averaging
# for each chain is driven by that chain's own acceptance probability, masked to
# zero when the chain diverged (the per-chain analog of the shared masked mean).
# Callback contract: the reported `step_size` is the mean of the per-chain step
# sizes at the start of the iteration.
function _batched_nuts_per_chain!(
    workspace::BatchedNUTSWorkspace,
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
    tree_depths::Matrix{Int},
    integration_steps_values::Matrix{Int},
    num_params::Int,
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int,
    total_iterations::Int,
    nuts_step_size::Float64,
    nuts_target_accept::Float64,
    nuts_max_delta_energy::Float64,
    max_tree_depth::Int,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    find_reasonable_step_size::Bool,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    callback,
    callback_every::Int,
    tree_strategy::Symbol,
    rng::AbstractRNG,
)
    inverse_mass_matrices = ones(num_params, num_chains)
    step_sizes = fill(nuts_step_size, num_chains)
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        step_size_workspace = BatchedHMCWorkspace(
            model, position, batch_args, batch_constraints, ones(num_params),
        )
        step_sizes = _find_reasonable_batched_step_size_per_chain(
            step_size_workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrices,
            batch_args,
            batch_constraints,
            nuts_step_size,
            rng,
        )
    end

    drivers = [
        WarmupDriver(
            num_params,
            num_warmup,
            step_sizes[chain_index],
            nuts_target_accept;
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
        ) for chain_index = 1:num_chains
    ]
    # Per-chain re-search: each chain re-runs the single-chain reasonable step-size
    # search on its own column at its own warmup window ends (chain-major RNG).
    refinds = [
        ScalarStepSizeSearch(
            model,
            workspace.column_gradient_caches[chain_index],
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

        if tree_strategy === :masked
            _batched_nuts_proposals_masked!(
                workspace,
                model,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrices,
                batch_args,
                batch_constraints,
                step_sizes,
                max_tree_depth,
                nuts_max_delta_energy,
                rng,
            )
        else
            _batched_nuts_proposals!(
                workspace,
                model,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrices,
                batch_args,
                batch_constraints,
                step_sizes,
                max_tree_depth,
                nuts_max_delta_energy,
                rng,
            )
        end

        for chain_index = 1:num_chains
            if workspace.control.accepted_step[chain_index]
                copyto!(view(position, :, chain_index), view(workspace.proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(workspace.proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = workspace.proposed_logjoint[chain_index]
            end
        end

        cumulative_divergences += count(workspace.control.divergent_step)

        if iteration <= num_warmup
            for chain_index = 1:num_chains
                mass_weight = _mass_adaptation_weight(
                    drivers[chain_index].variance_state,
                    false,
                    workspace.accept_prob[chain_index],
                    workspace.control.divergent_step[chain_index],
                )
                accept_statistic = workspace.control.divergent_step[chain_index] ? 0.0 :
                                   workspace.accept_prob[chain_index]
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
        else
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
                acceptance_stats[sample_index, chain_index] = workspace.accept_prob[chain_index]
                energies[sample_index, chain_index] = workspace.proposed_energy[chain_index]
                energy_errors[sample_index, chain_index] = workspace.energy_error[chain_index]
                accepted[sample_index, chain_index] = workspace.control.accepted_step[chain_index]
                divergent[sample_index, chain_index] = workspace.control.divergent_step[chain_index]
                tree_depths[sample_index, chain_index] = workspace.control.tree_depths[chain_index]
                integration_steps_values[sample_index, chain_index] = workspace.control.integration_steps[chain_index]
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, mean_step_size, cumulative_divergences)
        end
    end

    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
        mass_matrix = copy(drivers[chain_index].inverse_mass_matrix)
        chains[chain_index] = HMCChain(
            :nuts,
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
            0,
            max_tree_depth,
            vec(tree_depths[:, chain_index]),
            vec(integration_steps_values[:, chain_index]),
            nuts_target_accept,
            copy(drivers[chain_index].mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

# Pooled-mass / per-chain-step device masked NUTS (issue #137). The device leapfrog
# kernels consume a single SHARED diagonal inverse-mass vector, so the host per-
# chain-mass mode is not device-representable. This driver instead pools the mass
# across all chains (one running-variance state fed the full P x C positions) while
# giving each chain its own dual-averaged step size, uploaded to the device as a
# C-length `step` vector each iteration. This removes the prior-draw stranding the
# shared-step device path suffered (issue #137's gauss gate) without a per-chain
# mass. Only the doubling ROUND loop runs on the device; init/finalize reuse the
# host code (which already indexes a per-chain step vector).
function _batched_nuts_device_per_chain!(
    dws,
    workspace::BatchedNUTSWorkspace,
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
    tree_depths::Matrix{Int},
    integration_steps_values::Matrix{Int},
    num_params::Int,
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int,
    total_iterations::Int,
    nuts_step_size::Float64,
    nuts_target_accept::Float64,
    nuts_max_delta_energy::Float64,
    max_tree_depth::Int,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    find_reasonable_step_size::Bool,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    callback,
    callback_every::Int,
    rng::AbstractRNG,
)
    step_sizes = fill(nuts_step_size, num_chains)
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        step_size_workspace = BatchedHMCWorkspace(
            model, position, batch_args, batch_constraints, ones(num_params),
        )
        step_sizes = _find_reasonable_batched_step_size_per_chain(
            step_size_workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            ones(num_params, num_chains),
            batch_args,
            batch_constraints,
            nuts_step_size,
            rng,
        )
    end

    driver = PooledMassPerChainStepDriver(
        num_params,
        num_warmup,
        step_sizes,
        nuts_target_accept;
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_regularization=mass_matrix_regularization,
        mass_matrix_min_samples=mass_matrix_min_samples,
    )
    # Per-chain window-end re-search: each chain re-runs the single-chain reasonable
    # step-size search on its own column against the SHARED pooled mass (chain-major
    # RNG at warmup window ends), mirroring the host per-chain refind.
    refinds = [
        ScalarStepSizeSearch(
            model,
            workspace.column_gradient_caches[chain_index],
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            rng,
            collect(view(position, :, chain_index)),
            current_logjoint[chain_index],
        ) for chain_index = 1:num_chains
    ]
    accept_statistics = Vector{Float64}(undef, num_chains)

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        inverse_mass_matrix = driver.mass.inverse_mass_matrix
        mean_step_size = sum(driver.step_sizes) / num_chains

        _device_batched_nuts_proposals_masked!(
            dws,
            workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            driver.step_sizes,
            max_tree_depth,
            nuts_max_delta_energy,
            rng,
        )

        for chain_index = 1:num_chains
            if workspace.control.accepted_step[chain_index]
                copyto!(view(position, :, chain_index), view(workspace.proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(workspace.proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = workspace.proposed_logjoint[chain_index]
            end
        end

        cumulative_divergences += count(workspace.control.divergent_step)

        if iteration <= num_warmup
            @inbounds for chain_index = 1:num_chains
                workspace.mass_adaptation_weights[chain_index] = _mass_adaptation_weight(
                    driver.mass.variance_state,
                    false,
                    workspace.accept_prob[chain_index],
                    workspace.control.divergent_step[chain_index],
                )
                accept_statistics[chain_index] =
                    workspace.control.divergent_step[chain_index] ? 0.0 : workspace.accept_prob[chain_index]
                refinds[chain_index].position = collect(view(position, :, chain_index))
                refinds[chain_index].current_logjoint = current_logjoint[chain_index]
            end
            warmup_update!(
                driver,
                iteration,
                accept_statistics,
                position,
                workspace.mass_adaptation_weights,
                refinds,
            )
            if iteration == num_warmup
                warmup_finalize!(driver)
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, mean_step_size, cumulative_divergences)
        else
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
                acceptance_stats[sample_index, chain_index] = workspace.accept_prob[chain_index]
                energies[sample_index, chain_index] = workspace.proposed_energy[chain_index]
                energy_errors[sample_index, chain_index] = workspace.energy_error[chain_index]
                accepted[sample_index, chain_index] = workspace.control.accepted_step[chain_index]
                divergent[sample_index, chain_index] = workspace.control.divergent_step[chain_index]
                tree_depths[sample_index, chain_index] = workspace.control.tree_depths[chain_index]
                integration_steps_values[sample_index, chain_index] = workspace.control.integration_steps[chain_index]
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, mean_step_size, cumulative_divergences)
        end
    end

    mass_matrix = copy(driver.mass.inverse_mass_matrix)
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
        chains[chain_index] = HMCChain(
            :nuts,
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
            driver.step_sizes[chain_index],
            copy(mass_matrix),
            0,
            max_tree_depth,
            vec(tree_depths[:, chain_index]),
            vec(integration_steps_values[:, chain_index]),
            nuts_target_accept,
            copy(driver.mass.mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

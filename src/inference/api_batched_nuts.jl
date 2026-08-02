"""
    batched_nuts(model, args=(), constraints=choicemap(); num_chains, num_samples, kwargs...) -> HMCChains

Run many NUTS chains together over a dense parameter-by-chain layout — the
layout that lowers to the device backend. Returns an [`HMCChains`](@ref).

Key keyword arguments:

- `num_chains`, `num_samples` (required), `num_warmup`.
- `step_size`, `max_tree_depth`, `max_delta_energy`.
- `target_accept`, `adapt_step_size`, `adapt_mass_matrix`,
  `per_chain_adaptation` (per-chain step-size adaptation is the default).
- `init` (`:prior` or `:uniform`), `init_max_retries`, `initial_params`.
- `tree_strategy`: `:hybrid` (default), `:masked`, or `:persistent` (a
  device-only, one-launch-per-iteration tree kernel that requires `backend`).
- `backend`, `precision`, `persistent_gradient`: pass a
  `KernelAbstractions.Backend` to run the inner loop device-resident.
- `rng`.

See also [`nuts`](@ref), [`batched_hmc`](@ref), and [`batched_chees`](@ref).
"""
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
    init::Symbol=:prior,
    init_max_retries::Int=100,
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
    persistent_gradient::Symbol=:auto,
    device_sync_per_leaf::Bool=false,
    adtype::Symbol=:auto,
    rng::AbstractRNG=Random.default_rng(),
)
    tree_strategy in (:hybrid, :masked, :persistent) ||
        throw(ArgumentError("batched_nuts tree_strategy must be :hybrid, :masked, or :persistent, got $(repr(tree_strategy))"))
    # Host reverse-mode AD gradient selection (issue #268, A2). `:auto` uses
    # reverse mode when it is safe and beneficial (Enzyme loaded, the model is on
    # the generated-scorer path, the batch shares one posterior, the parameter
    # count clears the threshold, and a trial gradient compiles) and forward mode
    # otherwise; `:forward`/`:reverse` override. Reverse mode is host-only, so it
    # does not apply to the device (`backend`) path.
    adtype in (:auto, :forward, :reverse) ||
        throw(ArgumentError("batched_nuts adtype must be :auto, :forward, or :reverse, got $(repr(adtype))"))
    !(adtype === :reverse && backend !== nothing) ||
        throw(ArgumentError("batched_nuts adtype=:reverse is a host-only path and cannot be combined with a device `backend`"))
    # `:persistent` (issue #154 increment 2) is a device-only strategy: the whole tree
    # is built inside a device kernel, so it has no host fallback and REQUIRES a backend.
    !(tree_strategy === :persistent && backend === nothing) ||
        throw(ArgumentError("batched_nuts tree_strategy=:persistent requires a `backend` (it is a device-only path)"))
    init in (:prior, :uniform) ||
        throw(ArgumentError("batched_nuts init must be :prior or :uniform, got $(repr(init))"))
    init_max_retries >= 0 ||
        throw(ArgumentError("batched_nuts init_max_retries must be >= 0, got $init_max_retries"))

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
        tree_strategy in (:masked, :persistent) ||
            throw(
                ArgumentError(
                    "batched_nuts device backend requires tree_strategy=:masked or :persistent, got $(repr(tree_strategy))",
                ),
            )
        # `:persistent` (issue #154 increment 2): one kernel launch per iteration, each
        # grid lane building its own device-resident NUTS tree with on-device Philox
        # randomness. Statistically -- not bitwise -- equivalent to the host/masked
        # paths (RNG semantics differ; validated by the #121 gate). Diagonal mass only.
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
        tree_strategy === :persistent ?
        DevicePersistentNUTSWorkspace(
            model, num_chains, max_tree_depth;
            backend=backend, precision=device_precision, args=args, constraints=constraints,
            gradient_mode=persistent_gradient,
        ) :
        DeviceNUTSWorkspace(
            model, num_chains, max_tree_depth;
            backend=backend, precision=device_precision, args=args, constraints=constraints,
            sync_per_leaf=device_sync_per_leaf,
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
        init=init,
    )
    workspace = BatchedNUTSWorkspace(model, position, batch_args, batch_constraints, max_tree_depth; adtype=adtype)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = workspace.current_gradient
    # Retry non-finite starting points instead of throwing outright (issue #162):
    # heavy-tailed priors (e.g. the eight-schools HalfCauchy tau) can draw a
    # chain arbitrarily far out. Only re-drawable inits (prior/uniform/Pathfinder)
    # retry -- a fixed `initial_params` array reproduces the same value, so it
    # fails fast. `position` is re-drawn in place, so the sampler and workspace
    # (which read `position` directly) see the corrected starting point.
    local gradient
    init_attempt = 0
    while true
        _, gradient = _batched_logjoint_and_gradient_unconstrained!(
            current_logjoint,
            workspace.gradient_cache,
            position,
        )
        copyto!(current_gradient, gradient)
        bad_columns = _nonfinite_init_columns(current_logjoint, current_gradient)
        isempty(bad_columns) && break
        if !_init_is_redrawable(initial_params) || init_attempt >= init_max_retries
            retried = _init_is_redrawable(initial_params) ? " after $init_max_retries re-draw(s)" : ""
            throw(
                ArgumentError(
                    "initial batched NUTS parameters produced a non-finite unconstrained logjoint or gradient in $(length(bad_columns)) of $num_chains chain(s)$retried; try init=:uniform or supply finite initial_params, or check the constraint values for NaN/Inf",
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
            init,
            rng,
            num_params,
            constrained_num_params,
            num_chains,
        )
    end

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

    # Seed the warmup INITIAL diagonal inverse mass from the Pathfinder covariance
    # diagonal when a PathfinderResult is supplied (issue #162); otherwise the seed
    # is `ones(num_params)`, so every downstream path reproduces today's behavior
    # exactly. Threaded into the driver(s), the initial reasonable-step-size search,
    # and (device path) the initial shared inverse mass uploaded to the device.
    initial_inverse_mass =
        initial_params isa PathfinderResult ?
        _pathfinder_inverse_mass_seed(initial_params, model, num_params, mass_matrix_regularization) :
        ones(num_params)

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
            initial_inverse_mass,
            callback,
            callback_every,
            rng,
        )
    elseif per_chain_adaptation
        return _batched_nuts_host_pooled!(
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
            initial_inverse_mass,
            callback,
            callback_every,
            tree_strategy,
            rng,
        )
    end

    inverse_mass_matrix = copy(initial_inverse_mass)
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
        initial_inverse_mass_matrix=initial_inverse_mass,
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
        if device_nuts_workspace !== nothing
            _device_nuts_proposals_dispatch!(
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
                iteration,
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

# Pooled-mass / per-chain-step warmup adaptation for host batched NUTS (issue #158).
# This is the HOST batched default and the exact host analog of the device pooled
# path `_batched_nuts_device_per_chain!`: a SINGLE diagonal inverse-mass vector
# pooled across all chains (one running-variance state fed the full P x C positions)
# plus a PER-CHAIN dual-averaged step size. It supersedes the retired per-chain-MASS
# mode, which estimated every chain's mass column from only ~160 of its own draws
# (issue #168 / #158). Pooling feeds one running-variance state the full P x C draws,
# so the metric sees C x the samples and converges in the first slow window (a better,
# less marginal R-hat), while the per-chain step keeps the #137 stranding fix. It also
# unifies the meaning of per_chain_adaptation with the device path. The only difference
# from the device
# loop is the proposal generator: the host masked (`_batched_nuts_proposals_masked!`)
# or hybrid (`_batched_nuts_proposals!`) trajectory instead of the device kernel; both
# are threaded the shared P-vector mass and the C-length per-chain step vector, which
# the `_chain_inverse_mass`/`_chain_step_size` selectors already dispatch correctly.
# Callback contract: the reported `step_size` is the mean of the per-chain step sizes
# at the start of the iteration.
function _batched_nuts_host_pooled!(
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
    initial_inverse_mass::AbstractVector,
    callback,
    callback_every::Int,
    tree_strategy::Symbol,
    rng::AbstractRNG,
)
    # The starting shared diagonal inverse mass (issue #162): `ones` by default, or
    # the Pathfinder covariance-diagonal seed. Broadcast into the C mass columns the
    # per-chain proposal overloads consume, and used for the initial step search.
    initial_inverse_mass_columns = repeat(collect(Float64, initial_inverse_mass), 1, num_chains)
    # Keep the initial-search workspace ALIVE (issue #158 lever A) so the batched
    # window-end re-search reuses it instead of the retired C scalar searches.
    step_size_workspace = nothing
    step_sizes = fill(nuts_step_size, num_chains)
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        step_size_workspace = BatchedHMCWorkspace(
            model, position, batch_args, batch_constraints, collect(Float64, initial_inverse_mass),
        )
        step_sizes = _find_reasonable_batched_step_size_per_chain(
            step_size_workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            copy(initial_inverse_mass_columns),
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
        initial_inverse_mass_matrix=initial_inverse_mass,
    )
    # Per-chain window-end re-search (issue #158 lever A): ONE batched call re-searches
    # every chain's step against the SHARED pooled mass in a single vectorized doubling
    # loop (replacing the former C scalar searches). It holds live references to
    # position/current_logjoint/current_gradient (mutated in place below) and reuses
    # the initial-search workspace.
    refind = BatchedPerChainStepSizeSearch(
        step_size_workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        batch_args,
        batch_constraints,
        rng,
    )
    accept_statistics = Vector{Float64}(undef, num_chains)
    # The pooled shared mass is broadcast into every column so the fully-tested
    # per-chain proposal overloads (Matrix mass + C-length step vector) run with
    # identical mass columns -- numerically identical to a shared-mass integrator,
    # the host analog of the device path's single uploaded inverse-mass vector.
    # Seeded from `initial_inverse_mass` (issue #162); overwritten each iteration
    # from the driver's current shared mass, so the seed only matters transiently.
    inverse_mass_matrices = copy(initial_inverse_mass_columns)

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        @inbounds for chain_index = 1:num_chains
            copyto!(view(inverse_mass_matrices, :, chain_index), driver.mass.inverse_mass_matrix)
        end
        mean_step_size = sum(driver.step_sizes) / num_chains

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
                driver.step_sizes,
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
                driver.step_sizes,
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
                    driver.mass.variance_state,
                    false,
                    workspace.accept_prob[chain_index],
                    workspace.control.divergent_step[chain_index],
                )
                accept_statistics[chain_index] =
                    workspace.control.divergent_step[chain_index] ? 0.0 : workspace.accept_prob[chain_index]
            end
            warmup_update!(
                driver,
                iteration,
                accept_statistics,
                position,
                workspace.mass_adaptation_weights,
                refind,
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
    initial_inverse_mass::AbstractVector,
    callback,
    callback_every::Int,
    rng::AbstractRNG,
)
    # The starting shared diagonal inverse mass (issue #162): `ones` by default, or
    # the Pathfinder covariance-diagonal seed. It seeds the initial step search and
    # (via the driver's shared mass) is the first `inverse_mass` uploaded to the
    # device on iteration 1.
    initial_inverse_mass_columns = repeat(collect(Float64, initial_inverse_mass), 1, num_chains)
    # Keep the initial-search workspace ALIVE (issue #158 lever A) so the batched
    # window-end re-search reuses it instead of the retired C scalar searches.
    step_size_workspace = nothing
    step_sizes = fill(nuts_step_size, num_chains)
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        step_size_workspace = BatchedHMCWorkspace(
            model, position, batch_args, batch_constraints, collect(Float64, initial_inverse_mass),
        )
        step_sizes = _find_reasonable_batched_step_size_per_chain(
            step_size_workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            copy(initial_inverse_mass_columns),
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
        initial_inverse_mass_matrix=initial_inverse_mass,
    )
    # Per-chain window-end re-search (issue #158 lever A): ONE batched call re-searches
    # every chain's step against the SHARED pooled mass in a single vectorized doubling
    # loop (replacing the former C scalar searches). It holds live references to the
    # host mirrors position/current_logjoint/current_gradient (refreshed by the device
    # download each warmup iteration) and reuses the initial-search workspace.
    refind = BatchedPerChainStepSizeSearch(
        step_size_workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        batch_args,
        batch_constraints,
        rng,
    )
    accept_statistics = Vector{Float64}(undef, num_chains)

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        inverse_mass_matrix = driver.mass.inverse_mass_matrix
        mean_step_size = sum(driver.step_sizes) / num_chains

        _device_nuts_proposals_dispatch!(
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
            iteration,
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
            end
            warmup_update!(
                driver,
                iteration,
                accept_statistics,
                position,
                workspace.mass_adaptation_weights,
                refind,
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

# Progress-callback contract shared by all samplers:
#   `callback(info::NamedTuple)` where `info` has at least
#     phase::Symbol       -- :warmup or :sample (MCMC); :stage (SMC); :step (ADVI)
#     iteration::Int      -- 1-based index within the phase
#     total::Int          -- total iterations in the phase
#     step_size::Float64  -- current step size; NaN where meaningless
#     divergences::Int    -- cumulative divergences; 0 where untracked
# The callback fires every `callback_every` iterations and on the phase's final
# iteration. When `callback === nothing` there is zero overhead (each call site
# is guarded by `isnothing(callback) ||`). Callbacks never consume the RNG, so
# default runs remain bitwise reproducible.
@inline function _invoke_progress_callback(
    callback,
    callback_every::Int,
    phase::Symbol,
    iteration::Int,
    total::Int,
    step_size::Float64,
    divergences::Int,
)
    ((callback_every > 0 && iteration % callback_every == 0) || iteration == total) || return nothing
    callback((
        phase=phase,
        iteration=iteration,
        total=total,
        step_size=step_size,
        divergences=divergences,
    ))
    return nothing
end

"""
    hmc(model, args=(), constraints=choicemap(); num_samples, kwargs...) -> HMCChains

Run a single Hamiltonian Monte Carlo chain against `model` conditioned on
`constraints`, returning a one-chain [`HMCChains`](@ref) — the same result type
as [`hmc_chains`](@ref), so `summarize`, `rhat`, `ess`, `predict`, `loo`, and
the export helpers all work directly on the result.

Warmup adapts the step size (dual averaging to `target_accept`) and, optionally,
the mass matrix. Key keyword arguments:

- `num_samples` (required) and `num_warmup`: post-warmup and warmup iterations.
- `step_size`, `num_leapfrog_steps`: leapfrog integrator settings.
- `target_accept`, `adapt_step_size`, `adapt_mass_matrix`,
  `find_reasonable_step_size`: adaptation controls.
- `metric`: `:diag` (default) or `:dense` mass matrix.
- `initial_params`, `rng`: initialization and randomness.

Gradients always use ForwardDiff; there is no `adtype` selection here (only the
batched samplers accept `adtype`).

See also [`nuts`](@ref), [`hmc_chains`](@ref), and [`batched_hmc`](@ref).
"""
function hmc(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    metric::Symbol=:diag,
    callback=nothing,
    callback_every::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    metric in (:diag, :dense) || throw(ArgumentError("metric must be :diag or :dense, got :$metric"))
    layout = _conditioned_parameter_layout(model, constraints, args)
    num_params = parametercount(layout)
    constrained_num_params = parametervaluecount(layout)
    _validate_hmc_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )

    position = _initial_hmc_position(model, args, constraints, initial_params, rng)
    current_logjoint = logjoint_unconstrained(model, position, args, constraints)
    isfinite(current_logjoint) ||
        throw(
            ArgumentError(
                "initial HMC parameters produced a non-finite unconstrained logjoint; supply finite initial_params, or check the constraint values for NaN/Inf",
            ),
        )
    # sampler-owned evaluation: Stan-style reject semantics (issue #157)
    gradient_cache = _logjoint_gradient_cache(model, position, args, constraints; reject_invalid_parameters=true)

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, constrained_num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    acceptance_stats = Vector{Float64}(undef, num_samples)
    energies = Vector{Float64}(undef, num_samples)
    energy_errors = Vector{Float64}(undef, num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    total_iterations = num_warmup + num_samples
    hmc_step_size = Float64(step_size)
    hmc_target_accept = Float64(target_accept)
    hmc_divergence_threshold = Float64(divergence_threshold)
    inverse_mass_matrix = ones(num_params)
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        hmc_step_size = _find_reasonable_step_size(
            model,
            position,
            current_logjoint,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            hmc_step_size,
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
        metric=metric,
    )
    refind = ScalarStepSizeSearch(model, gradient_cache, args, constraints, rng, position, current_logjoint)

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        hmc_step_size = driver.step_size
        inverse_mass_matrix = _driver_metric(driver)
        momentum = _sample_momentum(rng, inverse_mass_matrix)
        proposal = _leapfrog(
            model,
            position,
            momentum,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            hmc_step_size,
            num_leapfrog_steps,
        )

        accepted_step = false
        log_accept_ratio = _proposal_log_accept_ratio(current_logjoint, momentum, proposal, inverse_mass_matrix)
        accept_prob = _acceptance_probability(log_accept_ratio)
        proposal_energy, energy_error, divergent_step = _proposal_diagnostics(
            current_logjoint,
            momentum,
            proposal,
            inverse_mass_matrix,
            hmc_divergence_threshold,
        )
        sample_energy = isnothing(proposal) ? _hamiltonian(current_logjoint, momentum, inverse_mass_matrix) : proposal_energy
        if !isnothing(proposal)
            proposed_position, _, proposed_logjoint = proposal

            if log(rand(rng)) < min(0.0, log_accept_ratio)
                position = proposed_position
                current_logjoint = proposed_logjoint
                accepted_step = true
            end
        end

        if !accepted_step
            sample_energy = _hamiltonian(current_logjoint, momentum, inverse_mass_matrix)
        end

        divergent_step && (cumulative_divergences += 1)

        if iteration <= num_warmup
            refind.position = position
            refind.current_logjoint = current_logjoint
            mass_weight = _mass_adaptation_weight(driver.variance_state, accepted_step, accept_prob, divergent_step)
            warmup_update!(driver, iteration, accept_prob, position, mass_weight, refind)
            if iteration == num_warmup
                warmup_finalize!(driver)
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, hmc_step_size, cumulative_divergences)
        else
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position, args, constraints)
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_prob
            energies[sample_index] = sample_energy
            energy_errors[sample_index] = energy_error
            accepted[sample_index] = accepted_step
            divergent[sample_index] = divergent_step
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, hmc_step_size, cumulative_divergences)
        end
    end

    chain = HMCChain(
        :hmc,
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        energies,
        energy_errors,
        accepted,
        divergent,
        driver.step_size,
        copy(driver.inverse_mass_matrix),
        num_leapfrog_steps,
        0,
        zeros(Int, num_samples),
        fill(num_leapfrog_steps, num_samples),
        hmc_target_accept,
        driver.mass_adaptation_windows,
        driver.metric_kind === :dense ? copy(driver.dense_metric.inverse_mass) : nothing,
    )
    # Single-chain runs return the SAME result type as multi-chain runs (issue
    # #337): a one-chain HMCChains, so every downstream consumer (summarize,
    # rhat, ess, posterior_array, predict, loo, ...) works without special
    # casing. The per-chain HMCChain stays reachable as `first(result)`.
    return HMCChains(model, args, constraints, [chain])
end

"""
    nuts(model, args=(), constraints=choicemap(); num_samples, kwargs...) -> HMCChains

Run a single No-U-Turn Sampler chain against `model` conditioned on
`constraints`, returning a one-chain [`HMCChains`](@ref) — the same result type
as [`nuts_chains`](@ref), so `summarize`, `rhat`, `ess`, `predict`, `loo`, and
the export helpers all work directly on the result. NUTS is the reference sampler:
it builds a per-iteration trajectory that adapts its own length instead of using
a fixed leapfrog-step count.

Key keyword arguments:

- `num_samples` (required) and `num_warmup`.
- `step_size`, `max_tree_depth`, `divergence_threshold`: trajectory controls.
- `target_accept`, `adapt_step_size`, `adapt_mass_matrix`,
  `find_reasonable_step_size`: adaptation controls.
- `metric`: `:diag` (default) or `:dense` mass matrix.
- `initial_params`, `rng`.

Gradients always use ForwardDiff; there is no `adtype` selection here (only the
batched samplers accept `adtype`).

See also [`hmc`](@ref), [`nuts_chains`](@ref), and [`batched_nuts`](@ref).
"""
function nuts(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    metric::Symbol=:diag,
    callback=nothing,
    callback_every::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    metric in (:diag, :dense) || throw(ArgumentError("metric must be :diag or :dense, got :$metric"))
    layout = _conditioned_parameter_layout(model, constraints, args)
    num_params = parametercount(layout)
    constrained_num_params = parametervaluecount(layout)
    _validate_nuts_arguments(
        num_params,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )

    position = _initial_hmc_position(model, args, constraints, initial_params, rng)
    current_logjoint = logjoint_unconstrained(model, position, args, constraints)
    isfinite(current_logjoint) ||
        throw(
            ArgumentError(
                "initial NUTS parameters produced a non-finite unconstrained logjoint; supply finite initial_params, or check the constraint values for NaN/Inf",
            ),
        )
    # sampler-owned evaluation: Stan-style reject semantics (issue #157)
    gradient_cache = _logjoint_gradient_cache(model, position, args, constraints; reject_invalid_parameters=true)
    current_gradient = copy(_logjoint_gradient!(gradient_cache, position))
    all(isfinite, current_gradient) ||
        throw(
            ArgumentError(
                "initial NUTS parameters produced a non-finite unconstrained gradient; supply finite initial_params, or check the constraint values for NaN/Inf",
            ),
        )

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, constrained_num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    acceptance_stats = Vector{Float64}(undef, num_samples)
    energies = Vector{Float64}(undef, num_samples)
    energy_errors = Vector{Float64}(undef, num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    tree_depths = zeros(Int, num_samples)
    integration_steps_per_sample = zeros(Int, num_samples)
    total_iterations = num_warmup + num_samples
    nuts_step_size = Float64(step_size)
    nuts_target_accept = Float64(target_accept)
    nuts_divergence_threshold = Float64(divergence_threshold)
    inverse_mass_matrix = ones(num_params)
    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        nuts_step_size = _find_reasonable_step_size(
            model,
            position,
            current_logjoint,
            gradient_cache,
            inverse_mass_matrix,
            args,
            constraints,
            nuts_step_size,
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
        metric=metric,
    )
    refind = ScalarStepSizeSearch(model, gradient_cache, args, constraints, rng, position, current_logjoint)

    sample_index = 0
    cumulative_divergences = 0
    nuts_target = ModelDensityTarget(model, args, constraints, gradient_cache)
    # Per-iteration NUTS scratch, hoisted out of the sampler loop (issue #159):
    # _nuts_proposal re-initializes both structs on entry, so they are reused
    # across every warmup and sampling iteration.
    nuts_tree_workspace = NUTSSubtreeWorkspace(num_params, max_tree_depth)
    nuts_continuation = NUTSContinuationState(num_params)
    for iteration = 1:total_iterations
        nuts_step_size = driver.step_size
        inverse_mass_matrix = _driver_metric(driver)
        proposal, accept_stat, tree_depth, integration_steps_used, proposal_energy, energy_error, divergent_step, moved_step =
            _nuts_proposal(
                nuts_tree_workspace,
                nuts_continuation,
                nuts_target,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrix,
                nuts_step_size,
                max_tree_depth,
                nuts_divergence_threshold,
                rng,
            )

        if moved_step
            # `proposal` aliases the hoisted continuation's state, which the
            # next iteration mutates in place -- copy the values out instead
            # of retaining the buffers.
            copyto!(position, proposal.position)
            current_logjoint = proposal.logjoint
            copyto!(current_gradient, proposal.gradient)
        end

        divergent_step && (cumulative_divergences += 1)

        if iteration <= num_warmup
            refind.position = position
            refind.current_logjoint = current_logjoint
            mass_weight = _mass_adaptation_weight(driver.variance_state, false, accept_stat, divergent_step)
            warmup_update!(driver, iteration, accept_stat, position, mass_weight, refind)
            if iteration == num_warmup
                warmup_finalize!(driver)
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, nuts_step_size, cumulative_divergences)
        else
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] = transform_to_constrained(model, position, args, constraints)
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_stat
            energies[sample_index] = proposal_energy
            energy_errors[sample_index] = energy_error
            accepted[sample_index] = moved_step
            divergent[sample_index] = divergent_step
            tree_depths[sample_index] = tree_depth
            integration_steps_per_sample[sample_index] = integration_steps_used
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, nuts_step_size, cumulative_divergences)
        end
    end

    chain = HMCChain(
        :nuts,
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        energies,
        energy_errors,
        accepted,
        divergent,
        driver.step_size,
        copy(driver.inverse_mass_matrix),
        0,
        max_tree_depth,
        tree_depths,
        integration_steps_per_sample,
        nuts_target_accept,
        driver.mass_adaptation_windows,
        driver.metric_kind === :dense ? copy(driver.dense_metric.inverse_mass) : nothing,
    )
    # One-chain HMCChains (issue #337): see the matching comment in `hmc`.
    return HMCChains(model, args, constraints, [chain])
end

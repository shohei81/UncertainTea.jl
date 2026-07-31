function batched_importance_sampling(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_particles::Int,
    proposal_loc=nothing,
    proposal_log_scale=0.0,
    rng::AbstractRNG=Random.default_rng(),
)
    # Signature-aware sizing (#95 PR-7): the latent set (and thus the particle
    # row count, proposal, and result columns) is a function of the conditioning
    # signature, not the syntactic default layout. Memoized in signature_cache.
    layout = _conditioned_parameter_layout(model, constraints)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_importance_sampling requires at least one parameterized latent choice"))
    num_particles > 0 || throw(ArgumentError("batched_importance_sampling requires num_particles > 0"))

    location = _resolve_unconstrained_point(model, args, constraints, proposal_loc, rng, "proposal_loc")
    log_scale = _resolve_scale_vector("proposal_log_scale", proposal_log_scale, parameter_total)
    particles = Matrix{Float64}(undef, parameter_total, num_particles)
    noise = similar(particles)
    log_proposal = Vector{Float64}(undef, num_particles)
    _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
    _gaussian_logdensity!(log_proposal, noise, log_scale)

    logjoint_values = batched_logjoint_unconstrained(model, particles, args, constraints)
    num_nonfinite = count(!isfinite, logjoint_values)
    num_nonfinite < num_particles ||
        throw(ArgumentError("batched_importance_sampling encountered only non-finite unconstrained logjoint values"))
    num_nonfinite == 0 ||
        @warn "batched_importance_sampling assigned zero weight to particles with a non-finite logjoint" num_nonfinite maxlog = 1

    logweights = Vector{Float64}(undef, num_particles)
    for particle_index in eachindex(logweights)
        value = logjoint_values[particle_index]
        logweights[particle_index] = isfinite(value) ? value - log_proposal[particle_index] : -Inf
    end

    normalized_weights, log_weight_total = _normalized_logweights(logweights)
    effective_sample_size = 0.0
    for weight in normalized_weights
        effective_sample_size += weight * weight
    end
    effective_sample_size = inv(effective_sample_size)

    constrained_particles = Matrix{Float64}(undef, parametervaluecount(layout), num_particles)
    _signature_batched_transform_to_constrained!(constrained_particles, model, particles, args, constraints)

    return ImportanceSamplingResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        particles,
        constrained_particles,
        logjoint_values,
        logweights,
        normalized_weights,
        log_weight_total - log(num_particles),
        effective_sample_size,
        _batched_evaluation_backend(model),
    )
end

function batched_sir(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_particles::Int,
    num_samples::Int=num_particles,
    proposal_loc=nothing,
    proposal_log_scale=0.0,
    resampling::Symbol=:systematic,
    rng::AbstractRNG=Random.default_rng(),
)
    importance = batched_importance_sampling(
        model,
        args,
        constraints;
        num_particles=num_particles,
        proposal_loc=proposal_loc,
        proposal_log_scale=proposal_log_scale,
        rng=rng,
    )
    ancestors = _resample_indices(resampling, importance.normalized_weights, num_samples, rng)
    unconstrained_samples = _resampled_particle_matrix(importance.unconstrained_particles, ancestors)
    constrained_samples = _resampled_particle_matrix(importance.constrained_particles, ancestors)
    return SIRResult(importance, ancestors, unconstrained_samples, constrained_samples)
end

function batched_smc(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_particles::Int,
    proposal_loc=nothing,
    proposal_log_scale=0.0,
    target_ess_ratio::Real=0.8,
    max_stages::Int=32,
    resample_final::Bool=false,
    move_kernel::Symbol=:random_walk,
    move_steps::Int=0,
    move_scale=0.1,
    move_step_size::Real=0.05,
    move_num_leapfrog_steps::Int=4,
    move_inverse_mass_matrix=1.0,
    move_max_tree_depth::Int=4,
    move_max_delta_energy::Real=1000.0,
    resampling::Symbol=:systematic,
    callback=nothing,
    callback_every::Int=10,
    adtype::Symbol=:auto,
    rng::AbstractRNG=Random.default_rng(),
)
    adtype in (:auto, :forward, :reverse) ||
        throw(ArgumentError("batched_smc adtype must be :auto, :forward, or :reverse, got $(repr(adtype))"))
    resampling in (:systematic, :stratified, :residual, :multinomial) ||
        throw(ArgumentError("batched_smc resampling must be :systematic, :stratified, :residual, or :multinomial"))
    # Signature-aware sizing (#95 PR-7): particle rows, proposal, tempering move
    # scratch, and result columns all derive from the conditioned latent set, not
    # the syntactic default layout. Memoized in signature_cache.
    layout = _conditioned_parameter_layout(model, constraints)
    parameter_total = parametercount(layout)
    parameter_total > 0 || throw(ArgumentError("batched_smc requires at least one parameterized latent choice"))
    num_particles > 0 || throw(ArgumentError("batched_smc requires num_particles > 0"))
    0 < target_ess_ratio <= 1 || throw(ArgumentError("batched_smc requires 0 < target_ess_ratio <= 1"))
    max_stages > 0 || throw(ArgumentError("batched_smc requires max_stages > 0"))
    move_steps >= 0 || throw(ArgumentError("batched_smc requires move_steps >= 0"))
    move_step_size > 0 || throw(ArgumentError("batched_smc requires move_step_size > 0"))
    move_num_leapfrog_steps > 0 || throw(ArgumentError("batched_smc requires move_num_leapfrog_steps > 0"))
    move_max_tree_depth > 0 || throw(ArgumentError("batched_smc requires move_max_tree_depth > 0"))
    move_max_delta_energy > 0 || throw(ArgumentError("batched_smc requires move_max_delta_energy > 0"))
    move_kernel in (:random_walk, :hmc, :nuts) ||
        throw(ArgumentError("batched_smc move_kernel must be :random_walk, :hmc, or :nuts"))

    location = _resolve_unconstrained_point(model, args, constraints, proposal_loc, rng, "proposal_loc")
    log_scale = _resolve_scale_vector("proposal_log_scale", proposal_log_scale, parameter_total)
    move_scale_vector = _resolve_scale_vector("move_scale", move_scale, parameter_total)
    move_inverse_mass = _resolve_scale_vector("move_inverse_mass_matrix", move_inverse_mass_matrix, parameter_total)
    particles = Matrix{Float64}(undef, parameter_total, num_particles)
    noise = similar(particles)
    logproposal_values = Vector{Float64}(undef, num_particles)
    logjoint_values = Vector{Float64}(undef, num_particles)
    logweights = Vector{Float64}(undef, num_particles)
    _draw_gaussian_particles!(particles, noise, location, log_scale, rng)
    _gaussian_logdensity!(logproposal_values, noise, log_scale)
    copyto!(logjoint_values, batched_logjoint_unconstrained(model, particles, args, constraints))
    num_nonfinite = count(!isfinite, logjoint_values)
    num_nonfinite < num_particles ||
        throw(ArgumentError("batched_smc encountered only non-finite unconstrained logjoint values"))
    if num_nonfinite > 0
        @warn "batched_smc assigned zero weight to particles with a non-finite logjoint" num_nonfinite maxlog = 1
        for index in eachindex(logjoint_values)
            isfinite(logjoint_values[index]) || (logjoint_values[index] = -Inf)
        end
    end

    log_ratio = Vector{Float64}(undef, num_particles)
    for index in eachindex(log_ratio)
        log_ratio[index] = logjoint_values[index] - logproposal_values[index]
    end

    normalized_weights = fill(1.0 / num_particles, num_particles)
    effective_sample_size = Float64(num_particles)
    min_effective_sample_size = Float64(target_ess_ratio) * num_particles
    beta = 0.0
    log_evidence_estimate = 0.0
    stages = SMCStageSummary[]
    ancestor_history = Vector{Vector{Int}}()
    nuts_move_workspace =
        move_kernel === :nuts && move_steps > 0 ?
        TemperedNUTSMoveWorkspace(model, particles, args, constraints, move_max_tree_depth; adtype=adtype) :
        nothing

    for stage_index = 1:max_stages
        beta_next = _adaptive_tempering_beta(log_ratio, beta, min_effective_sample_size, logweights)
        _incremental_tempering_weights!(logweights, log_ratio, beta, beta_next)
        normalized_weights, log_weight_total = _normalized_logweights(logweights)
        effective_sample_size = _effective_sample_size(normalized_weights)
        log_increment = log_weight_total - log(num_particles)
        log_evidence_estimate += log_increment
        resampled = beta_next < 1.0 || resample_final
        stage_move_steps = 0
        move_acceptance_rate = 0.0

        if resampled
            ancestors = _resample_indices(resampling, normalized_weights, num_particles, rng)
            push!(ancestor_history, ancestors)
            particles = _resampled_particle_matrix(particles, ancestors)
            logproposal_values = _resampled_particle_vector(logproposal_values, ancestors)
            logjoint_values = _resampled_particle_vector(logjoint_values, ancestors)
            log_ratio = _resampled_particle_vector(log_ratio, ancestors)
            fill!(normalized_weights, 1.0 / num_particles)
            fill!(logweights, 0.0)
            effective_sample_size = Float64(num_particles)
            if move_steps > 0
                if move_kernel === :random_walk
                    move_acceptance_rate = _batched_random_walk_move!(
                        particles,
                        logjoint_values,
                        logproposal_values,
                        log_ratio,
                        model,
                        args,
                        constraints,
                        location,
                        log_scale,
                        beta_next,
                        move_scale_vector,
                        move_steps,
                        rng,
                    )
                elseif move_kernel === :hmc
                    acceptance_sum = 0.0
                    for _ = 1:move_steps
                        acceptance_sum += _batched_hmc_move!(
                            particles,
                            logjoint_values,
                            logproposal_values,
                            log_ratio,
                            model,
                            args,
                            constraints,
                            location,
                            log_scale,
                            beta_next,
                            move_step_size,
                            move_num_leapfrog_steps,
                            move_inverse_mass,
                            rng,
                        )
                    end
                    move_acceptance_rate = acceptance_sum / move_steps
                else
                    acceptance_sum = 0.0
                    nuts_workspace = nuts_move_workspace::TemperedNUTSMoveWorkspace
                    for _ = 1:move_steps
                        acceptance_sum += _batched_nuts_move!(
                            nuts_workspace,
                            particles,
                            logjoint_values,
                            logproposal_values,
                            log_ratio,
                            model,
                            args,
                            constraints,
                            location,
                            log_scale,
                            beta_next,
                            move_step_size,
                            move_max_tree_depth,
                            move_max_delta_energy,
                            move_inverse_mass,
                            rng,
                        )
                    end
                    move_acceptance_rate = acceptance_sum / move_steps
                end
                stage_move_steps = move_steps
            end
        end

        push!(
            stages,
            SMCStageSummary(
                stage_index,
                beta,
                beta_next,
                effective_sample_size,
                log_increment,
                resampled,
                move_kernel,
                stage_move_steps,
                move_acceptance_rate,
            ),
        )

        # Shared callback contract: fire every `callback_every` stages and on
        # the final stage (schedule reached beta = 1 or max_stages exhausted).
        final_stage = beta_next >= 1.0 - 1e-12 || stage_index == max_stages
        if !isnothing(callback) &&
           ((callback_every > 0 && stage_index % callback_every == 0) || final_stage)
            callback((
                phase=:stage,
                iteration=stage_index,
                total=max_stages,
                step_size=NaN,
                divergences=0,
            ))
        end

        beta = beta_next
        beta >= 1.0 - 1e-12 && break
    end

    beta >= 1.0 - 1e-12 ||
        throw(ArgumentError("batched_smc reached max_stages=$max_stages before the tempering schedule reached 1.0"))

    if !isempty(stages) && last(stages).resampled
        logweights .= 0.0
        normalized_weights .= 1.0 / num_particles
        effective_sample_size = Float64(num_particles)
    end

    constrained_particles = Matrix{Float64}(undef, parametervaluecount(layout), num_particles)
    _signature_batched_transform_to_constrained!(constrained_particles, model, particles, args, constraints)
    importance = ImportanceSamplingResult(
        model,
        args,
        constraints,
        location,
        log_scale,
        particles,
        constrained_particles,
        logjoint_values,
        copy(logweights),
        copy(normalized_weights),
        log_evidence_estimate,
        effective_sample_size,
        _batched_evaluation_backend(model),
    )
    return SMCResult(importance, stages, ancestor_history)
end

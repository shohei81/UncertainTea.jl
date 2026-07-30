# Hand-derived analytic batched logjoint gradients: discrete families (bernoulli, poisson, geometric, binomial, negativebinomial, categorical).

function _accumulate_bernoulli_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
        derivative = value != 0 ? 1 / probability : -1 / (1 - probability)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

# Logit Bernoulli: the eta-gradient is well-conditioned, `d/d_eta = x - logistic(eta)`,
# so it stays finite where the sigmoid-spelling gradient (1/p or -1/(1-p)) blows
# up as p saturates. A non-0/1 value scores -Inf and contributes no gradient.
function _accumulate_bernoullilogit_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    eta_values::AbstractVector{T},
    eta_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        eta = eta_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_bernoullilogit_logpdf(eta, value)
        support = _bernoulli_value(value)
        isnothing(support) && continue
        derivative = (support ? one(T) : zero(T)) - _bernoullilogit_logistic(eta)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * eta_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBernoulliLogitChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support bernoullilogit latent parameters"))
    value_values = env.observed_values
    eta_values = _batched_numeric_scratch!(env, 1)
    eta_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(eta_values, eta_gradients, cache, env, step.eta, 2)
    _accumulate_bernoullilogit_gradient!(totals, gradients, eta_values, eta_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

function _accumulate_binomial_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    trials_values::AbstractVector{Int},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        trials = trials_values[batch_index]
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_binomial_logpdf(trials, probability, value)
        count = _poisson_count(value)
        if isnothing(count) || count > trials
            continue
        elseif count == 0
            derivative = -trials / (1 - probability)
        elseif count == trials
            derivative = count / probability
        else
            derivative = count / probability - (trials - count) / (1 - probability)
        end
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_poisson_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    lambda_values::AbstractVector{T},
    lambda_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        lambda = lambda_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_poisson_logpdf(lambda, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        derivative = count / lambda - 1
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * lambda_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_categorical_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    probability_values::Tuple,
    probability_gradients::Tuple,
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        probabilities = map(values -> values[batch_index], probability_values)
        value = value_values[batch_index]
        totals[batch_index] += _backend_categorical_logpdf(probabilities, value)
        index = _categorical_index(value, length(probabilities))
        isnothing(index) && continue
        derivative = 1 / probabilities[index]
        selected_gradients = probability_gradients[index]
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * selected_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBernoulliChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Bernoulli latent parameters"))
    value_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_bernoulli_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Binomial latent parameters"))
    value_values = env.observed_values
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_binomial_gradient!(totals, gradients, trials_values, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendCategoricalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support categorical latent parameters"))
    value_values = env.observed_values
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(step.probabilities))
    probability_gradients = ntuple(index -> _batched_backend_gradient_scratch!(cache, index), length(step.probabilities))
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    for (index, probability) in enumerate(step.probabilities)
        _eval_backend_numeric_expr_and_gradient!(
            probability_values[index],
            probability_gradients[index],
            cache,
            env,
            probability,
            length(step.probabilities) + index,
        )
    end
    _accumulate_categorical_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, length(step.probabilities) + 1)),
        )
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendPoissonChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Poisson latent parameters"))
    value_values = env.observed_values
    lambda_values = _batched_numeric_scratch!(env, 1)
    lambda_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(lambda_values, lambda_gradients, cache, env, step.lambda, 2)
    _accumulate_poisson_gradient!(totals, gradients, lambda_values, lambda_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

function _accumulate_geometric_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_geometric_logpdf(probability, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        # The count == 0 contribution of the -count / (1 - p) term is exactly
        # zero; skipping it keeps the gradient finite at p == 1 (issue #77).
        derivative = 1 / probability
        if count > 0
            derivative -= count / (1 - probability)
        end
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += derivative * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_negativebinomial_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    successes_values::AbstractVector{T},
    successes_gradients::AbstractMatrix{T},
    probability_values::AbstractVector{T},
    probability_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        successes = successes_values[batch_index]
        probability = probability_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_negativebinomial_logpdf(successes, probability, value)
        count = _poisson_count(value)
        isnothing(count) && continue
        dsuccesses = digamma(count + successes) - digamma(successes) + log(probability)
        # As in the geometric case, skip the exactly-zero count == 0 term so
        # p == 1 stays finite (issue #77).
        dprobability = successes / probability
        if count > 0
            dprobability -= count / (1 - probability)
        end
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dsuccesses * successes_gradients[parameter_index, batch_index] +
                dprobability * probability_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support Geometric latent parameters"))
    value_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 2)
    _accumulate_geometric_gradient!(totals, gradients, probability_values, probability_gradients, value_values)

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support NegativeBinomial latent parameters"))
    value_values = env.observed_values
    successes_values = _batched_numeric_scratch!(env, 1)
    successes_gradients = _batched_backend_gradient_scratch!(cache, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(successes_values, successes_gradients, cache, env, step.successes, 3)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, step.probability, 4)
    _accumulate_negativebinomial_gradient!(
        totals,
        gradients,
        successes_values,
        successes_gradients,
        probability_values,
        probability_gradients,
        value_values,
    )

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 5)),
        )
    end
    return totals, gradients
end

# Beta-Binomial (issue #231): the alpha/beta gradients are closed-form digamma
# differences. With s = alpha + beta,
#   d/dalpha = psi(k+a) - psi(n+s) - psi(a) + psi(s),
#   d/dbeta  = psi(n-k+b) - psi(n+s) - psi(b) + psi(s).
function _accumulate_betabinomial_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    trials_values::AbstractVector{Int},
    alpha_values::AbstractVector{T},
    alpha_gradients::AbstractMatrix{T},
    beta_values::AbstractVector{T},
    beta_gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        trials = trials_values[batch_index]
        alpha = alpha_values[batch_index]
        beta = beta_values[batch_index]
        value = value_values[batch_index]
        totals[batch_index] += _backend_betabinomial_logpdf(trials, alpha, beta, value)
        count = _poisson_count(value)
        (isnothing(count) || count > trials) && continue
        s = alpha + beta
        dgamma_ns = digamma(trials + s)
        dgamma_s = digamma(s)
        dalpha = digamma(count + alpha) - dgamma_ns - digamma(alpha) + dgamma_s
        dbeta = digamma(trials - count + beta) - dgamma_ns - digamma(beta) + dgamma_s
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dalpha * alpha_gradients[parameter_index, batch_index] +
                dbeta * beta_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBetaBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support BetaBinomial latent parameters"))
    value_values = env.observed_values
    trials_values = _batched_index_scratch!(env, 1)
    alpha_values = _batched_numeric_scratch!(env, 1)
    alpha_gradients = _batched_backend_gradient_scratch!(cache, 1)
    beta_values = _batched_numeric_scratch!(env, 2)
    beta_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 3)
    _eval_backend_numeric_expr_and_gradient!(alpha_values, alpha_gradients, cache, env, step.alpha, 4)
    _eval_backend_numeric_expr_and_gradient!(beta_values, beta_gradients, cache, env, step.beta, 5)
    _accumulate_betabinomial_gradient!(
        totals,
        gradients,
        trials_values,
        alpha_values,
        alpha_gradients,
        beta_values,
        beta_gradients,
        value_values,
    )

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 3)),
        )
    end
    return totals, gradients
end

# Discrete-uniform (issue #231): no continuous parameters, so the step only
# accumulates its log density into the totals and contributes no gradient.
function _score_backend_step_and_gradient!(
    step::BackendDiscreteUniformChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.parameter_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support DiscreteUniform latent parameters"))
    value_values = env.observed_values
    lower_values = _batched_index_scratch!(env, 1)
    upper_values = _batched_index_scratch!(env, 2)
    _eval_backend_index_value_expr!(lower_values, env, step.lower, 3)
    _eval_backend_index_value_expr!(upper_values, env, step.upper, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index in eachindex(totals)
        totals[batch_index] +=
            _backend_discreteuniform_logpdf(lower_values[batch_index], upper_values[batch_index], value_values[batch_index])
    end

    if !isnothing(step.binding_slot)
        _assign_backend_choice_value!(
            env,
            cache.slot_gradients,
            step.binding_slot,
            value_values,
            _zero_gradient!(_batched_backend_gradient_scratch!(cache, 1)),
        )
    end
    return totals, gradients
end

# --- marginalize=:enumerate analytic gradient (issue #13 PR-5) ----------------
#
# With branch terms t_v = log pmf(v) + suffix(v), the marginal gradient is the
# responsibility-weighted sum of branch gradients:
#   d logsumexp_v(t_v) = sum_v r_v * (d log pmf(v) + d suffix(v)),
#   r_v = exp(t_v - logsumexp(t)).
# Each branch scores its suffix into its OWN totals/gradients buffers from the
# same pre-branch environment and slot-gradient snapshot (the PR-3/PR-4 leak
# discipline extended to the gradient planes); a conditioned column takes its
# selected branch with weight one.
function _score_backend_step_and_gradient!(
    step::BackendMarginalizeChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    batch_size = env.batch_size
    support_size = length(step.support)
    parameter_count = size(gradients, 1)

    # pmf argument values and gradients into OWNED buffers: the branch bodies
    # below reuse the environment and cache scratch pools
    probability_values = [Vector{T}(undef, batch_size) for _ in step.probabilities]
    probability_gradients = [Matrix{T}(undef, parameter_count, batch_size) for _ in step.probabilities]
    for (argument, expr) in enumerate(step.probabilities)
        _eval_backend_numeric_expr_and_gradient!(
            probability_values[argument],
            probability_gradients[argument],
            cache,
            env,
            expr,
            1,
        )
    end
    log_pmf = Matrix{T}(undef, support_size, batch_size)
    for batch_index = 1:batch_size
        column = tuple((values[batch_index] for values in probability_values)...)
        for (branch, value) in enumerate(step.support)
            log_pmf[branch, batch_index] = _marginalize_choice_logpmf(step, column, value)
        end
    end

    constrained_branch = _marginalize_constrained_branches(step, env, constraints)

    environment_snapshot = _batched_environment_snapshot(env)
    slot_gradients_snapshot = copy(cache.slot_gradients)
    branch_totals = [zeros(T, batch_size) for _ = 1:support_size]
    branch_gradients = [zeros(T, parameter_count, batch_size) for _ = 1:support_size]
    for (branch, value) in enumerate(step.support)
        _marginalize_branch_needed(constrained_branch, log_pmf, branch) || continue
        _batched_environment_restore_snapshot!(env, environment_snapshot)
        copyto!(cache.slot_gradients, slot_gradients_snapshot)
        if !isnothing(step.binding_slot)
            _batched_environment_set_shared!(env, step.binding_slot, _marginalize_binding_value(step, value))
            # the branch value is a constant: a symbol rebound from an earlier
            # differentiable assignment must not leak that derivative into
            # suffix reads through its slot-gradient plane
            fill!(view(cache.slot_gradients, :, step.binding_slot, :), zero(T))
        end
        try
            _score_backend_steps_and_gradient!(
                step.body,
                branch_totals[branch],
                branch_gradients[branch],
                cache,
                env,
                params,
                constraints,
            )
        catch err
            err isa BatchedBackendFallback && rethrow()
            # a column whose result is later ignored may make the shared
            # branch body throw; the per-column fallback evaluates only the
            # branches each column needs (and re-raises genuine errors)
            throw(
                BatchedBackendFallback(
                    "marginalize branch $(value) suffix gradient failed for at least one column: $(sprint(showerror, err))",
                ),
            )
        end
    end
    _batched_environment_restore_snapshot!(env, environment_snapshot)
    copyto!(cache.slot_gradients, slot_gradients_snapshot)

    # d log pmf(v) / d theta through the probability-expression gradients
    pmf_gradient = function (branch, batch_index, parameter_row)
        if step.family === :bernoulli
            probability = probability_values[1][batch_index]
            scale = branch == 2 ? 1 / probability : -1 / (1 - probability)
            return scale * probability_gradients[1][parameter_row, batch_index]
        end
        return probability_gradients[branch][parameter_row, batch_index] /
               probability_values[branch][batch_index]
    end

    # an impossible branch (finite prior mass but a -Inf suffix, or zero
    # mass) carries zero responsibility, and its branch gradient may hold
    # NaN/Inf from the degenerate scoring -- exclude such branches by their
    # FULL term before reading the gradient buffers
    branch_term = function (branch, batch_index)
        pmf = log_pmf[branch, batch_index]
        return isfinite(pmf) ? pmf + branch_totals[branch][batch_index] : oftype(pmf, -Inf)
    end

    for batch_index = 1:batch_size
        selected = constrained_branch[batch_index]
        if selected != 0
            term = branch_term(selected, batch_index)
            if isfinite(term)
                totals[batch_index] += term
                for parameter_row = 1:parameter_count
                    gradients[parameter_row, batch_index] +=
                        pmf_gradient(selected, batch_index, parameter_row) +
                        branch_gradients[selected][parameter_row, batch_index]
                end
            else
                # a -Inf conditioned column has no usable gradient; leave the
                # other steps' contributions untouched (the sampler rejects
                # the state on the -Inf total)
                totals[batch_index] += -Inf
            end
        else
            shift = -Inf
            for branch = 1:support_size
                term = branch_term(branch, batch_index)
                isfinite(term) || continue
                shift = max(shift, term)
            end
            if !isfinite(shift)
                totals[batch_index] += -Inf
                continue
            end
            accumulator = zero(T)
            for branch = 1:support_size
                term = branch_term(branch, batch_index)
                isfinite(term) || continue
                accumulator += exp(term - shift)
            end
            column_total = shift + log(accumulator)
            totals[batch_index] += column_total
            for branch = 1:support_size
                term = branch_term(branch, batch_index)
                isfinite(term) || continue
                responsibility = exp(term - column_total)
                for parameter_row = 1:parameter_count
                    gradients[parameter_row, batch_index] +=
                        responsibility * (
                            pmf_gradient(branch, batch_index, parameter_row) +
                            branch_gradients[branch][parameter_row, batch_index]
                        )
                end
            end
        end
    end
    return totals, gradients
end

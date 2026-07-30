# Backend-native scalar and batched scoring: discrete families (bernoulli, poisson, geometric, binomial, negativebinomial, categorical).

function _backend_bernoulli_logpdf(p, x)
    probability = p
    zero(probability) <= probability <= one(probability) ||
        throw(ArgumentError("bernoulli requires 0 <= p <= 1"))
    value = _bernoulli_value(x)
    isnothing(value) && return oftype(float(probability), -Inf)
    return value ? log(probability) : log1p(-probability)
end

# Logit-parameterized Bernoulli (issues #149/#150), stable log-scale form
# `x*eta - log1p(exp(eta))`; mirrors the CPU `logpdf(::BernoulliLogitDist, x)`.
function _backend_bernoullilogit_logpdf(eta, x)
    value = _bernoulli_value(x)
    isnothing(value) && return oftype(float(eta), -Inf)
    log_normalizer = _bernoullilogit_log1p_exp(eta)
    return value ? eta - log_normalizer : -log_normalizer
end

function _score_backend_step!(
    step::BackendBernoulliLogitChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    eta = _eval_backend_numeric_expr(env, step.eta)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_bernoullilogit_logpdf(eta, value)
end

function _score_backend_step!(
    step::BackendBernoulliLogitChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    eta_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(eta_values, env, step.eta, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_bernoullilogit_logpdf(eta_values[batch_index], choice_values[batch_index])
        if !isnothing(step.binding_slot)
            value = choice_values[batch_index]
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                value isa Integer || throw(
                    BatchedBackendFallback("index backend slot $(step.binding_slot) received non-integer choice value"),
                )
                env.index_values[step.binding_slot, batch_index] = Int(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBernoulliLogitChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    eta_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(eta_values, env, step.eta, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_bernoullilogit_logpdf(eta_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _backend_poisson_logpdf(lambda, x)
    lambda > zero(lambda) || throw(ArgumentError("poisson requires lambda > 0"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(lambda), -Inf)
    return count * log(lambda) - lambda - _logfactorial_like(lambda, count)
end

function _score_backend_step!(
    step::BackendBernoulliChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    probability = _eval_backend_numeric_expr(env, step.probability)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_bernoulli_logpdf(probability, value)
end

function _score_backend_step!(
    step::BackendPoissonChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    lambda = _eval_backend_numeric_expr(env, step.lambda)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_poisson_logpdf(lambda, value)
end

function _score_backend_step!(
    step::BackendBernoulliChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                value isa Integer || throw(
                    BatchedBackendFallback("index backend slot $(step.binding_slot) received non-integer choice value"),
                )
                env.index_values[step.binding_slot, batch_index] = Int(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendPoissonChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    lambda_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(lambda_values, env, step.lambda, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        lambda = lambda_values[batch_index]
        totals[batch_index] += _backend_poisson_logpdf(lambda, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = Int(round(value))
            else
                env.generic_values[step.binding_slot][batch_index] = Int(round(value))
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBernoulliChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    probability_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_bernoulli_logpdf(probability, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendPoissonChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    lambda_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(lambda_values, env, step.lambda, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        lambda = lambda_values[batch_index]
        totals[batch_index] += _backend_poisson_logpdf(lambda, value)
    end
    return totals
end

function _backend_geometric_logpdf(probability, x)
    probability_ = float(probability)
    zero(probability_) < probability_ <= one(probability_) ||
        throw(ArgumentError("geometric requires 0 < p <= 1"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    if count == 0
        return log(probability_)
    elseif probability_ == one(probability_)
        return oftype(probability_, -Inf)
    end
    return log(probability_) + count * log1p(-probability_)
end

function _backend_negativebinomial_logpdf(successes, probability, x)
    successes_, probability_ = promote(successes, probability)
    successes_ > zero(successes_) || throw(ArgumentError("negativebinomial requires successes > 0"))
    zero(probability_) < probability_ <= one(probability_) ||
        throw(ArgumentError("negativebinomial requires 0 < p <= 1"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    if count == 0 && probability_ == one(probability_)
        return zero(probability_)
    elseif probability_ == one(probability_)
        return oftype(probability_, -Inf)
    end
    return loggamma(count + successes_) - loggamma(successes_) - _logfactorial_like(probability_, count) +
           successes_ * log(probability_) + count * log1p(-probability_)
end

function _backend_binomial_logpdf(trials, probability, x)
    probability_ = float(probability)
    zero(probability_) <= probability_ <= one(probability_) ||
        throw(ArgumentError("binomial requires 0 <= p <= 1"))
    trial_count = _binomial_trials(trials)
    isnothing(trial_count) && throw(ArgumentError("binomial requires integer trials >= 0"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    count <= trial_count || return oftype(probability_, -Inf)
    log_combination = _logbinomial_like(probability_, trial_count, count)
    if count == 0 && count == trial_count
        return log_combination
    elseif count == 0
        return log_combination + trial_count * log1p(-probability_)
    elseif count == trial_count
        return log_combination + count * log(probability_)
    end
    return log_combination +
           count * log(probability_) +
           (trial_count - count) * log1p(-probability_)
end

# Beta-Binomial (issue #231): shares `_betabinomial_logpdf_core` with the CPU
# reference; `alpha`/`beta` may be latent-flowing.
function _backend_betabinomial_logpdf(trials, alpha, beta, x)
    alpha_, beta_ = promote(float(alpha), float(beta))
    alpha_ > zero(alpha_) || throw(ArgumentError("betabinomial requires alpha > 0"))
    beta_ > zero(beta_) || throw(ArgumentError("betabinomial requires beta > 0"))
    trial_count = _binomial_trials(trials)
    isnothing(trial_count) && throw(ArgumentError("betabinomial requires integer trials >= 0"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(alpha_, -Inf)
    count <= trial_count || return oftype(alpha_, -Inf)
    return _betabinomial_logpdf_core(trial_count, alpha_, beta_, count)
end

# Discrete-uniform (issue #231): trivial `-log(b - a + 1)` density with no
# continuous parameters.
function _backend_discreteuniform_logpdf(lower, upper, x)
    lo = _discrete_integer(lower)
    hi = _discrete_integer(upper)
    (isnothing(lo) || isnothing(hi)) && throw(ArgumentError("discreteuniform requires integer bounds"))
    lo <= hi || throw(ArgumentError("discreteuniform requires a <= b"))
    value = _discrete_integer(x)
    (isnothing(value) || value < lo || value > hi) && return -Inf
    return -log(float(hi - lo + 1))
end

function _score_backend_step!(
    step::BackendBetaBinomialChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    trials = _eval_backend_index_value_expr(env, step.trials)
    alpha = _eval_backend_numeric_expr(env, step.alpha)
    beta = _eval_backend_numeric_expr(env, step.beta)
    count = _binomial_trials(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_betabinomial_logpdf(trials, alpha, beta, value)
end

function _score_backend_step!(
    step::BackendDiscreteUniformChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    lower = _eval_backend_index_value_expr(env, step.lower)
    upper = _eval_backend_index_value_expr(env, step.upper)
    count = _discrete_integer(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_discreteuniform_logpdf(lower, upper, value)
end

function _score_backend_step!(
    step::BackendBetaBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    trials_values = _batched_index_scratch!(env, 1)
    alpha_values = _batched_numeric_scratch!(env, 1)
    beta_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 3)
    _eval_backend_numeric_expr!(beta_values, env, step.beta, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        totals[batch_index] += _backend_betabinomial_logpdf(
            trials_values[batch_index],
            alpha_values[batch_index],
            beta_values[batch_index],
            value,
        )
        if !isnothing(step.binding_slot)
            count = _binomial_trials(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-betabinomial choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendDiscreteUniformChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    lower_values = _batched_index_scratch!(env, 1)
    upper_values = _batched_index_scratch!(env, 2)
    _eval_backend_index_value_expr!(lower_values, env, step.lower, 3)
    _eval_backend_index_value_expr!(upper_values, env, step.upper, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        totals[batch_index] += _backend_discreteuniform_logpdf(lower_values[batch_index], upper_values[batch_index], value)
        if !isnothing(step.binding_slot)
            count = _discrete_integer(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-discreteuniform choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBetaBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    trials_values = _batched_index_scratch!(env, 1)
    alpha_values = _batched_numeric_scratch!(env, 1)
    beta_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 3)
    _eval_backend_numeric_expr!(beta_values, env, step.beta, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_betabinomial_logpdf(
            trials_values[batch_index],
            alpha_values[batch_index],
            beta_values[batch_index],
            observed_values[batch_index],
        )
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendDiscreteUniformChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    lower_values = _batched_index_scratch!(env, 1)
    upper_values = _batched_index_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_index_value_expr!(lower_values, env, step.lower, 3)
    _eval_backend_index_value_expr!(upper_values, env, step.upper, 3)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_discreteuniform_logpdf(lower_values[batch_index], upper_values[batch_index], observed_values[batch_index])
    end
    return totals
end

# --- marginalize=:enumerate (docs/discrete-enumeration.md, PR-4) -------------

_marginalize_choice_logpmf(step::BackendMarginalizeChoicePlanStep, probabilities::Tuple, value) =
    step.family === :bernoulli ? _backend_bernoulli_logpdf(probabilities[1], value) :
    _backend_categorical_logpdf(probabilities, value)

# Branch values bind as plain integers: bernoulli lowers false/true to 0/1 for
# the numeric binding slot, categorical binds the category index.
_marginalize_binding_value(step::BackendMarginalizeChoicePlanStep, value) = value isa Bool ? Int(value) : value

# Per-column conditioning: a column whose constraints provide the latent
# scores the plain joint at that value (branch selection); the others (entry
# 0) marginalize. Branch selection is only faithful for values the family's
# own normalization accepts AND whose raw binding is arithmetic-equal to the
# branch value; anything else (a non-boolean bernoulli numeric, an
# out-of-support or non-real categorical value) binds the RAW value into the
# suffix on the reference path, which only the per-column fallback reproduces.
function _marginalize_constrained_branches(
    step::BackendMarginalizeChoicePlanStep,
    env::BatchedPlanEnvironment,
    constraints,
)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    constrained_branch = Vector{Int}(undef, env.batch_size)
    for batch_index = 1:env.batch_size
        address = _concrete_batched_address(address_parts, batch_index)
        found, value = _choice_tryget_normalized(_batched_constraint(constraints, batch_index), address)
        constrained_branch[batch_index] = if !found
            0
        else
            matched = if step.family === :bernoulli
                value isa Bool ? Int(value) + 1 :
                (value isa Real && (value == 0 || value == 1)) ? Int(value) + 1 : nothing
            else
                _categorical_index(value, length(step.support))
            end
            isnothing(matched) && throw(
                BatchedBackendFallback(
                    "marginalized $(step.family) conditioned on the unsupported value $value",
                ),
            )
            matched
        end
    end
    return constrained_branch
end

# A branch is scored when any column marginalizes over it with nonzero mass or
# is conditioned on it; an all-columns-dead branch is skipped (its suffix may
# be unevaluable).
function _marginalize_branch_needed(constrained_branch::Vector{Int}, log_pmf::AbstractMatrix, branch::Int)
    return any(
        batch_index ->
            constrained_branch[batch_index] == 0 ? isfinite(log_pmf[branch, batch_index]) :
            constrained_branch[batch_index] == branch,
        eachindex(constrained_branch),
    )
end

function _score_backend_step!(
    step::BackendMarginalizeChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    probabilities = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.probabilities)...)
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    if found
        isnothing(step.binding_slot) ||
            _environment_set!(env, step.binding_slot, _marginalize_binding_value(step, constrained_value))
        return _marginalize_choice_logpmf(step, probabilities, constrained_value) +
               _score_backend_steps(step.body, env, params, constraints)
    end

    snapshot = _environment_snapshot(env)
    terms = [
        begin
            _environment_restore_snapshot!(env, snapshot)
            choice_logpdf = _marginalize_choice_logpmf(step, probabilities, value)
            # zero-mass branches contribute nothing and their suffix may be
            # unevaluable (branch-dependent invalid parameters) -- skip them
            if isfinite(choice_logpdf)
                isnothing(step.binding_slot) ||
                    _environment_set!(env, step.binding_slot, _marginalize_binding_value(step, value))
                choice_logpdf + _score_backend_steps(step.body, env, params, constraints)
            else
                oftype(choice_logpdf, -Inf)
            end
        end for value in step.support
    ]
    _environment_restore_snapshot!(env, snapshot)
    shift = maximum(terms)
    isfinite(shift) || return oftype(shift, -Inf)
    total = zero(shift)
    for term in terms
        total += exp(term - shift)
    end
    return shift + log(total)
end

function _score_backend_step!(
    step::BackendMarginalizeChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    batch_size = env.batch_size
    support_size = length(step.support)
    element_type = eltype(env.numeric_values)

    # evaluate the pmf arguments into OWNED buffers first: the branch bodies
    # below reuse the environment scratch pool and would clobber them
    probability_values = [Vector{element_type}(undef, batch_size) for _ in step.probabilities]
    for (argument, expr) in enumerate(step.probabilities)
        _eval_backend_numeric_expr!(probability_values[argument], env, expr, 1)
    end
    # the flat ForwardDiff gradient tier runs this scorer with a Dual-typed
    # environment, so every buffer follows the environment element type
    log_pmf = Matrix{element_type}(undef, support_size, batch_size)
    for batch_index = 1:batch_size
        column = tuple((values[batch_index] for values in probability_values)...)
        for (branch, value) in enumerate(step.support)
            log_pmf[branch, batch_index] = _marginalize_choice_logpmf(step, column, value)
        end
    end

    constrained_branch = _marginalize_constrained_branches(step, env, constraints)

    snapshot = _batched_environment_snapshot(env)
    branch_totals = [fill!(similar(totals), zero(eltype(totals))) for _ = 1:support_size]
    for (branch, value) in enumerate(step.support)
        _marginalize_branch_needed(constrained_branch, log_pmf, branch) || continue
        _batched_environment_restore_snapshot!(env, snapshot)
        isnothing(step.binding_slot) ||
            _batched_environment_set_shared!(env, step.binding_slot, _marginalize_binding_value(step, value))
        try
            _score_backend_steps!(branch_totals[branch], step.body, env, params, constraints)
        catch err
            err isa BatchedBackendFallback && rethrow()
            # the branch body runs for every column, including columns whose
            # result is later ignored (conditioned on another branch, or
            # zero-mass only for them); a throwing scorer in such a column is
            # not a model error, so route to the per-column fallback, whose
            # marginalizer evaluates only the branches each column needs (and
            # faithfully re-raises genuine errors)
            throw(
                BatchedBackendFallback(
                    "marginalize branch $(value) suffix scoring failed for at least one column: $(sprint(showerror, err))",
                ),
            )
        end
    end
    _batched_environment_restore_snapshot!(env, snapshot)

    for batch_index = 1:batch_size
        selected = constrained_branch[batch_index]
        if selected != 0
            totals[batch_index] += log_pmf[selected, batch_index] + branch_totals[selected][batch_index]
        else
            shift = -Inf
            for branch = 1:support_size
                isfinite(log_pmf[branch, batch_index]) || continue
                shift = max(shift, log_pmf[branch, batch_index] + branch_totals[branch][batch_index])
            end
            if !isfinite(shift)
                totals[batch_index] += -Inf
                continue
            end
            accumulator = 0.0
            for branch = 1:support_size
                isfinite(log_pmf[branch, batch_index]) || continue
                accumulator += exp(log_pmf[branch, batch_index] + branch_totals[branch][batch_index] - shift)
            end
            totals[batch_index] += shift + log(accumulator)
        end
    end
    return totals
end

function _backend_categorical_logpdf(probabilities::Tuple, x)
    length(probabilities) > 0 || throw(ArgumentError("categorical requires at least one probability"))
    total = zero(float(first(probabilities)))
    for probability in probabilities
        probability_ = float(probability)
        zero(probability_) <= probability_ <= one(probability_) ||
            throw(ArgumentError("categorical requires 0 <= p <= 1"))
        total += probability_
    end
    tolerance = sqrt(eps(total)) * max(length(probabilities), 1) * 8
    abs(total - one(total)) <= tolerance || throw(ArgumentError("categorical probabilities must sum to 1"))
    index = _categorical_index(x, length(probabilities))
    isnothing(index) && return oftype(total, -Inf)
    return log(float(probabilities[index]))
end

function _score_backend_step!(
    step::BackendGeometricChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    probability = _eval_backend_numeric_expr(env, step.probability)
    count = _poisson_count(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_geometric_logpdf(probability, value)
end

function _score_backend_step!(
    step::BackendNegativeBinomialChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    successes = _eval_backend_numeric_expr(env, step.successes)
    probability = _eval_backend_numeric_expr(env, step.probability)
    count = _poisson_count(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_negativebinomial_logpdf(successes, probability, value)
end

function _score_backend_step!(
    step::BackendBinomialChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    trials = _eval_backend_index_value_expr(env, step.trials)
    probability = _eval_backend_numeric_expr(env, step.probability)
    count = _binomial_trials(value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(count) ? value : count)
    return _backend_binomial_logpdf(trials, probability, value)
end

function _score_backend_step!(
    step::BackendCategoricalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    probabilities = map(expr -> _eval_backend_numeric_expr(env, expr), step.probabilities)
    index = _categorical_index(value, length(probabilities))
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, isnothing(index) ? value : index)
    return _backend_categorical_logpdf(probabilities, value)
end

function _score_backend_step!(
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    probability_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_geometric_logpdf(probability, value)
        if !isnothing(step.binding_slot)
            count = _poisson_count(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-geometric choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    successes_values = _batched_numeric_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(successes_values, env, step.successes, 3)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        successes = successes_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_negativebinomial_logpdf(successes, probability, value)
        if !isnothing(step.binding_slot)
            count = _poisson_count(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-negativebinomial choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        trials = trials_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_binomial_logpdf(trials, probability, value)
        if !isnothing(step.binding_slot)
            count = _binomial_trials(value)
            isnothing(count) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-binomial choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), count)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = count
            else
                env.generic_values[step.binding_slot][batch_index] = count
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendCategoricalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(step.probabilities))
    for (index, probability) in enumerate(step.probabilities)
        _eval_backend_numeric_expr!(probability_values[index], env, probability, length(step.probabilities) + index)
    end
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        probabilities = map(values -> values[batch_index], probability_values)
        totals[batch_index] += _backend_categorical_logpdf(probabilities, value)
        if !isnothing(step.binding_slot)
            index = _categorical_index(value, length(probabilities))
            isnothing(index) && throw(
                BatchedBackendFallback("index backend slot $(step.binding_slot) received non-categorical choice value"),
            )
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), index)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = index
            else
                env.generic_values[step.binding_slot][batch_index] = index
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendGeometricChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    probability_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_geometric_logpdf(probability_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    successes_values = _batched_numeric_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(successes_values, env, step.successes, 3)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_negativebinomial_logpdf(
            successes_values[batch_index],
            probability_values[batch_index],
            observed_values[batch_index],
        )
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBinomialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_index_value_expr!(trials_values, env, step.trials, 2)
    _eval_backend_numeric_expr!(probability_values, env, step.probability, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        trials = trials_values[batch_index]
        probability = probability_values[batch_index]
        totals[batch_index] += _backend_binomial_logpdf(trials, probability, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendCategoricalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(step.probabilities))
    observed_values = env.observed_values
    for (index, probability) in enumerate(step.probabilities)
        _eval_backend_numeric_expr!(probability_values[index], env, probability, length(step.probabilities) + index)
    end
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        probabilities = map(values -> values[batch_index], probability_values)
        totals[batch_index] += _backend_categorical_logpdf(probabilities, value)
    end
    return totals
end

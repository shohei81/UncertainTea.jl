# Backend-native scalar and batched scoring: continuous scalar families (normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, studentt).
#
# Scale/positivity parameters (sigma, rate, shape, scale, alpha, beta, nu) are
# EXCEPTION-FREE here (issue #98): an out-of-support parameter scores NaN for
# that call instead of throwing, matching the device kernel contract in
# src/device/math.jl. These helpers back both the scalar backend path and the
# batched logjoint/gradient path used inside a leapfrog trajectory; a divergent
# chain that drives a log-transformed scale latent to exp(u) == 0 must invalidate
# only its own column (via the integrator's isfinite masking), not abort the run.
# The genuine user-input contract still throws at the distribution constructors
# in src/distributions/continuous.jl. A NaN guard is also required before the
# loggamma/digamma calls below, which raise DomainError on non-positive reals.

function _backend_normal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return -log(sigma_) - log(2 * pi) / 2 - z * z / 2
end

function _backend_lognormal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return _backend_normal_logpdf(mu_, sigma_, log(xx)) - log(xx)
end

function _backend_exponential_logpdf(rate, x)
    xx, rate_ = promote(x, rate)
    rate_ > zero(rate_) || return oftype(xx, NaN)
    xx >= zero(xx) || return oftype(xx, -Inf)
    return log(rate_) - rate_ * xx
end

function _backend_gamma_logpdf(shape, rate, x)
    xx, shape_, rate_ = promote(x, shape, rate)
    shape_ > zero(shape_) || return oftype(xx, NaN)
    rate_ > zero(rate_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return shape_ * log(rate_) - loggamma(shape_) + (shape_ - one(shape_)) * log(xx) - rate_ * xx
end

function _backend_studentt_logpdf(nu, mu, sigma, x)
    xx, nu_, mu_, sigma_ = promote(x, nu, mu, sigma)
    nu_ > zero(nu_) || return oftype(xx, NaN)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return _studentt_log_constant(nu_) - log(sigma_) -
           (nu_ + one(nu_)) * log1p((z * z) / nu_) / 2
end

function _backend_choice_value(parameter_slot::Union{Nothing,Int}, params::AbstractVector, constraints::ChoiceMap, address)
    if !isnothing(parameter_slot)
        return params[parameter_slot]
    end
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    found && return constrained_value
    throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
end

function _backend_choice_value(
    parameter_slot::Union{Nothing,Int},
    params::AbstractMatrix,
    constraint_map::ChoiceMap,
    address,
    batch_index::Int,
)
    if !isnothing(parameter_slot)
        return params[parameter_slot, batch_index]
    end
    found, constrained_value = _choice_tryget_normalized(constraint_map, address)
    found && return constrained_value
    throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
end

function _backend_observed_choice_value(constraint_map::ChoiceMap, address)
    found, constrained_value = _choice_tryget_normalized(constraint_map, address)
    found && return constrained_value
    throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
end

function _batched_observed_choice_values!(
    destination::AbstractVector,
    constraints::ChoiceMap,
    address,
)
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
    fill!(destination, _batched_backend_observed_value(constrained_value, eltype(destination)))
    return destination
end

function _batched_observed_choice_values!(
    destination::AbstractVector,
    constraints::AbstractVector,
    address,
)
    length(destination) == length(constraints) ||
        throw(DimensionMismatch("expected $(length(destination)) batched constraints, got $(length(constraints))"))
    for batch_index in eachindex(destination, constraints)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        destination[batch_index] = _batched_backend_observed_value(constrained_value, eltype(destination))
    end
    return destination
end

function _batched_choice_numeric_values!(
    destination::AbstractVector,
    ::Nothing,
    params::AbstractMatrix,
    constraints::ChoiceMap,
    address_parts::Tuple,
)
    size(params, 2) == length(destination) ||
        throw(DimensionMismatch("expected $(length(destination)) batched params columns, got $(size(params, 2))"))
    for batch_index in eachindex(destination)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        destination[batch_index] = _batched_backend_observed_value(constrained_value, eltype(destination))
    end
    return destination
end

function _batched_choice_numeric_values!(
    destination::AbstractVector,
    ::Nothing,
    params::AbstractMatrix,
    constraints::AbstractVector,
    address_parts::Tuple,
)
    size(params, 2) == length(destination) ||
        throw(DimensionMismatch("expected $(length(destination)) batched params columns, got $(size(params, 2))"))
    length(constraints) == length(destination) ||
        throw(DimensionMismatch("expected $(length(destination)) batched constraints, got $(length(constraints))"))
    for batch_index in eachindex(destination, constraints)
        address = _concrete_batched_address(address_parts, batch_index)
        found, constrained_value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        destination[batch_index] = _batched_backend_observed_value(constrained_value, eltype(destination))
    end
    return destination
end

function _score_backend_step!(
    step::BackendNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_normal_logpdf(mu, sigma, value)
end

# The scalar path always receives CONSTRAINED values (it backs the
# per-column fallback of the constrained batched_logjoint entry), so it
# scores the centered density on theta; only the BATCHED methods run in z
# space behind the identity pre-pass.
function _score_backend_step!(
    step::BackendNoncenteredNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_normal_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendLognormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_lognormal_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendExponentialChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    rate = _eval_backend_numeric_expr(env, step.rate)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_exponential_logpdf(rate, value)
end

function _score_backend_step!(
    step::BackendGammaChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    shape = _eval_backend_numeric_expr(env, step.shape)
    rate = _eval_backend_numeric_expr(env, step.rate)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_gamma_logpdf(shape, rate, value)
end

function _score_backend_step!(
    step::BackendStudentTChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    nu = _eval_backend_numeric_expr(env, step.nu)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_studentt_logpdf(nu, mu, sigma, value)
end

function _score_backend_step!(
    step::BackendNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
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
    step::BackendNoncenteredNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        z = choice_values[batch_index]
        sigma = sigma_values[batch_index]
        (isfinite(sigma) && sigma > 0) || throw(
            BatchedBackendFallback("noncentered normal requires a finite positive scale, got $sigma"),
        )
        totals[batch_index] += _backend_normal_logpdf(zero(z), one(z), z)
        if !isnothing(step.binding_slot)
            theta = mu_values[batch_index] + sigma * z
            env.numeric_slots[step.binding_slot] || throw(
                BatchedBackendFallback(
                    "noncentered normal binding slot $(step.binding_slot) must be numeric",
                ),
            )
            env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), theta)
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendLognormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
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
    step::BackendExponentialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    rate_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(rate_values, env, step.rate, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_exponential_logpdf(rate, value)
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
    step::BackendGammaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    shape_values = _batched_numeric_scratch!(env, 1)
    rate_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(rate_values, env, step.rate, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        shape = shape_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_gamma_logpdf(shape, rate, value)
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
    step::BackendStudentTChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    nu_values = _batched_numeric_scratch!(env, 1)
    mu_values = _batched_numeric_scratch!(env, 2)
    sigma_values = _batched_numeric_scratch!(env, 3)
    _eval_backend_numeric_expr!(nu_values, env, step.nu, 4)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 5)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 6)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        nu = nu_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_studentt_logpdf(nu, mu, sigma, value)
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

function _score_backend_observed_loop_choice!(
    step::BackendNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendLognormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendExponentialChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    rate_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(rate_values, env, step.rate, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_exponential_logpdf(rate, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendGammaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    shape_values = _batched_numeric_scratch!(env, 1)
    rate_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(rate_values, env, step.rate, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        shape = shape_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_gamma_logpdf(shape, rate, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendStudentTChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    nu_values = _batched_numeric_scratch!(env, 1)
    mu_values = _batched_numeric_scratch!(env, 2)
    sigma_values = _batched_numeric_scratch!(env, 3)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(nu_values, env, step.nu, 4)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 5)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 6)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        nu = nu_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_studentt_logpdf(nu, mu, sigma, value)
    end
    return totals
end
function _backend_laplace_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    return -log(2 * scale_) - abs(xx - mu_) / scale_
end

function _backend_inversegamma_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    shape_ > zero(shape_) || return oftype(xx, NaN)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return shape_ * log(scale_) - loggamma(shape_) -
           (shape_ + one(shape_)) * log(xx) -
           scale_ / xx
end

function _backend_weibull_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    shape_ > zero(shape_) || return oftype(xx, NaN)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx < zero(xx) && return oftype(xx, -Inf)
    if xx == zero(xx)
        if shape_ < one(shape_)
            return oftype(xx, Inf)
        elseif shape_ == one(shape_)
            return -log(scale_)
        end
        return oftype(xx, -Inf)
    end
    log_ratio = log(xx) - log(scale_)
    return log(shape_) + (shape_ - one(shape_)) * log(xx) -
           shape_ * log(scale_) - exp(shape_ * log_ratio)
end

function _backend_beta_logpdf(alpha, beta_parameter, x)
    xx, alpha_, beta_ = promote(x, alpha, beta_parameter)
    alpha_ > zero(alpha_) || return oftype(xx, NaN)
    beta_ > zero(beta_) || return oftype(xx, NaN)
    zero(xx) < xx < one(xx) || return oftype(xx, -Inf)
    return loggamma(alpha_ + beta_) - loggamma(alpha_) - loggamma(beta_) +
           (alpha_ - one(alpha_)) * log(xx) +
           (beta_ - one(beta_)) * log1p(-xx)
end

function _score_backend_step!(
    step::BackendLaplaceChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_laplace_logpdf(mu, scale, value)
end

function _score_backend_step!(
    step::BackendInverseGammaChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    shape = _eval_backend_numeric_expr(env, step.shape)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_inversegamma_logpdf(shape, scale, value)
end

function _score_backend_step!(
    step::BackendWeibullChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    shape = _eval_backend_numeric_expr(env, step.shape)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_weibull_logpdf(shape, scale, value)
end

function _score_backend_step!(
    step::BackendBetaChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    alpha = _eval_backend_numeric_expr(env, step.alpha)
    beta_parameter = _eval_backend_numeric_expr(env, step.beta)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_beta_logpdf(alpha, beta_parameter, value)
end

function _score_backend_step!(
    step::BackendLaplaceChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        mu = mu_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_laplace_logpdf(mu, scale, value)
        if !isnothing(step.binding_slot)
            env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendInverseGammaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_inversegamma_logpdf(shape, scale, value)
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
    step::BackendWeibullChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_weibull_logpdf(shape, scale, value)
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
    step::BackendBetaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    alpha_values = _batched_numeric_scratch!(env, 1)
    beta_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 3)
    _eval_backend_numeric_expr!(beta_values, env, step.beta, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        alpha = alpha_values[batch_index]
        beta_parameter = beta_values[batch_index]
        totals[batch_index] += _backend_beta_logpdf(alpha, beta_parameter, value)
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

function _score_backend_observed_loop_choice!(
    step::BackendLaplaceChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_laplace_logpdf(mu_values[batch_index], scale_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendInverseGammaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_inversegamma_logpdf(shape, scale, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendWeibullChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_weibull_logpdf(shape, scale, value)
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendBetaChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    alpha_values = _batched_numeric_scratch!(env, 1)
    beta_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 3)
    _eval_backend_numeric_expr!(beta_values, env, step.beta, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        value = observed_values[batch_index]
        alpha = alpha_values[batch_index]
        beta_parameter = beta_values[batch_index]
        totals[batch_index] += _backend_beta_logpdf(alpha, beta_parameter, value)
    end
    return totals
end

# ---- scalar prior families (issue #229): cauchy, halfnormal, halfcauchy, uniform, logistic, gumbel ----
# Scale/positivity parameters score NaN (not throw) out of support, matching the
# exception-free contract documented at the top of this file.

function _backend_cauchy_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return -log(oftype(xx, pi)) - log(sigma_) - log1p(z * z)
end

function _backend_halfnormal_logpdf(sigma, x)
    xx, sigma_ = promote(x, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    xx >= zero(xx) || return oftype(xx, -Inf)
    z = xx / sigma_
    return log(oftype(xx, 2)) - log(sigma_) - log(2 * oftype(xx, pi)) / 2 - z * z / 2
end

function _backend_halfcauchy_logpdf(scale, x)
    xx, scale_ = promote(x, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx >= zero(xx) || return oftype(xx, -Inf)
    z = xx / scale_
    return log(oftype(xx, 2)) - log(oftype(xx, pi)) - log(scale_) - log1p(z * z)
end

function _backend_uniform_logpdf(lower, upper, x)
    xx, lower_, upper_ = promote(x, lower, upper)
    upper_ > lower_ || return oftype(xx, NaN)
    lower_ <= xx <= upper_ || return oftype(xx, -Inf)
    return -log(upper_ - lower_)
end

function _backend_logistic_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    z = (xx - mu_) / scale_
    az = abs(z)
    return -log(scale_) - az - 2 * log1p(exp(-az))
end

function _backend_gumbel_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    z = (xx - mu_) / scale_
    return -log(scale_) - z - exp(-z)
end

# --- scalar (non-batched) scoring ---
function _score_backend_step!(
    step::BackendCauchyChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_cauchy_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendHalfNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    sigma = _eval_backend_numeric_expr(env, step.sigma)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_halfnormal_logpdf(sigma, value)
end

function _score_backend_step!(
    step::BackendHalfCauchyChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_halfcauchy_logpdf(scale, value)
end

function _score_backend_step!(
    step::BackendUniformChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    lower = _eval_backend_numeric_expr(env, step.lower)
    upper = _eval_backend_numeric_expr(env, step.upper)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_uniform_logpdf(lower, upper, value)
end

function _score_backend_step!(
    step::BackendLogisticChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_logistic_logpdf(mu, scale, value)
end

function _score_backend_step!(
    step::BackendGumbelChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_gumbel_logpdf(mu, scale, value)
end

# --- batched scoring ---
@inline function _backend_write_choice_binding!(env::BatchedPlanEnvironment, binding_slot, value, batch_index)
    isnothing(binding_slot) && return nothing
    if env.numeric_slots[binding_slot]
        env.numeric_values[binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
    elseif env.index_slots[binding_slot]
        value isa Integer ||
            throw(BatchedBackendFallback("index backend slot $(binding_slot) received non-integer choice value"))
        env.index_values[binding_slot, batch_index] = Int(value)
    else
        env.generic_values[binding_slot][batch_index] = value
    end
    return nothing
end

function _score_backend_step!(
    step::BackendCauchyChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_cauchy_logpdf(mu_values[batch_index], sigma_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendHalfNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    sigma_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_halfnormal_logpdf(sigma_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendHalfCauchyChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    scale_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_halfcauchy_logpdf(scale_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendUniformChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    lower_values = _batched_numeric_scratch!(env, 1)
    upper_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(lower_values, env, step.lower, 3)
    _eval_backend_numeric_expr!(upper_values, env, step.upper, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_uniform_logpdf(lower_values[batch_index], upper_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendLogisticChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_logistic_logpdf(mu_values[batch_index], scale_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendGumbelChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_gumbel_logpdf(mu_values[batch_index], scale_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

# --- observed-loop scoring ---
function _score_backend_observed_loop_choice!(
    step::BackendCauchyChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_cauchy_logpdf(mu_values[batch_index], sigma_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendHalfNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    sigma_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(sigma_values, env, step.sigma, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_halfnormal_logpdf(sigma_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendHalfCauchyChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    scale_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_halfcauchy_logpdf(scale_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendUniformChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    lower_values = _batched_numeric_scratch!(env, 1)
    upper_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(lower_values, env, step.lower, 3)
    _eval_backend_numeric_expr!(upper_values, env, step.upper, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_uniform_logpdf(lower_values[batch_index], upper_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendLogisticChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_logistic_logpdf(mu_values[batch_index], scale_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendGumbelChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_gumbel_logpdf(mu_values[batch_index], scale_values[batch_index], observed_values[batch_index])
    end
    return totals
end

# ---- positive-support / heavy-tail families (issue #230): pareto, frechet, rayleigh, inversegaussian ----
# Scale/positivity/shape parameters score NaN (not throw) out of support, matching
# the exception-free contract documented at the top of this file.

function _backend_pareto_logpdf(xm, alpha, x)
    xx, xm_, alpha_ = promote(x, xm, alpha)
    (xm_ > zero(xm_) && alpha_ > zero(alpha_)) || return oftype(xx, NaN)
    xx >= xm_ || return oftype(xx, -Inf)
    return log(alpha_) + alpha_ * log(xm_) - (alpha_ + one(alpha_)) * log(xx)
end

function _backend_frechet_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    (shape_ > zero(shape_) && scale_ > zero(scale_)) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    logz = log(xx) - log(scale_)
    return log(shape_) - log(scale_) - (one(shape_) + shape_) * logz - exp(-shape_ * logz)
end

function _backend_rayleigh_logpdf(scale, x)
    xx, scale_ = promote(x, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return log(xx) - 2 * log(scale_) - xx * xx / (2 * scale_ * scale_)
end

function _backend_inversegaussian_logpdf(mu, lambda, x)
    xx, mu_, lambda_ = promote(x, mu, lambda)
    (mu_ > zero(mu_) && lambda_ > zero(lambda_)) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    d = xx - mu_
    return log(lambda_) / 2 - log(2 * oftype(xx, pi)) / 2 - 3 * log(xx) / 2 -
           lambda_ * d * d / (2 * mu_ * mu_ * xx)
end

# --- scalar (non-batched) scoring ---
function _score_backend_step!(
    step::BackendParetoChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    xm = _eval_backend_numeric_expr(env, step.xm)
    alpha = _eval_backend_numeric_expr(env, step.alpha)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_pareto_logpdf(xm, alpha, value)
end

function _score_backend_step!(
    step::BackendFrechetChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    shape = _eval_backend_numeric_expr(env, step.shape)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_frechet_logpdf(shape, scale, value)
end

function _score_backend_step!(
    step::BackendRayleighChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    scale = _eval_backend_numeric_expr(env, step.scale)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_rayleigh_logpdf(scale, value)
end

function _score_backend_step!(
    step::BackendInverseGaussianChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    mu = _eval_backend_numeric_expr(env, step.mu)
    lambda = _eval_backend_numeric_expr(env, step.lambda)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_inversegaussian_logpdf(mu, lambda, value)
end

# --- batched scoring ---
function _score_backend_step!(
    step::BackendParetoChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    xm_values = _batched_numeric_scratch!(env, 1)
    alpha_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(xm_values, env, step.xm, 3)
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_pareto_logpdf(xm_values[batch_index], alpha_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendFrechetChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_frechet_logpdf(shape_values[batch_index], scale_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendRayleighChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    scale_values = _batched_numeric_scratch!(env, 1)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_rayleigh_logpdf(scale_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

function _score_backend_step!(
    step::BackendInverseGaussianChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = env.observed_values
    mu_values = _batched_numeric_scratch!(env, 1)
    lambda_values = _batched_numeric_scratch!(env, 2)
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(lambda_values, env, step.lambda, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_inversegaussian_logpdf(mu_values[batch_index], lambda_values[batch_index], choice_values[batch_index])
        _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

# --- observed-loop scoring ---
function _score_backend_observed_loop_choice!(
    step::BackendParetoChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    xm_values = _batched_numeric_scratch!(env, 1)
    alpha_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(xm_values, env, step.xm, 3)
    _eval_backend_numeric_expr!(alpha_values, env, step.alpha, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_pareto_logpdf(xm_values[batch_index], alpha_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendFrechetChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    shape_values = _batched_numeric_scratch!(env, 1)
    scale_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(shape_values, env, step.shape, 3)
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_frechet_logpdf(shape_values[batch_index], scale_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendRayleighChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    scale_values = _batched_numeric_scratch!(env, 1)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(scale_values, env, step.scale, 2)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] += _backend_rayleigh_logpdf(scale_values[batch_index], observed_values[batch_index])
    end
    return totals
end

function _score_backend_observed_loop_choice!(
    step::BackendInverseGaussianChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
    address,
)
    mu_values = _batched_numeric_scratch!(env, 1)
    lambda_values = _batched_numeric_scratch!(env, 2)
    observed_values = env.observed_values
    _eval_backend_numeric_expr!(mu_values, env, step.mu, 3)
    _eval_backend_numeric_expr!(lambda_values, env, step.lambda, 4)
    _batched_observed_choice_values!(observed_values, constraints, address)
    for batch_index = 1:env.batch_size
        totals[batch_index] +=
            _backend_inversegaussian_logpdf(mu_values[batch_index], lambda_values[batch_index], observed_values[batch_index])
    end
    return totals
end

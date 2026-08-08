# Backend-native scalar and batched scoring: continuous scalar families.
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
#
# The scalar logpdf kernels (_backend_<family>_logpdf) live in
# src/distributions/scalar_kernels.jl (issue #285): one declaration serves the
# CPU-reference logpdf methods, this scoring path, and the batched gradients.
# The per-family scalar/batched/observed-loop scoring methods below are
# @eval-generated from DISTRIBUTION_FAMILY_TABLE (issue #331 stage 2); the
# deviating bodies (noncentered normal, laplace's batched binding write) stay
# hand-written with a comment saying why.

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

# ---- table-generated per-family scoring (issue #331 stage 2) ------------------
#
# Three method families per distribution family, generated from the table's
# `step`/`params` fields and the issue-#285 kernels:
#
#   * scalar `_score_backend_step!(step, env::PlanEnvironment, ...)`
#   * batched `_score_backend_step!(step, totals, env::BatchedPlanEnvironment, ...)`
#     (laplace stays hand-written below: its historical binding write skips the
#     slot-kind dispatch of `_backend_write_choice_binding!`)
#   * `_score_backend_observed_loop_choice!(step, totals, env, ...)`
#
# Batched scratch layout matches the historical hand-written bodies exactly:
# parameter k evaluates into numeric scratch k with expression base nparams + k.
for family in (
    :normal,
    :lognormal,
    :exponential,
    :gamma,
    :inversegamma,
    :weibull,
    :beta,
    :studentt,
    :laplace,
    :cauchy,
    :halfnormal,
    :halfcauchy,
    :uniform,
    :logistic,
    :gumbel,
    :pareto,
    :frechet,
    :rayleigh,
    :inversegaussian,
)
    spec = _distribution_family_spec(family)
    parameters = spec.params
    parameter_count = length(parameters)
    step_type = Symbol("Backend", spec.step, "ChoicePlanStep")
    kernel_name = Symbol("_backend_", family, "_logpdf")
    values_names = [Symbol(parameter, "_values") for parameter in parameters]

    scalar_evaluations = [:($parameter = _eval_backend_numeric_expr(env, step.$parameter)) for parameter in parameters]
    @eval function _score_backend_step!(
        step::$step_type,
        env::PlanEnvironment,
        params::AbstractVector,
        constraints::ChoiceMap,
    )
        address = _concrete_address(env, step.address)
        value = _backend_choice_value(step.parameter_slot, params, constraints, address)
        $(scalar_evaluations...)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        return $kernel_name($(parameters...), value)
    end

    scratch = Any[]
    batched_evaluations = Any[]
    for (position, (parameter, values_name)) in enumerate(zip(parameters, values_names))
        push!(scratch, :($values_name = _batched_numeric_scratch!(env, $position)))
        push!(
            batched_evaluations,
            :(_eval_backend_numeric_expr!($values_name, env, step.$parameter, $(parameter_count + position))),
        )
    end
    batched_reads = [:($values_name[batch_index]) for values_name in values_names]

    family === :laplace || @eval function _score_backend_step!(
        step::$step_type,
        totals::AbstractVector,
        env::BatchedPlanEnvironment,
        params::AbstractMatrix,
        constraints,
    )
        choice_values = env.observed_values
        $(scratch...)
        $(batched_evaluations...)
        address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
        _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
        for batch_index = 1:env.batch_size
            totals[batch_index] += $kernel_name($(batched_reads...), choice_values[batch_index])
            _backend_write_choice_binding!(env, step.binding_slot, choice_values[batch_index], batch_index)
        end
        isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
        return totals
    end

    @eval function _score_backend_observed_loop_choice!(
        step::$step_type,
        totals::AbstractVector,
        env::BatchedPlanEnvironment,
        params::AbstractMatrix,
        constraints,
        address,
    )
        $(scratch...)
        observed_values = env.observed_values
        $(batched_evaluations...)
        _batched_observed_choice_values!(observed_values, constraints, address)
        for batch_index = 1:env.batch_size
            totals[batch_index] += $kernel_name($(batched_reads...), observed_values[batch_index])
        end
        return totals
    end
end

# ---- hand-written deviations ---------------------------------------------------

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
        (isfinite(sigma) && _primal(sigma) > 0) || throw(
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

# Laplace's batched binding write stays hand-written: it historically writes
# the numeric plane directly, skipping the numeric/index/generic slot-kind
# dispatch the generated bodies perform via `_backend_write_choice_binding!`.
# Behavior-neutral stage 2 (#331) preserves that as-is.
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

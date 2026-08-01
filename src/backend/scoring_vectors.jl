function _backend_dirichlet_logpdf(alpha, x)
    length(alpha) == length(x) ||
        throw(ArgumentError("dirichlet requires matching concentration and value lengths"))
    length(alpha) >= 2 || throw(ArgumentError("dirichlet requires at least two dimensions"))

    first_alpha = float(first(alpha))
    total_alpha = zero(first_alpha)
    accumulator = zero(first_alpha)
    for alpha_value in alpha
        alpha_float = float(alpha_value)
        alpha_float > zero(alpha_float) || throw(ArgumentError("dirichlet requires alpha > 0 in every dimension"))
        total_alpha += alpha_float
        accumulator -= loggamma(alpha_float)
    end
    accumulator += loggamma(total_alpha)

    total = zero(total_alpha)
    for (value, alpha_value) in zip(x, alpha)
        value_float = float(value)
        value_float > zero(value_float) || return oftype(accumulator, -Inf)
        total += value_float
        accumulator += (float(alpha_value) - 1) * log(value_float)
    end
    abs(total - one(total)) <= sqrt(eps(float(total))) * length(x) * 16 || return oftype(accumulator, -Inf)
    return accumulator
end

function _score_backend_step!(
    step::BackendDirichletChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    alpha = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.alpha)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_dirichlet_logpdf(alpha, value)
end

function _score_backend_step!(
    step::BackendDirichletChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:step.value_length]
    alpha_values = [_batched_numeric_scratch!(env, step.value_length + index) for index = 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index = 1:step.value_length
        _eval_backend_numeric_expr!(
            alpha_values[component_index],
            env,
            step.alpha[component_index],
            2 * step.value_length + 1,
        )
    end

    for batch_index = 1:env.batch_size
        total_alpha = 0.0
        accumulator = 0.0
        for component_index = 1:step.value_length
            alpha = alpha_values[component_index][batch_index]
            alpha > 0 || throw(ArgumentError("dirichlet requires alpha > 0 in every dimension"))
            total_alpha += alpha
            accumulator -= loggamma(alpha)
        end
        accumulator += loggamma(total_alpha)

        total = 0.0
        valid = true
        for component_index = 1:step.value_length
            value = choice_values[component_index][batch_index]
            value > 0 || begin
                valid = false
                break
            end
            total += value
            accumulator += (alpha_values[component_index][batch_index] - 1) * log(value)
        end

        totals[batch_index] += valid && abs(total - 1) <= sqrt(eps(total)) * step.value_length * 16 ? accumulator : -Inf
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end

# Mirror of the compiled `logpdf(::LKJCholeskyDist, x)` over the packed
# column-major lower triangle: an out-of-support value scores -Inf, while an
# invalid concentration throws (parameter error, like dirichlet's alpha check).
function _backend_lkjcholesky_logpdf(d::Int, eta, x)
    eta > 0 || throw(ArgumentError("lkjcholesky requires a concentration eta > 0"))
    accumulator = _lkj_log_normalizing_constant(d, eta) + zero(float(x[firstindex(x)]))
    tolerance = sqrt(eps(typeof(accumulator))) * d * 16
    for row = 1:d
        diagonal = x[_packed_lower_index(d, row, row)]
        diagonal > zero(diagonal) || return oftype(accumulator, -Inf)
        sum_sqs = zero(float(diagonal))
        for col = 1:row
            entry = x[_packed_lower_index(d, row, col)]
            sum_sqs += entry * entry
        end
        abs(sum_sqs - one(sum_sqs)) <= tolerance || return oftype(accumulator, -Inf)
        if row >= 2
            accumulator += (d - row + 2 * eta - 2) * log(diagonal)
        end
    end
    return accumulator
end

function _score_backend_step!(
    step::BackendLKJCholeskyChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    eta = _eval_backend_numeric_expr(env, step.eta)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_lkjcholesky_logpdf(step.d, eta, value)
end

function _score_backend_step!(
    step::BackendLKJCholeskyChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:step.value_length]
    eta_values = _batched_numeric_scratch!(env, step.value_length + 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    _eval_backend_numeric_expr!(eta_values, env, step.eta, step.value_length + 2)

    d = step.d
    for batch_index = 1:env.batch_size
        eta = eta_values[batch_index]
        eta > 0 || throw(ArgumentError("lkjcholesky requires a concentration eta > 0"))
        accumulator = _lkj_log_normalizing_constant(d, eta)
        # support tolerance scales with the working precision (dirichlet does
        # the same via sqrt(eps(total))): rows constructed by the Float32
        # transform are only unit vectors to Float32 rounding, so the compiled
        # path's Float64 tolerance would spuriously reject valid f32 latents
        tolerance = sqrt(eps(typeof(accumulator))) * d * 16
        valid = true
        for row = 1:d
            diagonal = choice_values[_packed_lower_index(d, row, row)][batch_index]
            diagonal > 0 || begin
                valid = false
                break
            end
            sum_sqs = zero(float(diagonal))
            for col = 1:row
                entry = choice_values[_packed_lower_index(d, row, col)][batch_index]
                sum_sqs += entry * entry
            end
            sum_sqs <= 1 + tolerance || begin
                valid = false
                break
            end
            if row >= 2
                accumulator += (d - row + 2 * eta - 2) * log(diagonal)
            end
        end
        totals[batch_index] += valid ? accumulator : oftype(accumulator, -Inf)
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end

# --- broadcast (vectorized) normal observation -------------------------------

# Scalar (single-column) element-aware evaluation of a broadcast argument. A vector
# environment value contributes its `element_index`-th entry; scalars broadcast.
function _eval_backend_broadcast_element(env::PlanEnvironment, expr::BackendLiteralExpr, element_index::Int)
    return _require_numeric_value(env, expr.value, "broadcast argument")
end

function _eval_backend_broadcast_element(env::PlanEnvironment, expr::BackendSlotExpr, element_index::Int)
    value = _environment_value(env, expr.slot)
    scalar = value isa AbstractVector ? value[element_index] : value
    return _require_numeric_value(env, scalar, "broadcast argument slot")
end

function _eval_backend_broadcast_element(env::PlanEnvironment, expr::BackendPrimitiveExpr, element_index::Int)
    (expr.op === Symbol(":") || expr.op === Symbol("=>")) &&
        _backend_numeric_error(env, "broadcast argument cannot use `$(expr.op)`")
    arguments = tuple((_eval_backend_broadcast_element(env, arg, element_index) for arg in expr.arguments)...)
    return _require_numeric_value(env, _backend_primitive(expr.op, arguments...), "broadcast primitive")
end

function _eval_backend_broadcast_element(env::PlanEnvironment, expr::BackendBlockExpr, element_index::Int)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_broadcast_element(env, arg, element_index)
    end
    return value
end

function _backend_broadcast_observed_vector(value)
    values = value isa Tuple ? collect(value) : value
    values isa AbstractVector ||
        throw(ArgumentError("broadcast normal expects a vector or tuple observation value"))
    return values
end

function _score_backend_step!(
    step::BackendBroadcastNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    values = _backend_broadcast_observed_vector(_backend_observed_choice_value(constraints, address))
    n = length(values)
    n >= 1 || throw(ArgumentError("broadcast normal requires a non-empty observation at $(address)"))
    mu1 = _eval_backend_broadcast_element(env, step.mu, 1)
    sigma1 = _eval_backend_broadcast_element(env, step.sigma, 1)
    total = _backend_normal_logpdf(mu1, sigma1, float(values[1]))
    for element_index = 2:n
        mu = _eval_backend_broadcast_element(env, step.mu, element_index)
        sigma = _eval_backend_broadcast_element(env, step.sigma, element_index)
        total += _backend_normal_logpdf(mu, sigma, float(values[element_index]))
    end
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, values)
    return total
end

# Batched element-aware evaluation of a broadcast argument, reusing the numeric
# vectorized primitive kernels for a fixed data element.
function _eval_backend_broadcast_numeric!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendLiteralExpr,
    element_index::Int,
    depth::Int,
)
    fill!(destination, _require_numeric_value(env, expr.value, "batched broadcast argument"))
    return destination
end

function _eval_backend_broadcast_numeric!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendSlotExpr,
    element_index::Int,
    depth::Int,
)
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        copyto!(destination, view(env.numeric_values, expr.slot, :))
        return destination
    elseif env.index_slots[expr.slot]
        for batch_index in eachindex(destination)
            destination[batch_index] = convert(eltype(destination), env.index_values[expr.slot, batch_index])
        end
        return destination
    end
    storage = env.generic_values[expr.slot]
    for batch_index in eachindex(destination)
        value = storage[batch_index]
        scalar = value isa AbstractVector ? value[element_index] : value
        destination[batch_index] = _require_numeric_value(env, scalar, "batched broadcast slot")
    end
    return destination
end

function _eval_backend_broadcast_numeric!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendPrimitiveExpr,
    element_index::Int,
    depth::Int,
)
    (expr.op === Symbol(":") || expr.op === Symbol("=>")) &&
        _backend_numeric_error(env, "batched broadcast argument cannot use `$(expr.op)`")
    isempty(expr.arguments) && _backend_numeric_error(env, "batched broadcast primitive requires arguments")

    _eval_backend_broadcast_numeric!(destination, env, first(expr.arguments), element_index, depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_unary!(destination, env, expr.op)
    elseif expr.op === :clamp
        length(expr.arguments) == 3 ||
            _backend_numeric_error(env, "batched broadcast `clamp` expects exactly 3 arguments")
        middle = _batched_numeric_scratch!(env, depth)
        rhs = _batched_numeric_scratch!(env, depth + 1)
        _eval_backend_broadcast_numeric!(middle, env, expr.arguments[2], element_index, depth + 2)
        _eval_backend_broadcast_numeric!(rhs, env, expr.arguments[3], element_index, depth + 2)
        return _apply_backend_numeric_ternary!(destination, env, expr.op, middle, rhs)
    end

    temp = _batched_numeric_scratch!(env, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_broadcast_numeric!(temp, env, argument, element_index, depth + 1)
        _apply_backend_numeric_binary!(destination, env, expr.op, temp)
    end
    return destination
end

function _eval_backend_broadcast_numeric!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendBlockExpr,
    element_index::Int,
    depth::Int,
)
    for argument in expr.arguments
        _eval_backend_broadcast_numeric!(destination, env, argument, element_index, depth)
    end
    return destination
end

# Fetch the observed broadcast vectors for every batch column (as element-typed
# vectors), then validate a shared length across the batch.
function _batched_broadcast_observed_values(
    env::BatchedPlanEnvironment,
    address_parts::Tuple,
    constraints::ChoiceMap,
)
    T = eltype(env.numeric_values)
    observed = Vector{Vector{T}}(undef, env.batch_size)
    for batch_index = 1:env.batch_size
        address = _concrete_batched_address(address_parts, batch_index)
        found, value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_broadcast_observed_vector(value)
        observed[batch_index] = T[convert(T, float(item)) for item in values]
    end
    return observed
end

function _batched_broadcast_observed_values(
    env::BatchedPlanEnvironment,
    address_parts::Tuple,
    constraints::AbstractVector,
)
    length(constraints) == env.batch_size ||
        throw(DimensionMismatch("expected $(env.batch_size) batched constraints, got $(length(constraints))"))
    T = eltype(env.numeric_values)
    observed = Vector{Vector{T}}(undef, env.batch_size)
    for batch_index = 1:env.batch_size
        address = _concrete_batched_address(address_parts, batch_index)
        found, value = _choice_tryget_normalized(constraints[batch_index], address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values = _backend_broadcast_observed_vector(value)
        observed[batch_index] = T[convert(T, float(item)) for item in values]
    end
    return observed
end

function _broadcast_uniform_length(observed::AbstractVector{<:AbstractVector})
    isempty(observed) && return 0
    n = length(observed[1])
    n >= 1 || throw(BatchedBackendFallback("broadcast normal requires a non-empty observation"))
    for values in observed
        length(values) == n ||
            throw(BatchedBackendFallback("broadcast normal requires a shared observation length across the batch"))
    end
    return n
end

function _score_backend_step!(
    step::BackendBroadcastNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    observed = _batched_broadcast_observed_values(env, address_parts, constraints)
    n = _broadcast_uniform_length(observed)
    mu_values = _batched_numeric_scratch!(env, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    for element_index = 1:n
        _eval_backend_broadcast_numeric!(mu_values, env, step.mu, element_index, 3)
        _eval_backend_broadcast_numeric!(sigma_values, env, step.sigma, element_index, 5)
        for batch_index = 1:env.batch_size
            totals[batch_index] += _backend_normal_logpdf(
                mu_values[batch_index],
                sigma_values[batch_index],
                observed[batch_index][element_index],
            )
        end
    end
    if !isnothing(step.binding_slot)
        env.generic_slots[step.binding_slot] ||
            throw(BatchedBackendFallback("broadcast normal binding slot must be generic"))
        storage = env.generic_values[step.binding_slot]
        for batch_index = 1:env.batch_size
            storage[batch_index] = observed[batch_index]
        end
        env.assigned[step.binding_slot] = true
    end
    return totals
end

# --- family-generic broadcast scalar observation (issue #287) -----------------

function _score_backend_step!(
    step::BackendBroadcastScalarChoicePlanStep{F},
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
) where {F}
    address = _concrete_address(env, step.address)
    values = _backend_broadcast_observed_vector(_backend_observed_choice_value(constraints, address))
    n = length(values)
    n >= 1 || throw(ArgumentError("broadcast $(F) requires a non-empty observation at $(address)"))
    total = _broadcast_element_logpdf(
        Val(F), float(values[1]),
        map(arg -> _eval_backend_broadcast_element(env, arg, 1), step.arguments)...,
    )
    for element_index = 2:n
        total += _broadcast_element_logpdf(
            Val(F), float(values[element_index]),
            map(arg -> _eval_backend_broadcast_element(env, arg, element_index), step.arguments)...,
        )
    end
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, values)
    return total
end

function _score_backend_step!(
    step::BackendBroadcastScalarChoicePlanStep{F},
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
) where {F}
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    observed = _batched_broadcast_observed_values(env, address_parts, constraints)
    n = _broadcast_uniform_length(observed)
    arg_count = length(step.arguments)
    arg_values = ntuple(k -> _batched_numeric_scratch!(env, k), arg_count)
    for element_index = 1:n
        for k = 1:arg_count
            _eval_backend_broadcast_numeric!(arg_values[k], env, step.arguments[k], element_index, arg_count + k)
        end
        for batch_index = 1:env.batch_size
            totals[batch_index] += _broadcast_element_logpdf(
                Val(F),
                observed[batch_index][element_index],
                ntuple(k -> arg_values[k][batch_index], arg_count)...,
            )
        end
    end
    if !isnothing(step.binding_slot)
        env.generic_slots[step.binding_slot] ||
            throw(BatchedBackendFallback("broadcast $(F) binding slot must be generic"))
        storage = env.generic_values[step.binding_slot]
        for batch_index = 1:env.batch_size
            storage[batch_index] = observed[batch_index]
        end
        env.assigned[step.binding_slot] = true
    end
    return totals
end

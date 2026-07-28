function _eval_backend_expr(env::PlanEnvironment, expr::BackendLiteralExpr)
    return expr.value
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendLiteralExpr, batch_index::Int)
    return expr.value
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendSlotExpr)
    return _environment_value(env, expr.slot)
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendSlotExpr, batch_index::Int)
    env.assigned[expr.slot] || throw(ArgumentError("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        return env.numeric_values[expr.slot, batch_index]
    elseif env.index_slots[expr.slot]
        return env.index_values[expr.slot, batch_index]
    end
    return env.generic_values[expr.slot][batch_index]
end

function _backend_primitive(op::Symbol, args...)
    if op === Symbol(":")
        return getfield(Base, Symbol(":"))(args...)
    elseif op === Symbol("=>")
        length(args) == 2 || throw(ArgumentError("`=>` expects exactly 2 arguments"))
        return args[1] => args[2]
    elseif op === :+
        return +(args...)
    elseif op === :-
        return -(args...)
    elseif op === :*
        return *(args...)
    elseif op === :/
        return /(args...)
    elseif op === :^
        return ^(args...)
    elseif op === :%
        return %(args...)
    elseif op === :exp
        length(args) == 1 || throw(ArgumentError("`exp` expects exactly 1 argument"))
        return exp(args[1])
    elseif op === :log
        length(args) == 1 || throw(ArgumentError("`log` expects exactly 1 argument"))
        return log(args[1])
    elseif op === :log1p
        length(args) == 1 || throw(ArgumentError("`log1p` expects exactly 1 argument"))
        return log1p(args[1])
    elseif op === :sqrt
        length(args) == 1 || throw(ArgumentError("`sqrt` expects exactly 1 argument"))
        return sqrt(args[1])
    elseif op === :abs
        length(args) == 1 || throw(ArgumentError("`abs` expects exactly 1 argument"))
        return abs(args[1])
    elseif op === :min
        return min(args...)
    elseif op === :max
        return max(args...)
    elseif op === :clamp
        length(args) == 3 || throw(ArgumentError("`clamp` expects exactly 3 arguments"))
        return clamp(args...)
    end

    throw(ArgumentError("unsupported backend primitive `$op`"))
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    arguments = tuple((_eval_backend_expr(env, arg) for arg in expr.arguments)...)
    return _backend_primitive(expr.op, arguments...)
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    arguments = tuple((_eval_backend_expr(env, arg, batch_index) for arg in expr.arguments)...)
    return _backend_primitive(expr.op, arguments...)
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendTupleExpr)
    return tuple((_eval_backend_expr(env, arg) for arg in expr.arguments)...)
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendTupleExpr, batch_index::Int)
    return tuple((_eval_backend_expr(env, arg, batch_index) for arg in expr.arguments)...)
end

function _eval_backend_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_expr(env, arg)
    end
    return value
end

function _eval_backend_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_expr(env, arg, batch_index)
    end
    return value
end

# Like `_backend_index_error` below: a numeric-typed position holding a
# non-real value is a backend capability gap, and the scalar path signals the
# same fallback as the batched one so the per-column workspace scoring can
# drop to the compiled plan (which reproduces the reference semantics for such
# values, including their errors).
function _backend_numeric_error(env::PlanEnvironment, message::String)
    throw(BatchedBackendFallback(message))
end

function _backend_numeric_error(env::BatchedPlanEnvironment, message::String)
    throw(BatchedBackendFallback(message))
end

function _require_numeric_value(env, value, context::String)
    value isa Real && !(value isa Bool) && return float(value)
    _backend_numeric_error(env, "$context requires real values, got $(typeof(value))")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendLiteralExpr)
    return _require_numeric_value(env, expr.value, "backend numeric expression")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendLiteralExpr, batch_index::Int)
    return _require_numeric_value(env, expr.value, "batched backend numeric expression")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendSlotExpr)
    return _require_numeric_value(env, _environment_value(env, expr.slot), "backend numeric slot")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendSlotExpr, batch_index::Int)
    return _require_numeric_value(env, _eval_backend_expr(env, expr, batch_index), "batched backend numeric slot")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _backend_numeric_error(env, "backend numeric expression cannot use `$(expr.op)`")
    end
    arguments = tuple((_eval_backend_numeric_expr(env, arg) for arg in expr.arguments)...)
    return _require_numeric_value(env, _backend_primitive(expr.op, arguments...), "backend numeric primitive")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _backend_numeric_error(env, "batched backend numeric expression cannot use `$(expr.op)`")
    end
    arguments = tuple((_eval_backend_numeric_expr(env, arg, batch_index) for arg in expr.arguments)...)
    return _require_numeric_value(
        env,
        _backend_primitive(expr.op, arguments...),
        "batched backend numeric primitive",
    )
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendTupleExpr)
    _backend_numeric_error(env, "backend numeric expression cannot be a tuple")
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendTupleExpr, batch_index::Int)
    _backend_numeric_error(env, "batched backend numeric expression cannot be a tuple")
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_numeric_expr(env, arg)
    end
    return value
end

function _eval_backend_numeric_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_numeric_expr(env, arg, batch_index)
    end
    return value
end

# --- fused GLM linear predictor eta = intercept + sum(coef .* Matrix[:, index]) ---

function _linear_predictor_coef_matrix(env, coef, matrix, context::String)
    coef isa AbstractVector ||
        _backend_numeric_error(env, "$context coefficient must be a vector, got $(typeof(coef))")
    matrix isa AbstractMatrix ||
        _backend_numeric_error(env, "$context covariate must be a matrix, got $(typeof(matrix))")
    return coef, matrix
end

function _eval_backend_numeric_expr(env::PlanEnvironment, expr::BackendLinearPredictorExpr)
    coef, matrix = _linear_predictor_coef_matrix(
        env,
        _environment_value(env, expr.coef_slot),
        _environment_value(env, expr.matrix_slot),
        "backend linear predictor",
    )
    column = _eval_backend_index_value_expr(env, expr.index)
    accumulator = coef[1] * matrix[1, column]
    for d = 2:expr.coef_length
        accumulator += coef[d] * matrix[d, column]
    end
    isnothing(expr.intercept) && return accumulator
    return _eval_backend_numeric_expr(env, expr.intercept) + accumulator
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendLinearPredictorExpr,
    depth::Int=1,
)
    if isnothing(expr.intercept)
        fill!(destination, zero(eltype(destination)))
    else
        _eval_backend_numeric_expr!(destination, env, expr.intercept, depth)
    end
    env.assigned[expr.coef_slot] ||
        throw(BatchedBackendFallback("linear predictor coefficient slot $(expr.coef_slot) is not assigned"))
    env.assigned[expr.matrix_slot] ||
        throw(BatchedBackendFallback("linear predictor covariate slot $(expr.matrix_slot) is not assigned"))
    columns = _batched_index_scratch!(env, depth)
    _eval_backend_index_value_expr!(columns, env, expr.index, depth + 1)
    coef_storage = env.generic_values[expr.coef_slot]
    matrix_storage = env.generic_values[expr.matrix_slot]
    for batch_index = 1:env.batch_size
        coef, matrix = _linear_predictor_coef_matrix(
            env,
            coef_storage[batch_index],
            matrix_storage[batch_index],
            "batched backend linear predictor",
        )
        column = columns[batch_index]
        accumulator = destination[batch_index]
        @inbounds for d = 1:expr.coef_length
            accumulator += coef[d] * matrix[d, column]
        end
        destination[batch_index] = accumulator
    end
    return destination
end

function _batched_numeric_scratch!(env::BatchedPlanEnvironment, depth::Int)
    depth > 0 || throw(ArgumentError("batched numeric scratch depth must be positive"))
    while length(env.numeric_scratch) < depth
        push!(env.numeric_scratch, Vector{eltype(env.numeric_values)}(undef, env.batch_size))
    end
    buffer = env.numeric_scratch[depth]
    length(buffer) == env.batch_size || resize!(buffer, env.batch_size)
    return buffer
end

function _batched_index_scratch!(env::BatchedPlanEnvironment, depth::Int)
    depth > 0 || throw(ArgumentError("batched index scratch depth must be positive"))
    while length(env.index_scratch) < depth
        push!(env.index_scratch, Vector{Int}(undef, env.batch_size))
    end
    buffer = env.index_scratch[depth]
    length(buffer) == env.batch_size || resize!(buffer, env.batch_size)
    return buffer
end

function _apply_backend_numeric_unary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
)
    for batch_index in eachindex(destination)
        destination[batch_index] = _require_numeric_value(
            env,
            _backend_primitive(op, destination[batch_index]),
            "batched backend numeric primitive",
        )
    end
    return destination
end

function _apply_backend_numeric_binary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
    rhs::AbstractVector,
)
    length(destination) == length(rhs) ||
        throw(
            DimensionMismatch("expected backend numeric vectors of matching length, got $(length(destination)) and $(length(rhs))"),
        )
    for batch_index in eachindex(destination, rhs)
        destination[batch_index] = _require_numeric_value(
            env,
            _backend_primitive(op, destination[batch_index], rhs[batch_index]),
            "batched backend numeric primitive",
        )
    end
    return destination
end

function _apply_backend_numeric_ternary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
    middle::AbstractVector,
    rhs::AbstractVector,
)
    length(destination) == length(middle) == length(rhs) ||
        throw(
            DimensionMismatch(
                "expected backend numeric vectors of matching length, got $(length(destination)), $(length(middle)), and $(length(rhs))",
            ),
        )
    for batch_index in eachindex(destination, middle, rhs)
        destination[batch_index] = _require_numeric_value(
            env,
            _backend_primitive(op, destination[batch_index], middle[batch_index], rhs[batch_index]),
            "batched backend numeric primitive",
        )
    end
    return destination
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendLiteralExpr,
    depth::Int=1,
)
    fill!(destination, _require_numeric_value(env, expr.value, "batched backend numeric expression"))
    return destination
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendSlotExpr,
    depth::Int=1,
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
    _backend_numeric_error(env, "batched backend numeric slot $(expr.slot) is not numeric")
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _backend_numeric_error(env, "batched backend numeric expression cannot use `$(expr.op)`")
    end
    isempty(expr.arguments) && _backend_numeric_error(env, "batched backend numeric primitive requires arguments")

    _eval_backend_numeric_expr!(destination, env, first(expr.arguments), depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_unary!(destination, env, expr.op)
    elseif expr.op === :clamp
        length(expr.arguments) == 3 ||
            _backend_numeric_error(env, "batched backend numeric `clamp` expects exactly 3 arguments")
        middle = _batched_numeric_scratch!(env, depth)
        rhs = _batched_numeric_scratch!(env, depth + 1)
        _eval_backend_numeric_expr!(middle, env, expr.arguments[2], depth + 2)
        _eval_backend_numeric_expr!(rhs, env, expr.arguments[3], depth + 2)
        return _apply_backend_numeric_ternary!(destination, env, expr.op, middle, rhs)
    end

    temp = _batched_numeric_scratch!(env, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_numeric_expr!(temp, env, argument, depth + 1)
        _apply_backend_numeric_binary!(destination, env, expr.op, temp)
    end
    return destination
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendTupleExpr,
    depth::Int=1,
)
    _backend_numeric_error(env, "batched backend numeric expression cannot be a tuple")
end

function _eval_backend_numeric_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendBlockExpr,
    depth::Int=1,
)
    for arg in expr.arguments
        _eval_backend_numeric_expr!(destination, env, arg, depth)
    end
    return destination
end

# An index-typed slot holding a non-integer (a Pair/Tuple address argument,
# say) is a backend capability gap, not a model error: the scalar path signals
# the same fallback as the batched one so the per-column workspace scoring can
# drop to the compiled plan (which handles those values natively).
function _backend_index_error(env::PlanEnvironment, message::String)
    throw(BatchedBackendFallback(message))
end

function _backend_index_error(env::BatchedPlanEnvironment, message::String)
    throw(BatchedBackendFallback(message))
end

function _require_index_value(env, value, context::String)
    value isa Integer && return Int(value)
    _backend_index_error(env, "$context requires integer values, got $(typeof(value))")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendLiteralExpr)
    return _require_index_value(env, expr.value, "backend index expression")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendLiteralExpr, batch_index::Int)
    return _require_index_value(env, expr.value, "batched backend index expression")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendSlotExpr)
    return _require_index_value(env, _environment_value(env, expr.slot), "backend index slot")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendSlotExpr, batch_index::Int)
    return _require_index_value(env, _eval_backend_expr(env, expr, batch_index), "batched backend index slot")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    expr.op === Symbol(":") && _backend_index_error(env, "backend index value expression cannot be a range")
    expr.op === Symbol("=>") && _backend_index_error(env, "backend index value expression cannot be a pair")
    arguments = tuple((_eval_backend_index_value_expr(env, arg) for arg in expr.arguments)...)
    value = _backend_primitive(expr.op, arguments...)
    return _require_index_value(env, value, "backend index primitive")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    expr.op === Symbol(":") && _backend_index_error(env, "batched backend index value expression cannot be a range")
    expr.op === Symbol("=>") && _backend_index_error(env, "batched backend index value expression cannot be a pair")
    arguments = tuple((_eval_backend_index_value_expr(env, arg, batch_index) for arg in expr.arguments)...)
    value = _backend_primitive(expr.op, arguments...)
    return _require_index_value(env, value, "batched backend index primitive")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendTupleExpr)
    _backend_index_error(env, "backend index value expression cannot be a tuple")
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendTupleExpr, batch_index::Int)
    _backend_index_error(env, "batched backend index value expression cannot be a tuple")
end

function _eval_backend_index_value_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_index_value_expr(env, arg)
    end
    return value
end

function _eval_backend_index_value_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = _eval_backend_index_value_expr(env, arg, batch_index)
    end
    return value
end

function _apply_backend_index_unary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
)
    for batch_index in eachindex(destination)
        destination[batch_index] = _require_index_value(
            env,
            _backend_primitive(op, destination[batch_index]),
            "batched backend index primitive",
        )
    end
    return destination
end

function _apply_backend_index_binary!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    op::Symbol,
    rhs::AbstractVector,
)
    length(destination) == length(rhs) ||
        throw(DimensionMismatch("expected backend index vectors of matching length, got $(length(destination)) and $(length(rhs))"))
    for batch_index in eachindex(destination, rhs)
        destination[batch_index] = _require_index_value(
            env,
            _backend_primitive(op, destination[batch_index], rhs[batch_index]),
            "batched backend index primitive",
        )
    end
    return destination
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendLiteralExpr,
    depth::Int=1,
)
    fill!(destination, _require_index_value(env, expr.value, "batched backend index expression"))
    return destination
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendSlotExpr,
    depth::Int=1,
)
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.index_slots[expr.slot]
        copyto!(destination, view(env.index_values, expr.slot, :))
        return destination
    elseif env.numeric_slots[expr.slot]
        for batch_index in eachindex(destination)
            destination[batch_index] = _require_index_value(
                env,
                env.numeric_values[expr.slot, batch_index],
                "batched backend index slot",
            )
        end
        return destination
    end
    _backend_index_error(env, "batched backend index slot $(expr.slot) is not index-compatible")
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    expr.op === Symbol(":") && _backend_index_error(env, "batched backend index value expression cannot be a range")
    expr.op === Symbol("=>") && _backend_index_error(env, "batched backend index value expression cannot be a pair")
    isempty(expr.arguments) && _backend_index_error(env, "batched backend index primitive requires arguments")

    _eval_backend_index_value_expr!(destination, env, first(expr.arguments), depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_index_unary!(destination, env, expr.op)
    end

    temp = _batched_index_scratch!(env, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_index_value_expr!(temp, env, argument, depth + 1)
        _apply_backend_index_binary!(destination, env, expr.op, temp)
    end
    return destination
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendTupleExpr,
    depth::Int=1,
)
    _backend_index_error(env, "batched backend index value expression cannot be a tuple")
end

function _eval_backend_index_value_expr!(
    destination::AbstractVector,
    env::BatchedPlanEnvironment,
    expr::BackendBlockExpr,
    depth::Int=1,
)
    for arg in expr.arguments
        _eval_backend_index_value_expr!(destination, env, arg, depth)
    end
    return destination
end

function _eval_backend_index_iterable_expr(env::PlanEnvironment, expr::BackendPrimitiveExpr)
    expr.op === Symbol(":") || _backend_index_error(env, "backend loop iterable must lower to `:`")
    arguments = tuple((_eval_backend_index_value_expr(env, arg) for arg in expr.arguments)...)
    return getfield(Base, Symbol(":"))(arguments...)
end

function _eval_backend_index_iterable_expr(env::BatchedPlanEnvironment, expr::BackendPrimitiveExpr, batch_index::Int)
    expr.op === Symbol(":") || _backend_index_error(env, "batched backend loop iterable must lower to `:`")
    arguments = tuple((_eval_backend_index_value_expr(env, arg, batch_index) for arg in expr.arguments)...)
    return getfield(Base, Symbol(":"))(arguments...)
end

function _eval_backend_index_iterable_expr(env::PlanEnvironment, expr::BackendBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = if arg isa BackendPrimitiveExpr
            _eval_backend_index_iterable_expr(env, arg)
        else
            _backend_index_error(env, "backend loop iterable block must end in `:`")
        end
    end
    return value
end

function _eval_backend_index_iterable_expr(env::BatchedPlanEnvironment, expr::BackendBlockExpr, batch_index::Int)
    value = nothing
    for arg in expr.arguments
        value = if arg isa BackendPrimitiveExpr
            _eval_backend_index_iterable_expr(env, arg, batch_index)
        else
            _backend_index_error(env, "batched backend loop iterable block must end in `:`")
        end
    end
    return value
end

function _batched_index_iterable_reference(
    env::BatchedPlanEnvironment,
    expr::BackendPrimitiveExpr,
    depth::Int=1,
)
    expr.op === Symbol(":") || _backend_index_error(env, "batched backend loop iterable must lower to `:`")
    reserved_depth = depth + length(expr.arguments)
    argument_buffers = ntuple(length(expr.arguments)) do argument_index
        buffer = _batched_index_scratch!(env, depth + argument_index - 1)
        _eval_backend_index_value_expr!(buffer, env, expr.arguments[argument_index], reserved_depth)
        buffer
    end

    reference_arguments = ntuple(length(argument_buffers)) do argument_index
        values = argument_buffers[argument_index]
        reference_value = values[1]
        for batch_index = 2:env.batch_size
            values[batch_index] == reference_value || throw(
                BatchedBackendFallback(
                    "batched backend evaluation requires synchronized loop iterables across the batch",
                ),
            )
        end
        reference_value
    end
    return getfield(Base, Symbol(":"))(reference_arguments...)
end

function _batched_index_iterable_reference(
    env::BatchedPlanEnvironment,
    expr::BackendBlockExpr,
    depth::Int=1,
)
    value = nothing
    for arg in expr.arguments
        value = if arg isa BackendPrimitiveExpr
            _batched_index_iterable_reference(env, arg, depth)
        else
            _backend_index_error(env, "batched backend loop iterable block must end in `:`")
        end
    end
    return value
end

_concrete_backend_address_parts(env::PlanEnvironment, ::Tuple{}) = ()

function _concrete_backend_address_parts(env::PlanEnvironment, parts::Tuple)
    part = first(parts)
    head = if part isa BackendAddressLiteralPart
        part.value
    else
        _eval_backend_index_value_expr(env, part.expr)
    end
    return (head, _concrete_backend_address_parts(env, Base.tail(parts))...)
end

function _concrete_address(env::PlanEnvironment, address::BackendAddressSpec)
    return _normalize_concrete_address(_concrete_backend_address_parts(env, address.parts))
end

_concrete_backend_address_parts(env::BatchedPlanEnvironment, ::Tuple{}, batch_index::Int) = ()

function _concrete_backend_address_parts(env::BatchedPlanEnvironment, parts::Tuple, batch_index::Int)
    part = first(parts)
    head = if part isa BackendAddressLiteralPart
        part.value
    else
        _eval_backend_index_value_expr(env, part.expr, batch_index)
    end
    return (head, _concrete_backend_address_parts(env, Base.tail(parts), batch_index)...)
end

function _concrete_address(env::BatchedPlanEnvironment, address::BackendAddressSpec, batch_index::Int)
    return _normalize_concrete_address(_concrete_backend_address_parts(env, address.parts, batch_index))
end

_batched_backend_address_parts(env::BatchedPlanEnvironment, ::Tuple{}, depth::Int=1) = ()

function _batched_backend_address_parts(env::BatchedPlanEnvironment, parts::Tuple, depth::Int=1)
    part = first(parts)
    head = if part isa BackendAddressLiteralPart
        part.value
    else
        values = _batched_index_scratch!(env, depth)
        _eval_backend_index_value_expr!(values, env, part.expr, depth + 1)
        values
    end
    next_depth = part isa BackendAddressLiteralPart ? depth : depth + 1
    return (head, _batched_backend_address_parts(env, Base.tail(parts), next_depth)...)
end

_concrete_batched_address(::Tuple{}, batch_index::Int) = ()

function _concrete_batched_address(parts::Tuple, batch_index::Int)
    return _normalize_concrete_address(_concrete_batched_address_parts(parts, batch_index))
end

_concrete_batched_address_parts(::Tuple{}, batch_index::Int) = ()

function _concrete_batched_address_parts(parts::Tuple, batch_index::Int)
    source = first(parts)
    head = source isa AbstractVector ? source[batch_index] : source
    return (head, _concrete_batched_address_parts(Base.tail(parts), batch_index)...)
end

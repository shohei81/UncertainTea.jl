const BACKEND_GRADIENT_SUPPORTED_PRIMITIVES =
    Set([:+, :-, :*, :/, :^, :%, :exp, :log, :log1p, :sqrt, :abs, :min, :max, :clamp])

_backend_gradient_supported_expr(::BackendLiteralExpr) = true
_backend_gradient_supported_expr(::BackendSlotExpr) = true
_backend_gradient_supported_expr(::BackendTupleExpr) = false

# The linear-predictor node emits its own per-parameter partials (X[d, index]
# onto coef's rows, plus the intercept's own gradient); only the intercept
# sub-expr must be gradient-differentiable.
_backend_gradient_supported_expr(expr::BackendLinearPredictorExpr) =
    isnothing(expr.intercept) || _backend_gradient_supported_expr(expr.intercept)

function _backend_gradient_supported_constant_expr(expr::BackendLiteralExpr)
    return expr.value isa Real && !(expr.value isa Bool)
end

_backend_gradient_supported_constant_expr(expr::AbstractBackendExpr) = false

function _backend_gradient_supported_expr(expr::BackendPrimitiveExpr)
    if expr.op === :^ || expr.op === :%
        length(expr.arguments) == 2 || return false
        return _backend_gradient_supported_expr(expr.arguments[1]) &&
               _backend_gradient_supported_constant_expr(expr.arguments[2])
    elseif expr.op === :clamp
        length(expr.arguments) == 3 || return false
    end
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES || return false
    return all(_backend_gradient_supported_expr, expr.arguments)
end

function _backend_gradient_supported_expr(expr::BackendBlockExpr)
    return all(_backend_gradient_supported_expr, expr.arguments)
end

function _backend_gradient_supported_step(step::BackendNormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendNoncenteredNormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendLaplaceChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.scale)
end

function _backend_gradient_supported_step(step::BackendLognormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) && _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendExponentialChoicePlanStep)
    return _backend_gradient_supported_expr(step.rate)
end

function _backend_gradient_supported_step(step::BackendGammaChoicePlanStep)
    return _backend_gradient_supported_expr(step.shape) && _backend_gradient_supported_expr(step.rate)
end

function _backend_gradient_constant_index_expr(expr::BackendLiteralExpr, numeric_slots::BitVector)
    return true
end

function _backend_gradient_constant_index_expr(expr::BackendSlotExpr, numeric_slots::BitVector)
    return !numeric_slots[expr.slot]
end

function _backend_gradient_constant_index_expr(expr::BackendPrimitiveExpr, numeric_slots::BitVector)
    return all(arg -> _backend_gradient_constant_index_expr(arg, numeric_slots), expr.arguments)
end

function _backend_gradient_constant_index_expr(expr::BackendBlockExpr, numeric_slots::BitVector)
    return all(arg -> _backend_gradient_constant_index_expr(arg, numeric_slots), expr.arguments)
end

_backend_gradient_constant_index_expr(expr::AbstractBackendExpr, numeric_slots::BitVector) = false

function _backend_gradient_supported_step(step::BackendInverseGammaChoicePlanStep)
    return _backend_gradient_supported_expr(step.shape) && _backend_gradient_supported_expr(step.scale)
end

function _backend_gradient_supported_step(step::BackendWeibullChoicePlanStep)
    return _backend_gradient_supported_expr(step.shape) && _backend_gradient_supported_expr(step.scale)
end

function _backend_gradient_supported_step(step::BackendBetaChoicePlanStep)
    return _backend_gradient_supported_expr(step.alpha) && _backend_gradient_supported_expr(step.beta)
end

function _backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendBernoulliLogitChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.eta)
end

function _backend_gradient_supported_step(step::BackendGeometricChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendBinomialChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendNegativeBinomialChoicePlanStep)
    return isnothing(step.parameter_slot) &&
           _backend_gradient_supported_expr(step.successes) &&
           _backend_gradient_supported_expr(step.probability)
end

function _backend_gradient_supported_step(step::BackendCategoricalChoicePlanStep)
    return isnothing(step.parameter_slot) && all(_backend_gradient_supported_expr, step.probabilities)
end

# betabinomial: alpha/beta gradients are analytic digamma differences; trials is
# an integer index expression carrying no gradient (checked constant below).
function _backend_gradient_supported_step(step::BackendBetaBinomialChoicePlanStep)
    return isnothing(step.parameter_slot) &&
           _backend_gradient_supported_expr(step.alpha) &&
           _backend_gradient_supported_expr(step.beta)
end

# discreteuniform: no continuous parameters, so the step contributes no
# gradient; bounds are integer index expressions.
function _backend_gradient_supported_step(step::BackendDiscreteUniformChoicePlanStep)
    return isnothing(step.parameter_slot)
end

function _backend_gradient_supported_step(step::BackendPoissonChoicePlanStep)
    return isnothing(step.parameter_slot) && _backend_gradient_supported_expr(step.lambda)
end

function _backend_gradient_supported_step(step::BackendStudentTChoicePlanStep)
    return _backend_gradient_supported_expr(step.nu) &&
           _backend_gradient_supported_expr(step.mu) &&
           _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendMvNormalChoicePlanStep)
    return all(_backend_gradient_supported_expr, step.mu) &&
           all(_backend_gradient_supported_expr, step.sigma)
end

function _backend_gradient_supported_step(step::BackendTruncatedNormalChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) &&
           _backend_gradient_supported_expr(step.sigma) &&
           _backend_gradient_supported_expr(step.lower) &&
           _backend_gradient_supported_expr(step.upper)
end

# nu is a lowering-guaranteed literal constant (its d/dnu term is intractable and
# omitted); only mu/sigma/lower/upper must be gradient-differentiable.
function _backend_gradient_supported_step(step::BackendTruncatedStudentTChoicePlanStep)
    return _backend_gradient_supported_expr(step.mu) &&
           _backend_gradient_supported_expr(step.sigma) &&
           _backend_gradient_supported_expr(step.lower) &&
           _backend_gradient_supported_expr(step.upper)
end

function _backend_gradient_supported_step(step::BackendMixtureNormalChoicePlanStep)
    return all(_backend_gradient_supported_expr, step.weights) &&
           all(_backend_gradient_supported_expr, step.mus) &&
           all(_backend_gradient_supported_expr, step.sigmas)
end

# scale_tril is a generic constant matrix (zero gradient); only mu must be
# gradient-differentiable.
function _backend_gradient_supported_step(step::BackendMvNormalDenseChoicePlanStep)
    return all(_backend_gradient_supported_expr, step.mu)
end

function _backend_gradient_supported_step(step::BackendDirichletChoicePlanStep)
    return all(_backend_gradient_supported_expr, step.alpha)
end

function _backend_gradient_supported_step(step::BackendLKJCholeskyChoicePlanStep)
    return _backend_gradient_supported_expr(step.eta)
end

# the marginalize step's analytic gradient needs differentiable pmf
# expressions and a fully supported suffix (it scores the body per branch)
_backend_gradient_supported_step(step::BackendMarginalizeChoicePlanStep, numeric_slots::BitVector) =
    all(_backend_gradient_supported_expr, step.probabilities) &&
    all(inner -> _backend_gradient_supported_step(inner, numeric_slots), step.body)

function _backend_gradient_supported_step(step::BackendBroadcastNormalChoicePlanStep)
    return isnothing(step.binding_slot) &&
           _backend_gradient_supported_expr(step.mu) &&
           _backend_gradient_supported_expr(step.sigma)
end

function _backend_gradient_supported_step(step::BackendDeterministicPlanStep, numeric_slots::BitVector)
    return numeric_slots[step.binding_slot] ? _backend_gradient_supported_expr(step.expr) : true
end

function _backend_gradient_supported_step(step::BackendLoopPlanStep, numeric_slots::BitVector)
    return all(inner -> _backend_gradient_supported_step(inner, numeric_slots), step.body)
end

_backend_gradient_supported_step(step::BackendNormalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendNoncenteredNormalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendLaplaceChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendLognormalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendExponentialChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendGammaChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendInverseGammaChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendWeibullChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBetaChoicePlanStep, numeric_slots::BitVector) = _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBernoulliChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBernoulliLogitChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendGeometricChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBinomialChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step) &&
    _backend_gradient_constant_index_expr(step.trials, numeric_slots)
_backend_gradient_supported_step(step::BackendNegativeBinomialChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendCategoricalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBetaBinomialChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step) && _backend_gradient_constant_index_expr(step.trials, numeric_slots)
_backend_gradient_supported_step(step::BackendDiscreteUniformChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step) &&
    _backend_gradient_constant_index_expr(step.lower, numeric_slots) &&
    _backend_gradient_constant_index_expr(step.upper, numeric_slots)
_backend_gradient_supported_step(step::BackendPoissonChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendStudentTChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendMvNormalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendTruncatedNormalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendTruncatedStudentTChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendMixtureNormalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendMvNormalDenseChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendDirichletChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendLKJCholeskyChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)
_backend_gradient_supported_step(step::BackendBroadcastNormalChoicePlanStep, numeric_slots::BitVector) =
    _backend_gradient_supported_step(step)

# Latent vector bindings (mvnormal/dirichlet/lkjcholesky/mvnormaldense) are
# stored as generic per-column vectors with no per-row gradients, so any
# gradient-bearing expression that reads one (a broadcast argument, an
# mvnormaldense scale_tril, ...) would silently treat the latent as constant
# data. Mark those binding slots so the support gate below can reject the
# model to the per-column fallback instead.
function _mark_backend_latent_vector_bindings!(tainted::BitVector, steps)
    for step in steps
        if step isa BackendLoopPlanStep
            _mark_backend_latent_vector_bindings!(tainted, step.body)
        elseif step isa BackendMarginalizeChoicePlanStep
            _mark_backend_latent_vector_bindings!(tainted, step.body)
        elseif step isa Union{
            BackendMvNormalChoicePlanStep,
            BackendMvNormalDenseChoicePlanStep,
            BackendDirichletChoicePlanStep,
            BackendLKJCholeskyChoicePlanStep,
        }
            isnothing(step.parameter_slot) || isnothing(step.binding_slot) || (tainted[step.binding_slot] = true)
        end
    end
    return tainted
end

function _backend_step_reads_tainted_slot(step, tainted::BitVector)
    step isa BackendLoopPlanStep &&
        return any(inner -> _backend_step_reads_tainted_slot(inner, tainted), step.body)
    if step isa BackendMarginalizeChoicePlanStep
        # the suffix needs the per-step walk (a broadcast step inside it has
        # its mu/sigma reads intentionally unmarked by the slot-kind
        # collection), plus the step's own probability-expression reads
        any(inner -> _backend_step_reads_tainted_slot(inner, tainted), step.body) && return true
        referenced_numeric = falses(length(tainted))
        referenced_index = falses(length(tainted))
        referenced_generic = falses(length(tainted))
        _mark_backend_choice_address_slots!(step.address, referenced_numeric, referenced_index, referenced_generic)
        for expr in step.probabilities
            _mark_backend_generic_expr_slots!(expr, referenced_numeric, referenced_index, referenced_generic)
        end
        return any((referenced_numeric .| referenced_index .| referenced_generic) .& tainted)
    end
    referenced_numeric = falses(length(tainted))
    referenced_index = falses(length(tainted))
    referenced_generic = falses(length(tainted))
    _collect_backend_slot_kinds!(step, referenced_numeric, referenced_index, referenced_generic)
    # the broadcast step's slot-kind collection intentionally leaves its
    # mu/sigma slots unmarked (they may be scalar-numeric or generic vectors),
    # so its reads have to be collected explicitly here
    if step isa BackendBroadcastNormalChoicePlanStep
        _mark_backend_generic_expr_slots!(step.mu, referenced_numeric, referenced_index, referenced_generic)
        _mark_backend_generic_expr_slots!(step.sigma, referenced_numeric, referenced_index, referenced_generic)
    end
    # a step's own binding mark is a write, not a read
    if step isa BackendChoicePlanStep && !isnothing(step.binding_slot)
        referenced_numeric[step.binding_slot] = false
        referenced_index[step.binding_slot] = false
        referenced_generic[step.binding_slot] = false
    end
    return any((referenced_numeric .| referenced_index .| referenced_generic) .& tainted)
end

function _backend_gradient_supported(plan::BackendExecutionPlan)
    all(step -> _backend_gradient_supported_step(step, plan.numeric_slots), plan.steps) || return false
    tainted = falses(length(plan.numeric_slots))
    _mark_backend_latent_vector_bindings!(tainted, plan.steps)
    any(tainted) || return true
    return !any(step -> _backend_step_reads_tainted_slot(step, tainted), plan.steps)
end

function _batched_backend_gradient_scratch!(cache::BatchedBackendGradientCache, depth::Int)
    depth > 0 || throw(ArgumentError("batched backend gradient scratch depth must be positive"))
    element_type = eltype(cache.slot_gradients)
    parameter_count = size(cache.slot_gradients, 1)
    batch_size = size(cache.slot_gradients, 3)
    while length(cache.gradient_scratch) < depth
        push!(cache.gradient_scratch, Matrix{element_type}(undef, parameter_count, batch_size))
    end
    buffer = cache.gradient_scratch[depth]
    if size(buffer) != (parameter_count, batch_size)
        buffer = Matrix{element_type}(undef, parameter_count, batch_size)
        cache.gradient_scratch[depth] = buffer
    end
    return buffer
end

function _zero_gradient!(destination::AbstractMatrix{T}) where {T<:AbstractFloat}
    fill!(destination, zero(T))
    return destination
end

function _copy_slot_gradient!(
    destination::AbstractMatrix{T},
    slot_gradients::Array{T,3},
    slot::Int,
) where {T<:AbstractFloat}
    for batch_index in axes(destination, 2), parameter_index in axes(destination, 1)
        destination[parameter_index, batch_index] = slot_gradients[parameter_index, slot, batch_index]
    end
    return destination
end

function _store_slot_gradient!(
    slot_gradients::Array{T,3},
    slot::Int,
    source::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in axes(source, 2), parameter_index in axes(source, 1)
        slot_gradients[parameter_index, slot, batch_index] = source[parameter_index, batch_index]
    end
    return slot_gradients
end

# The step's `parameter_slot` is the slot's constrained VALUE row (what the
# constrained-matrix readers need); gradient buffers are indexed by
# UNCONSTRAINED rows, so the derivative seed routes through the cache's
# value-row -> seed-row map (issue #36). A zero map entry means the value row
# has no unconstrained counterpart (an extra simplex/cholesky value row);
# scalar and dimension-preserving vector steps can never hold such a row, so
# hitting one is a convention violation, not a model shape to fall back on.
function _fill_choice_gradient!(
    destination::AbstractMatrix{T},
    parameter_slot::Union{Nothing,Int},
    seed_rows::Vector{Int},
) where {T<:AbstractFloat}
    fill!(destination, zero(T))
    isnothing(parameter_slot) && return destination
    seed_row = seed_rows[parameter_slot]
    seed_row > 0 || error("no unconstrained seed row for value row $parameter_slot (issue #36 convention violation)")
    for batch_index in axes(destination, 2)
        destination[seed_row, batch_index] = one(T)
    end
    return destination
end

function _fill_choice_vector_gradient!(
    destination::AbstractMatrix{T},
    value_index::Union{Nothing,Int},
    component_index::Int,
    seed_rows::Vector{Int},
) where {T<:AbstractFloat}
    fill!(destination, zero(T))
    isnothing(value_index) && return destination
    value_row = value_index + component_index - 1
    seed_row = seed_rows[value_row]
    seed_row > 0 || error("no unconstrained seed row for value row $value_row (issue #36 convention violation)")
    for batch_index in axes(destination, 2)
        destination[seed_row, batch_index] = one(T)
    end
    return destination
end

function _apply_backend_numeric_gradient_unary!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    env::BatchedPlanEnvironment{T},
    op::Symbol,
) where {T<:AbstractFloat}
    if op === :-
        for batch_index in eachindex(values)
            values[batch_index] = -values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] = -gradients[parameter_index, batch_index]
        end
    elseif op === :exp
        for batch_index in eachindex(values)
            value = exp(values[batch_index])
            values[batch_index] = value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] *= value
            end
        end
    elseif op === :log
        for batch_index in eachindex(values)
            value = values[batch_index]
            values[batch_index] = log(value)
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= value
            end
        end
    elseif op === :log1p
        for batch_index in eachindex(values)
            value = values[batch_index]
            values[batch_index] = log1p(value)
            denominator = 1 + value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= denominator
            end
        end
    elseif op === :sqrt
        for batch_index in eachindex(values)
            root = sqrt(values[batch_index])
            values[batch_index] = root
            factor = 2 * root
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] /= factor
            end
        end
    elseif op === :abs
        for batch_index in eachindex(values)
            original = values[batch_index]
            values[batch_index] = abs(original)
            factor = original > 0 ? one(T) : (original < 0 ? -one(T) : zero(T))
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] *= factor
            end
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support unary primitive `$(op)`")
    end
    return values, gradients
end

function _apply_backend_numeric_gradient_binary!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    env::BatchedPlanEnvironment{T},
    op::Symbol,
    rhs_values::AbstractVector{T},
    rhs_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    if op === :+
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] += rhs_values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += rhs_gradients[parameter_index, batch_index]
        end
    elseif op === :-
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] -= rhs_values[batch_index]
        end
        for batch_index in axes(gradients, 2), parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] -= rhs_gradients[parameter_index, batch_index]
        end
    elseif op === :*
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] =
                    gradients[parameter_index, batch_index] * rhs_value +
                    lhs_value * rhs_gradients[parameter_index, batch_index]
            end
            values[batch_index] = lhs_value * rhs_value
        end
    elseif op === :/
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            denominator = rhs_value * rhs_value
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] =
                    (
                        gradients[parameter_index, batch_index] * rhs_value -
                        lhs_value * rhs_gradients[parameter_index, batch_index]
                    ) / denominator
            end
            values[batch_index] = lhs_value / rhs_value
        end
    elseif op === :^
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            exponent = rhs_values[batch_index]
            power = lhs_value ^ exponent
            factor = exponent * (lhs_value ^ (exponent - 1))
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] *= factor
            end
            values[batch_index] = power
        end
    elseif op === :%
        for batch_index in eachindex(values, rhs_values)
            values[batch_index] = values[batch_index] % rhs_values[batch_index]
        end
    elseif op === :min
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            if lhs_value < rhs_value
                values[batch_index] = lhs_value
            elseif lhs_value > rhs_value
                values[batch_index] = rhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = rhs_gradients[parameter_index, batch_index]
                end
            else
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        T(0.5) * (gradients[parameter_index, batch_index] + rhs_gradients[parameter_index, batch_index])
                end
            end
        end
    elseif op === :max
        for batch_index in eachindex(values, rhs_values)
            lhs_value = values[batch_index]
            rhs_value = rhs_values[batch_index]
            if lhs_value > rhs_value
                values[batch_index] = lhs_value
            elseif lhs_value < rhs_value
                values[batch_index] = rhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = rhs_gradients[parameter_index, batch_index]
                end
            else
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        T(0.5) * (gradients[parameter_index, batch_index] + rhs_gradients[parameter_index, batch_index])
                end
            end
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support binary primitive `$(op)`")
    end
    return values, gradients
end

function _apply_backend_numeric_gradient_ternary!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    env::BatchedPlanEnvironment{T},
    op::Symbol,
    middle_values::AbstractVector{T},
    middle_gradients::AbstractMatrix{T},
    rhs_values::AbstractVector{T},
    rhs_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    if op === :clamp
        for batch_index in eachindex(values, middle_values, rhs_values)
            lhs_value = values[batch_index]
            middle_value = middle_values[batch_index]
            rhs_value = rhs_values[batch_index]
            if lhs_value < middle_value
                values[batch_index] = middle_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = middle_gradients[parameter_index, batch_index]
                end
            elseif lhs_value > rhs_value
                values[batch_index] = rhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] = rhs_gradients[parameter_index, batch_index]
                end
            elseif lhs_value == middle_value && lhs_value == rhs_value
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        (
                            gradients[parameter_index, batch_index] +
                            middle_gradients[parameter_index, batch_index] +
                            rhs_gradients[parameter_index, batch_index]
                        ) / 3
                end
            elseif lhs_value == middle_value
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        T(0.5) * (gradients[parameter_index, batch_index] + middle_gradients[parameter_index, batch_index])
                end
            elseif lhs_value == rhs_value
                values[batch_index] = lhs_value
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] =
                        T(0.5) * (gradients[parameter_index, batch_index] + rhs_gradients[parameter_index, batch_index])
                end
            else
                values[batch_index] = lhs_value
            end
        end
    else
        _backend_numeric_error(env, "batched backend gradient does not support ternary primitive `$(op)`")
    end
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendLiteralExpr,
    depth::Int=1,
) where {T<:AbstractFloat}
    fill!(values, T(_require_numeric_value(env, expr.value, "batched backend numeric expression")))
    fill!(gradients, zero(T))
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendSlotExpr,
    depth::Int=1,
) where {T<:AbstractFloat}
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        copyto!(values, view(env.numeric_values, expr.slot, :))
        _copy_slot_gradient!(gradients, cache.slot_gradients, expr.slot)
        return values, gradients
    elseif env.index_slots[expr.slot]
        for batch_index in eachindex(values)
            values[batch_index] = T(env.index_values[expr.slot, batch_index])
        end
        fill!(gradients, zero(T))
        return values, gradients
    end
    _backend_numeric_error(env, "batched backend gradient slot $(expr.slot) is not numeric")
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendPrimitiveExpr,
    depth::Int=1,
) where {T<:AbstractFloat}
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES ||
        _backend_numeric_error(env, "batched backend gradient does not support primitive `$(expr.op)`")
    isempty(expr.arguments) && _backend_numeric_error(env, "batched backend gradient primitive requires arguments")

    _eval_backend_numeric_expr_and_gradient!(values, gradients, cache, env, first(expr.arguments), depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_gradient_unary!(values, gradients, env, expr.op)
    elseif expr.op === :clamp
        length(expr.arguments) == 3 ||
            _backend_numeric_error(env, "batched backend gradient `clamp` expects exactly 3 arguments")
        middle_values = _batched_numeric_scratch!(env, depth)
        middle_gradients = _batched_backend_gradient_scratch!(cache, depth)
        rhs_values = _batched_numeric_scratch!(env, depth + 1)
        rhs_gradients = _batched_backend_gradient_scratch!(cache, depth + 1)
        _eval_backend_numeric_expr_and_gradient!(middle_values, middle_gradients, cache, env, expr.arguments[2], depth + 2)
        _eval_backend_numeric_expr_and_gradient!(rhs_values, rhs_gradients, cache, env, expr.arguments[3], depth + 2)
        return _apply_backend_numeric_gradient_ternary!(
            values,
            gradients,
            env,
            expr.op,
            middle_values,
            middle_gradients,
            rhs_values,
            rhs_gradients,
        )
    end

    temp_values = _batched_numeric_scratch!(env, depth)
    temp_gradients = _batched_backend_gradient_scratch!(cache, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_numeric_expr_and_gradient!(temp_values, temp_gradients, cache, env, argument, depth + 1)
        _apply_backend_numeric_gradient_binary!(values, gradients, env, expr.op, temp_values, temp_gradients)
    end
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendTupleExpr,
    depth::Int=1,
) where {T<:AbstractFloat}
    _backend_numeric_error(env, "batched backend gradient expression cannot be a tuple")
end

# Fused GLM linear predictor value + gradient. The intercept contributes its own
# gradient plane (e.g. d_eta/d_alpha through alpha's slot gradient); the
# coefficient part contributes eta += coef[d]*X[d, index] and seeds
# d_eta/d_coef[d] = X[d, index] directly onto coef's UNCONSTRAINED rows via the
# cache's value-row -> seed-row map (X is constant data, zero gradient).
function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendLinearPredictorExpr,
    depth::Int=1,
) where {T<:AbstractFloat}
    if isnothing(expr.intercept)
        fill!(values, zero(T))
        fill!(gradients, zero(T))
    else
        _eval_backend_numeric_expr_and_gradient!(values, gradients, cache, env, expr.intercept, depth)
    end
    env.assigned[expr.coef_slot] ||
        throw(BatchedBackendFallback("linear predictor coefficient slot $(expr.coef_slot) is not assigned"))
    env.assigned[expr.matrix_slot] ||
        throw(BatchedBackendFallback("linear predictor covariate slot $(expr.matrix_slot) is not assigned"))
    columns = _batched_index_scratch!(env, depth)
    _eval_backend_index_value_expr!(columns, env, expr.index, depth + 1)
    coef_storage = env.generic_values[expr.coef_slot]
    matrix_storage = env.generic_values[expr.matrix_slot]
    seed_rows = cache.seed_rows
    for batch_index = 1:env.batch_size
        coef = coef_storage[batch_index]
        matrix = matrix_storage[batch_index]
        column = columns[batch_index]
        accumulator = values[batch_index]
        @inbounds for d = 1:expr.coef_length
            covariate = T(matrix[d, column])
            accumulator += T(coef[d]) * covariate
            seed_row = seed_rows[expr.coef_value_index+d-1]
            seed_row > 0 || throw(
                BatchedBackendFallback(
                    "linear predictor coefficient row $(expr.coef_value_index + d - 1) has no gradient seed row",
                ),
            )
            gradients[seed_row, batch_index] += covariate
        end
        values[batch_index] = accumulator
    end
    return values, gradients
end

function _eval_backend_numeric_expr_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendBlockExpr,
    depth::Int=1,
) where {T<:AbstractFloat}
    for argument in expr.arguments
        _eval_backend_numeric_expr_and_gradient!(values, gradients, cache, env, argument, depth)
    end
    return values, gradients
end

function _set_numeric_binding!(
    env::BatchedPlanEnvironment{T},
    slot_gradients::Array{T,3},
    slot::Int,
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    copyto!(view(env.numeric_values, slot, :), values)
    _store_slot_gradient!(slot_gradients, slot, gradients)
    env.assigned[slot] = true
    return env
end

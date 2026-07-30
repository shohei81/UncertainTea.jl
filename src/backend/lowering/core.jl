# Lowering of static models to the backend expression IR: shared machinery.

const GPU_BACKEND_SUPPORTED_PRIMITIVES = Symbol[
    Symbol(":"),
    Symbol("=>"),
    :+,
    :-,
    :*,
    :/,
    :^,
    :%,
    :exp,
    :log,
    :log1p,
    :sqrt,
    :abs,
    :min,
    :max,
    :clamp,
]

abstract type AbstractBackendExpr end
abstract type AbstractBackendAddressPart end
abstract type AbstractBackendPlanStep end
abstract type BackendChoicePlanStep <: AbstractBackendPlanStep end

struct BackendLiteralExpr{T} <: AbstractBackendExpr
    value::T
end

struct BackendSlotExpr <: AbstractBackendExpr
    slot::Int
end

struct BackendPrimitiveExpr{A<:Tuple} <: AbstractBackendExpr
    op::Symbol
    arguments::A
end

struct BackendTupleExpr{A<:Tuple} <: AbstractBackendExpr
    arguments::A
end

# A fused GLM linear predictor `eta = intercept + sum(coef .* Matrix[:, index])`
# (issues #149/#150). Unlike any other backend expr this produces PER-PARAMETER
# gradients from an indexed covariate column times a latent coefficient VECTOR:
# `d_eta/d_intercept` flows through the intercept sub-expr, and
# `d_eta/d_coef[d] = Matrix[d, index]` is seeded directly onto the coefficient's
# unconstrained parameter rows (X is constant data, zero gradient). The node is
# family-agnostic (it yields a scalar eta and its gradient) so a future
# normal/linreg lowering can reuse it; only bernoulli-logit is wired now.
#
# `intercept` is an optional numeric sub-expr (`nothing` when absent). `coef_slot`
# is the coefficient latent's environment binding slot (its stored constrained
# vector supplies the value); `coef_value_index`/`coef_length` locate its
# constrained value rows so the gradient can be seeded through the cache's
# value-row -> seed-row map. `matrix_slot` is the covariate model-argument
# (generic) slot; `index` lowers the loop iterator.
struct BackendLinearPredictorExpr{I,X<:AbstractBackendExpr} <: AbstractBackendExpr
    intercept::I
    coef_slot::Int
    coef_value_index::Int
    coef_length::Int
    matrix_slot::Int
    index::X
end

struct BackendBlockExpr{A<:Tuple} <: AbstractBackendExpr
    arguments::A
end

struct BackendAddressLiteralPart{T} <: AbstractBackendAddressPart
    value::T
end

struct BackendAddressExprPart{E<:AbstractBackendExpr} <: AbstractBackendAddressPart
    expr::E
end

struct BackendAddressSpec{P<:Tuple}
    parts::P
end

struct BackendDeterministicPlanStep{E<:AbstractBackendExpr} <: AbstractBackendPlanStep
    binding_slot::Int
    expr::E
end

struct BackendLoopPlanStep{I<:AbstractBackendExpr,B<:Tuple} <: AbstractBackendPlanStep
    iterator_slot::Int
    iterable::I
    body::B
end

struct BackendExecutionPlan{S<:Tuple}
    target::Symbol
    steps::S
    numeric_slots::BitVector
    index_slots::BitVector
    generic_slots::BitVector
end

struct BackendLoweringReport
    target::Symbol
    supported::Bool
    issues::Vector{String}
    supported_families::Vector{Symbol}
    supported_primitives::Vector{Symbol}
end

struct BackendLoweringResult
    report::BackendLoweringReport
    plan::Union{Nothing,BackendExecutionPlan}
end

struct BatchedBackendFallback <: Exception
    message::String
end

mutable struct BatchedPlanEnvironment{T<:Real}
    layout::EnvironmentLayout
    numeric_slots::BitVector
    index_slots::BitVector
    generic_slots::BitVector
    numeric_values::Matrix{T}
    index_values::Matrix{Int}
    generic_values::Vector{Vector{Any}}
    numeric_scratch::Vector{Vector{T}}
    index_scratch::Vector{Vector{Int}}
    observed_values::Vector{T}
    assigned::BitVector
    batch_size::Int
end

function BatchedPlanEnvironment(
    layout::EnvironmentLayout,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    batch_size::Int,
    ::Type{T}=Float64,
) where {T<:Real}
    numeric_values = Matrix{T}(undef, length(layout.symbols), batch_size)
    index_values = zeros(Int, length(layout.symbols), batch_size)
    generic_values = [Vector{Any}(undef, batch_size) for _ in layout.symbols]
    return BatchedPlanEnvironment(
        layout,
        copy(numeric_slots),
        copy(index_slots),
        copy(generic_slots),
        numeric_values,
        index_values,
        generic_values,
        Vector{Vector{T}}(),
        Vector{Vector{Int}}(),
        Vector{T}(undef, batch_size),
        falses(length(layout.symbols)),
        batch_size,
    )
end

function Base.show(io::IO, report::BackendLoweringReport)
    print(
        io,
        "BackendLoweringReport(target=",
        report.target,
        ", supported=",
        report.supported,
        ", issues=",
        length(report.issues),
        ")",
    )
end

function Base.show(io::IO, plan::BackendExecutionPlan)
    print(
        io,
        "BackendExecutionPlan(target=",
        plan.target,
        ", steps=",
        length(plan.steps),
        ", numeric_slots=",
        count(identity, plan.numeric_slots),
        ", index_slots=",
        count(identity, plan.index_slots),
        ")",
    )
end

function Base.showerror(io::IO, err::BatchedBackendFallback)
    print(io, err.message)
end

function _backend_issue!(issues::Vector{String}, message::String)
    push!(issues, message)
    return nothing
end

function _supported_backend_primitive(op::Symbol)
    return op in GPU_BACKEND_SUPPORTED_PRIMITIVES
end

function _supported_backend_distribution(family::Symbol)
    return family in GPU_BACKEND_SUPPORTED_DISTRIBUTIONS
end

# Per-model plan-BUILD de-specialization (issue #155 part 2). Like the compiled
# plan builders in evaluator.jl, backend lowering runs ONCE per (model,
# signature) -- its `BackendLoweringResult` is memoized in
# `ResolvedSignaturePlan.backend_lowering` -- but used to recompile for every
# distinct `TeaModel{M,F,S}` because the model's `impl` type `F` is unique per
# `@tea` model. `model` is a DATA source here (evaluation module, name, IR); it
# never enters the type of the lowered backend plan (that follows the step
# structure). `@nospecialize(model)` lets each lowering method compile once
# against the abstract `TeaModel` and be reused across models, leaving the
# lowered plan and every numeric result unchanged. The device/backed hot
# scoring kernels consume the concrete lowered plan and are untouched.
function _backend_primitive_name(@nospecialize(model::TeaModel), callee)
    if callee isa Symbol
        return _supported_backend_primitive(callee) ? callee : nothing
    elseif callee isa GlobalRef
        if (callee.mod === Base || callee.mod === _evaluation_module(model)) &&
           _supported_backend_primitive(callee.name)
            return callee.name
        end
    end
    return nothing
end

function _backend_lower_expr(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    expr,
    issues::Vector{String},
    context::String,
)
    if expr isa QuoteNode
        return BackendLiteralExpr(expr.value)
    elseif expr isa Symbol
        slot = _environment_slot(layout, expr)
        if !isnothing(slot)
            return BackendSlotExpr(slot)
        end
        _backend_issue!(issues, "unsupported free symbol `$expr` in $context")
        return nothing
    elseif expr isa GlobalRef
        value = getfield(expr.mod, expr.name)
        if value isa Number || value isa Bool || value isa Char || value isa String
            return BackendLiteralExpr(value)
        end
        _backend_issue!(issues, "unsupported global reference `$(expr.mod).$(expr.name)` in $context")
        return nothing
    elseif expr isa LineNumberNode
        return nothing
    elseif expr isa Expr
        if expr.head == :call
            op = _backend_primitive_name(model, expr.args[1])
            if isnothing(op)
                _backend_issue!(issues, "unsupported call `$(expr.args[1])` in $context")
                return nothing
            end
            arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args[2:end])
            any(isnothing, arguments) && return nothing
            return BackendPrimitiveExpr(op, tuple(arguments...))
        elseif expr.head == :tuple
            arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args)
            any(isnothing, arguments) && return nothing
            return BackendTupleExpr(tuple(arguments...))
        elseif expr.head == :block
            arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args)
            filtered = Tuple(arg for arg in arguments if !isnothing(arg))
            return BackendBlockExpr(filtered)
        end

        _backend_issue!(issues, "unsupported expression head `$(expr.head)` in $context")
        return nothing
    elseif expr isa Number || expr isa Bool || expr isa Char || expr isa String
        return BackendLiteralExpr(expr)
    end

    _backend_issue!(issues, "unsupported literal `$(repr(expr))` in $context")
    return nothing
end

function _backend_lower_address(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    address::AddressSpec,
    issues::Vector{String},
)
    parts = map(address.parts) do part
        if part isa AddressLiteralPart
            BackendAddressLiteralPart(part.value)
        else
            lowered = _backend_lower_expr(model, layout, part.value, issues, "address")
            isnothing(lowered) && return nothing
            BackendAddressExprPart(lowered)
        end
    end
    any(isnothing, parts) && return nothing
    return BackendAddressSpec(tuple(parts...))
end

# Scalar family steps store their slot's VALUE row (issue #36): that is what
# the batched constrained-matrix readers index with. Gradient seeding, which
# needs the UNCONSTRAINED row, resolves it through the gradient cache's
# value-row -> seed-row map, so dimension-changing predecessors
# (simplex/cholesky) no longer force a fallback.
# The parameter layout here is the layout of the PLAN being lowered (issue #95,
# PR-4): a signature-specific plan classifies latents/observations differently
# from the model's syntactic default, so the slot indices carried by the plan's
# steps must be resolved against the plan's own layout, not `parameterlayout(model)`.
function _backend_scalar_parameter_row(parameter_layout::ParameterLayout, parameter_slot, issues::Vector{String})
    isnothing(parameter_slot) && return nothing, true
    slot = parameter_layout.slots[parameter_slot]
    return slot.value_index, true
end

function _backend_lower_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    if step.rhs isa BroadcastDistributionSpec
        return _backend_lower_broadcast_normal_choice_step(model, layout, step, issues)
    end
    step.rhs isa DistributionSpec || begin
        _backend_issue!(issues, "unsupported choice RHS $(typeof(step.rhs)) in backend lowering")
        return nothing
    end
    _supported_backend_distribution(step.rhs.family) || begin
        _backend_issue!(issues, "unsupported distribution family `$(step.rhs.family)` in backend lowering")
        return nothing
    end
    step.rhs.marginalize === :none || begin
        # unreachable through _backend_lower_steps (which routes flagged
        # choices to the suffix-consuming marginalize lowering before this
        # dispatch) and loop bodies reject the flag at plan build; kept as a
        # defense against future call paths
        _backend_issue!(
            issues,
            "marginalize=:enumerate choices must lower through the suffix-consuming marginalize path",
        )
        return nothing
    end
    parameter_row, parameter_row_ok = _backend_scalar_parameter_row(parameter_layout, step.parameter_slot, issues)
    parameter_row_ok || return nothing
    if step.rhs.reparam === :noncentered && step.rhs.family !== :normal
        _backend_issue!(
            issues,
            "reparam=:noncentered is lowered for normal only; $(step.rhs.family) is pending (docs/noncentered-reparam.md)",
        )
        return nothing
    end
    step.rhs.family === :mvnormal && return _backend_lower_mvnormal_choice_step(model, layout, parameter_layout, step, issues)
    step.rhs.family === :dirichlet && return _backend_lower_dirichlet_choice_step(model, layout, parameter_layout, step, issues)
    step.rhs.family === :lkjcholesky &&
        return _backend_lower_lkjcholesky_choice_step(model, layout, parameter_layout, step, issues)
    step.rhs.family === :truncatednormal &&
        return _backend_lower_truncatednormal_choice_step(model, layout, parameter_layout, step, issues)
    step.rhs.family === :truncatedstudentt &&
        return _backend_lower_truncatedstudentt_choice_step(model, layout, parameter_layout, step, issues)
    step.rhs.family === :mixture && return _backend_lower_mixture_choice_step(model, layout, parameter_layout, step, issues)
    step.rhs.family === :mvnormaldense &&
        return _backend_lower_mvnormaldense_choice_step(model, layout, parameter_layout, step, issues)
    step.rhs.family === :bernoullilogit &&
        return _backend_lower_bernoullilogit_choice_step(model, layout, parameter_layout, step, issues)

    address = _backend_lower_address(model, layout, step.address, issues)
    arguments = map(arg -> _backend_lower_expr(model, layout, arg, issues, "distribution argument"), step.rhs.arguments)
    (isnothing(address) || any(isnothing, arguments)) && return nothing
    if step.rhs.family === :normal
        length(arguments) == 2 || begin
            _backend_issue!(issues, "normal expects exactly 2 backend arguments")
            return nothing
        end
        step.rhs.reparam === :noncentered && return BackendNoncenteredNormalChoicePlanStep(
            step.binding_slot,
            address,
            arguments[1],
            arguments[2],
            parameter_row,
        )
        return BackendNormalChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :laplace
        length(arguments) == 2 || begin
            _backend_issue!(issues, "laplace expects exactly 2 backend arguments")
            return nothing
        end
        return BackendLaplaceChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :lognormal
        length(arguments) == 2 || begin
            _backend_issue!(issues, "lognormal expects exactly 2 backend arguments")
            return nothing
        end
        return BackendLognormalChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :exponential
        length(arguments) == 1 || begin
            _backend_issue!(issues, "exponential expects exactly 1 backend argument")
            return nothing
        end
        return BackendExponentialChoicePlanStep(step.binding_slot, address, arguments[1], parameter_row)
    elseif step.rhs.family === :gamma
        length(arguments) == 2 || begin
            _backend_issue!(issues, "gamma expects exactly 2 backend arguments")
            return nothing
        end
        return BackendGammaChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :inversegamma
        length(arguments) == 2 || begin
            _backend_issue!(issues, "inversegamma expects exactly 2 backend arguments")
            return nothing
        end
        return BackendInverseGammaChoicePlanStep(
            step.binding_slot,
            address,
            arguments[1],
            arguments[2],
            parameter_row,
        )
    elseif step.rhs.family === :weibull
        length(arguments) == 2 || begin
            _backend_issue!(issues, "weibull expects exactly 2 backend arguments")
            return nothing
        end
        return BackendWeibullChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :beta
        length(arguments) == 2 || begin
            _backend_issue!(issues, "beta expects exactly 2 backend arguments")
            return nothing
        end
        return BackendBetaChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :bernoulli
        length(arguments) == 1 || begin
            _backend_issue!(issues, "bernoulli expects exactly 1 backend argument")
            return nothing
        end
        return BackendBernoulliChoicePlanStep(step.binding_slot, address, arguments[1], parameter_row)
    elseif step.rhs.family === :binomial
        length(arguments) == 2 || begin
            _backend_issue!(issues, "binomial expects exactly 2 backend arguments")
            return nothing
        end
        return BackendBinomialChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :betabinomial
        length(arguments) == 3 || begin
            _backend_issue!(issues, "betabinomial expects exactly 3 backend arguments")
            return nothing
        end
        return BackendBetaBinomialChoicePlanStep(
            step.binding_slot,
            address,
            arguments[1],
            arguments[2],
            arguments[3],
            parameter_row,
        )
    elseif step.rhs.family === :discreteuniform
        length(arguments) == 2 || begin
            _backend_issue!(issues, "discreteuniform expects exactly 2 backend arguments")
            return nothing
        end
        return BackendDiscreteUniformChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :geometric
        length(arguments) == 1 || begin
            _backend_issue!(issues, "geometric expects exactly 1 backend argument")
            return nothing
        end
        return BackendGeometricChoicePlanStep(step.binding_slot, address, arguments[1], parameter_row)
    elseif step.rhs.family === :negativebinomial
        length(arguments) == 2 || begin
            _backend_issue!(issues, "negativebinomial expects exactly 2 backend arguments")
            return nothing
        end
        return BackendNegativeBinomialChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :categorical
        isempty(arguments) && begin
            _backend_issue!(issues, "categorical expects at least 1 backend argument")
            return nothing
        end
        return BackendCategoricalChoicePlanStep(step.binding_slot, address, tuple(arguments...), parameter_row)
    elseif step.rhs.family === :poisson
        length(arguments) == 1 || begin
            _backend_issue!(issues, "poisson expects exactly 1 backend argument")
            return nothing
        end
        return BackendPoissonChoicePlanStep(step.binding_slot, address, arguments[1], parameter_row)
    elseif step.rhs.family === :studentt
        length(arguments) == 3 || begin
            _backend_issue!(issues, "studentt expects exactly 3 backend arguments")
            return nothing
        end
        return BackendStudentTChoicePlanStep(
            step.binding_slot,
            address,
            arguments[1],
            arguments[2],
            arguments[3],
            parameter_row,
        )
    elseif step.rhs.family === :cauchy
        length(arguments) == 2 || begin
            _backend_issue!(issues, "cauchy expects exactly 2 backend arguments")
            return nothing
        end
        return BackendCauchyChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :halfnormal
        length(arguments) == 1 || begin
            _backend_issue!(issues, "halfnormal expects exactly 1 backend argument")
            return nothing
        end
        return BackendHalfNormalChoicePlanStep(step.binding_slot, address, arguments[1], parameter_row)
    elseif step.rhs.family === :halfcauchy
        length(arguments) == 1 || begin
            _backend_issue!(issues, "halfcauchy expects exactly 1 backend argument")
            return nothing
        end
        return BackendHalfCauchyChoicePlanStep(step.binding_slot, address, arguments[1], parameter_row)
    elseif step.rhs.family === :uniform
        length(arguments) == 2 || begin
            _backend_issue!(issues, "uniform expects exactly 2 backend arguments")
            return nothing
        end
        # A latent uniform draws through a bounded parameter transform that the
        # batched backend unconstrained-transform layer does not implement (the
        # same limitation the truncated families hit), so fall back for latents.
        # Observations stay backend-native.
        isnothing(step.parameter_slot) || begin
            _backend_issue!(issues, "latent uniform is not supported in backend lowering (bounded transform)")
            return nothing
        end
        return BackendUniformChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :logistic
        length(arguments) == 2 || begin
            _backend_issue!(issues, "logistic expects exactly 2 backend arguments")
            return nothing
        end
        return BackendLogisticChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :gumbel
        length(arguments) == 2 || begin
            _backend_issue!(issues, "gumbel expects exactly 2 backend arguments")
            return nothing
        end
        return BackendGumbelChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :pareto
        length(arguments) == 2 || begin
            _backend_issue!(issues, "pareto expects exactly 2 backend arguments")
            return nothing
        end
        # A latent pareto draws through a lower-bounded parameter transform that
        # the batched backend unconstrained-transform layer does not implement
        # (the same limitation uniform/truncated families hit), so fall back for
        # latents. Observations stay backend-native.
        isnothing(step.parameter_slot) || begin
            _backend_issue!(issues, "latent pareto is not supported in backend lowering (lower-bounded transform)")
            return nothing
        end
        return BackendParetoChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :frechet
        length(arguments) == 2 || begin
            _backend_issue!(issues, "frechet expects exactly 2 backend arguments")
            return nothing
        end
        return BackendFrechetChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    elseif step.rhs.family === :rayleigh
        length(arguments) == 1 || begin
            _backend_issue!(issues, "rayleigh expects exactly 1 backend argument")
            return nothing
        end
        return BackendRayleighChoicePlanStep(step.binding_slot, address, arguments[1], parameter_row)
    elseif step.rhs.family === :inversegaussian
        length(arguments) == 2 || begin
            _backend_issue!(issues, "inversegaussian expects exactly 2 backend arguments")
            return nothing
        end
        return BackendInverseGaussianChoicePlanStep(step.binding_slot, address, arguments[1], arguments[2], parameter_row)
    end

    _backend_issue!(issues, "unsupported distribution family `$(step.rhs.family)` in backend lowering")
    return nothing
end

function _backend_lower_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::DeterministicPlanStep,
    issues::Vector{String},
)
    expr = _backend_lower_expr(model, layout, step.expr, issues, "deterministic assignment")
    isnothing(expr) && return nothing
    return BackendDeterministicPlanStep(step.binding_slot, expr)
end

function _backend_lower_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::LoopPlanStep,
    issues::Vector{String},
)
    iterable = _backend_lower_expr(model, layout, step.iterable, issues, "loop iterable")
    body = map(inner -> _backend_lower_step(model, layout, parameter_layout, inner, issues), step.body)
    (isnothing(iterable) || any(isnothing, body)) && return nothing
    return BackendLoopPlanStep(step.iterator_slot, iterable, tuple(body...))
end

function _mark_backend_numeric_slot!(
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    slot::Int,
)
    (index_slots[slot] || generic_slots[slot]) && return false
    numeric_slots[slot] && return false
    numeric_slots[slot] = true
    return true
end

function _mark_backend_index_slot!(
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    slot::Int,
)
    changed = false
    if !index_slots[slot]
        index_slots[slot] = true
        changed = true
    end
    if numeric_slots[slot]
        numeric_slots[slot] = false
        changed = true
    end
    if generic_slots[slot]
        generic_slots[slot] = false
        changed = true
    end
    return changed
end

function _mark_backend_generic_slot!(
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
    slot::Int,
)
    changed = false
    index_slots[slot] && return false
    if !generic_slots[slot]
        generic_slots[slot] = true
        changed = true
    end
    if numeric_slots[slot]
        numeric_slots[slot] = false
        changed = true
    end
    return changed
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendLiteralExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    if expr.op === Symbol(":") || expr.op === Symbol("=>")
        _mark_backend_index_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
        return nothing
    end
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_numeric_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_numeric_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

# A linear-predictor node reads its intercept (numeric), the covariate matrix
# (generic model arg), and the loop index. The coefficient latent binding slot
# is deliberately NOT marked as a read: it is a latent-vector binding marked
# generic by its own prior step, and this node supplies coef's per-parameter
# gradient explicitly, so the latent-vector taint gate (manual_gradient.jl) must
# not see it as a plain (gradient-blind) read.
function _mark_backend_numeric_expr_slots!(
    expr::BackendLinearPredictorExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    isnothing(expr.intercept) ||
        _mark_backend_numeric_expr_slots!(expr.intercept, numeric_slots, index_slots, generic_slots)
    _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, expr.matrix_slot)
    _mark_backend_index_expr_slots!(expr.index, numeric_slots, index_slots, generic_slots)
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendLiteralExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_index_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_index_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_index_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_index_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendLiteralExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendSlotExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, expr.slot)
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendPrimitiveExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendTupleExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _mark_backend_generic_expr_slots!(
    expr::BackendBlockExpr,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for arg in expr.arguments
        _mark_backend_generic_expr_slots!(arg, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _backend_expr_is_numeric(expr::BackendLiteralExpr)
    return expr.value isa Real && !(expr.value isa Bool)
end

_backend_expr_is_numeric(expr::BackendSlotExpr) = true

function _backend_expr_is_numeric(expr::BackendPrimitiveExpr)
    (expr.op === Symbol(":") || expr.op === Symbol("=>")) && return false
    return all(_backend_expr_is_numeric, expr.arguments)
end

function _backend_expr_is_numeric(expr::BackendTupleExpr)
    return false
end

function _backend_expr_is_numeric(expr::BackendBlockExpr)
    return all(_backend_expr_is_numeric, expr.arguments)
end

function _mark_backend_choice_address_slots!(
    address::BackendAddressSpec,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    for part in address.parts
        if part isa BackendAddressExprPart
            _mark_backend_index_expr_slots!(part.expr, numeric_slots, index_slots, generic_slots)
        end
    end
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendDeterministicPlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    if _backend_expr_is_numeric(step.expr)
        _mark_backend_numeric_expr_slots!(step.expr, numeric_slots, index_slots, generic_slots)
        _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    else
        _mark_backend_generic_expr_slots!(step.expr, numeric_slots, index_slots, generic_slots)
        _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    end
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendLoopPlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.iterator_slot)
    _mark_backend_index_expr_slots!(step.iterable, numeric_slots, index_slots, generic_slots)
    for inner in step.body
        _collect_backend_slot_kinds!(inner, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

_backend_iterator_only_index_expr(::BackendLiteralExpr, iterator_slot::Int) = true
_backend_iterator_only_index_expr(expr::BackendSlotExpr, iterator_slot::Int) = expr.slot == iterator_slot

function _backend_iterator_only_index_expr(expr::BackendPrimitiveExpr, iterator_slot::Int)
    expr.op === Symbol(":") && return false
    expr.op === Symbol("=>") && return false
    return all(arg -> _backend_iterator_only_index_expr(arg, iterator_slot), expr.arguments)
end

_backend_iterator_only_index_expr(::BackendTupleExpr, iterator_slot::Int) = false

function _backend_iterator_only_index_expr(expr::BackendBlockExpr, iterator_slot::Int)
    return all(arg -> _backend_iterator_only_index_expr(arg, iterator_slot), expr.arguments)
end

function _backend_iterator_only_address(address::BackendAddressSpec, iterator_slot::Int)
    for part in address.parts
        if part isa BackendAddressExprPart &&
           !_backend_iterator_only_index_expr(part.expr, iterator_slot)
            return false
        end
    end
    return true
end

function _backend_loop_observed_choice(step::BackendLoopPlanStep)
    length(step.body) == 1 || return nothing
    choice = first(step.body)
    choice isa BackendChoicePlanStep || return nothing
    choice isa BackendMvNormalChoicePlanStep && return nothing
    choice isa BackendMarginalizeChoicePlanStep && return nothing
    choice isa BackendMvNormalDenseChoicePlanStep && return nothing
    choice isa BackendLKJCholeskyChoicePlanStep && return nothing
    choice isa BackendTruncatedNormalChoicePlanStep && return nothing
    choice isa BackendTruncatedStudentTChoicePlanStep && return nothing
    choice isa BackendMixtureNormalChoicePlanStep && return nothing
    isnothing(choice.parameter_slot) || return nothing
    isnothing(choice.binding_slot) || return nothing
    _backend_iterator_only_address(choice.address, step.iterator_slot) || return nothing
    return choice
end

function _derive_backend_slot_kinds(layout::EnvironmentLayout, steps::Tuple)
    numeric_slots = falses(length(layout.symbols))
    index_slots = falses(length(layout.symbols))
    generic_slots = falses(length(layout.symbols))
    for step in steps
        _collect_backend_slot_kinds!(step, numeric_slots, index_slots, generic_slots)
    end
    return numeric_slots, index_slots, generic_slots
end

_backend_marginalized_choice_step(step) =
    step isa ChoicePlanStep && step.rhs isa DistributionSpec && step.rhs.marginalize === :enumerate

# Sequential lowering: a marginalize=:enumerate choice consumes the remaining
# steps as its continuation (docs/discrete-enumeration.md), so the plan's tail
# becomes the marginalize step's body and nested flagged latents nest as
# nested steps. Everything else lowers step-by-step as before.
function _backend_lower_steps(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    steps::AbstractVector,
    issues::Vector{String},
)
    lowered = Any[]
    for (position, step) in enumerate(steps)
        if _backend_marginalized_choice_step(step)
            push!(
                lowered,
                _backend_lower_marginalize_choice_step(
                    model,
                    layout,
                    parameter_layout,
                    step,
                    steps[(position+1):end],
                    issues,
                ),
            )
            return lowered
        end
        push!(lowered, _backend_lower_step(model, layout, parameter_layout, step, issues))
    end
    return lowered
end

function _lower_backend_execution_plan(@nospecialize(model::TeaModel); target::Symbol=:gpu)
    return _lower_backend_execution_plan(model, executionplan(model); target=target)
end

# Lower an arbitrary execution plan (the model's syntactic default OR a
# conditioning-signature-specific plan; see `_signature_execution_plan`). The
# latent/observation split -- and therefore whether a choice step carries a
# `parameter_slot` -- is baked into `plan` already, so the backend classification
# stays consistent with the CPU signature layout by construction (issue #95).
function _lower_backend_execution_plan(@nospecialize(model::TeaModel), plan::ExecutionPlan; target::Symbol=:gpu)
    target === :gpu || throw(ArgumentError("only :gpu backend lowering is currently supported"))
    if model.branchful
        # a (dynamic-mode) body with if/else control flow: the linear plan
        # holds both branches, so no backend lowering of it can be trusted
        report = BackendLoweringReport(
            target,
            false,
            [
                "model contains branchful control flow (if/else); the linear execution plan " *
                "cannot represent it -- use generate/assess",
            ],
            copy(GPU_BACKEND_SUPPORTED_DISTRIBUTIONS),
            copy(GPU_BACKEND_SUPPORTED_PRIMITIVES),
        )
        return BackendLoweringResult(report, nothing)
    end
    issues = String[]
    steps = _backend_lower_steps(model, plan.environment_layout, plan.parameter_layout, plan.steps, issues)
    report = BackendLoweringReport(
        target,
        isempty(issues),
        issues,
        copy(GPU_BACKEND_SUPPORTED_DISTRIBUTIONS),
        copy(GPU_BACKEND_SUPPORTED_PRIMITIVES),
    )
    if any(isnothing, steps)
        return BackendLoweringResult(report, nothing)
    end
    if !_backend_noncentered_dependencies_ok!(issues, tuple(steps...))
        return BackendLoweringResult(
            BackendLoweringReport(target, false, issues, report.supported_families, report.supported_primitives),
            nothing,
        )
    end
    numeric_slots, index_slots, generic_slots = _derive_backend_slot_kinds(plan.environment_layout, tuple(steps...))
    return BackendLoweringResult(
        report,
        BackendExecutionPlan(target, tuple(steps...), numeric_slots, index_slots, generic_slots),
    )
end


# A noncentered location/scale may only depend on model arguments and slotted
# latents -- the CPU plan-walk transform rejects anything else, so the z-space
# backend plan must too (codex review on PR #35). Taint propagates from
# slotless choice bindings through deterministic bindings.
_backend_expr_uses_tainted(expr::BackendLiteralExpr, tainted::BitSet) = false
_backend_expr_uses_tainted(expr::BackendSlotExpr, tainted::BitSet) = expr.slot in tainted
_backend_expr_uses_tainted(expr::BackendPrimitiveExpr, tainted::BitSet) =
    any(_backend_expr_uses_tainted(arg, tainted) for arg in expr.arguments)
_backend_expr_uses_tainted(expr::BackendTupleExpr, tainted::BitSet) =
    any(_backend_expr_uses_tainted(arg, tainted) for arg in expr.arguments)
_backend_expr_uses_tainted(expr::BackendBlockExpr, tainted::BitSet) =
    any(_backend_expr_uses_tainted(arg, tainted) for arg in expr.arguments)
_backend_expr_uses_tainted(expr, tainted::BitSet) = false

function _backend_noncentered_dependencies_ok!(issues::Vector{String}, steps::Tuple, tainted::BitSet=BitSet())
    ok = true
    for step in steps
        if step isa BackendDeterministicPlanStep
            _backend_expr_uses_tainted(step.expr, tainted) && push!(tainted, step.binding_slot)
        elseif step isa BackendLoopPlanStep
            ok &= _backend_noncentered_dependencies_ok!(issues, step.body, tainted)
        elseif step isa BackendNoncenteredNormalChoicePlanStep
            if _backend_expr_uses_tainted(step.mu, tainted) || _backend_expr_uses_tainted(step.sigma, tainted)
                _backend_issue!(
                    issues,
                    "reparam=:noncentered location/scale depends on a choice without a parameter slot; " *
                    "the CPU transform rejects this, so backend lowering does too",
                )
                ok = false
            end
        elseif step isa BackendMarginalizeChoicePlanStep
            # the enumerated binding is a slotless choice, and the audit must
            # keep walking the suffix the step owns
            isnothing(step.binding_slot) || push!(tainted, step.binding_slot)
            ok &= _backend_noncentered_dependencies_ok!(issues, step.body, tainted)
        elseif step isa BackendChoicePlanStep
            if isnothing(step.parameter_slot) && !isnothing(step.binding_slot)
                push!(tainted, step.binding_slot)
            end
        end
    end
    return ok
end

function _backend_lowering(@nospecialize(model::TeaModel); target::Symbol=:gpu)
    target === :gpu || throw(ArgumentError("only :gpu backend lowering is currently supported"))
    cached = model.backend_cache[]
    if isnothing(cached)
        cached = _lower_backend_execution_plan(model; target=target)
        model.backend_cache[] = cached
    end
    return cached::BackendLoweringResult
end

function _backend_execution_plan(@nospecialize(model::TeaModel); target::Symbol=:gpu)
    return _backend_lowering(model; target=target).plan
end

# Backend lowering for a conditioning-signature-specific plan (issue #95, PR-4).
# Lowered from `resolved.plan`, so a bound-but-constrained choice is an observation
# and a bound-but-unconstrained choice is a latent, exactly matching the CPU
# signature layout. Memoized on the resolved plan itself (which lives in the
# model's `signature_cache`), so repeated batched/device runs at the same
# conditioning reuse the lowering.
function _signature_backend_lowering(
    @nospecialize(model::TeaModel),
    resolved::ResolvedSignaturePlan;
    target::Symbol=:gpu,
)
    target === :gpu || throw(ArgumentError("only :gpu backend lowering is currently supported"))
    cached = resolved.backend_lowering[]
    if isnothing(cached)
        cached = _lower_backend_execution_plan(model, resolved.plan; target=target)
        resolved.backend_lowering[] = cached
    end
    return cached::BackendLoweringResult
end

_signature_backend_plan(@nospecialize(model::TeaModel), resolved::ResolvedSignaturePlan; target::Symbol=:gpu) =
    _signature_backend_lowering(model, resolved; target=target).plan

function backend_report(@nospecialize(model::TeaModel); target::Symbol=:gpu)
    return _backend_lowering(model; target=target).report
end

function backend_execution_plan(@nospecialize(model::TeaModel); target::Symbol=:gpu)
    result = _backend_lowering(model; target=target)
    isnothing(result.plan) && throw(
        ArgumentError("model $(model.name) is not supported for $(target) backend lowering; see backend_report(model)"),
    )
    return result.plan
end

# Lower a truncated-family bound expression. Static bounds (numeric literals and
# `Inf`/`-Inf`, which arrive as the symbol `:Inf` in the captured AST) become
# literal backend expressions; anything else (a latent-flowing bound) reuses the
# standard numeric lowering.
function _backend_lower_bound_expr(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    expr,
    issues::Vector{String},
    context::String,
)
    static_value = _static_bound_value(expr)
    isnothing(static_value) || return BackendLiteralExpr(float(static_value))
    return _backend_lower_expr(model, layout, expr, issues, context)
end

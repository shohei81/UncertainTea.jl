# Lowering of static models to the backend expression IR: discrete families (bernoulli, poisson, geometric, binomial, negativebinomial, categorical).

struct BackendBernoulliChoicePlanStep{P<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probability::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendCategoricalChoicePlanStep{P<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probabilities::P
    parameter_slot::Union{Nothing,Int}
end

# Logit-parameterized Bernoulli observation (issues #149/#150). `eta` is either a
# fused `BackendLinearPredictorExpr` (the GLM linear predictor) or a generic
# numeric backend expr (so `bernoullilogit(scalar_expr)` still lowers). Scored in
# the stable `x*eta - log1p(exp(eta))` form with the well-conditioned
# `d/d_eta = x - logistic(eta)` gradient.
struct BackendBernoulliLogitChoicePlanStep{E,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    eta::E
    parameter_slot::Union{Nothing,Int}
end

struct BackendPoissonChoicePlanStep{L<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    lambda::L
    parameter_slot::Union{Nothing,Int}
end

function _collect_backend_slot_kinds!(
    step::BackendBernoulliChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendBernoulliLogitChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.eta, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

# Recognize the GLM linear-predictor form `[intercept +] sum(coef .* Matrix[:, index])`
# and lower it to a fused `BackendLinearPredictorExpr`; returns `nothing` (without
# recording an issue) when the shape does not match exactly, so the caller falls
# back to generic numeric lowering. `coef` must resolve to a latent VECTOR
# parameter slot (VectorIdentityTransform), `Matrix` to a model-argument env slot,
# and `index` lowers as an index expr (the loop iterator).
function _is_glm_linear_predictor_sum(expr)
    (expr isa Expr && expr.head == :call && length(expr.args) == 2 && expr.args[1] === :sum) || return false
    dot = expr.args[2]
    (dot isa Expr && dot.head == :call && length(dot.args) == 3 && dot.args[1] === Symbol(".*")) || return false
    ref = dot.args[3]
    (ref isa Expr && ref.head == :ref && length(ref.args) == 3 && ref.args[2] === Symbol(":")) || return false
    return true
end

function _backend_lower_linear_predictor_arg(
    model::TeaModel,
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    expr,
    issues::Vector{String},
)
    intercept_ast = nothing
    sum_ast = expr
    if expr isa Expr && expr.head == :call && length(expr.args) == 3 && expr.args[1] === :+
        left, right = expr.args[2], expr.args[3]
        if _is_glm_linear_predictor_sum(right)
            intercept_ast, sum_ast = left, right
        elseif _is_glm_linear_predictor_sum(left)
            intercept_ast, sum_ast = right, left
        else
            return nothing
        end
    elseif _is_glm_linear_predictor_sum(expr)
        sum_ast = expr
    else
        return nothing
    end

    dot = sum_ast.args[2]
    coef_symbol = dot.args[2]
    ref = dot.args[3]
    matrix_symbol = ref.args[1]
    index_ast = ref.args[3]
    (coef_symbol isa Symbol && matrix_symbol isa Symbol) || return nothing

    coef_slot = _environment_slot(layout, coef_symbol)
    isnothing(coef_slot) && return nothing
    coef_parameter_slot = nothing
    for slot in parameter_layout.slots
        if slot.binding === coef_symbol
            coef_parameter_slot = slot
            break
        end
    end
    isnothing(coef_parameter_slot) && return nothing
    coef_parameter_slot.transform isa VectorIdentityTransform || return nothing

    matrix_slot = _environment_slot(layout, matrix_symbol)
    isnothing(matrix_slot) && return nothing

    # sub-lowering uses a throwaway issue buffer: a mismatch here is not a model
    # error, just a fall-through to generic lowering
    scratch_issues = String[]
    index = _backend_lower_expr(model, layout, index_ast, scratch_issues, "linear predictor index")
    isnothing(index) && return nothing
    intercept = nothing
    if !isnothing(intercept_ast)
        intercept = _backend_lower_expr(model, layout, intercept_ast, scratch_issues, "linear predictor intercept")
        isnothing(intercept) && return nothing
    end
    return BackendLinearPredictorExpr(
        intercept,
        coef_slot,
        coef_parameter_slot.value_index,
        coef_parameter_slot.value_length,
        matrix_slot,
        index,
    )
end

function _backend_lower_bernoullilogit_choice_step(
    model::TeaModel,
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 1 || begin
        _backend_issue!(issues, "bernoullilogit expects exactly 1 backend argument")
        return nothing
    end
    isnothing(step.parameter_slot) || begin
        _backend_issue!(issues, "bernoullilogit latents are not supported in backend lowering")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    argument = step.rhs.arguments[1]
    eta = _backend_lower_linear_predictor_arg(model, layout, parameter_layout, argument, issues)
    if isnothing(eta)
        eta = _backend_lower_expr(model, layout, argument, issues, "bernoullilogit eta")
        isnothing(eta) && return nothing
    end
    return BackendBernoulliLogitChoicePlanStep(step.binding_slot, address, eta, step.parameter_slot)
end

function _collect_backend_slot_kinds!(
    step::BackendCategoricalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for probability in step.probabilities
        _mark_backend_numeric_expr_slots!(probability, numeric_slots, index_slots, generic_slots)
    end
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendPoissonChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.lambda, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

# marginalize=:enumerate (docs/discrete-enumeration.md, issue #13 PR-4): the
# step owns the lowered plan SUFFIX and scores it once per compile-time
# support value, combining with a per-column logsumexp. `probabilities` holds
# the family's lowered numeric argument exprs (bernoulli: `(p,)`; categorical:
# the K literal entries); `parameter_slot` is always `nothing` (an enumerated
# latent has no slot) and exists only for the choice-step field convention.
struct BackendMarginalizeChoicePlanStep{P<:Tuple,S<:Tuple,B<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    family::Symbol
    probabilities::P
    support::S
    body::B
    parameter_slot::Union{Nothing,Int}
end

# Nested enumerated latents multiply the suffix cost; the backend rejects
# support products beyond this bound honestly instead of compiling them.
const BACKEND_MARGINALIZE_SUPPORT_LIMIT = 32

function _backend_marginalize_support_product(steps)
    product = 1
    for step in steps
        if step isa BackendMarginalizeChoicePlanStep
            product *= length(step.support) * _backend_marginalize_support_product(step.body)
        elseif step isa BackendLoopPlanStep
            product *= _backend_marginalize_support_product(step.body)
        end
    end
    return product
end

function _collect_backend_slot_kinds!(
    step::BackendMarginalizeChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for probability in step.probabilities
        _mark_backend_numeric_expr_slots!(probability, numeric_slots, index_slots, generic_slots)
    end
    if !isnothing(step.binding_slot)
        # bernoulli branches bind 0/1 numerics; categorical branches bind the
        # category itself, which integer consumers (binomial trials, loop
        # bounds) require to stay an index slot
        if step.family === :categorical
            _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
        else
            _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
        end
    end
    for inner in step.body
        _collect_backend_slot_kinds!(inner, numeric_slots, index_slots, generic_slots)
    end
    return nothing
end

function _backend_lower_marginalize_choice_step(
    model::TeaModel,
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    suffix::AbstractVector,
    issues::Vector{String},
)
    isnothing(step.parameter_slot) || begin
        _backend_issue!(issues, "marginalized choices carry no parameter slot")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    family = step.rhs.family
    probabilities = if family === :bernoulli
        length(step.rhs.arguments) == 1 || begin
            _backend_issue!(issues, "bernoulli expects exactly 1 backend argument")
            return nothing
        end
        probability = _backend_lower_expr(model, layout, step.rhs.arguments[1], issues, "bernoulli probability")
        isnothing(probability) && return nothing
        (probability,)
    elseif family === :categorical
        lowered = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[1], issues, "categorical probabilities")
        isnothing(lowered) && return nothing
        lowered
    else
        _backend_issue!(
            issues,
            "marginalize=:enumerate supports bernoulli and categorical in backend lowering, got `$family`",
        )
        return nothing
    end
    support = family === :bernoulli ? (false, true) : ntuple(identity, length(probabilities))

    body = _backend_lower_steps(model, layout, parameter_layout, suffix, issues)
    any(isnothing, body) && return nothing
    body_steps = tuple(body...)
    support_product = length(support) * _backend_marginalize_support_product(body_steps)
    support_product <= BACKEND_MARGINALIZE_SUPPORT_LIMIT || begin
        _backend_issue!(
            issues,
            "nested marginalize=:enumerate support product $support_product exceeds the backend limit " *
            "$BACKEND_MARGINALIZE_SUPPORT_LIMIT",
        )
        return nothing
    end
    return BackendMarginalizeChoicePlanStep(
        step.binding_slot,
        address,
        family,
        probabilities,
        support,
        body_steps,
        nothing,
    )
end

struct BackendBinomialChoicePlanStep{N<:AbstractBackendExpr,P<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    trials::N
    probability::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendGeometricChoicePlanStep{P<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    probability::P
    parameter_slot::Union{Nothing,Int}
end

struct BackendNegativeBinomialChoicePlanStep{R<:AbstractBackendExpr,P<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    successes::R
    probability::P
    parameter_slot::Union{Nothing,Int}
end

function _collect_backend_slot_kinds!(
    step::BackendBinomialChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_index_expr_slots!(step.trials, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendGeometricChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendNegativeBinomialChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.successes, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.probability, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_index_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

abstract type AbstractAddressPart end

struct AddressLiteralPart <: AbstractAddressPart
    value::Any
end

struct AddressDynamicPart <: AbstractAddressPart
    value::Any
end

struct AddressSpec
    parts::Tuple{Vararg{AbstractAddressPart}}
end

abstract type AbstractChoiceRhsSpec end

struct DistributionSpec <: AbstractChoiceRhsSpec
    family::Symbol
    arguments::Vector{Any}
    # For registered user families: the builder captured when the model was
    # defined, so later re-registration cannot desynchronize an existing model
    # from its parameter layout. `nothing` for built-ins (resolved by family).
    builder::Any
    # :centered (default) or :noncentered -- the staged reparameterization of
    # docs/noncentered-reparam.md (issue #19).
    reparam::Symbol
    # :none (default) or :enumerate -- marginalize a finite-support discrete
    # latent out of the logjoint (docs/discrete-enumeration.md, issue #13).
    marginalize::Symbol
end

DistributionSpec(family::Symbol, arguments) = DistributionSpec(family, arguments, nothing, :centered, :none)
DistributionSpec(family::Symbol, arguments, builder) = DistributionSpec(family, arguments, builder, :centered, :none)
DistributionSpec(family::Symbol, arguments, builder, reparam::Symbol) =
    DistributionSpec(family, arguments, builder, reparam, :none)

struct GenerativeCallSpec <: AbstractChoiceRhsSpec
    callee::Any
    arguments::Vector{Any}
end

# A dot-call distribution observation, e.g. `{:y} ~ normal.(mu, sigma)`. `arguments`
# holds the broadcast-elementwise argument expressions (mirrors `DistributionSpec`).
# Only `family === :normal` is supported initially.
struct BroadcastDistributionSpec <: AbstractChoiceRhsSpec
    family::Symbol
    arguments::Vector{Any}
end

struct RawChoiceRhsSpec <: AbstractChoiceRhsSpec
    expr::Any
end

struct LoopScopeSpec
    iterator::Symbol
    iterable::Any
    shape_specialized::Bool
end

abstract type AbstractParameterTransform end

struct IdentityTransform <: AbstractParameterTransform end
struct VectorIdentityTransform <: AbstractParameterTransform
    size::Int

    function VectorIdentityTransform(size::Int)
        size >= 1 || throw(ArgumentError("vector identity transform requires size >= 1"))
        return new(size)
    end
end
struct LogTransform <: AbstractParameterTransform end
struct LogitTransform <: AbstractParameterTransform end
# Marker for reparam=:noncentered latents (docs/noncentered-reparam.md): the
# slot holds the standardized z; the constrained value loc + scale * z is
# materialized during the plan walk (PR-3), not by the slot transform pass.
struct NoncenteredTransform <: AbstractParameterTransform end
# iid vector variant: an n-wide slot of standardized z's (docs/noncentered-reparam.md, PR-6)
struct VectorNoncenteredTransform <: AbstractParameterTransform
    size::Int

    function VectorNoncenteredTransform(size::Int)
        size >= 1 || throw(ArgumentError("vector noncentered transform requires size >= 1"))
        return new(size)
    end
end
# Per-element unconstrained -> positive transform for `iid` latents over families
# whose scalar transform is `LogTransform` (lognormal/exponential/gamma/...).
struct VectorLogTransform <: AbstractParameterTransform
    size::Int

    function VectorLogTransform(size::Int)
        size >= 1 || throw(ArgumentError("vector log transform requires size >= 1"))
        return new(size)
    end
end
# Per-element unconstrained -> (0, 1) transform for `iid` latents over `beta`.
struct VectorLogitTransform <: AbstractParameterTransform
    size::Int

    function VectorLogitTransform(size::Int)
        size >= 1 || throw(ArgumentError("vector logit transform requires size >= 1"))
        return new(size)
    end
end
struct SimplexTransform <: AbstractParameterTransform
    size::Int

    function SimplexTransform(size::Int)
        size >= 2 || throw(ArgumentError("simplex transform requires size >= 2"))
        return new(size)
    end
end

# Unconstrained -> Cholesky factor of a correlation matrix (Stan's canonical
# partial correlation parameterization). The constrained value is the PACKED
# column-major lower triangle of the factor, length size*(size+1)/2 (diagonal
# included); the unconstrained vector holds one entry per below-diagonal
# element, length size*(size-1)/2, consumed in row-major below-diagonal order
# ((2,1), (3,1), (3,2), ...).
struct CholeskyCorrTransform <: AbstractParameterTransform
    size::Int

    function CholeskyCorrTransform(size::Int)
        size >= 2 || throw(ArgumentError("cholesky correlation transform requires size >= 2"))
        return new(size)
    end
end

# Unconstrained -> Cholesky factor of a positive-definite (covariance) matrix
# ("log-Cholesky" parameterization). The constrained value is the PACKED
# column-major lower triangle of the factor L, length size*(size+1)/2 (diagonal
# included); the unconstrained vector holds the SAME packed layout, but with the
# diagonal coordinates stored as log(L[i,i]) (exp on constrain) and the
# below-diagonal coordinates stored directly (identity). The log-abs-det of the
# unconstrained -> L map is sum_i log(L[i,i]) = sum over the diagonal packed
# coordinates. Wishart/inverse-Wishart latents score their PD-matrix density over
# this factor.
struct CholeskyCovTransform <: AbstractParameterTransform
    size::Int

    function CholeskyCovTransform(size::Int)
        size >= 1 || throw(ArgumentError("cholesky covariance transform requires size >= 1"))
        return new(size)
    end
end

# Position of entry (row, col), row >= col, inside the column-major packed
# lower triangle of a `size` x `size` matrix (diagonal included).
function _packed_lower_index(size::Int, row::Integer, col::Integer)
    return (col - 1) * size - ((col - 1) * (col - 2)) ÷ 2 + (row - col + 1)
end

struct BoundedTransform{T<:Real} <: AbstractParameterTransform
    lower::T
    upper::T

    function BoundedTransform(lower::Real, upper::Real)
        promoted_lower, promoted_upper = promote(float(lower), float(upper))
        promoted_lower < promoted_upper ||
            throw(ArgumentError("bounded transform requires lower < upper"))
        return new{typeof(promoted_lower)}(promoted_lower, promoted_upper)
    end
end

struct LowerBoundedTransform{T<:Real} <: AbstractParameterTransform
    lower::T

    LowerBoundedTransform(lower::Real) = new{typeof(float(lower))}(float(lower))
end

struct UpperBoundedTransform{T<:Real} <: AbstractParameterTransform
    upper::T

    UpperBoundedTransform(upper::Real) = new{typeof(float(upper))}(float(upper))
end

struct ChoiceSpec
    binding::Union{Nothing,Symbol}
    address::AddressSpec
    rhs::AbstractChoiceRhsSpec
    scopes::Vector{LoopScopeSpec}
end

struct ParameterSlotSpec
    choice_index::Int
    binding::Symbol
    address::AddressSpec
    index::Int
    dimension::Int
    value_index::Int
    value_length::Int
    transform::AbstractParameterTransform
end

struct ParameterLayout
    slots::Vector{ParameterSlotSpec}
    parameter_count::Int
    value_count::Int
end

struct EnvironmentLayout
    symbols::Vector{Symbol}
    slot_by_symbol::Dict{Symbol,Int}
    argument_slots::Vector{Int}
end

abstract type AbstractPlanStep end

struct ChoicePlanStep <: AbstractPlanStep
    choice_index::Int
    binding::Union{Nothing,Symbol}
    binding_slot::Union{Nothing,Int}
    address::AddressSpec
    rhs::AbstractChoiceRhsSpec
    scopes::Vector{LoopScopeSpec}
    parameter_slot::Union{Nothing,Int}
end

function ChoicePlanStep(choice_index::Int, binding, address::AddressSpec, rhs::AbstractChoiceRhsSpec, scopes, parameter_slot)
    normalized_scopes = LoopScopeSpec[scope for scope in scopes]
    return ChoicePlanStep(choice_index, binding, nothing, address, rhs, normalized_scopes, parameter_slot)
end

struct DeterministicPlanStep <: AbstractPlanStep
    binding::Symbol
    binding_slot::Union{Nothing,Int}
    expr::Any
    scopes::Vector{LoopScopeSpec}
end

DeterministicPlanStep(binding::Symbol, expr::Any) = DeterministicPlanStep(binding, nothing, expr, LoopScopeSpec[])
DeterministicPlanStep(binding::Symbol, binding_slot::Union{Nothing,Int}, expr::Any) =
    DeterministicPlanStep(binding, binding_slot, expr, LoopScopeSpec[])

struct LoopPlanStep <: AbstractPlanStep
    iterator::Symbol
    iterator_slot::Union{Nothing,Int}
    iterable::Any
    body::Vector{AbstractPlanStep}
end

function LoopPlanStep(iterator::Symbol, iterable::Any, body)
    normalized_body = AbstractPlanStep[step for step in body]
    return LoopPlanStep(iterator, nothing, iterable, normalized_body)
end

struct ExecutionPlan
    model_name::Symbol
    steps::Vector{AbstractPlanStep}
    parameter_layout::ParameterLayout
    environment_layout::EnvironmentLayout
end

struct ModelSpec
    name::Symbol
    mode::Symbol
    arguments::Vector{Symbol}
    choices::Vector{ChoiceSpec}
    shape_specialized::Bool
    parameter_layout::ParameterLayout
    return_expr::Any
    execution_plan::ExecutionPlan
end

function ModelSpec(
    name::Symbol,
    mode::Symbol,
    arguments,
    choices::Vector{ChoiceSpec},
    shape_specialized::Bool,
    parameter_layout::ParameterLayout,
    return_expr::Any,
)
    argument_symbols = Symbol[arg for arg in arguments]
    environment_layout = EnvironmentLayout(
        copy(argument_symbols),
        Dict(arg => idx for (idx, arg) in enumerate(argument_symbols)),
        collect(eachindex(argument_symbols)),
    )
    plan = ExecutionPlan(name, AbstractPlanStep[], parameter_layout, environment_layout)
    return ModelSpec(name, mode, argument_symbols, choices, shape_specialized, parameter_layout, return_expr, plan)
end

function ModelSpec(
    name::Symbol,
    mode::Symbol,
    arguments,
    choices::Vector{ChoiceSpec},
    shape_specialized::Bool,
    parameter_layout::ParameterLayout,
    return_expr::Any,
    plan_steps,
)
    argument_symbols = Symbol[arg for arg in arguments]
    raw_steps = AbstractPlanStep[step for step in plan_steps]
    plan = build_execution_plan(name, argument_symbols, raw_steps, parameter_layout, return_expr)
    return ModelSpec(name, mode, argument_symbols, choices, shape_specialized, plan.parameter_layout, return_expr, plan)
end

function modelspec(model)
    return model.spec
end

function parameterlayout(model)
    return model.spec.parameter_layout
end

function executionplan(model)
    return model.spec.execution_plan
end

isstaticaddress(address::AddressSpec) = all(part -> part isa AddressLiteralPart, address.parts)
isaddresstemplate(address::AddressSpec) = !isstaticaddress(address)
isrepeatedchoice(choice::ChoiceSpec) = !isempty(choice.scopes)
hasrepeatedchoices(spec::ModelSpec) = any(isrepeatedchoice, spec.choices)
parametercount(layout::ParameterLayout) = layout.parameter_count
parametervaluecount(layout::ParameterLayout) = layout.value_count

parameterindices(slot::ParameterSlotSpec) = slot.index:(slot.index+slot.dimension-1)
parametervalueindices(slot::ParameterSlotSpec) = slot.value_index:(slot.value_index+slot.value_length-1)
isscalarparameterslot(slot::ParameterSlotSpec) = slot.dimension == 1 && slot.value_length == 1

function _parameter_slot_index(layout::ParameterLayout, choice_index::Int)
    for (slot_index, slot) in enumerate(layout.slots)
        if slot.choice_index == choice_index
            return slot_index
        end
    end
    return nothing
end

function _push_environment_symbol!(symbols::Vector{Symbol}, seen::Set{Symbol}, symbol::Symbol)
    symbol in seen && return nothing
    push!(symbols, symbol)
    push!(seen, symbol)
    return nothing
end

function _collect_environment_symbols!(steps::Vector{AbstractPlanStep}, symbols::Vector{Symbol}, seen::Set{Symbol})
    for step in steps
        if step isa ChoicePlanStep
            isnothing(step.binding) || _push_environment_symbol!(symbols, seen, step.binding)
        elseif step isa DeterministicPlanStep
            _push_environment_symbol!(symbols, seen, step.binding)
        elseif step isa LoopPlanStep
            _push_environment_symbol!(symbols, seen, step.iterator)
            _collect_environment_symbols!(step.body, symbols, seen)
        end
    end
    return nothing
end

function _build_environment_layout(arguments::Vector{Symbol}, steps::Vector{AbstractPlanStep})
    symbols = Symbol[]
    seen = Set{Symbol}()
    for argument in arguments
        _push_environment_symbol!(symbols, seen, argument)
    end
    _collect_environment_symbols!(steps, symbols, seen)

    slot_by_symbol = Dict{Symbol,Int}()
    for (idx, symbol) in enumerate(symbols)
        slot_by_symbol[symbol] = idx
    end
    argument_slots = Int[slot_by_symbol[argument] for argument in arguments]
    return EnvironmentLayout(symbols, slot_by_symbol, argument_slots)
end

function _annotate_environment_slots(step::ChoicePlanStep, layout::EnvironmentLayout)
    binding_slot = isnothing(step.binding) ? nothing : layout.slot_by_symbol[step.binding]
    return ChoicePlanStep(step.choice_index, step.binding, binding_slot, step.address, step.rhs, step.scopes, step.parameter_slot)
end

function _annotate_environment_slots(step::DeterministicPlanStep, layout::EnvironmentLayout)
    return DeterministicPlanStep(step.binding, layout.slot_by_symbol[step.binding], step.expr)
end

function _annotate_environment_slots(step::LoopPlanStep, layout::EnvironmentLayout)
    body = AbstractPlanStep[_annotate_environment_slots(inner, layout) for inner in step.body]
    return LoopPlanStep(step.iterator, layout.slot_by_symbol[step.iterator], step.iterable, body)
end

function _annotate_environment_slots(steps::Vector{AbstractPlanStep}, layout::EnvironmentLayout)
    return AbstractPlanStep[_annotate_environment_slots(step, layout) for step in steps]
end

function _substitute_expr(expr, substitutions::Dict{Symbol,Any})
    if expr isa QuoteNode
        return expr
    elseif expr isa Symbol
        return get(substitutions, expr, expr)
    elseif expr isa Expr
        return Expr(expr.head, map(arg -> _substitute_expr(arg, substitutions), expr.args)...)
    elseif expr isa Tuple
        return tuple((_substitute_expr(arg, substitutions) for arg in expr)...)
    end
    return expr
end

function _substitute_address(address::AddressSpec, substitutions::Dict{Symbol,Any})
    parts = map(address.parts) do part
        if part isa AddressLiteralPart
            return part
        end
        return AddressDynamicPart(_substitute_expr(part.value, substitutions))
    end
    return AddressSpec(tuple(parts...))
end

function _prefix_address(prefix::AddressSpec, address::AddressSpec)
    return AddressSpec((prefix.parts..., address.parts...))
end

function _collect_bound_symbols!(steps::Vector{AbstractPlanStep}, bindings::Set{Symbol}, iterators::Set{Symbol})
    for step in steps
        if step isa ChoicePlanStep
            isnothing(step.binding) || push!(bindings, step.binding)
        elseif step isa DeterministicPlanStep
            push!(bindings, step.binding)
        elseif step isa LoopPlanStep
            push!(iterators, step.iterator)
            _collect_bound_symbols!(step.body, bindings, iterators)
        end
    end
    return nothing
end

function _substitute_rhs(rhs::DistributionSpec, substitutions::Dict{Symbol,Any})
    return DistributionSpec(
        rhs.family,
        Any[_substitute_expr(arg, substitutions) for arg in rhs.arguments],
        rhs.builder,
        rhs.reparam,
        rhs.marginalize,
    )
end

function _substitute_rhs(rhs::GenerativeCallSpec, substitutions::Dict{Symbol,Any})
    callee = rhs.callee isa TeaModel ? rhs.callee : _substitute_expr(rhs.callee, substitutions)
    arguments = Any[_substitute_expr(arg, substitutions) for arg in rhs.arguments]
    return GenerativeCallSpec(callee, arguments)
end

function _substitute_rhs(rhs::BroadcastDistributionSpec, substitutions::Dict{Symbol,Any})
    return BroadcastDistributionSpec(rhs.family, Any[_substitute_expr(arg, substitutions) for arg in rhs.arguments])
end

_substitute_rhs(rhs::RawChoiceRhsSpec, substitutions::Dict{Symbol,Any}) =
    RawChoiceRhsSpec(_substitute_expr(rhs.expr, substitutions))

function _substitute_loop_scopes(scopes::Vector{LoopScopeSpec}, substitutions::Dict{Symbol,Any})
    replaced = LoopScopeSpec[]
    for scope in scopes
        iterator = get(substitutions, scope.iterator, scope.iterator)
        iterator isa Symbol || throw(ArgumentError("loop iterator substitution must stay a Symbol"))
        push!(replaced, LoopScopeSpec(iterator, _substitute_expr(scope.iterable, substitutions), scope.shape_specialized))
    end
    return replaced
end

function _substitute_step(
    step::ChoicePlanStep,
    substitutions::Dict{Symbol,Any};
    prefix::Union{Nothing,AddressSpec}=nothing,
    parameter_slot=nothing,
)
    address = _substitute_address(step.address, substitutions)
    prefixed = isnothing(prefix) ? address : _prefix_address(prefix, address)
    binding = isnothing(step.binding) ? nothing : get(substitutions, step.binding, step.binding)
    if !isnothing(binding) && !(binding isa Symbol)
        throw(ArgumentError("choice binding substitution must stay a Symbol"))
    end
    return ChoicePlanStep(
        step.choice_index,
        binding,
        prefixed,
        _substitute_rhs(step.rhs, substitutions),
        _substitute_loop_scopes(step.scopes, substitutions),
        parameter_slot,
    )
end

function _substitute_step(
    step::DeterministicPlanStep,
    substitutions::Dict{Symbol,Any};
    prefix::Union{Nothing,AddressSpec}=nothing,
    parameter_slot=nothing,
)
    binding = get(substitutions, step.binding, step.binding)
    binding isa Symbol || throw(ArgumentError("deterministic binding substitution must stay a Symbol"))
    return DeterministicPlanStep(
        binding,
        nothing,
        _substitute_expr(step.expr, substitutions),
        _substitute_loop_scopes(step.scopes, substitutions),
    )
end

function _substitute_step(
    step::LoopPlanStep,
    substitutions::Dict{Symbol,Any};
    prefix::Union{Nothing,AddressSpec}=nothing,
    parameter_slot=nothing,
)
    iterator = get(substitutions, step.iterator, step.iterator)
    iterator isa Symbol || throw(ArgumentError("loop iterator substitution must stay a Symbol"))
    body = AbstractPlanStep[_substitute_step(inner, substitutions; prefix=prefix, parameter_slot=nothing) for inner in step.body]
    return LoopPlanStep(iterator, _substitute_expr(step.iterable, substitutions), body)
end

function _wrap_steps_with_scopes(steps::Vector{AbstractPlanStep}, scopes::Vector{LoopScopeSpec})
    result = steps
    for scope in reverse(scopes)
        result = AbstractPlanStep[LoopPlanStep(scope.iterator, scope.iterable, result)]
    end
    return result
end

function _merge_loop_steps(steps::Vector{AbstractPlanStep})
    merged = AbstractPlanStep[]
    for step in steps
        if step isa LoopPlanStep
            body = _merge_loop_steps(step.body)
            if !isempty(merged) && merged[end] isa LoopPlanStep
                previous = merged[end]
                if previous.iterator == step.iterator && previous.iterable == step.iterable
                    merged[end] = LoopPlanStep(previous.iterator, previous.iterable, vcat(previous.body, body))
                    continue
                end
            end
            push!(merged, LoopPlanStep(step.iterator, step.iterable, body))
        else
            push!(merged, step)
        end
    end
    return merged
end

function _static_bound_value(expr)
    if expr isa Number
        return float(expr)
    elseif expr === :Inf || expr === :Inf64 || expr === :Inf32 || expr === :Inf16
        return Inf
    elseif expr isa Expr && expr.head == :call && length(expr.args) == 2 && expr.args[1] === :-
        inner = _static_bound_value(expr.args[2])
        return isnothing(inner) ? nothing : -inner
    end
    return nothing
end

function _truncated_bound_exprs(family::Symbol, arguments)
    if family === :truncatednormal
        length(arguments) == 4 || return nothing
        return arguments[3], arguments[4]
    elseif family === :truncatedstudentt
        length(arguments) == 5 || return nothing
        return arguments[4], arguments[5]
    end
    return nothing
end

function _truncated_static_bounds(family::Symbol, arguments)
    bound_exprs = _truncated_bound_exprs(family, arguments)
    isnothing(bound_exprs) && return nothing
    lower = _static_bound_value(bound_exprs[1])
    upper = _static_bound_value(bound_exprs[2])
    (isnothing(lower) || isnothing(upper)) && return nothing
    return (lower, upper)
end

# uniform(a, b) needs literal finite bounds for a latent slot, reusing the
# static-bounds discipline the truncated families established. Returns
# (lower, upper) when both arguments are static finite Numbers, else nothing.
function _uniform_static_bounds(arguments)
    length(arguments) == 2 || return nothing
    lower = _static_bound_value(arguments[1])
    upper = _static_bound_value(arguments[2])
    (isnothing(lower) || isnothing(upper)) && return nothing
    (isfinite(lower) && isfinite(upper) && upper > lower) || return nothing
    return (lower, upper)
end

# pareto(x_m, alpha) has support x >= x_m, so a latent slot unconstrains through
# a LowerBoundedTransform whose bound must be a literal (static) positive Number,
# reusing the static-bounds discipline of the truncated/uniform families. Returns
# the static x_m when arg 1 is a static finite positive Number, else nothing.
function _pareto_static_lower(arguments)
    length(arguments) == 2 || return nothing
    xm = _static_bound_value(arguments[1])
    isnothing(xm) && return nothing
    (isfinite(xm) && xm > 0) || return nothing
    return xm
end

function _truncated_bound_transform(lower::Real, upper::Real)
    lower_finite = isfinite(lower)
    upper_finite = isfinite(upper)
    if lower_finite && upper_finite
        return BoundedTransform(lower, upper)
    elseif lower_finite
        return LowerBoundedTransform(lower)
    elseif upper_finite
        return UpperBoundedTransform(upper)
    end
    return IdentityTransform()
end

const _MIXTURE_REAL_LINE_FAMILIES = (:normal, :laplace, :studentt)

# Collect the callee symbols of a mixture's component constructor calls. `arguments`
# is the full mixture argument list (first entry is the weights expression, the rest
# are components). Returns `nothing` if any component is not a plain constructor call.
function _mixture_component_callees(arguments)
    length(arguments) >= 2 || return nothing
    callees = Symbol[]
    for component in arguments[2:end]
        if component isa Expr && component.head == :call && !isempty(component.args) &&
           component.args[1] isa Symbol
            push!(callees, component.args[1])
        else
            return nothing
        end
    end
    return callees
end

# A mixture is eligible for a latent parameter slot only when every component is one of
# the real-line location-scale families, so an IdentityTransform is exact.
function _mixture_latent_eligible(arguments)
    callees = _mixture_component_callees(arguments)
    isnothing(callees) && return false
    return all(callee -> callee in _MIXTURE_REAL_LINE_FAMILIES, callees)
end

function _parameter_transform(rhs::DistributionSpec)
    if rhs.reparam === :noncentered
        # family eligibility was validated at macro expansion
        if rhs.family === :iid
            size = _iid_static_size(rhs.arguments)
            isnothing(size) && return nothing
            return VectorNoncenteredTransform(size)
        end
        return NoncenteredTransform()
    end
    if rhs.family === :normal || rhs.family === :laplace || rhs.family === :studentt ||
       rhs.family === :cauchy || rhs.family === :logistic || rhs.family === :gumbel
        return IdentityTransform()
    elseif rhs.family === :halfnormal || rhs.family === :halfcauchy
        return LogTransform()
    elseif rhs.family === :uniform
        bounds = _uniform_static_bounds(rhs.arguments)
        isnothing(bounds) && return nothing
        return BoundedTransform(bounds[1], bounds[2])
    elseif rhs.family === :mvnormal
        size = _mvnormal_static_size(rhs.arguments)
        isnothing(size) || return VectorIdentityTransform(size)
    elseif rhs.family === :mvnormaldense
        size = _mvnormaldense_static_size(rhs.arguments)
        isnothing(size) || return VectorIdentityTransform(size)
    elseif rhs.family === :mvstudentt
        size = _mvstudentt_static_size(rhs.arguments)
        isnothing(size) || return VectorIdentityTransform(size)
    elseif rhs.family === :mvstudenttdense
        size = _mvstudenttdense_static_size(rhs.arguments)
        isnothing(size) || return VectorIdentityTransform(size)
    elseif rhs.family === :wishart || rhs.family === :inversewishart
        size = _wishart_static_size(rhs.arguments)
        isnothing(size) || return CholeskyCovTransform(size)
    elseif rhs.family === :lognormal || rhs.family === :exponential || rhs.family === :gamma ||
           rhs.family === :inversegamma || rhs.family === :weibull ||
           rhs.family === :frechet || rhs.family === :rayleigh || rhs.family === :inversegaussian
        return LogTransform()
    elseif rhs.family === :pareto
        xm = _pareto_static_lower(rhs.arguments)
        isnothing(xm) && return nothing
        return LowerBoundedTransform(xm)
    elseif rhs.family === :beta
        return LogitTransform()
    elseif rhs.family === :dirichlet
        size = _dirichlet_static_size(rhs.arguments)
        isnothing(size) || return SimplexTransform(size)
    elseif rhs.family === :lkjcholesky
        size = _lkjcholesky_static_size(rhs.arguments)
        isnothing(size) || return CholeskyCorrTransform(size)
    elseif rhs.family === :truncatednormal || rhs.family === :truncatedstudentt
        bounds = _truncated_static_bounds(rhs.family, rhs.arguments)
        isnothing(bounds) && return nothing
        return _truncated_bound_transform(bounds[1], bounds[2])
    elseif rhs.family === :mixture
        _mixture_latent_eligible(rhs.arguments) && return IdentityTransform()
        return nothing
    elseif rhs.family === :iid
        return _iid_parameter_transform(rhs.arguments)
    end
    registration = _registered_user_distribution(rhs.family)
    isnothing(registration) || return registration.transform
    return nothing
end

_parameter_transform(::AbstractChoiceRhsSpec) = nothing

# Base family of an `iid(base_call, n)` right-hand side, or `nothing` if the first
# argument is not a plain distribution constructor call.
function _iid_base_family(arguments)
    length(arguments) == 2 || return nothing
    base = arguments[1]
    base isa Expr && base.head == :call && !isempty(base.args) && base.args[1] isa Symbol || return nothing
    return base.args[1]
end

function _iid_static_size(arguments)
    length(arguments) == 2 || return nothing
    n = arguments[2]
    n isa Integer || return nothing
    return Int(n)
end

function _iid_parameter_transform(arguments)
    family = _iid_base_family(arguments)
    size = _iid_static_size(arguments)
    (isnothing(family) || isnothing(size)) && return nothing
    if family === :normal || family === :laplace || family === :studentt ||
       family === :cauchy || family === :logistic || family === :gumbel
        return VectorIdentityTransform(size)
    elseif family === :lognormal || family === :exponential || family === :gamma ||
           family === :inversegamma || family === :weibull ||
           family === :halfnormal || family === :halfcauchy ||
           family === :frechet || family === :rayleigh || family === :inversegaussian
        return VectorLogTransform(size)
    elseif family === :beta
        return VectorLogitTransform(size)
    end
    return nothing
end

_parameter_dimensions(::IdentityTransform) = (1, 1)
_parameter_dimensions(::NoncenteredTransform) = (1, 1)
_parameter_dimensions(transform::VectorNoncenteredTransform) = (transform.size, transform.size)
_parameter_dimensions(transform::VectorIdentityTransform) = (transform.size, transform.size)
_parameter_dimensions(::LogTransform) = (1, 1)
_parameter_dimensions(::LogitTransform) = (1, 1)
_parameter_dimensions(transform::VectorLogTransform) = (transform.size, transform.size)
_parameter_dimensions(transform::VectorLogitTransform) = (transform.size, transform.size)
_parameter_dimensions(transform::SimplexTransform) = (transform.size - 1, transform.size)
_parameter_dimensions(transform::CholeskyCorrTransform) =
    ((transform.size * (transform.size - 1)) ÷ 2, (transform.size * (transform.size + 1)) ÷ 2)
_parameter_dimensions(transform::CholeskyCovTransform) =
    ((transform.size * (transform.size + 1)) ÷ 2, (transform.size * (transform.size + 1)) ÷ 2)
_parameter_dimensions(::BoundedTransform) = (1, 1)
_parameter_dimensions(::LowerBoundedTransform) = (1, 1)
_parameter_dimensions(::UpperBoundedTransform) = (1, 1)

function _static_length(expr)
    if expr isa Expr
        if expr.head == :vect || expr.head == :tuple
            return length(expr.args)
        end
    elseif expr isa QuoteNode
        value = expr.value
        if value isa Tuple || value isa AbstractVector
            return length(value)
        end
    elseif expr isa Tuple || expr isa AbstractVector
        return length(expr)
    end
    return nothing
end

function _dirichlet_static_size(arguments::Vector)
    isempty(arguments) && return nothing
    if length(arguments) == 1
        return _static_length(arguments[1])
    end
    return length(arguments)
end

function _mvnormal_static_size(arguments::Vector)
    length(arguments) == 2 || return nothing
    mu_size = _static_length(arguments[1])
    sigma_size = _static_length(arguments[2])
    if !isnothing(mu_size) && !isnothing(sigma_size)
        mu_size == sigma_size || throw(ArgumentError("mvnormal requires mean and scale vectors with the same static length"))
        return mu_size
    end
    return something(mu_size, sigma_size)
end

# Static dimension of `lkjcholesky(d, eta)`. The frontend enforces a literal
# integer `d` at macro time, but the runtime `DistributionSpec` argument vector
# is built from a `[d, eta]` literal, so an integer `d` may arrive promoted to
# a Float64 (e.g. `lkjcholesky(2, 2.0)` stores `Any[2.0, 2.0]`); accept any
# integer-valued literal number.
function _lkjcholesky_static_size(arguments::Vector)
    length(arguments) == 2 || return nothing
    d = arguments[1]
    d isa Real || return nothing
    isinteger(d) || return nothing
    dimension = Int(d)
    dimension >= 2 || return nothing
    return dimension
end

# Static size of `mvnormaldense(mu, scale_tril)` from the mu argument only; the
# scale factor is an arbitrary matrix expression the frontend never introspects.
function _mvnormaldense_static_size(arguments::Vector)
    length(arguments) == 2 || return nothing
    return _static_length(arguments[1])
end

# Static size of `mvstudentt(nu, mu, sigma)` from the mean/scale vector arguments
# (argument 1 is the degrees of freedom `nu`).
function _mvstudentt_static_size(arguments::Vector)
    length(arguments) == 3 || return nothing
    mu_size = _static_length(arguments[2])
    sigma_size = _static_length(arguments[3])
    if !isnothing(mu_size) && !isnothing(sigma_size)
        mu_size == sigma_size ||
            throw(ArgumentError("mvstudentt requires mean and scale vectors with the same static length"))
        return mu_size
    end
    return something(mu_size, sigma_size, Some(nothing))
end

# Static size of `mvstudenttdense(nu, mu, scale_tril)` from the mean argument.
function _mvstudenttdense_static_size(arguments::Vector)
    length(arguments) == 3 || return nothing
    return _static_length(arguments[2])
end

# Static dimension d of a d x d matrix LITERAL, or `nothing`. `[a b; c d]` parses
# as a `:vcat` of `:row`s; a single-row `[a b]` is an `:hcat`. Used to resolve
# `wishart(nu, S)` / `inversewishart(nu, S)` latents, whose value dimension is the
# side length of the scale matrix S (there is no vector argument to read it from).
function _matrix_literal_dim(expr)
    if expr isa Expr && expr.head == :vcat
        rows = expr.args
        d = length(rows)
        for row in rows
            if row isa Expr && row.head == :row
                length(row.args) == d || return nothing
            else
                # a `:vcat` of scalars is a column vector, not a square matrix
                return nothing
            end
        end
        return d
    elseif expr isa Expr && expr.head == :hcat
        # a single row: square only when 1x1
        return length(expr.args) == 1 ? 1 : nothing
    elseif expr isa QuoteNode && expr.value isa AbstractMatrix
        m = expr.value
        size(m, 1) == size(m, 2) || return nothing
        return size(m, 1)
    elseif expr isa AbstractMatrix
        size(expr, 1) == size(expr, 2) || return nothing
        return size(expr, 1)
    end
    return nothing
end

# Static dimension of `wishart(nu, S)` / `inversewishart(nu, S)` from a scale
# matrix literal S (argument 2). Returns `nothing` for a non-literal S (e.g. a
# model argument), which makes the family observation-only.
function _wishart_static_size(arguments::Vector)
    length(arguments) == 2 || return nothing
    return _matrix_literal_dim(arguments[2])
end

function _parameterize_step(
    step::ChoicePlanStep,
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
    parameter_counter::Base.RefValue{Int},
    value_counter::Base.RefValue{Int},
)
    step_index = step_counter[]
    step_counter[] += 1
    transform = isnothing(step.binding) ? nothing : _parameter_transform(step.rhs)

    if isnothing(transform) || !isempty(step.scopes) || !isstaticaddress(step.address)
        step.rhs isa DistributionSpec && step.rhs.reparam === :noncentered &&
            throw(
                ArgumentError(
                    "reparam=:noncentered requires a bound, static-address, unscoped latent choice; " *
                    "the choice at $(step.address) gets no parameter slot",
                ),
            )
        return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, nothing)
    end

    slot_index = slot_counter[]
    slot_counter[] += 1
    dimension, value_length = _parameter_dimensions(transform)
    parameter_index = parameter_counter[]
    value_index = value_counter[]
    parameter_counter[] += dimension
    value_counter[] += value_length
    push!(
        slots,
        ParameterSlotSpec(
            step_index,
            step.binding,
            step.address,
            parameter_index,
            dimension,
            value_index,
            value_length,
            transform,
        ),
    )
    return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, slot_index)
end

function _parameterize_plan_steps(
    steps::Vector{AbstractPlanStep},
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
    parameter_counter::Base.RefValue{Int},
    value_counter::Base.RefValue{Int},
)
    parameterized = AbstractPlanStep[]
    for step in steps
        if step isa ChoicePlanStep
            push!(
                parameterized,
                _parameterize_step(
                    step,
                    slots,
                    step_counter,
                    slot_counter,
                    parameter_counter,
                    value_counter,
                ),
            )
        elseif step isa DeterministicPlanStep
            push!(parameterized, step)
        elseif step isa LoopPlanStep
            body = _parameterize_plan_steps(
                step.body,
                slots,
                step_counter,
                slot_counter,
                parameter_counter,
                value_counter,
            )
            push!(parameterized, LoopPlanStep(step.iterator, step.iterable, body))
        else
            throw(ArgumentError("unsupported plan step in parameterization: $(typeof(step))"))
        end
    end
    return parameterized
end

function _assign_parameter_layout(steps::Vector{AbstractPlanStep})
    slots = ParameterSlotSpec[]
    step_counter = Ref(1)
    slot_counter = Ref(1)
    parameter_counter = Ref(1)
    value_counter = Ref(1)
    parameterized = _parameterize_plan_steps(
        steps,
        slots,
        step_counter,
        slot_counter,
        parameter_counter,
        value_counter,
    )
    return parameterized, ParameterLayout(slots, parameter_counter[] - 1, value_counter[] - 1)
end

# --- signature-driven latent/observation classification (issue #95) ---------
#
# The syntactic layout above (`_assign_parameter_layout`) remains the model's
# default. docs/constraint-driven-conditioning.md makes the split a function of
# the CONDITIONING SIGNATURE instead: a static-address, unscoped choice whose
# address is present in the constraints is an OBSERVATION (no slot); every other
# choice is a LATENT and is given a parameter slot when it is structurally
# slot-eligible (has a parameter transform, static address, unscoped). Binding
# is orthogonal, so an UNBOUND latent (`{:a} ~ dist` left unconstrained) also
# gets a slot -- unlike the syntactic pass, this classification does not consult
# `step.binding`.
#
# `observed` is any collection supporting `in` and holds the normalized
# addresses classified as observations for the current signature.

# Normalized address of a fully static choice step, or `nothing` when the
# address is templated (loop/tuple-dynamic) and therefore never slot-eligible.
function _static_choice_address(step::ChoicePlanStep)
    isstaticaddress(step.address) || return nothing
    return normalize_address(tuple((part.value for part in step.address.parts)...))
end

function _parameterize_step_for_signature(
    step::ChoicePlanStep,
    observed,
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
    parameter_counter::Base.RefValue{Int},
    value_counter::Base.RefValue{Int},
)
    step_index = step_counter[]
    step_counter[] += 1

    static = isempty(step.scopes) && isstaticaddress(step.address)
    is_observation = static && (_static_choice_address(step) in observed)
    transform = _parameter_transform(step.rhs)
    slot_eligible = !is_observation && static && !isnothing(transform)

    if !slot_eligible
        # A latent that is structurally ineligible for a slot cannot carry a
        # dependent noncentered transform. (The default build pass already
        # rejects the ineligible shapes at construction, so this only guards
        # the residual case of a noncentered latent whose address is templated.)
        if !is_observation &&
           step.rhs isa DistributionSpec &&
           step.rhs.reparam === :noncentered &&
           isnothing(transform)
            throw(
                ArgumentError(
                    "reparam=:noncentered requires a static-address, unscoped latent choice; " *
                    "the choice at $(step.address) gets no parameter slot",
                ),
            )
        end
        return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, nothing)
    end

    slot_index = slot_counter[]
    slot_counter[] += 1
    dimension, value_length = _parameter_dimensions(transform)
    parameter_index = parameter_counter[]
    value_index = value_counter[]
    parameter_counter[] += dimension
    value_counter[] += value_length
    push!(
        slots,
        ParameterSlotSpec(
            step_index,
            isnothing(step.binding) ? Symbol("") : step.binding,
            step.address,
            parameter_index,
            dimension,
            value_index,
            value_length,
            transform,
        ),
    )
    return ChoicePlanStep(step.choice_index, step.binding, step.address, step.rhs, step.scopes, slot_index)
end

function _parameterize_plan_steps_for_signature(
    steps::Vector{AbstractPlanStep},
    observed,
    slots::Vector{ParameterSlotSpec},
    step_counter::Base.RefValue{Int},
    slot_counter::Base.RefValue{Int},
    parameter_counter::Base.RefValue{Int},
    value_counter::Base.RefValue{Int},
)
    parameterized = AbstractPlanStep[]
    for step in steps
        if step isa ChoicePlanStep
            push!(
                parameterized,
                _parameterize_step_for_signature(
                    step,
                    observed,
                    slots,
                    step_counter,
                    slot_counter,
                    parameter_counter,
                    value_counter,
                ),
            )
        elseif step isa DeterministicPlanStep
            push!(parameterized, step)
        elseif step isa LoopPlanStep
            body = _parameterize_plan_steps_for_signature(
                step.body,
                observed,
                slots,
                step_counter,
                slot_counter,
                parameter_counter,
                value_counter,
            )
            push!(parameterized, LoopPlanStep(step.iterator, step.iterable, body))
        else
            throw(ArgumentError("unsupported plan step in signature parameterization: $(typeof(step))"))
        end
    end
    return parameterized
end

# Re-parameterize an already-built (inlined + env-annotated) execution plan for
# a conditioning signature, producing a fresh `ExecutionPlan` whose parameter
# layout reflects the signature's latent/observation split. The environment
# layout is signature-independent (it depends only on bindings), so it is
# reused; binding slots are re-annotated because reparameterization rebuilds the
# choice steps.
# Automatic marginalization of un-annotated discrete latents (issue #264). A
# finite-support discrete choice has no parameter transform, so if it is neither
# observed nor annotated `marginalize=:enumerate` it currently reaches the
# "requires a provided value" error at score time -- there is no valid way to
# HMC/NUTS it. Auto-detect such latents and route them through the same
# enumeration path the explicit annotation uses. Bounded by the support-product
# cap so a model with many discrete latents cannot silently create an
# intractable enumeration (all-or-nothing: past the cap they stay erroring so the
# user must be explicit).
const AUTO_MARGINALIZE_SUPPORT_LIMIT = 32

# Support size of a discrete latent that can be auto-marginalized (bernoulli, or
# categorical with a literal probability vector so K is known), and only when it
# is not already annotated; `nothing` otherwise. Mirrors `_marginalize_support`.
function _auto_marginalize_support_size(rhs)
    rhs isa DistributionSpec || return nothing
    rhs.marginalize === :none || return nothing
    if rhs.family === :bernoulli
        return 2
    elseif rhs.family === :categorical && length(rhs.arguments) == 1
        probs = rhs.arguments[1]
        if probs isa Expr && probs.head === :vect
            return length(probs.args)
        elseif probs isa AbstractVector
            return length(probs)
        end
    end
    return nothing
end

# Flip eligible unobserved discrete latents to `marginalize=:enumerate` in place
# of the "requires a value" error. Discrete choices carry no parameter slot, so
# this never changes the parameter layout; it only rewrites the choice rhs.
function _auto_marginalize_discrete_latents(steps::Vector{AbstractPlanStep}, observed)
    candidates = Int[]
    support_product = 1
    for (i, step) in enumerate(steps)
        step isa ChoicePlanStep || continue
        step.parameter_slot === nothing || continue                 # continuous latents keep a slot
        (isempty(step.scopes) && isstaticaddress(step.address)) || continue
        (_static_choice_address(step) in observed) && continue      # observed choice: leave as data
        size = _auto_marginalize_support_size(step.rhs)
        size === nothing && continue
        push!(candidates, i)
        support_product *= size
    end
    (isempty(candidates) || support_product > AUTO_MARGINALIZE_SUPPORT_LIMIT) && return steps
    out = copy(steps)
    for i in candidates
        s = steps[i]
        r = s.rhs
        out[i] = ChoicePlanStep(
            s.choice_index, s.binding, s.binding_slot, s.address,
            DistributionSpec(r.family, r.arguments, r.builder, r.reparam, :enumerate),
            s.scopes, s.parameter_slot,
        )
    end
    return out
end

function _signature_execution_plan(base_plan::ExecutionPlan, observed)
    slots = ParameterSlotSpec[]
    step_counter = Ref(1)
    slot_counter = Ref(1)
    parameter_counter = Ref(1)
    value_counter = Ref(1)
    reparameterized = _parameterize_plan_steps_for_signature(
        base_plan.steps,
        observed,
        slots,
        step_counter,
        slot_counter,
        parameter_counter,
        value_counter,
    )
    reparameterized = _auto_marginalize_discrete_latents(reparameterized, observed)
    layout = ParameterLayout(slots, parameter_counter[] - 1, value_counter[] - 1)
    annotated = _annotate_environment_slots(reparameterized, base_plan.environment_layout)
    return ExecutionPlan(base_plan.model_name, annotated, layout, base_plan.environment_layout)
end

function _inline_plan_steps(steps::Vector{AbstractPlanStep})
    expanded = AbstractPlanStep[]
    for step in steps
        append!(expanded, _inline_plan_step(step))
    end
    return _merge_loop_steps(expanded)
end

function _inline_plan_step(step::DeterministicPlanStep)
    isempty(step.scopes) && return AbstractPlanStep[step]
    stripped = DeterministicPlanStep(step.binding, step.binding_slot, step.expr, LoopScopeSpec[])
    return _wrap_steps_with_scopes(AbstractPlanStep[stripped], step.scopes)
end

function _inline_plan_step(step::LoopPlanStep)
    body = _inline_plan_steps(step.body)
    return AbstractPlanStep[LoopPlanStep(step.iterator, step.iterable, body)]
end

function _inline_plan_step(step::ChoicePlanStep)
    if step.rhs isa DistributionSpec || step.rhs isa RawChoiceRhsSpec || step.rhs isa BroadcastDistributionSpec
        return _wrap_steps_with_scopes(
            AbstractPlanStep[ChoicePlanStep(
                step.choice_index,
                step.binding,
                step.address,
                step.rhs,
                LoopScopeSpec[],
                step.parameter_slot,
            )],
            step.scopes,
        )
    elseif step.rhs isa GenerativeCallSpec
        callee = step.rhs.callee
        callee isa TeaModel || throw(ArgumentError("generative call inlining requires a TeaModel callee, got $(typeof(callee))"))

        callee_spec = modelspec(callee)
        substitutions = Dict{Symbol,Any}()
        for (argname, argexpr) in zip(callee_spec.arguments, step.rhs.arguments)
            substitutions[argname] = argexpr
        end

        bound_symbols = Set{Symbol}()
        iterators = Set{Symbol}()
        _collect_bound_symbols!(executionplan(callee).steps, bound_symbols, iterators)
        for sym in union(bound_symbols, iterators)
            substitutions[sym] = gensym(sym)
        end

        inlined = AbstractPlanStep[]
        for inner_step in executionplan(callee).steps
            substituted = _substitute_step(inner_step, substitutions; prefix=step.address, parameter_slot=nothing)
            append!(inlined, _inline_plan_step(substituted))
        end

        if !isnothing(step.binding)
            return_expr = _substitute_expr(callee_spec.return_expr, substitutions)
            push!(inlined, DeterministicPlanStep(step.binding, return_expr))
        end

        return _wrap_steps_with_scopes(inlined, step.scopes)
    end

    throw(ArgumentError("unsupported choice RHS in execution-plan inlining: $(typeof(step.rhs))"))
end

# marginalize=:enumerate is rejected on loop-scoped choices at macro time, but
# submodel inlining can move a flagged choice into a LoopPlanStep afterwards
# (a child model with an enumerated latent called from inside a parent loop);
# re-validate on the inlined plan so the rejection cannot be bypassed.
function _reject_loop_scoped_marginalize(steps, in_loop::Bool)
    for step in steps
        if step isa LoopPlanStep
            _reject_loop_scoped_marginalize(step.body, true)
        elseif in_loop &&
               step isa ChoicePlanStep &&
               step.rhs isa DistributionSpec &&
               step.rhs.marginalize === :enumerate
            throw(
                ArgumentError(
                    "marginalize=:enumerate is not supported on loop-scoped choices " *
                    "(including choices inlined from a submodel called inside a loop; " *
                    "docs/discrete-enumeration.md), found choice at $(step.address)",
                ),
            )
        end
    end
    return nothing
end

# Duplicate choice addresses would silently overwrite earlier choices at
# runtime (and double-score in the compiled plan). Fully static addresses are
# detectable at model construction: two distinct plan steps sharing one
# collide whenever both execute, so reject them here. Loop-generated
# (template) addresses are checked at execution time by the runtime `choice`
# recording instead.
function _reject_duplicate_static_addresses(name::Symbol, steps, seen)
    for step in steps
        if step isa LoopPlanStep
            _reject_duplicate_static_addresses(name, step.body, seen)
        elseif step isa ChoicePlanStep && isstaticaddress(step.address)
            address = normalize_address(tuple((part.value for part in step.address.parts)...))
            if address in seen
                throw(
                    ArgumentError(
                        "duplicate choice address $(address) in model `$(name)`: every random " *
                        "choice needs a unique address, but two choices in the model use this one " *
                        "(the second would silently overwrite the first)",
                    ),
                )
            end
            push!(seen, address)
        end
    end
    return nothing
end

function build_execution_plan(
    name::Symbol,
    arguments::Vector{Symbol},
    raw_steps::Vector{AbstractPlanStep},
    layout::ParameterLayout,
    return_expr::Any,
)
    if isempty(raw_steps)
        environment_layout = _build_environment_layout(arguments, AbstractPlanStep[])
        return ExecutionPlan(name, AbstractPlanStep[], layout, environment_layout)
    end

    steps = _inline_plan_steps(raw_steps)
    _reject_duplicate_static_addresses(name, steps, Set{Any}())
    _reject_loop_scoped_marginalize(steps, false)
    parameterized_steps, parameterized_layout = _assign_parameter_layout(steps)
    environment_layout = _build_environment_layout(arguments, parameterized_steps)
    annotated_steps = _annotate_environment_slots(parameterized_steps, environment_layout)
    return ExecutionPlan(name, annotated_steps, parameterized_layout, environment_layout)
end

function Base.show(io::IO, part::AddressLiteralPart)
    print(io, repr(part.value))
end

function Base.show(io::IO, part::AddressDynamicPart)
    print(io, "\$", repr(part.value))
end

function Base.show(io::IO, address::AddressSpec)
    print(io, "AddressSpec(")
    for (idx, part) in enumerate(address.parts)
        idx > 1 && print(io, " => ")
        show(io, part)
    end
    print(io, ")")
end

function Base.show(io::IO, spec::ChoiceSpec)
    print(io, "ChoiceSpec(")
    isnothing(spec.binding) || print(io, "binding=", spec.binding, ", ")
    print(io, "address=")
    show(io, spec.address)
    print(io, ", rhs=", nameof(typeof(spec.rhs)))
    isempty(spec.scopes) || print(io, ", scopes=", length(spec.scopes))
    print(io, ")")
end

function Base.show(io::IO, spec::ParameterSlotSpec)
    print(
        io,
        "ParameterSlotSpec(index=",
        spec.index,
        ", dimension=",
        spec.dimension,
        ", value_index=",
        spec.value_index,
        ", value_length=",
        spec.value_length,
        ", binding=",
        spec.binding,
        ", choice=",
        spec.choice_index,
        ", transform=",
        nameof(typeof(spec.transform)),
        ")",
    )
end

function Base.show(io::IO, layout::ParameterLayout)
    print(
        io,
        "ParameterLayout(",
        length(layout.slots),
        " slots, parameters=",
        layout.parameter_count,
        ", values=",
        layout.value_count,
        ")",
    )
end

function Base.show(io::IO, step::ChoicePlanStep)
    print(io, "ChoicePlanStep(choice=", step.choice_index)
    isnothing(step.parameter_slot) || print(io, ", parameter_slot=", step.parameter_slot)
    isempty(step.scopes) || print(io, ", scopes=", length(step.scopes))
    print(io, ")")
end

function Base.show(io::IO, step::DeterministicPlanStep)
    print(io, "DeterministicPlanStep(binding=", step.binding, ")")
end

function Base.show(io::IO, step::LoopPlanStep)
    print(io, "LoopPlanStep(iterator=", step.iterator, ", body=", length(step.body), ")")
end

function Base.show(io::IO, plan::ExecutionPlan)
    print(io, "ExecutionPlan(", plan.model_name, ", steps=", length(plan.steps), ")")
end

function Base.show(io::IO, spec::ModelSpec)
    print(io, "ModelSpec(", spec.name, ", mode=", spec.mode, ", choices=", length(spec.choices), ")")
end

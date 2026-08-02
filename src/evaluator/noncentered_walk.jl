# --- dependent-transform plan walk (reparam=:noncentered) ---------------------
#
# Noncentered slots hold the standardized z while the constrained space keeps
# theta = location + scale * z, where location/scale are expressions over
# model arguments and earlier latents. The transform therefore runs as a walk
# over the compiled plan with an environment, visiting slots in execution
# order; centered slots use the ordinary per-slot transforms. Observations
# are bound as NaN poison: a location/scale that depends on one fails loudly.

# positions of (location, scale) among the family's arguments
_noncentered_location_scale_indices(family::Symbol) = family === :studentt ? (2, 3) : (1, 2)

# Poison bound to slotless choices during the walk. Unlike NaN it cannot be
# swallowed by comparisons or branching (`NaN > 0` is silently false): it is
# not a Number, so arithmetic and ordered comparisons raise MethodError, and
# equality is overloaded to throw outright.
struct _TransformUnknownValue end

const _TRANSFORM_UNKNOWN_MESSAGE =
    "reparam=:noncentered location/scale expressions may only depend on model arguments, " *
    "earlier latents with parameter slots, and observed (constrained) values; this model " *
    "routes a choice with no fixed value during the change of variables -- a " *
    "marginalize=:enumerate discrete latent or a loop-scoped binding -- into a noncentered " *
    "location/scale"

Base.:(==)(::_TransformUnknownValue, ::Any) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.:(==)(::Any, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.:(==)(::_TransformUnknownValue, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::_TransformUnknownValue, ::Any) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::Any, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::_TransformUnknownValue, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))

# Evaluate an expression during the walk, converting the poison's MethodError
# into the informative rejection.
function _walk_transform_eval(env::PlanEnvironment, expr)
    try
        return _eval_compiled_expr(env, expr)
    catch err
        if err isa MethodError && any(arg isa _TransformUnknownValue for arg in err.args)
            throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
        end
        rethrow()
    end
end

# --- reachability: which bindings a noncentered loc/scale actually needs ------
#
# Only expressions reachable from a reparam=:noncentered location/scale take part
# in the change of variables, so only those deterministic steps and loops need to
# run during the transform walk (issue #100). We compute, over the compiled plan,
# the set of environment slots a noncentered loc/scale transitively depends on;
# the walk then evaluates a deterministic step (or a loop) only when it populates
# one of those slots. A step that a poisoned (slotless) binding feeds but that
# never reaches a noncentered loc/scale is left unevaluated -- exactly the steps
# whose eager evaluation used to throw spuriously.

_collect_slot_refs!(::Set{Int}, ::CompiledLiteralExpr) = nothing
function _collect_slot_refs!(acc::Set{Int}, expr::CompiledSlotExpr)
    push!(acc, expr.slot)
    return nothing
end
function _collect_slot_refs!(acc::Set{Int}, expr::CompiledCallExpr)
    _collect_slot_refs!(acc, expr.callee)
    for arg in expr.arguments
        _collect_slot_refs!(acc, arg)
    end
    return nothing
end
function _collect_slot_refs!(acc::Set{Int}, expr::Union{CompiledTupleExpr,CompiledVectorExpr,CompiledBlockExpr})
    for arg in expr.arguments
        _collect_slot_refs!(acc, arg)
    end
    return nothing
end
# CompiledNoncentered stores loc/scale untyped; dispatch resolves on the concrete
# compiled-expr subtype at runtime.
_collect_slot_refs!(acc::Set{Int}, expr) = nothing

# Seed = slots read directly by any noncentered loc/scale; `defs` maps a slot to
# the slots the step producing it reads (deterministic bindings and loop
# iterators), recursing into loop bodies.
_collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, steps::Tuple) =
    (foreach(step -> _collect_walk_dependencies!(seed, defs, step), steps); nothing)
function _collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, step::CompiledDeterministicPlanStep)
    refs = Set{Int}()
    _collect_slot_refs!(refs, step.expr)
    defs[step.binding_slot] = refs
    return nothing
end
function _collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, step::CompiledChoicePlanStep)
    if !isnothing(step.noncentered)
        _collect_slot_refs!(seed, step.noncentered.location)
        _collect_slot_refs!(seed, step.noncentered.scale)
    end
    return nothing
end
function _collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, step::CompiledLoopPlanStep)
    refs = Set{Int}()
    _collect_slot_refs!(refs, step.iterable)
    defs[step.iterator_slot] = refs
    _collect_walk_dependencies!(seed, defs, step.body)
    return nothing
end

function _required_walk_slots(steps::Tuple)
    seed = Set{Int}()
    defs = Dict{Int,Set{Int}}()
    _collect_walk_dependencies!(seed, defs, steps)
    required = Set{Int}()
    isempty(seed) && return required
    worklist = collect(seed)
    while !isempty(worklist)
        slot = pop!(worklist)
        slot in required && continue
        push!(required, slot)
        edges = get(defs, slot, nothing)
        isnothing(edges) && continue
        for dep in edges
            dep in required || push!(worklist, dep)
        end
    end
    return required
end

# Whether walking this step (or its loop subtree) can populate a required slot.
_subtree_defines_required(step::CompiledDeterministicPlanStep, required::Set{Int}) =
    step.binding_slot in required
_subtree_defines_required(step::CompiledChoicePlanStep, required::Set{Int}) =
    !isnothing(step.binding_slot) && step.binding_slot in required
function _subtree_defines_required(step::CompiledLoopPlanStep, required::Set{Int})
    step.iterator_slot in required && return true
    for inner in step.body
        _subtree_defines_required(inner, required) && return true
    end
    return false
end

function _dependent_transform_walk!(
    destination::AbstractVector,
    model::TeaModel,
    plan::ExecutionPlan,
    compiled_plan::CompiledExecutionPlan,
    params::AbstractVector,
    args::Tuple,
    inverse::Bool,
    constraints::ChoiceMap=choicemap();
    reject_invalid_parameters::Bool=false,
)
    args = _complete_model_args(model, args)
    env = PlanEnvironment(plan.environment_layout; reject_invalid_parameters=reject_invalid_parameters)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end
    return _walk_transform_steps!(
        destination,
        compiled_plan.steps,
        env,
        plan.parameter_layout,
        params,
        inverse,
        constraints,
        compiled_plan.required_walk_slots,
    )
end

function _dependent_transform_walk!(
    destination::AbstractVector,
    model::TeaModel,
    params::AbstractVector,
    args::Tuple,
    inverse::Bool,
    constraints::ChoiceMap=choicemap();
    reject_invalid_parameters::Bool=false,
)
    return _dependent_transform_walk!(
        destination,
        model,
        executionplan(model),
        _compiled_execution_plan(model),
        params,
        args,
        inverse,
        constraints;
        reject_invalid_parameters=reject_invalid_parameters,
    )
end

# Signature-aware constrained transform used by `logjoint_unconstrained`: it
# transforms against the resolved signature layout so the unconstrained
# parameter vector and the scored plan agree on which choices are latent.
function _transform_to_constrained_with_logabsdet(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap=choicemap();
    reject_invalid_parameters::Bool=false,
)
    layout = resolved.plan.parameter_layout
    expected = parametercount(layout)
    length(params) == expected ||
        throw(_signature_length_error(model, layout, constraints, expected, length(params)))

    constrained = similar(params, parametervaluecount(layout))
    if _has_dependent_transforms(layout)
        logabsdet = _dependent_transform_walk!(
            constrained,
            model,
            resolved.plan,
            resolved.compiled,
            params,
            args,
            false,
            constraints;
            reject_invalid_parameters=reject_invalid_parameters,
        )
        return constrained, logabsdet
    end
    logabsdet = expected == 0 ? 0.0 : zero(params[firstindex(params)])
    for slot in layout.slots
        logabsdet += _transform_slot_to_constrained!(constrained, slot, params)
    end
    return constrained, logabsdet
end

_walk_transform_steps!(destination, ::Tuple{}, env, layout, params, inverse, constraints, required) =
    zero(eltype(destination))

function _walk_transform_steps!(destination, steps::Tuple, env, layout, params, inverse, constraints, required)
    return _walk_transform_step!(destination, first(steps), env, layout, params, inverse, constraints, required) +
           _walk_transform_steps!(destination, Base.tail(steps), env, layout, params, inverse, constraints, required)
end

function _walk_transform_step!(
    destination,
    step::CompiledDeterministicPlanStep,
    env,
    layout,
    params,
    inverse,
    constraints,
    required,
)
    # Only evaluate when the binding feeds a noncentered loc/scale (issue #100);
    # otherwise the step is irrelevant to the change of variables and evaluating
    # it would poison the walk on unrelated slotless choices.
    if step.binding_slot in required
        _environment_set!(env, step.binding_slot, _walk_transform_eval(env, step.expr))
    end
    return zero(eltype(destination))
end

function _walk_transform_step!(destination, step::CompiledLoopPlanStep, env, layout, params, inverse, constraints, required)
    # Skip the loop entirely unless walking it populates a required binding; a
    # loop-scoped binding never reaches a noncentered loc/scale otherwise.
    _subtree_defines_required(step, required) || return zero(eltype(destination))
    iterable = _walk_transform_eval(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    total = zero(eltype(destination))
    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        total += _walk_transform_steps!(destination, step.body, env, layout, params, inverse, constraints, required)
    end
    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return total
end

# Reject-mode handling for a noncentered step whose location/scale evaluated to a
# non-finite Real (issue #202). Mirrors `_compiled_distribution` returning
# `nothing` (scored as -Inf): the change of variables cannot be evaluated here, so
# this step contributes -Inf to the total unconstrained log-joint, which the
# sampler already treats as a divergent/rejected proposal. The destination slot is
# still filled with a finite placeholder so the downstream (reject-mode) scoring
# walk reads valid memory and returns a finite value -- `finite + (-Inf)` is -Inf,
# whereas leaving the non-finite affine in place could make it `NaN`. The binding
# is left unset: a genuinely-unknown binding is only consulted when it feeds
# another noncentered loc/scale (it would be in `required` and poison loudly), and
# the -Inf already forces rejection.
function _reject_noncentered_transform_step!(destination, slot, inverse)
    placeholder = zero(eltype(destination))
    indices = inverse ? parameterindices(slot) : parametervalueindices(slot)
    for index in indices
        destination[index] = placeholder
    end
    return -oftype(placeholder, Inf)
end

function _walk_transform_step!(destination, step::CompiledChoicePlanStep, env, layout, params, inverse, constraints, required)
    if isnothing(step.parameter_slot)
        # Slotless choice. Unified value resolution (issue #95, doc section 2):
        # an OBSERVATION resolves to its constrained value and is bound so
        # downstream deterministic steps and noncentered loc/scale expressions
        # see the real value; a genuinely-unknown slotless latent (a
        # marginalized discrete site) has no single value during the transform
        # and is poisoned so any dependence fails loudly.
        if !isnothing(step.binding_slot)
            address = _concrete_address(env, step.address)
            found, constrained_value = _choice_tryget_normalized(constraints, address)
            _environment_set!(env, step.binding_slot, found ? constrained_value : _TransformUnknownValue())
        end
        return zero(eltype(destination))
    end
    slot = layout.slots[step.parameter_slot]
    if isnothing(step.noncentered)
        if inverse
            _transform_slot_to_unconstrained!(destination, slot, params)
            value = _parameter_slot_value(parametervalueindices(slot), params)
            isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
            return zero(eltype(destination))
        end
        logabsdet = _transform_slot_to_constrained!(destination, slot, params)
        value = _parameter_slot_value(parametervalueindices(slot), destination)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        return logabsdet
    end

    location = _walk_transform_eval(env, step.noncentered.location)
    scale = _walk_transform_eval(env, step.noncentered.scale)
    (location isa _TransformUnknownValue || scale isa _TransformUnknownValue) &&
        throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
    if !(location isa Real && scale isa Real && isfinite(location) && isfinite(scale))
        # A non-finite (Inf/NaN) location/scale is a #157-class parameter-VALUE
        # failure: a leapfrog trajectory on a funnel/heavy-tailed model overshoots
        # to where a finite unconstrained tau maps to a non-finite constrained one,
        # and the noncentered change of variables can no longer be evaluated. In
        # reject mode (sampler-owned workspaces) treat it exactly like
        # `_compiled_distribution` treats an invalid distribution: drive the
        # log-joint to -Inf so the sampler rejects the proposal as divergent,
        # instead of throwing out of the batched gradient. Gated strictly on
        # `env.reject_invalid_parameters`, and -- mirroring the narrowness of the
        # #157 catch (ArgumentError/DomainError only, never structural errors) --
        # only when both are Reals, so a non-Real loc/scale (a genuine structural
        # bug) still throws. Outside reject mode the check throws unchanged, so it
        # keeps catching real model bugs.
        if env.reject_invalid_parameters && location isa Real && scale isa Real
            return _reject_noncentered_transform_step!(destination, slot, inverse)
        end
        throw(
            ArgumentError(
                "reparam=:noncentered location/scale must evaluate to finite reals; got " *
                "location=$location, scale=$scale for the choice bound to :$(slot.binding)",
            ),
        )
    end
    logspace = step.noncentered.logspace
    if inverse
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            constrained_value = params[value_index]
            unconstrained_value = logspace ? log(constrained_value) : constrained_value
            destination[parameter_index] = (unconstrained_value - location) / scale
        end
        value = _parameter_slot_value(parametervalueindices(slot), params)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        return zero(eltype(destination))
    end
    logabsdet = zero(eltype(destination))
    for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
        affine = location + scale * params[parameter_index]
        destination[value_index] = logspace ? exp(affine) : affine
        logabsdet += log(scale) + (logspace ? affine : zero(affine))
    end
    value = _parameter_slot_value(parametervalueindices(slot), destination)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return logabsdet
end

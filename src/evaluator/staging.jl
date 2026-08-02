# --- dense observed-value staging (issue #145) --------------------------------
#
# A dense `Vector{Float64}` read costs ~0.5 ns; the equivalent per-observation
# ChoiceMap lookup assembles a `(literal..., i)` tuple and hashes it (~137 ns +
# 5 allocs). Constraints are bound once per LogjointGradientCache and assumed
# immutable for its lifetime, so the cache pre-resolves every stage-marked
# loop observation (`CompiledChoicePlanStep.stage_index`) into a dense
# per-site vector indexed by the loop index. Anything dynamic -- non-Int loop
# indices, non-Float64 values, indices the staging walk never visited -- falls
# back to the live ChoiceMap lookup, and the assumed immutability is verified
# per evaluation via the ChoiceMap mutation counter (see `_stage_is_current`).

struct StagedObservationSite
    values::Vector{Float64}
    filled::BitVector
end

struct ObservationStage
    sites::Vector{StagedObservationSite}
    constraints::ChoiceMap
    constraints_mutation_count::Int
end

# The staging contract: the stage may only be consulted for the exact
# constraints object it snapshot, and only while that object has not been
# mutated since (gibbs replaces discrete-site values in its merged constraints
# in place between gradient evaluations -- staged values would be stale).
_stage_is_current(stage::ObservationStage, constraints::ChoiceMap) =
    stage.constraints === constraints && stage.constraints_mutation_count == constraints.mutation_count

# Staged value for a stage-marked choice step, or `nothing` when the site (or
# this particular index) must fall back to the ChoiceMap.
@inline function _staged_observation_value(stage::ObservationStage, step::CompiledChoicePlanStep, env::PlanEnvironment)
    # static whole-vector sites (iterator slot -1, issue #288) are consumed only
    # by the generated scorer; the interpreter keeps its ChoiceMap lookup
    step.stage_iterator_slot == -1 && return nothing
    site = @inbounds stage.sites[step.stage_index]
    index = _environment_value(env, step.stage_iterator_slot)
    index isa Int || return nothing
    (1 <= index <= length(site.filled) && @inbounds(site.filled[index])) || return nothing
    return @inbounds site.values[index]
end

const _MaybeObservationStage = Union{Nothing,ObservationStage}

# --- staging walk (issue #145) -------------------------------------------------
#
# Build the dense per-site observation vectors by walking the compiled plan
# once, exactly like a scoring pass but recording constrained values instead of
# summing logpdfs (distributions are never constructed). Latent bindings come
# from the constrained-space `params`; a loop iterable that somehow yields
# different indices at other parameter values is harmless because scoring falls
# back to the ChoiceMap for any index the walk did not fill. Any anomaly aborts
# the walk (`nothing` -- full ChoiceMap fallback); a non-Int index or a
# non-Float64 value deactivates just that site.

function _stage_observations(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    unconstrained_seed::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    compiled = resolved.compiled
    compiled.stage_count == 0 && return nothing
    # NaN guard (issue #346), stamped per mutation_count: batched paths resolve
    # the plan against a representative column only, and gibbs re-stages after
    # in-place mutation, so staging is the chokepoint that sees every column's
    # actual values. Deliberately OUTSIDE the `try` below — the catch means
    # "fall back to live lookups", not "swallow a data bug".
    _validate_constraint_values_not_nan(constraints)
    mutation_count = constraints.mutation_count
    sites = [StagedObservationSite(Float64[], BitVector()) for _ = 1:compiled.stage_count]
    active = trues(compiled.stage_count)
    try
        seed = collect(Float64, unconstrained_seed)
        constrained, _ = _transform_to_constrained_with_logabsdet(model, resolved, seed, args, constraints)
        complete_args = _complete_model_args(model, args)
        env = PlanEnvironment(resolved.plan.environment_layout)
        for (slot, value) in zip(resolved.plan.environment_layout.argument_slots, complete_args)
            _environment_set!(env, slot, value)
        end
        _stage_walk_steps!(sites, active, compiled.steps, env, constrained, constraints)
    catch
        return nothing
    end
    for site_index in eachindex(active)
        if !active[site_index]
            empty!(sites[site_index].values)
            empty!(sites[site_index].filled)
        end
    end
    return ObservationStage(sites, constraints, mutation_count)
end

function _stage_walk_steps!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    steps::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    for step in steps
        _stage_walk_step!(sites, active, step, env, params, constraints)
    end
    return nothing
end

function _stage_walk_step!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    step::CompiledChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    value = if !isnothing(step.parameter_value_indices)
        _parameter_slot_value(step.parameter_value_indices, params)
    else
        address = _concrete_address(env, step.address)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        if found
            constrained_value
        elseif !isnothing(step.marginalize)
            # unconstrained enumerated latent: bind a representative support
            # value so the walk stays linear -- staged values are a pure
            # function of (site, loop index) and the fallback covers any
            # branch-dependent loop pattern
            first(step.marginalize.support)
        else
            # scoring would throw here; abort staging so it does
            throw(ArgumentError("staging walk found no value for choice $(address)"))
        end
    end
    if step.stage_index != 0 && active[step.stage_index] && step.stage_iterator_slot == -1
        # static whole-vector observation site (issue #288): the constrained
        # value is the dense Float64 vector itself (the same Float64-only rule
        # the loop sites apply; anything else deactivates the site)
        site = sites[step.stage_index]
        # Float64 vectors stage directly; INTEGER-valued vectors (Int counts,
        # Bool labels -- the natural spelling of count/binary data, issue #308)
        # convert EXACTLY to Float64, since every count/support classifier
        # (_poisson_count, _bernoulli_value) and scalar kernel scores 3 and 3.0
        # identically. Float32 stays excluded: its promotion-sensitive scoring
        # is not identical under densification.
        if value isa Vector{Float64} && !isempty(value)
            n = length(value)
            resize!(site.values, n)
            copyto!(site.values, value)
            resize!(site.filled, n)
            fill!(site.filled, true)
        elseif value isa AbstractVector{<:Integer} && !isempty(value)
            n = length(value)
            resize!(site.values, n)
            for i = 1:n
                site.values[i] = Float64(value[i])
            end
            resize!(site.filled, n)
            fill!(site.filled, true)
        else
            active[step.stage_index] = false
        end
    elseif step.stage_index != 0 && active[step.stage_index]
        index = _environment_value(env, step.stage_iterator_slot)
        # integer-valued observations densify exactly (issue #308, mirroring the
        # static-vector rule above); Float32 stays excluded
        staged = value isa Float64 ? value : (value isa Integer ? Float64(value) : nothing)
        if index isa Int && index >= 1 && staged isa Float64
            value = staged
            site = sites[step.stage_index]
            if index > length(site.values)
                previous_length = length(site.values)
                resize!(site.values, index)
                resize!(site.filled, index)
                for unfilled = (previous_length+1):index
                    site.filled[unfilled] = false
                end
            end
            site.values[index] = value
            site.filled[index] = true
        else
            active[step.stage_index] = false
        end
    end
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return nothing
end

function _stage_walk_step!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    step::CompiledDeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    _environment_set!(env, step.binding_slot, _eval_compiled_expr(env, step.expr))
    return nothing
end

function _stage_walk_step!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    step::CompiledLoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    iterable = _eval_compiled_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        _stage_walk_steps!(sites, active, step.body, env, params, constraints)
    end
    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return nothing
end

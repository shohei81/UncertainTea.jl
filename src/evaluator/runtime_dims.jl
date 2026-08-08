# --- runtime-dimension resolution walk (issue #289, PR-2) ---------------------
#
# `_resolve_runtime_dims` (evaluator/signature.jl) resolves the length of each
# LATENT runtime-dimension candidate -- an mv-family choice whose size-bearing
# argument is not a literal, e.g. `theta ~ mvnormal(zeros(n), ones(n))` with a
# model argument `n` -- from the model ARGUMENTS alone, so the resolved dims
# tuple can key the signature cache and size the late-constructed
# `VectorIdentityTransform`. The size expression may reference model arguments
# and deterministic bindings derived from them, so resolution is a walk over
# the compiled base plan restricted to the dependency cone of the size
# expressions (the reachability discipline of the noncentered transform walk,
# issue #100): a deterministic step evaluates only when it transitively feeds a
# size expression, and every choice binding is poisoned with
# `_TransformUnknownValue` so a latent-dependent length fails loudly instead of
# resolving from an arbitrary value.

function _runtime_dim_latent_dependent_error(address)
    return ArgumentError(
        "the runtime length of the latent vector at address `$(address)` depends on a random " *
        "choice; a latent vector dimension must be computable from model arguments alone " *
        "(issue #289)",
    )
end

# Whether `err` is the choice-binding poison surfacing: either the MethodError
# arithmetic raises on the non-Number poison, or the ArgumentError its equality
# overloads throw (whose message names the noncentered walk this walk borrows
# the poison from).
function _runtime_dims_is_poison_error(err)
    err isa MethodError && any(arg isa _TransformUnknownValue for arg in err.args) && return true
    err isa ArgumentError && err.msg == _TRANSFORM_UNKNOWN_MESSAGE && return true
    return false
end

# Evaluate `expr` during the dims walk, converting the poison into the
# latent-dependent-length rejection naming the candidate `address` it feeds.
function _runtime_dims_eval(env::PlanEnvironment, expr, address)
    value = try
        _eval_compiled_expr(env, expr)
    catch err
        _runtime_dims_is_poison_error(err) && throw(_runtime_dim_latent_dependent_error(address))
        rethrow()
    end
    value isa _TransformUnknownValue && throw(_runtime_dim_latent_dependent_error(address))
    return value
end

# Transitive closure of the environment slots reachable from `seed` through the
# compiled plan's definition edges (deterministic bindings and loop iterators).
# Reuses the noncentered walk's dependency collector for the edges but seeds
# from the size expressions instead of the noncentered loc/scale reads.
function _runtime_dims_required_slots(steps::Tuple, seed::Set{Int})
    noncentered_seed = Set{Int}()  # collected but deliberately unused here
    defs = Dict{Int,Set{Int}}()
    _collect_walk_dependencies!(noncentered_seed, defs, steps)
    required = Set{Int}()
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

# The address of a candidate whose dependency cone contains `slot`, for error
# naming when a poisoned deterministic step inside the cone fails.
function _runtime_dims_owner_address(slot::Int, required_per, candidates)
    for (index, required) in enumerate(required_per)
        slot in required && return candidates[index].address
    end
    return candidates[1].address
end

# `length` of the candidate's evaluated size expression, validated to a
# positive Int.
function _runtime_dims_candidate_length(env::PlanEnvironment, candidate, compiled_size)
    value = _runtime_dims_eval(env, compiled_size, candidate.address)
    dim = try
        Int(length(value))
    catch err
        _runtime_dims_is_poison_error(err) && throw(_runtime_dim_latent_dependent_error(candidate.address))
        throw(
            ArgumentError(
                "could not resolve the runtime length of the latent `$(candidate.family)` at " *
                "address `$(candidate.address)` from the model arguments (issue #289): `length` " *
                "of the evaluated size-bearing argument failed ($(typeof(err)))",
            ),
        )
    end
    dim >= 1 || throw(
        ArgumentError(
            "the runtime length of the latent `$(candidate.family)` at address " *
            "`$(candidate.address)` resolved to $(dim); a latent vector needs a positive length",
        ),
    )
    return dim
end

function _runtime_dims_walk_steps!(dims, steps::Tuple, env, required, required_per, candidates, compiled_sizes, pending)
    for step in steps
        _runtime_dims_walk_step!(dims, step, env, required, required_per, candidates, compiled_sizes, pending)
    end
    return nothing
end

function _runtime_dims_walk_step!(
    dims,
    step::CompiledDeterministicPlanStep,
    env,
    required,
    required_per,
    candidates,
    compiled_sizes,
    pending,
)
    step.binding_slot in required || return nothing
    address = _runtime_dims_owner_address(step.binding_slot, required_per, candidates)
    _environment_set!(env, step.binding_slot, _runtime_dims_eval(env, step.expr, address))
    return nothing
end

function _runtime_dims_walk_step!(
    dims,
    step::CompiledLoopPlanStep,
    env,
    required,
    required_per,
    candidates,
    compiled_sizes,
    pending,
)
    # walk the loop only when its subtree populates a required binding; a
    # candidate never lives inside a loop (a static address there collides at
    # runtime), so skipping is safe for the pending set
    _subtree_defines_required(step, required) || return nothing
    address = _runtime_dims_owner_address(step.iterator_slot, required_per, candidates)
    iterable = _runtime_dims_eval(env, step.iterable, address)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        _runtime_dims_walk_steps!(dims, step.body, env, required, required_per, candidates, compiled_sizes, pending)
    end
    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return nothing
end

function _runtime_dims_walk_step!(
    dims,
    step::CompiledChoicePlanStep,
    env,
    required,
    required_per,
    candidates,
    compiled_sizes,
    pending,
)
    # a candidate's size expression is evaluated AT its step position, so a
    # later rebinding of a name it reads cannot corrupt the resolved length
    if !isempty(pending) && all(part -> part isa CompiledAddressLiteralPart, step.address.parts)
        index = pop!(pending, _concrete_address(env, step.address), 0)
        if index != 0
            dims[index] = _runtime_dims_candidate_length(env, candidates[index], compiled_sizes[index])
        end
    end
    # every choice binding is poison: latents have no value during resolution,
    # and observed values are excluded on purpose (the dims must be computable
    # from the model arguments alone to stay a value-independent cache key)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, _TransformUnknownValue())
    return nothing
end

# Resolve the runtime length of each latent candidate (in plan order) against
# `args`. Called only on the resolution cold path (a signature-cache miss plus
# one cheap re-walk per resolution call); correctness over speed.
function _runtime_dims_walk(
    @nospecialize(model::TeaModel),
    candidates::Vector{RuntimeDimCandidate},
    @nospecialize(args::Tuple),
)
    plan = executionplan(model)
    compiled = _compiled_execution_plan(model)
    layout = plan.environment_layout
    completed_args = _complete_model_args(model, args)

    compiled_sizes = Any[_compile_plan_expr(model, layout, c.size_expr) for c in candidates]
    # per-candidate dependency cones: the union picks which deterministic steps
    # to evaluate; the per-candidate sets name the candidate when a poisoned
    # step inside its cone fails
    seeds = [Set{Int}() for _ in candidates]
    for (seed, compiled_size) in zip(seeds, compiled_sizes)
        _collect_slot_refs!(seed, compiled_size)
    end
    required_per = [_runtime_dims_required_slots(compiled.steps, seed) for seed in seeds]
    required = Set{Int}()
    for per in required_per
        union!(required, per)
    end

    env = PlanEnvironment(layout)
    for (slot, value) in zip(layout.argument_slots, completed_args)
        _environment_set!(env, slot, value)
    end

    pending = Dict{Any,Int}(candidate.address => index for (index, candidate) in enumerate(candidates))
    dims = fill(-1, length(candidates))
    _runtime_dims_walk_steps!(dims, compiled.steps, env, required, required_per, candidates, compiled_sizes, pending)
    # a candidate whose choice step was not matched during the walk (defensive:
    # e.g. a plan shape the matcher does not recognize) resolves against the
    # final environment -- argument-derived bindings persist after the walk
    for (index, dim) in enumerate(dims)
        dim == -1 || continue
        dims[index] = _runtime_dims_candidate_length(env, candidates[index], compiled_sizes[index])
    end
    return tuple(dims...)
end

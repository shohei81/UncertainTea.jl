# --- type-stable generated single-chain scorer (issue #144) -------------------
#
# The compiled-plan interpreter in evaluator.jl (`_score_compiled_steps` /
# `_score_plan_step!`) infers `Any`: `PlanEnvironment.values::Vector{Any}`
# boxes every environment read, and per-observation ChoiceMap lookups assemble
# and hash `(literal..., i)` tuples. `Base.return_types(_score_compiled_steps,
# ...)` is `Any`, so ForwardDiff differentiates boxed code.
#
# This file emits, per resolved signature plan, a straight-line typed Julia
# function directly from the concrete `CompiledExecutionPlan`: environment
# slots become `local` variables (unboxed, inference-friendly), distributions
# are constructed from concrete constructors, and dense loop observations are
# read from a per-site `Vector{Float64}` (built once from the constraints via
# the existing #145 staging walk) instead of hashing the ChoiceMap. The
# accumulator is `zero(eltype(params))`, so ForwardDiff sees unboxed Duals.
#
# The generated scorer is used only for the SINGLE-CHAIN CPU scoring entry
# points (`logjoint` / `logjoint_unconstrained` / `logjoint_gradient_unconstrained`
# and the single-chain samplers' interpreter gradient cache). The batched and
# device paths are untouched. Any plan the generator cannot represent falls
# back to the interpreter, which stays the source of truth for numerics.
#
# --- which model classes are generated ----------------------------------------
#
# Generated (structural gate `_gen_structural_ok`):
#   * every choice step is either PARAMETER-VALUED (a latent read from `params`,
#     including reparam=:noncentered latents, whose change of variables is done
#     by the interpreted transform walk before scoring) or a STAGED loop
#     observation (`stage_index != 0`, a `(literal..., loop-index)` address with
#     a Float64 value);
#   * no marginalize=:enumerate site (the logsumexp suffix ownership stays
#     interpreted);
#   * the emitted body (loop bodies counted once) is under `_GEN_MAX_STEPS`, to
#     bound first-call compile time (issue #155 / the TTFX watch).
#
# Interpreted fallback (still fully correct, just not accelerated):
#   * scalar / non-loop observations, broadcast (`y .~`) observations, and any
#     observation whose constrained value is not Float64 (e.g. the Float32
#     fixtures) -- these are not densifiable, so the obs-vector build returns
#     `nothing`;
#   * marginalize=:enumerate models;
#   * dynamic-mode / branchful models (already rejected before compilation);
#   * anything the emitter or the obs-vector build cannot handle (it aborts to
#     `nothing`).

# Global switch + observability, used by the numerical-identity tests to toggle
# generated vs interpreter in-process and to assert the generated path actually
# ran. Not part of the public API.
const _USE_GENERATED_SCORER = Ref(true)
const _GEN_SCORER_LAST_USED = Ref(false)

# Bound on emitted straight-line statements (loop bodies counted once). A plan
# with thousands of distinct un-looped steps would blow up first-call compile
# time; such plans fall back to the interpreter.
const _GEN_MAX_STEPS = 400

struct _GeneratedScorerCache
    structural_ok::Bool
    # emitted scorers keyed by reject flag; `nothing` until first requested
    scorer_noreject::Base.RefValue{Any}
    scorer_reject::Base.RefValue{Any}
end

_GeneratedScorerCache(structural_ok::Bool) =
    _GeneratedScorerCache(structural_ok, Ref{Any}(nothing), Ref{Any}(nothing))

# --- structural analysis ------------------------------------------------------

_gen_choice_ok(step::CompiledChoicePlanStep) =
    isnothing(step.marginalize) && (!isnothing(step.parameter_value_indices) || step.stage_index != 0)

_gen_steps_structural_ok(::Tuple{}) = true
function _gen_steps_structural_ok(steps::Tuple)
    return _gen_step_structural_ok(first(steps)) && _gen_steps_structural_ok(Base.tail(steps))
end

_gen_step_structural_ok(step::CompiledChoicePlanStep) = _gen_choice_ok(step)
_gen_step_structural_ok(step::CompiledDeterministicPlanStep) = true
_gen_step_structural_ok(step::CompiledLoopPlanStep) = _gen_steps_structural_ok(step.body)

# emitted-statement budget (loop bodies counted once)
_gen_step_count(::Tuple{}) = 0
_gen_step_count(steps::Tuple) = _gen_one_count(first(steps)) + _gen_step_count(Base.tail(steps))
_gen_one_count(::CompiledChoicePlanStep) = 1
_gen_one_count(::CompiledDeterministicPlanStep) = 1
_gen_one_count(step::CompiledLoopPlanStep) = 1 + _gen_step_count(step.body)

function _gen_structural_ok(compiled::CompiledExecutionPlan)
    compiled.stage_count == 0 && return false
    _gen_steps_structural_ok(compiled.steps) || return false
    _gen_step_count(compiled.steps) <= _GEN_MAX_STEPS || return false
    return true
end

# --- slot collection (which env slots become `local`s vs loop iterators) ------

function _gen_collect_slots!(binding::Set{Int}, iterators::Set{Int}, steps::Tuple)
    for step in steps
        _gen_collect_slots_step!(binding, iterators, step)
    end
    return nothing
end
function _gen_collect_slots_step!(binding, iterators, step::CompiledChoicePlanStep)
    isnothing(step.binding_slot) || push!(binding, step.binding_slot)
    return nothing
end
function _gen_collect_slots_step!(binding, iterators, step::CompiledDeterministicPlanStep)
    push!(binding, step.binding_slot)
    return nothing
end
function _gen_collect_slots_step!(binding, iterators, step::CompiledLoopPlanStep)
    push!(iterators, step.iterator_slot)
    _gen_collect_slots!(binding, iterators, step.body)
    return nothing
end

_gen_slot_sym(slot::Int) = Symbol("slot_", slot)

# --- expression emission ------------------------------------------------------
#
# Compiled-expr -> Julia AST. Literal VALUES are inlined as constants (Symbols
# and Exprs are `QuoteNode`d so they stay data, not variable references);
# function/distribution objects are inlined directly, so the emitted code calls
# the identical functions the interpreter does and the numerics match.

_gen_lit(v) = (v isa Symbol || v isa Expr) ? QuoteNode(v) : v

_gen_emit_expr(e::CompiledLiteralExpr) = _gen_lit(e.value)
_gen_emit_expr(e::CompiledSlotExpr) = _gen_slot_sym(e.slot)
_gen_emit_expr(e::CompiledCallExpr) =
    Expr(:call, _gen_emit_expr(e.callee), map(_gen_emit_expr, e.arguments)...)
_gen_emit_expr(e::CompiledTupleExpr) = Expr(:tuple, map(_gen_emit_expr, e.arguments)...)
_gen_emit_expr(e::CompiledVectorExpr) = Expr(:ref, :Any, map(_gen_emit_expr, e.arguments)...)
_gen_emit_expr(e::CompiledBlockExpr) = Expr(:block, map(_gen_emit_expr, e.arguments)...)

# --- step emission ------------------------------------------------------------

# Push the score contribution for one distribution `dist_expr` observed/latent
# at `value_expr` onto `body`. `reject` mirrors the interpreter's Stan-style
# reject semantics (issue #157): a parameter-validation error (ArgumentError /
# DomainError) from argument evaluation or the constructor scores that draw as
# -Inf with clean (zero) partials instead of throwing.
function _gen_push_score!(body::Vector{Any}, dist_expr, value_expr, reject::Bool)
    if !reject
        d = gensym(:dist)
        push!(body, Expr(:(=), d, dist_expr))
        push!(body, Expr(:(+=), :acc, Expr(:call, logpdf, d, value_expr)))
        return nothing
    end
    d = gensym(:dist)
    ok = gensym(:ok)
    e = gensym(:err)
    push!(body, Expr(:local, d))
    push!(body, Expr(:(=), ok, true))
    push!(
        body,
        Expr(
            :try,
            Expr(:block, Expr(:(=), d, dist_expr)),
            e,
            Expr(
                :block,
                Expr(
                    :||,
                    Expr(:call, |, Expr(:call, isa, e, ArgumentError), Expr(:call, isa, e, DomainError)),
                    Expr(:call, rethrow),
                ),
                Expr(:(=), ok, false),
            ),
        ),
    )
    push!(
        body,
        Expr(
            :if,
            ok,
            Expr(:(+=), :acc, Expr(:call, logpdf, d, value_expr)),
            Expr(:(+=), :acc, Expr(:call, oftype, :acc, -Inf)),
        ),
    )
    return nothing
end

_gen_dist_expr(step::CompiledChoicePlanStep) =
    Expr(:call, step.constructor, map(_gen_emit_expr, step.arguments)...)

function _gen_choice_value_expr(step::CompiledChoicePlanStep)
    if !isnothing(step.parameter_value_indices)
        indices = step.parameter_value_indices
        if length(indices) == 1
            return Expr(:ref, :params, first(indices))
        end
        return Expr(:call, collect, Expr(:call, view, :params, indices))
    end
    # staged loop observation: dense per-site Float64 vector indexed by the
    # enclosing loop iterator slot
    return Expr(:ref, Expr(:ref, :obs, step.stage_index), _gen_slot_sym(step.stage_iterator_slot))
end

function _gen_emit_step!(body::Vector{Any}, step::CompiledChoicePlanStep, reject::Bool)
    value_expr = _gen_choice_value_expr(step)
    if isnothing(step.binding_slot)
        _gen_push_score!(body, _gen_dist_expr(step), value_expr, reject)
    else
        v = gensym(:val)
        push!(body, Expr(:(=), v, value_expr))
        push!(body, Expr(:(=), _gen_slot_sym(step.binding_slot), v))
        _gen_push_score!(body, _gen_dist_expr(step), v, reject)
    end
    return nothing
end

function _gen_emit_step!(body::Vector{Any}, step::CompiledDeterministicPlanStep, reject::Bool)
    push!(body, Expr(:(=), _gen_slot_sym(step.binding_slot), _gen_emit_expr(step.expr)))
    return nothing
end

function _gen_emit_step!(body::Vector{Any}, step::CompiledLoopPlanStep, reject::Bool)
    inner = Any[]
    for s in step.body
        _gen_emit_step!(inner, s, reject)
    end
    push!(
        body,
        Expr(:for, Expr(:(=), _gen_slot_sym(step.iterator_slot), _gen_emit_expr(step.iterable)), Expr(:block, inner...)),
    )
    return nothing
end

# --- function body assembly + @eval ------------------------------------------

function _gen_build_function(resolved::ResolvedSignaturePlan, reject::Bool)
    compiled = resolved.compiled
    layout = resolved.plan.environment_layout
    binding = Set{Int}()
    iterators = Set{Int}()
    _gen_collect_slots!(binding, iterators, compiled.steps)

    body = Any[]
    # `local`-declare argument and binding slots so loop-body bindings update
    # the enclosing local (persisting after the loop, as the interpreter keeps
    # non-iterator slots). Loop iterators are owned by their `for`.
    locals = sort!(collect(union(binding, Set(layout.argument_slots))))
    locals = filter(s -> !(s in iterators), locals)
    isempty(locals) || push!(body, Expr(:local, map(_gen_slot_sym, locals)...))
    # bind argument slots positionally from the completed args tuple
    for (k, slot) in enumerate(layout.argument_slots)
        push!(body, Expr(:(=), _gen_slot_sym(slot), Expr(:ref, :args, k)))
    end
    # accumulator promoted from the parameter eltype (Float64 for logjoint,
    # Dual under ForwardDiff), matching the interpreter's `0.0` start to <1 ulp
    push!(body, Expr(:(=), :acc, Expr(:call, zero, Expr(:call, eltype, :params))))
    for step in compiled.steps
        _gen_emit_step!(body, step, reject)
    end
    push!(body, Expr(:return, :acc))

    name = gensym(reject ? :gen_scorer_reject : :gen_scorer)
    fnexpr = Expr(
        :function,
        Expr(:call, name, Expr(:(::), :args, :Tuple), :params, Expr(:(::), :obs, :(Vector{Vector{Float64}}))),
        Expr(:block, body...),
    )
    return @eval $fnexpr
end

# --- cache + accessors --------------------------------------------------------

function _generated_scorer_cache(resolved::ResolvedSignaturePlan)
    cached = resolved.generated_scorer_cache[]
    if !isnothing(cached)
        return cached::_GeneratedScorerCache
    end
    return lock(_PLAN_MEMO_LOCK) do
        again = resolved.generated_scorer_cache[]
        if isnothing(again)
            again = _GeneratedScorerCache(_gen_structural_ok(resolved.compiled))
            resolved.generated_scorer_cache[] = again
        end
        again::_GeneratedScorerCache
    end
end

# The emitted scorer function for this plan (generating it once per reject
# variant under the plan memo lock), or `nothing` when the plan is not
# structurally generatable or the generated path is switched off.
function _generated_scorer(resolved::ResolvedSignaturePlan, reject::Bool)
    _USE_GENERATED_SCORER[] || return nothing
    cache = _generated_scorer_cache(resolved)
    cache.structural_ok || return nothing
    slot = reject ? cache.scorer_reject : cache.scorer_noreject
    existing = slot[]
    isnothing(existing) || return existing
    return lock(_PLAN_MEMO_LOCK) do
        again = slot[]
        if isnothing(again)
            again = _gen_build_function(resolved, reject)
            slot[] = again
        end
        again
    end
end

# --- dense observation-vector build ------------------------------------------
#
# The generated scorer reads observations from `Vector{Vector{Float64}}` built
# from the constraints. This reuses the #145 staging walk; the observation
# values are constants (independent of the Dual `params`), so one build from a
# Float64 seed is valid for every parameter vector of a given constraints
# state. When the constraints object is mutated in place (gibbs), the reused
# gradient objective re-stages via `_gen_refresh_obs!`, matching the
# interpreter's per-call `_stage_is_current` fallback.

# Extract the dense per-site vectors from staging sites, or `nothing` if any
# site is inactive or has a gap (fall back to the interpreter).
function _gen_obs_from_sites(sites::Vector{StagedObservationSite}, active::BitVector)
    vecs = Vector{Vector{Float64}}(undef, length(sites))
    for k in eachindex(sites)
        active[k] || return nothing
        site = sites[k]
        (length(site.filled) == length(site.values) && all(site.filled)) || return nothing
        vecs[k] = site.values
    end
    return vecs
end

# Build dense obs vectors from an already-built `ObservationStage` (the
# gradient path, which stages once from the seed).
function _gen_obs_from_stage(stage::_MaybeObservationStage)
    stage isa ObservationStage || return nothing
    isempty(stage.sites) && return nothing
    vecs = Vector{Vector{Float64}}(undef, length(stage.sites))
    for k in eachindex(stage.sites)
        site = stage.sites[k]
        (!isempty(site.values) && length(site.filled) == length(site.values) && all(site.filled)) || return nothing
        vecs[k] = site.values
    end
    return vecs
end

# Build dense obs vectors directly from a set of constrained-space parameters
# (the scalar `logjoint` path, which has no pre-built stage). `constrained`
# supplies latent bindings for loop bounds; the stored values are the
# constrained observation values.
function _gen_obs_from_params(
    resolved::ResolvedSignaturePlan,
    constrained::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    compiled = resolved.compiled
    compiled.stage_count == 0 && return nothing
    sites = [StagedObservationSite(Float64[], BitVector()) for _ = 1:compiled.stage_count]
    active = trues(compiled.stage_count)
    try
        env = PlanEnvironment(resolved.plan.environment_layout)
        for (slot, value) in zip(resolved.plan.environment_layout.argument_slots, args)
            _environment_set!(env, slot, value)
        end
        _stage_walk_steps!(sites, active, compiled.steps, env, constrained, constraints)
    catch
        return nothing
    end
    return _gen_obs_from_sites(sites, active)
end

# --- gradient objective (type-stable capture) ---------------------------------
#
# A concrete callable struct capturing the generated scorer AND the dense obs
# vectors with their concrete types, so `LogjointGradientCache{F}` is
# type-concrete and ForwardDiff differentiates unboxed code. Being a distinct
# type also lets `_logjoint_gradient!` recognize the generated objective and
# cross the world-age boundary (the scorer method is emitted after the caller's
# world) with a single `invokelatest`, leaving the interpreter objective on its
# direct call path. The interpreted transform walk still runs (O(latents); it
# contributes one dynamic add of `logabsdet`, never per-observation boxing).
#
# The dense `obs` vectors are refreshed by `_gen_refresh_obs!` whenever the
# constraints object is mutated (gibbs replaces merged-constraint values in
# place between gradient evaluations); the mutable `obs`/`obs_mutation_count`
# and the retained unconstrained `seed` (used to re-run the staging walk) mirror
# the interpreter's per-call `_stage_is_current` fallback so a reused cache
# never scores stale observations.
mutable struct _GenGradientObjective{S,M,R,A,C}
    scorer::S
    model::M
    resolved::R
    args::A
    constraints::C
    seed::Vector{Float64}
    obs::Vector{Vector{Float64}}
    obs_mutation_count::Int
    reject::Bool
end

function (objective::_GenGradientObjective)(theta)
    constrained, logabsdet = _transform_to_constrained_with_logabsdet(
        objective.model, objective.resolved, theta, objective.args, objective.constraints,
    )
    return objective.scorer(objective.args, constrained, objective.obs) + logabsdet
end

function _gen_gradient_objective(
    scorer,
    model,
    resolved,
    args,
    constraints::ChoiceMap,
    seed::Vector{Float64},
    obs::Vector{Vector{Float64}};
    reject::Bool=false,
)
    return _GenGradientObjective(
        scorer, model, resolved, args, constraints, seed, obs, constraints.mutation_count, reject,
    )
end

# Re-stage the dense obs vectors when the constraints mutated since the last
# build; `true` on success (obs current), `false` when the mutated constraints
# can no longer be densified (the caller must fall back to the interpreter).
function _gen_refresh_obs!(objective::_GenGradientObjective)
    objective.constraints.mutation_count == objective.obs_mutation_count && return true
    stage = _stage_observations(
        objective.model, objective.resolved, objective.seed, objective.args, objective.constraints,
    )
    new_obs = _gen_obs_from_stage(stage)
    isnothing(new_obs) && return false
    objective.obs = new_obs
    objective.obs_mutation_count = objective.constraints.mutation_count
    return true
end

# Interpreter gradient for a generated cache whose constraints mutated into a
# no-longer-densifiable shape (rare; gibbs keeps observations Float64). Uses the
# live-lookup interpreter objective and a one-off config.
function _gen_interpreter_gradient!(objective::_GenGradientObjective, buffer, params)
    reject = objective.reject
    interp =
        theta -> _logjoint_unconstrained(
            objective.model, objective.resolved, theta, objective.args, objective.constraints, nothing;
            reject_invalid_parameters=reject,
        )
    ForwardDiff.gradient!(buffer, interp, params)
    return buffer
end

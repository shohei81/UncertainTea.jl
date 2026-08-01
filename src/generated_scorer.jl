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

# Sufficient-statistics fusion switch (issue #138). When on (production), a
# fusable observation loop (single staged observed choice, exponential-family
# family with LOOP-INVARIANT parameter expressions, no binding/reparam/
# marginalize) is scored in O(1) from statistics computed ONCE at obs-build
# time, instead of the O(observations) per-iteration logpdf loop. The emitted
# scorer reads this Ref so the numerical-identity tests can compare the fused
# and unfused generated paths in the same process without re-emitting: off ->
# the per-observation loop (the #144 unfused path); on with data the closed
# form cannot represent exactly (recorded as empty stats at build time) also
# takes the per-observation loop.
const _GEN_SCORER_SUFFSTATS = Ref(true)

# Bound on emitted straight-line statements (loop bodies counted once). A plan
# with thousands of distinct un-looped steps would blow up first-call compile
# time; such plans fall back to the interpreter.
const _GEN_MAX_STEPS = 400

struct _GeneratedScorerCache
    structural_ok::Bool
    # per stage-index sufficient-statistics fusion family (issue #138): `:normal`
    # / `:exponential` / `:poisson` for a fusable loop observation, `:none`
    # otherwise. Length `stage_count`; drives which stages get statistics built
    # (`_gen_build_stats`), consistently with what the emitter fuses (both derive
    # from `_gen_loop_fusable_family`).
    stage_fusion::Vector{Symbol}
    # emitted scorers keyed by reject flag; `nothing` until first requested
    scorer_noreject::Base.RefValue{Any}
    scorer_reject::Base.RefValue{Any}
end

_GeneratedScorerCache(structural_ok::Bool, stage_fusion::Vector{Symbol}) =
    _GeneratedScorerCache(structural_ok, stage_fusion, Ref{Any}(nothing), Ref{Any}(nothing))

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

# --- sufficient-statistics fusion: structural analysis (issue #138) -----------
#
# A loop observation is fusable when its body is a single staged observed choice
# of an exponential-family family (normal / exponential / poisson) whose
# parameter expressions are LOOP-INVARIANT (never read the loop iterator). Then
# the whole per-observation logpdf loop collapses to a closed form over
# statistics of the data alone (computed once at obs-build time), read in O(1)
# by the emitted scorer -- mirroring the batched #146 fusion. Anything else
# (binding, reparam, marginalize, a parameter that reads the iterator, an
# unsupported family) keeps the per-observation loop.

# Does a compiled expression read the given environment slot? Conservative
# (returns `true`) on any node the walker does not recognize, so an unanalyzable
# parameter expression is never mistaken for loop-invariant.
_gen_expr_reads_slot(e::CompiledLiteralExpr, slot::Int) = false
_gen_expr_reads_slot(e::CompiledSlotExpr, slot::Int) = e.slot == slot
_gen_expr_reads_slot(e::CompiledCallExpr, slot::Int) =
    _gen_expr_reads_slot(e.callee, slot) || any(a -> _gen_expr_reads_slot(a, slot), e.arguments)
_gen_expr_reads_slot(e::CompiledTupleExpr, slot::Int) = any(a -> _gen_expr_reads_slot(a, slot), e.arguments)
_gen_expr_reads_slot(e::CompiledVectorExpr, slot::Int) = any(a -> _gen_expr_reads_slot(a, slot), e.arguments)
_gen_expr_reads_slot(e::CompiledBlockExpr, slot::Int) = any(a -> _gen_expr_reads_slot(a, slot), e.arguments)
_gen_expr_reads_slot(::Any, ::Int) = true

# Map a choice-step constructor to its fusable family tag, or `nothing`. Only the
# exact module builders match; a custom `builder` distribution keeps the loop.
_gen_fusable_family(constructor) =
    constructor === normal ? :normal :
    constructor === exponential ? :exponential : constructor === poisson ? :poisson : nothing

# The fusable family of a loop step, or `nothing` if the loop is not fusable.
function _gen_loop_fusable_family(step::CompiledLoopPlanStep)
    length(step.body) == 1 || return nothing
    choice = step.body[1]
    choice isa CompiledChoicePlanStep || return nothing
    choice.stage_index == 0 && return nothing                 # must be a staged observation
    isnothing(choice.binding_slot) || return nothing          # value not bound to a variable
    isnothing(choice.parameter_value_indices) || return nothing  # observation, not a latent
    isnothing(choice.marginalize) || return nothing
    isnothing(choice.noncentered) || return nothing
    family = _gen_fusable_family(choice.constructor)
    isnothing(family) && return nothing
    for arg in choice.arguments
        _gen_expr_reads_slot(arg, step.iterator_slot) && return nothing  # loop-invariant params only
    end
    return family
end

# Per stage-index fusion family for the whole plan (`:none` where not fusable).
function _gen_stage_fusion(compiled::CompiledExecutionPlan)
    fusion = fill(:none, compiled.stage_count)
    _gen_collect_fusion!(fusion, compiled.steps)
    return fusion
end
_gen_collect_fusion!(::Vector{Symbol}, ::Tuple{}) = nothing
function _gen_collect_fusion!(fusion::Vector{Symbol}, steps::Tuple)
    _gen_collect_fusion_step!(fusion, first(steps))
    _gen_collect_fusion!(fusion, Base.tail(steps))
end
_gen_collect_fusion_step!(::Vector{Symbol}, ::CompiledChoicePlanStep) = nothing
_gen_collect_fusion_step!(::Vector{Symbol}, ::CompiledDeterministicPlanStep) = nothing
function _gen_collect_fusion_step!(fusion::Vector{Symbol}, step::CompiledLoopPlanStep)
    family = _gen_loop_fusable_family(step)
    isnothing(family) || (fusion[step.body[1].stage_index] = family)
    _gen_collect_fusion!(fusion, step.body)  # nested loops when this one is not fusable
    return nothing
end

# --- sufficient-statistics fusion: data statistics (issue #138) ---------------
#
# Built ONCE per obs-vector build (obs-build time), from the Float64 observation
# data alone -- never from the parameters -- so a single build is valid for
# every parameter vector (Float64 or Dual) the cached scorer is later called
# with. Each stage's stats are encoded as a small `Vector{Float64}` (kept type-
# stable, unlike an Any-boxed struct, so the scorer reads them without dynamic
# dispatch); an EMPTY vector means "not fused" -- either the stage is not a
# fusable family or its data cannot take the closed form (an empty loop, a
# non-finite normal value, a negative exponential value, a non-count poisson
# value), in which case the emitted scorer runs the per-observation loop.

# Normal: CENTERED statistics (n, ybar, residual = sum(y - ybar),
# S2c = sum((y - ybar)^2)). The residual carries the ~1 ulp the stored mean drops
# so the moment identity in `_gen_fused_sum_expr` is exact about `ybar`; the
# centered second moment makes the sum cancellation-free even when |ybar| >>
# sigma (the naive power-sum form is not). Mirrors `_NormalObservationStats`.
function _gen_normal_stats(y::Vector{Float64})
    n = length(y)
    (n > 0 && all(isfinite, y)) || return Float64[]
    mean = sum(y) / n
    residual = sum(v -> v - mean, y)
    centered_sum_squares = sum(v -> abs2(v - mean), y)
    return Float64[n, mean, residual, centered_sum_squares]
end

# Exponential: (n, sum(y)). A negative observation scores -Inf in the per-obs
# form, which the closed form cannot represent, so keep those loops unfused.
function _gen_exponential_stats(y::Vector{Float64})
    n = length(y)
    (n > 0 && all(v -> isfinite(v) && v >= 0.0, y)) || return Float64[]
    return Float64[n, sum(y)]
end

# Poisson: (n, sum(y), sum(log y!)); the log-factorial total is a data constant
# accumulated with the same `_logfactorial_like` the per-observation logpdf uses.
# A non-count observation scores -Inf per-obs, so keep those loops unfused.
function _gen_poisson_stats(y::Vector{Float64})
    n = length(y)
    n > 0 || return Float64[]
    total = 0.0
    log_factorial_total = 0.0
    for v in y
        count = _poisson_count(v)
        isnothing(count) && return Float64[]
        total += count
        log_factorial_total += _logfactorial_like(1.0, count)
    end
    return Float64[n, total, log_factorial_total]
end

function _gen_compute_stats(family::Symbol, y::Vector{Float64})
    family === :normal && return _gen_normal_stats(y)
    family === :exponential && return _gen_exponential_stats(y)
    family === :poisson && return _gen_poisson_stats(y)
    return Float64[]
end

# Statistics for every stage, parallel to the dense obs vectors.
function _gen_build_stats(stage_fusion::Vector{Symbol}, obs::Vector{Vector{Float64}})
    stats = Vector{Vector{Float64}}(undef, length(obs))
    for k in eachindex(obs)
        stats[k] = _gen_compute_stats(stage_fusion[k], obs[k])
    end
    return stats
end

# Statistics for the given obs vectors under a resolved plan's fusion map.
function _gen_stats(resolved::ResolvedSignaturePlan, obs::Vector{Vector{Float64}})
    return _gen_build_stats(_generated_scorer_cache(resolved).stage_fusion, obs)
end

# Dense obs vectors + their statistics for a memoized `ObservationStage`, cached
# single-slot on the resolved plan (issue #138). The public
# `logjoint_gradient_unconstrained` builds a fresh objective per call; without
# this memo the O(observations) statistics scan would run every call, keeping
# the entry point O(observations) even though the fused gradient evaluation is
# O(1). Reused while the SAME stage object is current; a new/rebuilt stage
# rebuilds obs+stats once. Returns `(obs, stats)` or `nothing` when the stage is
# not densifiable.
function _gen_obs_and_stats_for_stage(resolved::ResolvedSignaturePlan, stage::_MaybeObservationStage)
    cached = resolved.generated_obs_stats_cache[]
    if cached isa Tuple{Any,Vector{Vector{Float64}},Vector{Vector{Float64}}} && cached[1] === stage
        return (cached[2], cached[3])
    end
    obs = _gen_obs_from_stage(stage)
    isnothing(obs) && return nothing
    stats = _gen_stats(resolved, obs)
    resolved.generated_obs_stats_cache[] = (stage, obs, stats)
    return (obs, stats)
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
    # static whole-vector observation (issue #288): the dense site vector itself
    step.stage_iterator_slot == -1 && return Expr(:ref, :obs, step.stage_index)
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
    family = _gen_loop_fusable_family(step)
    if isnothing(family)
        _gen_emit_plain_loop!(body, step, reject)
    else
        _gen_emit_fusable_loop!(body, step, family, reject)
    end
    return nothing
end

# The per-observation loop: the fallback tier (also the whole body of a
# non-fusable loop). Emits `for iter = iterable; <body>; end`.
function _gen_emit_plain_loop!(body::Vector{Any}, step::CompiledLoopPlanStep, reject::Bool)
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

# A fusable loop: read the stage's statistics once and, gated on the runtime
# suffstats switch AND non-empty (closed-form-representable) stats, add the O(1)
# fused sum; otherwise run the per-observation loop. The two branches contribute
# the same accumulator type, so the scorer stays type-stable for ForwardDiff.
function _gen_emit_fusable_loop!(body::Vector{Any}, step::CompiledLoopPlanStep, family::Symbol, reject::Bool)
    choice = step.body[1]
    statvec = gensym(:stats)
    push!(body, Expr(:(=), statvec, Expr(:ref, :stats, choice.stage_index)))
    fused = Any[]
    _gen_push_fused_score!(fused, _gen_dist_expr(choice), d -> _gen_fused_sum_expr(family, d, statvec), reject)
    fallback = Any[]
    _gen_emit_plain_loop!(fallback, step, reject)
    condition = Expr(:&&, Expr(:ref, :_GEN_SCORER_SUFFSTATS), Expr(:call, !, Expr(:call, isempty, statvec)))
    push!(body, Expr(:if, condition, Expr(:block, fused...), Expr(:block, fallback...)))
    return nothing
end

# Push the fused score contribution: construct the distribution ONCE (giving the
# interpreter's parameter validation -- non-positive sigma/rate/lambda throw in
# non-reject mode, score -Inf in reject mode, exactly as the per-observation
# construction does), then add the closed-form sum built from the constructed
# distribution's fields and the stored statistics. `sum_fn(d)` returns the sum
# expression for the distribution symbol `d`. Mirrors `_gen_push_score!`.
function _gen_push_fused_score!(body::Vector{Any}, dist_expr, sum_fn, reject::Bool)
    d = gensym(:dist)
    if !reject
        push!(body, Expr(:(=), d, dist_expr))
        push!(body, Expr(:(+=), :acc, sum_fn(d)))
        return nothing
    end
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
            Expr(:(+=), :acc, sum_fn(d)),
            Expr(:(+=), :acc, Expr(:call, oftype, :acc, -Inf)),
        ),
    )
    return nothing
end

# The closed-form logpdf sum for a fusable family, over the constructed
# distribution `d` (a symbol) and the stage statistics vector `statvec` (a
# symbol). ForwardDiff differentiates these expressions w.r.t. the Dual
# parameters carried by `d`'s fields, recovering the analytic gradient; the
# result reassociates the per-observation sum, so it matches the interpreter to
# tolerance, not bitwise. Normal uses the cancellation-free centered form.
function _gen_fused_sum_expr(family::Symbol, d::Symbol, statvec::Symbol)
    if family === :normal
        return quote
            let n = $statvec[1], ybar = $statvec[2], residual = $statvec[3], centered_sum_squares = $statvec[4], mu = $d.mu,
                sigma = $d.sigma

                delta = ybar - mu
                squared_z_sum = (centered_sum_squares + delta * (2 * residual + n * delta)) / (sigma * sigma)
                -n * (log(sigma) + log(2 * pi) / 2) - squared_z_sum / 2
            end
        end
    elseif family === :exponential
        return quote
            let n = $statvec[1], total = $statvec[2], rate = $d.rate
                n * log(rate) - rate * total
            end
        end
    else # :poisson
        return quote
            let n = $statvec[1], total = $statvec[2], log_factorial_total = $statvec[3], lambda = $d.lambda
                total * log(lambda) - n * lambda - log_factorial_total
            end
        end
    end
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
        Expr(
            :call,
            name,
            Expr(:(::), :args, :Tuple),
            :params,
            Expr(:(::), :obs, :(Vector{Vector{Float64}})),
            Expr(:(::), :stats, :(Vector{Vector{Float64}})),
        ),
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
            again = _GeneratedScorerCache(
                _gen_structural_ok(resolved.compiled),
                _gen_stage_fusion(resolved.compiled),
            )
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
    # sufficient statistics parallel to `obs` (issue #138), rebuilt with `obs`
    # whenever the constraints mutate. Data-derived, so one build serves every
    # parameter vector the cached scorer is called with.
    stats::Vector{Vector{Float64}}
    obs_mutation_count::Int
    reject::Bool
end

function (objective::_GenGradientObjective)(theta)
    constrained, logabsdet = _transform_to_constrained_with_logabsdet(
        objective.model, objective.resolved, theta, objective.args, objective.constraints,
    )
    return objective.scorer(objective.args, constrained, objective.obs, objective.stats) + logabsdet
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
    stats::Union{Nothing,Vector{Vector{Float64}}}=nothing,
)
    resolved_stats = isnothing(stats) ? _gen_stats(resolved, obs) : stats
    return _GenGradientObjective(
        scorer, model, resolved, args, constraints, seed, obs, resolved_stats,
        constraints.mutation_count, reject,
    )
end

# Build the type-stable generated gradient objective for a model, or `nothing`
# when the model is not on the generated-scorer path (scalar/broadcast/non-Float64
# observations, marginalize sites, or a body over `_GEN_MAX_STEPS`). Shared by the
# reverse-mode entry points (issue #268): both `reverse_mode_gradient(model, ...)`
# and the batched per-column reverse path need the same Enzyme-differentiable
# objective the forward `logjoint_gradient_unconstrained` builds.
function _generated_gradient_objective_or_nothing(model, seed, args, constraints)
    resolved = _resolve_signature_plan(model, constraints)
    stage = _memoized_observation_stage(model, resolved, seed, args, constraints)
    scorer = _generated_scorer(resolved, false)
    isnothing(scorer) && return nothing
    obs_stats = _gen_obs_and_stats_for_stage(resolved, stage)
    isnothing(obs_stats) && return nothing
    obs, stats = obs_stats
    return _gen_gradient_objective(
        scorer, model, resolved, _complete_model_args(model, args), constraints, seed, obs; stats=stats,
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
    objective.stats = _gen_stats(objective.resolved, new_obs)
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

# Interpreter VALUE for a generated cache whose constraints mutated into a
# no-longer-densifiable shape: live-lookup logjoint, same reject flag as the
# cache, mirroring `_gen_interpreter_gradient!` on the value path (issue #188).
function _gen_interpreter_value(objective::_GenGradientObjective, position::AbstractVector)
    return _logjoint_unconstrained(
        objective.model, objective.resolved, position, objective.args, objective.constraints, nothing;
        reject_invalid_parameters=objective.reject,
    )
end

# Sampler VALUE path (issue #188): evaluate the unconstrained logjoint VALUE
# through the observations the gradient cache already staged, instead of the
# public `logjoint_unconstrained`, which re-stages all N observations (re-walks
# the plan rebuilding `(:y,i)` addresses) on every call. Observations are
# constant data, so the cache stages them once and this reuses them with an O(1)
# staleness check; the numerics are the cache objective's own value evaluation,
# so seeded draws stay bitwise identical. Reject semantics (issue #157) are
# carried by the objective (the cache is built reject-on for the samplers).
#
# Generated objective: refresh the dense obs against an in-place constraint
# mutation first (gibbs), matching `_logjoint_gradient!`; on a no-longer-
# densifiable mutation fall back to the live-lookup interpreter value. The
# scorer method is emitted after this caller's world, so the generated call
# crosses the world-age boundary via `invokelatest` (as the gradient path does).
function _logjoint_value_from_cache(cache::LogjointGradientCache, position::AbstractVector)
    objective = cache.objective
    if objective isa _GenGradientObjective
        if _gen_refresh_obs!(objective)
            return Base.invokelatest(objective, position)
        end
        return _gen_interpreter_value(objective, position)
    end
    # Interpreter objective (a plain closure over the pre-built ObservationStage):
    # calling it reuses that stage, dropping back to live lookups only when the
    # ChoiceMap mutated (its internal `_stage_is_current` check).
    return objective(position)
end

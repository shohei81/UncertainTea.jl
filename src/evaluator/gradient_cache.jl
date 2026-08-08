struct LogjointGradientCache{F,C,V}
    objective::F
    config::C
    buffer::V
end

# The gradient objective binds the resolved signature plan and the dense
# observation stage once, so per-evaluation work is the compiled scoring walk
# alone (no signature re-resolution, no per-observation ChoiceMap hashing for
# staged sites). Building it through a named helper keeps the closure type a
# pure function of the captured types, which the public gradient-config
# memoization keys on.
function _gradient_objective(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    args::Tuple,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage;
    reject_invalid_parameters::Bool=false,
)
    return theta -> _logjoint_unconstrained(
        model, resolved, theta, args, constraints, stage;
        reject_invalid_parameters=reject_invalid_parameters,
    )
end

function _logjoint_gradient_cache(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
    buffer::AbstractVector=similar(collect(params));
    reject_invalid_parameters::Bool=false,
)
    seed = collect(params)
    length(buffer) == length(seed) ||
        throw(DimensionMismatch("expected gradient buffer of length $(length(seed)), got $(length(buffer))"))
    resolved = _resolve_signature_plan(model, constraints, args)
    stage = _stage_observations(model, resolved, seed, args, constraints)
    # Prefer the type-stable generated scorer (issue #144) when the plan and its
    # dense observation vectors are representable; otherwise the interpreter
    # objective. The choice is fixed for the cache's lifetime, so its objective
    # type -- and the `LogjointGradientCache{F}` -- stays concrete.
    scorer = _generated_scorer(resolved, reject_invalid_parameters)
    obs = isnothing(scorer) ? nothing : _gen_obs_from_stage(stage)
    objective = if !isnothing(scorer) && !isnothing(obs)
        _gen_gradient_objective(
            scorer, model, resolved, _complete_model_args(model, args), constraints, seed, obs;
            reject=reject_invalid_parameters,
        )
    else
        _gradient_objective(
            model, resolved, args, constraints, stage;
            reject_invalid_parameters=reject_invalid_parameters,
        )
    end
    config = ForwardDiff.GradientConfig(objective, seed)
    return LogjointGradientCache(objective, config, buffer)
end

function _logjoint_gradient!(cache::LogjointGradientCache, params::AbstractVector)
    # The generated objective's scorer method is emitted after this caller's
    # world (see generated_scorer.jl); `invokelatest` runs the differentiation
    # in the latest world so the concrete scorer resolves and stays type-stable.
    # The interpreter objective keeps its direct, world-stable call path. A
    # cache reused across an in-place constraint mutation (gibbs) re-stages its
    # dense observations first; if the mutated constraints are no longer
    # densifiable, it falls back to the interpreter for that call.
    if cache.objective isa _GenGradientObjective
        if _gen_refresh_obs!(cache.objective)
            Base.invokelatest(ForwardDiff.gradient!, cache.buffer, cache.objective, params, cache.config)
        else
            Base.invokelatest(_gen_interpreter_gradient!, cache.objective, cache.buffer, params)
        end
    else
        ForwardDiff.gradient!(cache.buffer, cache.objective, params, cache.config)
    end
    return cache.buffer
end

# Memoized `ForwardDiff.GradientConfig` for the public gradient entry point
# (issue #145): the config (seed duals + chunk buffers, ~80 kB for the audit
# models) depends only on the objective's type, the seed eltype, and the
# parameter count, all of which are stable per resolved signature -- not on
# the constraint or argument VALUES -- so it lives on the memoized
# `ResolvedSignaturePlan` instead of being rebuilt per call.
function _cached_gradient_config(resolved::ResolvedSignaturePlan, objective, seed::AbstractVector)
    store = resolved.gradient_config_cache[]
    if isnothing(store)
        store = Dict{Tuple{DataType,DataType,Int},Any}()
        resolved.gradient_config_cache[] = store
    end
    dict = store::Dict{Tuple{DataType,DataType,Int},Any}
    key = (typeof(objective), eltype(seed), length(seed))
    config = get(dict, key, nothing)
    isnothing(config) || return config
    config = ForwardDiff.GradientConfig(objective, seed)
    dict[key] = config
    return config
end

# Stage memo for the public gradient entry point: reuse the last stage while
# it is still current (same constraints object, no mutations since); rebuild
# otherwise. A cache miss costs one plan walk -- about half a gradient
# evaluation -- and every hit removes the per-observation ChoiceMap lookups
# from every subsequent call.
function _memoized_observation_stage(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    seed::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    resolved.compiled.stage_count == 0 && return nothing
    cached = resolved.observation_stage_cache[]
    if cached isa ObservationStage && _stage_is_current(cached, constraints)
        return cached
    end
    stage = _stage_observations(model, resolved, seed, args, constraints)
    isnothing(stage) || (resolved.observation_stage_cache[] = stage)
    return stage
end

function logjoint_gradient_unconstrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    seed = collect(params)
    resolved = _resolve_signature_plan(model, constraints, args)
    stage = _memoized_observation_stage(model, resolved, seed, args, constraints)
    scorer = _generated_scorer(resolved, false)
    obs_stats = isnothing(scorer) ? nothing : _gen_obs_and_stats_for_stage(resolved, stage)
    if !isnothing(scorer) && !isnothing(obs_stats)
        obs, stats = obs_stats
        _GEN_SCORER_LAST_USED[] = true
        objective = _gen_gradient_objective(
            scorer, model, resolved, _complete_model_args(model, args), constraints, seed, obs; stats=stats,
        )
        config = _cached_gradient_config(resolved, objective, seed)
        gradient = similar(seed)
        Base.invokelatest(ForwardDiff.gradient!, gradient, objective, seed, config)
        return gradient
    end
    _GEN_SCORER_LAST_USED[] = false
    objective = _gradient_objective(model, resolved, args, constraints, stage)
    config = _cached_gradient_config(resolved, objective, seed)
    gradient = similar(seed)
    ForwardDiff.gradient!(gradient, objective, seed, config)
    return gradient
end

# Reverse-mode gradient of a model's unconstrained logjoint (issue #268, part A).
#
# The forward-mode `logjoint_gradient_unconstrained` above is `O(P)` in the
# parameter count; on high-dimensional non-analytic models a reverse-mode pass is
# `O(1)`. The type-stable GENERATED scorer (`_GenGradientObjective`, the same
# objective the forward path differentiates) is Enzyme-differentiable, so the
# reverse-mode gradient is exactly the forward-mode gradient computed the cheaper
# way — no separate evaluator. The interpreter fallback objective is NOT
# type-stable enough for Enzyme, so a model that falls off the generated-scorer
# path is rejected with a clear message rather than silently taking a slow path.
#
# The actual Enzyme call lives in `reverse_mode_gradient(f, x)`, supplied by
# UncertainTeaEnzymeExt (loaded via `using Enzyme`); it is reached through
# `invokelatest` because the generated scorer is emitted at runtime (the
# forward path crosses the same world-age boundary). Without Enzyme loaded the
# inner call raises a MethodError — the intended "load Enzyme" signal.
function reverse_mode_gradient(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    seed = collect(params)
    objective = _generated_gradient_objective_or_nothing(model, seed, args, constraints)
    if isnothing(objective)
        throw(
            ArgumentError(
                "reverse_mode_gradient(model, ...) currently supports only models on the " *
                "type-stable generated-scorer path; this model falls back to the interpreter. " *
                "Use logjoint_gradient_unconstrained (forward-mode) instead.",
            ),
        )
    end
    return Base.invokelatest(reverse_mode_gradient, objective, seed)
end

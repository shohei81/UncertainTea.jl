# Pointwise log-likelihood extraction.
#
# This mirrors the compiled scoring walk in `evaluator.jl` (`_score_compiled_steps` /
# `_score_plan_step!`) but instead of summing every choice's logpdf it *records* the
# per-observation logpdf for every observation.
#
# The observation/latent split is taken from the CONDITIONING SIGNATURE (issue #95),
# exactly as `logjoint` and the batched/device paths do: the walk runs over the
# signature-resolved compiled plan, in which a static choice constrained at inference
# time carries no parameter slot (so it reads its value from the constraints and is
# recorded), while any unconstrained static choice is a slotted latent. This keeps the
# recorded observations identical to `observation_addresses` and to the scoring paths.
#
# marginalize=:enumerate latents (issue #88, docs/discrete-enumeration.md): an
# enumerated discrete latent is NEVER an observation (it has no slot and is not
# constrained), so it is not recorded as a data column. Instead the walk computes the
# correct MARGINAL pointwise log-likelihood for the observations in its enumeration
# suffix, integrating the latent out with the same logsumexp the marginalized logjoint
# uses. See `_record_marginalized_choice!` for the supported/rejected structures.

_record_compiled_steps(::Tuple{}, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap, records) = records

function _record_compiled_steps(steps::Tuple, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap, records)
    head = first(steps)
    tail = Base.tail(steps)
    if head isa CompiledChoicePlanStep && !isnothing(head.marginalize)
        # the enumerated latent owns its suffix (mirrors `_score_compiled_steps`)
        _record_marginalized_choice!(head, tail, env, params, constraints, records)
    else
        _record_plan_step!(head, env, params, constraints, records)
        _record_compiled_steps(tail, env, params, constraints, records)
    end
    return records
end

function _record_plan_step!(
    step::CompiledChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    records,
)
    address = _concrete_address(env, step.address)
    is_constrained = isnothing(step.parameter_value_indices)
    value = if !is_constrained
        _parameter_slot_value(step.parameter_value_indices, params)
    else
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("pointwise loglikelihood requires a provided value for choice $(address)"))
        constrained_value
    end

    dist = _compiled_distribution(step, env)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    lp = logpdf(dist, value)
    if is_constrained
        push!(records, address => lp)
    end
    return records
end

function _record_plan_step!(
    step::CompiledDeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    records,
)
    _environment_set!(env, step.binding_slot, _eval_compiled_expr(env, step.expr))
    return records
end

function _record_plan_step!(
    step::CompiledLoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    records,
)
    iterable = _eval_compiled_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing

    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        _record_compiled_steps(step.body, env, params, constraints, records)
    end

    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return records
end

# marginalize=:enumerate pointwise likelihood (issue #88).
#
# For an unconstrained enumerated latent `z`, the correct per-observation marginal
# likelihood integrates `z` out. Writing the observations in `z`'s suffix as `y_k`, the
# marginal joint of the suffix data is
#
#     p(data | theta) = sum_z p(z) * prod_k p(y_k | z, theta).
#
# Splitting the observations into those whose density does NOT depend on `z` (set `I`)
# and those that do (set `D`),
#
#     p(data | theta) = [prod_{k in I} p(y_k)] * sum_z p(z) prod_{k in D} p(y_k | z).
#
# The z-independent factors decompose per observation directly. The dependent block is a
# SINGLE joint term over all of `D`; it decomposes into per-observation pointwise terms
# only when |D| <= 1:
#
#   * |D| == 0: the enumerated latent affects no recorded observation; every observation
#     is z-independent and recorded as its own logpdf (`sum_z p(z) == 1`).
#   * |D| == 1: the one dependent observation `y*` gets the marginal
#     `log sum_z p(z) p(y*|z) = logsumexp_z(logpdf(z) + logpdf(y*|z))`; the rest are
#     recorded directly. This is the common case (the #88 repro: one enumerated
#     indicator local to one observation).
#   * |D| >= 2: `log sum_z p(z) prod_k p(y_k|z)` does not factorize into per-observation
#     terms (the observations are dependent given theta once z is summed out); a clear
#     error is raised.
#
# Nested marginalize=:enumerate latents in the suffix are handled by recursion, so the
# supported set includes the factorizing nested cases: independent enumerated latents
# feeding separate observations decompose per column, and several enumerated latents all
# feeding a SINGLE observation give that column its full multi-sum marginal. A nested
# structure whose observations are genuinely entangled is still caught by the |D| >= 2
# check at whichever enumeration level shares two observations.
#
# Structures rejected up front (non-factorizing for pointwise purposes):
#   * a free (slotted) latent inside the enumeration suffix -- its prior can depend on
#     `z`, which would reweight the enumeration in a way the per-observation
#     decomposition cannot see; place the enumerated site after all continuous latents;
#   * an enumerated latent whose value changes the SET of downstream observations.
#
# A CONSTRAINED enumerated latent is an observation like any other fixed value: it is
# recorded and its suffix is walked once at that value (this factorizes trivially).
function _record_marginalized_choice!(
    step::CompiledChoicePlanStep,
    tail::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    records,
)
    address = _concrete_address(env, step.address)
    dist = _compiled_distribution(step, env)
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    if found
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, constrained_value)
        push!(records, address => logpdf(dist, constrained_value))
        _record_compiled_steps(tail, env, params, constraints, records)
        return records
    end

    if _marginal_suffix_has_free_latent(tail)
        throw(
            ArgumentError(
                "pointwise loglikelihood is not defined for the marginalize=:enumerate latent " *
                "$(address): its enumeration suffix contains a free (slotted) latent, whose prior " *
                "may depend on the enumerated value, so integrating the latent out does not " *
                "factorize into per-observation terms. Place the enumerated site after all " *
                "continuous latents so its suffix scores only observations.",
            ),
        )
    end

    snapshot = _environment_snapshot(env)
    branch_weights = Float64[]
    branch_records = Vector{Pair{Any,Float64}}[]
    for value in step.marginalize.support
        _environment_restore_snapshot!(env, snapshot)
        choice_logpdf = logpdf(dist, value)
        # a zero-mass support value contributes nothing to the marginal and its suffix
        # may be unevaluable; skip it (mirrors `_marginalized_suffix_terms`)
        isfinite(choice_logpdf) || continue
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        branch = Pair{Any,Float64}[]
        _record_compiled_steps(tail, env, params, constraints, branch)
        push!(branch_weights, Float64(choice_logpdf))
        push!(branch_records, branch)
    end
    _environment_restore_snapshot!(env, snapshot)

    _combine_marginal_branches!(records, address, branch_weights, branch_records)
    return records
end

# A free (slotted) latent anywhere in the enumeration suffix (including inside a loop
# body, or the suffix of a nested marginalized latent) makes the pointwise marginal
# ill-defined: its prior can depend on the enumerated value and would reweight the
# enumeration, which the per-observation decomposition cannot represent. A nested
# marginalize step is not itself a free latent; the recursion handles it.
#
# The scan runs over a `Vector{Any}` rather than the heterogeneous compiled-step
# `Tuple`: a `for` loop over such a tuple combined with the recursion into loop bodies
# (each a differently-typed tuple) blows Julia 1.10's type inference recursion bound
# ("fatal error in type inference (type bound)"). The `Vector{Any}` function barrier
# fixes the element and recursion-argument types, so inference stays bounded on both
# 1.10 and 1.12. This is a structural check run off the hot path, so the boxing is fine.
_marginal_suffix_has_free_latent(steps::Tuple) = _marginal_suffix_has_free_latent(collect(Any, steps))

function _marginal_suffix_has_free_latent(steps::Vector{Any})
    for step in steps
        if step isa CompiledChoicePlanStep
            isnothing(step.marginalize) && !isnothing(step.parameter_value_indices) && return true
        elseif step isa CompiledLoopPlanStep
            _marginal_suffix_has_free_latent(collect(Any, step.body)) && return true
        end
    end
    return false
end

function _pointwise_logsumexp(terms::AbstractVector)
    shift = maximum(terms)
    isfinite(shift) || return shift
    total = 0.0
    for term in terms
        total += exp(term - shift)
    end
    return shift + log(total)
end

# Combine the per-branch observation records of an enumerated latent into the marginal
# pointwise log-likelihoods (see `_record_marginalized_choice!` for the math).
function _combine_marginal_branches!(records, address, weights::Vector{Float64}, branch_records)
    isempty(branch_records) && return records
    reference = Any[first(pair) for pair in branch_records[1]]
    for branch in branch_records
        Any[first(pair) for pair in branch] == reference || throw(
            ArgumentError(
                "pointwise loglikelihood is not defined for the marginalize=:enumerate latent " *
                "$(address): its value changes the set of downstream observations.",
            ),
        )
    end

    n_obs = length(reference)
    dependent = Int[]
    for k = 1:n_obs
        base = branch_records[1][k].second
        for b = 2:length(branch_records)
            if branch_records[b][k].second != base
                push!(dependent, k)
                break
            end
        end
    end

    if length(dependent) > 1
        shared = join((string(reference[k]) for k in dependent), ", ")
        throw(
            ArgumentError(
                "pointwise loglikelihood is not defined for the marginalize=:enumerate latent " *
                "$(address): it is shared across multiple observations ($(shared)), so " *
                "log sum_z p(z) prod_i p(y_i | z) does not factorize into independent per-observation " *
                "terms. Give each observation its own local enumerated latent, or constrain the latent.",
            ),
        )
    end

    for k = 1:n_obs
        if k in dependent
            terms = Float64[weights[b] + branch_records[b][k].second for b = 1:length(branch_records)]
            push!(records, reference[k] => _pointwise_logsumexp(terms))
        else
            push!(records, reference[k] => branch_records[1][k].second)
        end
    end
    return records
end

function _record_execution!(records, model::TeaModel, params::AbstractVector, args::Tuple, constraints::ChoiceMap)
    resolved = _resolve_signature_plan(model, constraints)
    plan = resolved.plan
    compiled_plan = resolved.compiled
    args = _complete_model_args(model, args)

    env = PlanEnvironment(plan.environment_layout)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end

    _record_compiled_steps(compiled_plan.steps, env, params, constraints, records)
    return records
end

# Initial parameter vector matching the signature layout (not the default
# layout): observation_addresses only needs a validly-shaped latent vector to
# walk the plan and read the constrained addresses, so any prior draw of the
# signature latents will do. Drawn by the latent-only plan walk
# (`_sample_prior_steps!`, issue #156), which consumes the RNG at exactly the
# sites a traced `generate` would and therefore produces the identical draw
# without re-tracing every observation address.
function _signature_initial_parameters(
    model::TeaModel,
    args::Tuple,
    resolved,
    constraints::ChoiceMap;
    rng::AbstractRNG=Random.default_rng(),
)
    plan = resolved.plan
    layout = plan.parameter_layout
    completed_args = _complete_model_args(model, args)

    env = PlanEnvironment(plan.environment_layout)
    for (slot, value) in zip(plan.environment_layout.argument_slots, completed_args)
        _environment_set!(env, slot, value)
    end

    params = Vector{Float64}(undef, parametervaluecount(layout))
    _sample_prior_steps!(resolved.compiled.steps, env, layout, params, constraints, rng)
    return params
end

"""
    observation_addresses(model, args=(), constraints=choicemap()) -> Vector

Return the ordered list of observation choice addresses in execution-plan order. Under the
constraint-driven rule (issue #95, `docs/constraint-driven-conditioning.md`) an observation
is exactly a choice whose address is constrained-and-present, regardless of whether the
choice is bound; binding is orthogonal. This order defines the column order of
`pointwise_loglikelihood` and is deterministic across draws.
"""
function observation_addresses(model::TeaModel, args::Tuple=(), constraints::ChoiceMap=choicemap())
    resolved = _resolve_signature_plan(model, constraints)
    params = _signature_initial_parameters(model, args, resolved, constraints)
    records = Pair{Any,Float64}[]
    _record_execution!(records, model, params, args, constraints)
    return Any[first(r) for r in records]
end

"""
    pointwise_loglikelihood(model, args, constraints, result; kwargs...) -> Matrix{Float64}
    pointwise_loglikelihood(model, args, constraints, draws::AbstractMatrix) -> Matrix{Float64}

Compute the pointwise (per-observation) log-likelihood matrix of size `S x N_obs`,
where `S` is the number of posterior draws and `N_obs` is the number of constrained
(observation) choices. `result` is any inference result implementing
`constrained_draws` (see `UncertainTea.Diagnostics.constrained_draws`; keyword
arguments are forwarded to it), or a raw `num_params x S` matrix whose columns are
*constrained* parameter vectors. Columns of the output follow
`observation_addresses(model, args, constraints)`.
"""
function pointwise_loglikelihood(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    draws::AbstractMatrix,
)
    addresses = observation_addresses(model, args, constraints)
    n_obs = length(addresses)
    column_of = Dict{Any,Int}()
    for (i, a) in enumerate(addresses)
        column_of[a] = i
    end

    ll = Matrix{Float64}(undef, size(draws, 2), n_obs)
    records = Pair{Any,Float64}[]
    for (s, j) in enumerate(axes(draws, 2))
        empty!(records)
        _record_execution!(records, model, view(draws, :, j), args, constraints)
        for (address, lp) in records
            ll[s, column_of[address]] = lp
        end
    end
    return ll
end

# MH-within-Gibbs for discrete latent sites (docs/mh-within-gibbs.md, issue
# #13 track 2): alternate symmetric single-site Metropolis-Hastings updates on
# discrete latents with NUTS transitions on the continuous block. Conditioning
# flows through ONE mutable ChoiceMap shared by reference with the NUTS target
# and gradient cache, so upserting a discrete value needs no rebuild.

const _GIBBS_DISCRETE_FAMILIES = (:bernoulli, :categorical, :poisson, :geometric, :negativebinomial, :binomial)

# Unconstrained initial continuous position from a prior trace, read through the
# signature-specific continuous layout (#95 PR-6) so a bound continuous
# observation does not leak into the sampled block.
function _gibbs_trace_unconstrained(
    model::TeaModel,
    trace::TeaTrace,
    layout::ParameterLayout,
    args::Tuple,
    constraints::ChoiceMap,
)
    constrained = Vector{Float64}(undef, parametervaluecount(layout))
    for slot in layout.slots
        _write_slot_value!(constrained, slot, trace[_static_address(slot.address)])
    end
    return transform_to_unconstrained(model, constrained, args, constraints)
end

struct GibbsSite
    address::Any
    kind::Symbol      # :flip (bernoulli) | :uniform_other (categorical) | :walk (integer supports)
    lower::Int        # static lower support bound for :walk pre-rejection
    categories::Int   # K for :uniform_other
end

# Sites are matched against the INLINED execution plan's choice steps, not
# the raw spec: submodel calls are expanded there with prefixed addresses
# ((:sub, :z) for a child latent), and loop-scoped or dynamic address parts
# match as templates (literal parts must agree, dynamic parts match anything).
function _gibbs_collect_choice_steps!(collected::Vector{ChoicePlanStep}, steps)
    for step in steps
        if step isa ChoicePlanStep
            push!(collected, step)
        elseif step isa LoopPlanStep
            _gibbs_collect_choice_steps!(collected, step.body)
        end
    end
    return collected
end

function _gibbs_step_matches(step::ChoicePlanStep, address::Tuple; exact::Bool)
    parts = step.address.parts
    length(parts) == length(address) || return false
    for index in eachindex(address)
        part = parts[index]
        if part isa AddressLiteralPart
            part.value == address[index] || return false
        elseif exact
            return false
        end
    end
    return true
end

# Classification of a matched step for one site: whether it stays
# marginalized, the family, and the proposal-relevant support size.
function _gibbs_step_classification(step::ChoicePlanStep, address::Tuple)
    step.rhs isa DistributionSpec || throw(
        ArgumentError(
            "gibbs found the latent choice $address but could not match it to a distribution " *
            "choice in the model's execution plan",
        ),
    )
    step.rhs.marginalize === :enumerate && return (true, :none, 0)
    family = step.rhs.family
    categories = 0
    if family === :categorical
        arguments = step.rhs.arguments
        categories = if length(arguments) > 1
            # vararg form categorical(p1, p2, ...): K is the argument count
            length(arguments)
        elseif length(arguments) == 1 && arguments[1] isa Expr && arguments[1].head == :vect
            length(arguments[1].args)
        elseif length(arguments) == 1 && arguments[1] isa Real
            1
        else
            0
        end
    end
    return (false, family, categories)
end

# Classify one candidate latent address: `nothing` for marginalize=:enumerate
# sites (they stay marginalized inside the logjoint), a GibbsSite for discrete
# families, a loud error otherwise. An exact all-literal address match wins
# outright; otherwise every template match (dynamic parts as wildcards) must
# agree on the classification, because a wildcard template can shadow an
# unrelated concrete site. Matched steps accumulate into `site_steps` for the
# shape validation.
function _gibbs_site(choice_steps::Vector{ChoicePlanStep}, address::Tuple, site_steps::Set{ChoicePlanStep})
    matched_step = nothing
    for step in choice_steps
        if _gibbs_step_matches(step, address; exact=true)
            matched_step = step
            break
        end
    end
    classification = if !isnothing(matched_step)
        push!(site_steps, matched_step)
        _gibbs_step_classification(matched_step, address)
    else
        candidates = [step for step in choice_steps if _gibbs_step_matches(step, address; exact=false)]
        isempty(candidates) && throw(
            ArgumentError(
                "gibbs could not match the latent choice $address to any choice in the model's " *
                "execution plan (a dynamic address expression that normalizes to multiple parts, " *
                "e.g. a Pair-valued address argument, cannot be matched; use static or " *
                "single-part addresses for Gibbs sites)",
            ),
        )
        classifications = unique(_gibbs_step_classification(step, address) for step in candidates)
        length(classifications) == 1 || throw(
            ArgumentError(
                "gibbs matched the latent choice $address to multiple template choices with " *
                "conflicting classifications $classifications; disambiguate the addresses",
            ),
        )
        for step in candidates
            push!(site_steps, step)
        end
        classifications[1]
    end

    marginalized, family, categories = classification
    marginalized && return nothing
    family in _GIBBS_DISCRETE_FAMILIES || throw(
        ArgumentError(
            "gibbs requires every non-observed slotless choice to be a finite- or integer-support " *
            "discrete latent; choice $address has family `$family` (continuous slotless latents " *
            "have no sampler)",
        ),
    )
    if family === :bernoulli
        return GibbsSite(address, :flip, 0, 0)
    elseif family === :categorical
        # a singleton support can never move: self-transition instead of
        # sampling from an empty proposal range
        categories == 1 && return GibbsSite(address, :fixed, 1, 1)
        categories > 1 && return GibbsSite(address, :uniform_other, 1, categories)
        # a non-literal probability vector hides the support size, and a +-1
        # walk cannot cross zero-probability categories between positive
        # ones -- reject rather than sample a disconnected component
        throw(
            ArgumentError(
                "gibbs requires categorical sites to have a literal probability vector so the " *
                "uniform proposal can reach every category; choice $address has a dynamic one " *
                "(marginalize the site or use a literal vector)",
            ),
        )
    end
    return GibbsSite(address, :walk, 0, 0)
end

# Discrete Gibbs sites = prior-trace addresses minus continuous parameter
# slots minus observations minus marginalized sites (the SBC observation
# detection pattern, complemented).
function _gibbs_discrete_sites(model::TeaModel, trace::TeaTrace, constraints::ChoiceMap)
    latent_map = parameterchoicemap(model, parameter_vector(trace))
    choice_steps = _gibbs_collect_choice_steps!(ChoicePlanStep[], executionplan(model).steps)
    sites = GibbsSite[]
    site_steps = Set{ChoicePlanStep}()
    for entry in trace.choices.entries
        address = first(entry)
        haskey(latent_map, address) && continue
        haskey(constraints, address) && continue
        site = _gibbs_site(choice_steps, address, site_steps)
        isnothing(site) || push!(sites, site)
    end
    _gibbs_validate_static_shape(model, constraints)
    return sites
end

_gibbs_expr_uses(expr, symbols::Set{Symbol}) = false
_gibbs_expr_uses(expr::Symbol, symbols::Set{Symbol}) = expr in symbols
_gibbs_expr_uses(expr::Expr, symbols::Set{Symbol}) = any(arg -> _gibbs_expr_uses(arg, symbols), expr.args)

# A choice step that could be a latent Gibbs site: slotless, not
# marginalized, and not verifiably observed. This is judged from the PLAN
# template, not from one trace's discovered sites -- a shape-dependent model
# can draw a shape where the controlled sites do not exist at all (`for i =
# 1:z` with z drawn 0), and the validation must still fire. Only an
# ALL-LITERAL address matching a constraint counts as observed: a
# dynamic-address template may be only partially constrained (one (:w, i)
# observed, later ones latent), which a static check cannot verify.
function _gibbs_step_potentially_latent(step::ChoicePlanStep, constraints::ChoiceMap)
    if step.rhs isa BroadcastDistributionSpec
        # broadcast observations have no latent form; a constrained
        # all-literal one is plain data
        all(part -> part isa AddressLiteralPart, step.address.parts) || return true
        static_address = Tuple(part.value for part in step.address.parts)
        return !haskey(constraints, static_address)
    end
    step.rhs isa DistributionSpec || return true
    step.rhs.marginalize === :enumerate && return false
    isnothing(step.parameter_slot) || return false
    all(part -> part isa AddressLiteralPart, step.address.parts) || return true
    static_address = Tuple(part.value for part in step.address.parts)
    return !haskey(constraints, static_address)
end

function _gibbs_contains_potential_site(steps, constraints::ChoiceMap)
    for step in steps
        step isa ChoicePlanStep && _gibbs_step_potentially_latent(step, constraints) && return true
        step isa LoopPlanStep && _gibbs_contains_potential_site(step.body, constraints) && return true
    end
    return false
end

# The set of Gibbs-site addresses must stay constant across the chain: a loop
# bound or dynamic address that shapes a potential SITE and depends on any
# latent quantity (a Gibbs site's binding OR a continuous parameter) makes
# the model trans-dimensional -- up-moves would require missing choices and
# down-moves would leave stale ones behind, silently biasing the chain.
# Reject such models at construction. Model arguments and observed loops are
# fixed and stay allowed.
function _gibbs_validate_static_shape(model::TeaModel, constraints::ChoiceMap)
    # the walk runs regardless of the seed trace's drawn shape (it can draw
    # zero controlled sites), and bindings are tainted AT their sampling step
    # (a clean earlier assignment to the same symbol must not launder a later
    # sample), so it starts from an empty set
    _gibbs_validate_static_shape_steps(executionplan(model).steps, Set{Symbol}(), constraints, false, false)
    return nothing
end

# Whether a choice step SAMPLES its binding: everything except an observed
# SLOTLESS all-literal address (fixed by data) counts -- continuous slots
# are sampled from the position even when a constraint names their address,
# and discrete sites and unobserved marginalize=:enumerate choices are random
# by definition.
function _gibbs_step_samples_binding(step::ChoicePlanStep, constraints::ChoiceMap)
    if step.rhs isa Union{DistributionSpec,BroadcastDistributionSpec} &&
       isnothing(step.parameter_slot) &&
       all(part -> part isa AddressLiteralPart, step.address.parts)
        static_address = Tuple(part.value for part in step.address.parts)
        haskey(constraints, static_address) && return false
    end
    return true
end

function _gibbs_validate_static_shape_steps(
    steps,
    tainted::Set{Symbol},
    constraints::ChoiceMap,
    under_tainted_loop::Bool,
    in_loop::Bool,
)
    for step in steps
        if step isa DeterministicPlanStep
            # any assignment under a tainted loop depends on the sampled
            # iteration count (`last = i` inside `for i = 1:z`), whatever its
            # right-hand side mentions; a STRAIGHT-LINE reassignment from a
            # clean expression clears the symbol (`n = z; n = 2`), but a
            # reassignment inside any loop may execute zero times and must
            # not clear the outer taint
            if under_tainted_loop || _gibbs_expr_uses(step.expr, tainted)
                push!(tainted, step.binding)
            elseif !in_loop
                delete!(tainted, step.binding)
            end
        elseif step isa LoopPlanStep
            loop_tainted = under_tainted_loop || _gibbs_expr_uses(step.iterable, tainted)
            # the iterator's taint is scoped to this loop (the evaluator
            # shadows and restores iterators too): inside a FIXED loop the
            # iterator is fixed even when it shadows a tainted outer symbol,
            # and a later loop reusing a tainted iterator name must not
            # inherit it
            iterator_was_tainted = step.iterator in tainted
            if loop_tainted
                push!(tainted, step.iterator)
            else
                delete!(tainted, step.iterator)
            end
            if loop_tainted && _gibbs_contains_potential_site(step.body, constraints)
                throw(
                    ArgumentError(
                        "gibbs does not support models where the set of discrete latent sites " *
                        "changes with a latent value (a loop bound around a latent choice " *
                        "depends on a sampled quantity); marginalize the site or restructure " *
                        "the model",
                    ),
                )
            end
            _gibbs_validate_static_shape_steps(step.body, tainted, constraints, loop_tainted, true)
            if iterator_was_tainted
                push!(tainted, step.iterator)
            else
                delete!(tainted, step.iterator)
            end
        elseif step isa ChoicePlanStep
            if _gibbs_step_potentially_latent(step, constraints)
                for part in step.address.parts
                    part isa AddressDynamicPart || continue
                    _gibbs_expr_uses(part.value, tainted) && throw(
                        ArgumentError(
                            "gibbs does not support models where the set of discrete latent " *
                            "sites changes with a latent value (the address of " *
                            "$(step.address) depends on a sampled quantity); marginalize the " *
                            "site or restructure the model",
                        ),
                    )
                end
            end
            # taint the binding at the sampling point (temporal correctness)
            if !isnothing(step.binding) && _gibbs_step_samples_binding(step, constraints)
                push!(tainted, step.binding)
            end
        end
    end
    return nothing
end

# Exceptions a PROPOSAL evaluation may convert into a zero-density rejection:
# support/domain failures a proposed branch can legitimately trigger.
# Everything else (interrupts, genuine model bugs) rethrows diagnostically.
_gibbs_rejectable_error(err) = err isa Union{ArgumentError,DomainError,BoundsError,InexactError}

function _gibbs_propose(rng::AbstractRNG, site::GibbsSite, value, discrete_tail::Float64)
    if site.kind === :flip
        current = value isa Bool ? value : value != 0
        return !current
    elseif site.kind === :uniform_other
        shifted = Int(value) + rand(rng, 1:(site.categories-1))
        return shifted > site.categories ? shifted - site.categories : shifted
    end
    # symmetric integer walk; with probability discrete_tail the step gets a
    # geometric tail (s = 1 + Geometric(1/2)) so large-count posteriors mix
    # without breaking symmetry (the magnitude is drawn independently of the
    # direction)
    step = 1
    if discrete_tail > 0 && rand(rng) < discrete_tail
        # clamp the uniform away from zero (as the geometric sampler does) so
        # a valid rand(rng) == 0.0 cannot floor(Int, Inf)
        threshold = max(rand(rng), floatmin(Float64))
        step += floor(Int, log(threshold) / log(0.5))
    end
    return Int(value) + rand(rng, (-1, 1)) * step
end

"""
    discrete_ess(chain::GibbsChain)

Split-chain effective sample sizes of the discrete sites, aligned with
`chain.discrete_addresses`.
"""
function discrete_ess(chain::GibbsChain)
    return map(1:length(chain.discrete_addresses)) do site_index
        draws = Float64.(view(chain.discrete_samples, site_index, :))
        # _split_ess expects rows that ARE the split chains: hand it the two
        # halves so first-vs-second-half nonstationarity registers
        half = length(draws) ÷ 2
        half >= 1 || return Float64(length(draws))
        _split_ess(permutedims(reshape(draws[1:(2*half)], half, 2)))
    end
end

"""
    gibbs(model, args=(), constraints=choicemap(); num_samples, kwargs...)

MH-within-Gibbs for models with discrete latent sites (docs/mh-within-gibbs.md):
each iteration sweeps the discrete sites with symmetric single-site
Metropolis-Hastings updates, then advances the continuous block with one NUTS
transition conditioned on the current discrete values. Discrete sites are
discovered automatically (every non-observed choice without a parameter slot
that is not `marginalize=:enumerate`); a model without discrete sites reduces
to plain NUTS, and a model without continuous slots to pure single-site MH.

Returns a [`GibbsChain`](@ref) with the continuous samples and per-site
discrete samples. The NUTS keyword arguments mirror [`nuts`](@ref); the
prior initial state is retried when observations rule it out, and
`initial_discrete` (a choicemap over site addresses) seeds specific
discrete values when the prior cannot reach the posterior's support.
"""
function gibbs(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    initial_discrete::Union{Nothing,ChoiceMap}=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    metric::Symbol=:diag,
    discrete_tail::Real=0.0,
    rng::AbstractRNG=Random.default_rng(),
)
    metric in (:diag, :dense) || throw(ArgumentError("metric must be :diag or :dense, got :$metric"))
    0 <= discrete_tail < 1 || throw(ArgumentError("discrete_tail must be in [0, 1)"))
    discrete_tail_probability = Float64(discrete_tail)
    # The continuous block conditions on `merged` (observations + the current
    # discrete-site values). Discrete sites carry no parameter slot, so the
    # continuous latent layout is the same whether keyed on `constraints` or on
    # `merged`; size it from the conditioning signature (#95 PR-6) rather than
    # the syntactic default so a bound continuous observation drops its slot.
    continuous_layout = _conditioned_parameter_layout(model, constraints, args)
    num_params = parametercount(continuous_layout)
    constrained_num_params = parametervaluecount(continuous_layout)
    has_continuous = num_params > 0
    # a pure-discrete model legitimately has zero continuous parameters; the
    # remaining NUTS options are validated uniformly either way
    _validate_nuts_arguments(
        has_continuous ? num_params : 1,
        num_samples,
        num_warmup,
        step_size,
        max_tree_depth,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
    )

    # one prior trace seeds BOTH the discrete values and (absent
    # initial_params) the continuous position, so the initial state is a
    # coherent draw from the prior conditioned on the observations.
    # Observations can rule most prior draws of a discrete site out, and a
    # ruled-out draw may even make `generate` itself throw (a downstream
    # consumer of the drawn value, like ps[z + 1]); retry the prior seed a
    # bounded number of times, applying initial_discrete overrides each time.
    initialization_limit = 100
    sites = GibbsSite[]
    merged = choicemap((first(entry), last(entry)) for entry in constraints.entries)
    position = Float64[]
    current_logjoint = -Inf
    initialized = false
    initialization_attempt = 0
    # seed values enter the prior draws as constraints, so a seeded site can
    # never take an unevaluable prior value in the first place; likewise a
    # provided initial_params pins the continuous slots, so the discovery
    # draws cannot wander into throwing regions the caller already avoided
    seeded_position = isnothing(initial_params) ? nothing :
                      _initial_hmc_position(model, args, constraints, initial_params, rng)
    generation_constraints = if isnothing(initial_discrete) && isnothing(seeded_position)
        constraints
    else
        seeded = choicemap((first(entry), last(entry)) for entry in constraints.entries)
        if !isnothing(initial_discrete)
            for entry in initial_discrete.entries
                _pushchoice!(seeded, first(entry), last(entry))
            end
        end
        if !isnothing(seeded_position) && has_continuous
            seeded_constrained = transform_to_constrained(model, seeded_position, args, constraints)
            for entry in parameterchoicemap(model, seeded_constrained, constraints).entries
                _pushchoice!(seeded, first(entry), last(entry))
            end
        end
        seeded
    end
    while initialization_attempt < initialization_limit
        initialization_attempt += 1
        trace = try
            generate(model, args, generation_constraints; rng=rng)[1]
        catch err
            _gibbs_rejectable_error(err) || rethrow()
            continue
        end
        if !initialized
            sites = _gibbs_discrete_sites(model, trace, constraints)
            if !isnothing(initial_discrete)
                for entry in initial_discrete.entries
                    address = first(entry)
                    any(site -> site.address == address, sites) || throw(
                        ArgumentError(
                            "initial_discrete provides a value for $address, which is not a " *
                            "discovered Gibbs site (sites: $(Any[site.address for site in sites]))",
                        ),
                    )
                end
            end
            initialized = true
        end
        for site in sites
            _pushchoice!(merged, site.address, trace.choices[site.address])
        end
        if !isnothing(initial_discrete)
            for entry in initial_discrete.entries
                _pushchoice!(merged, first(entry), last(entry))
            end
        end
        position = if isnothing(seeded_position)
            has_continuous ? _gibbs_trace_unconstrained(model, trace, continuous_layout, args, constraints) :
            Float64[]
        else
            seeded_position
        end
        current_logjoint = try
            logjoint_unconstrained(model, position, args, merged)
        catch err
            _gibbs_rejectable_error(err) || rethrow()
            -Inf
        end
        isfinite(current_logjoint) && break
    end
    isfinite(current_logjoint) || throw(
        ArgumentError(
            "gibbs could not find a finite initial state from the prior after " *
            "$initialization_limit attempts; pass initial_discrete (and initial_params) " *
            "to seed a supported state",
        ),
    )

    nuts_step_size = Float64(step_size)
    nuts_target_accept = Float64(target_accept)
    nuts_divergence_threshold = Float64(divergence_threshold)
    if has_continuous
        # sampler-owned evaluation: Stan-style reject semantics (issue #157)
        gradient_cache = _logjoint_gradient_cache(model, position, args, merged; reject_invalid_parameters=true)
        current_gradient = copy(_logjoint_gradient!(gradient_cache, position))
        all(isfinite, current_gradient) ||
            throw(
                ArgumentError(
                    "initial gibbs state produced a non-finite unconstrained gradient; supply finite initial_params (and initial_discrete), or check the constraint values for NaN/Inf",
                ),
            )
        inverse_mass_matrix = ones(num_params)
        if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
            nuts_step_size = _find_reasonable_step_size(
                model,
                position,
                current_logjoint,
                gradient_cache,
                inverse_mass_matrix,
                args,
                merged,
                nuts_step_size,
                rng,
            )
        end
        driver = WarmupDriver(
            num_params,
            num_warmup,
            nuts_step_size,
            nuts_target_accept;
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            metric=metric,
        )
        refind = ScalarStepSizeSearch(model, gradient_cache, args, merged, rng, position, current_logjoint)
        nuts_target = ModelDensityTarget(model, args, merged, gradient_cache)
        # Per-iteration NUTS scratch, hoisted out of the sampler loop (issue
        # #159): _nuts_proposal re-initializes both structs on entry.
        nuts_tree_workspace = NUTSSubtreeWorkspace(num_params, max_tree_depth)
        nuts_continuation = NUTSContinuationState(num_params)
    end

    unconstrained_samples = Matrix{Float64}(undef, num_params, num_samples)
    constrained_samples = Matrix{Float64}(undef, constrained_num_params, num_samples)
    logjoint_values = Vector{Float64}(undef, num_samples)
    acceptance_stats = zeros(num_samples)
    accepted = falses(num_samples)
    divergent = falses(num_samples)
    discrete_samples = Matrix{Int}(undef, length(sites), num_samples)
    discrete_accepted = zeros(Int, length(sites))

    total_iterations = num_warmup + num_samples
    sample_index = 0
    for iteration = 1:total_iterations
        # --- discrete pass: symmetric single-site MH ----------------------
        for (site_index, site) in enumerate(sites)
            site.kind === :fixed && continue
            current_value = merged[site.address]
            proposed = _gibbs_propose(rng, site, current_value, discrete_tail_probability)
            # static-lower-bound pre-rejection (the compiled scorer would
            # bind the invalid value and a suffix consumer may throw)
            site.kind === :walk && proposed < site.lower && continue
            _pushchoice!(merged, site.address, proposed)
            # a proposal the model cannot evaluate is a zero-density region
            # (dynamic support bounds); the CURRENT state's scoring is never
            # caught, and only support/domain failures map to rejection --
            # interrupts and unrelated errors rethrow
            proposal_logjoint = try
                logjoint_unconstrained(model, position, args, merged)
            catch err
                _gibbs_rejectable_error(err) || rethrow()
                -Inf
            end
            if log(rand(rng)) < proposal_logjoint - current_logjoint
                current_logjoint = proposal_logjoint
                discrete_accepted[site_index] += 1
            else
                _pushchoice!(merged, site.address, current_value)
            end
        end

        # --- continuous pass: one NUTS transition -------------------------
        divergent_step = false
        moved_step = false
        accept_stat = 0.0
        if has_continuous
            # the discrete pass changed the conditioning under the unchanged
            # position, so the cached logjoint/gradient must be refreshed
            current_gradient = copy(_logjoint_gradient!(gradient_cache, position))
            nuts_step_size = driver.step_size
            inverse_mass_matrix = _driver_metric(driver)
            proposal, accept_stat, _, _, _, _, divergent_step, moved_step = _nuts_proposal(
                nuts_tree_workspace,
                nuts_continuation,
                nuts_target,
                position,
                current_logjoint,
                current_gradient,
                inverse_mass_matrix,
                nuts_step_size,
                max_tree_depth,
                nuts_divergence_threshold,
                rng,
            )
            if moved_step
                # `proposal` aliases the hoisted continuation's state, which
                # the next iteration mutates in place -- copy the values out
                # instead of retaining the buffers.
                copyto!(position, proposal.position)
                current_logjoint = proposal.logjoint
                copyto!(current_gradient, proposal.gradient)
            end
            if iteration <= num_warmup
                refind.position = position
                refind.current_logjoint = current_logjoint
                mass_weight = _mass_adaptation_weight(driver.variance_state, false, accept_stat, divergent_step)
                warmup_update!(driver, iteration, accept_stat, position, mass_weight, refind)
                iteration == num_warmup && warmup_finalize!(driver)
            end
        end

        if iteration > num_warmup
            sample_index += 1
            unconstrained_samples[:, sample_index] = position
            constrained_samples[:, sample_index] =
                has_continuous ? transform_to_constrained(model, position, args, constraints) : Float64[]
            logjoint_values[sample_index] = current_logjoint
            acceptance_stats[sample_index] = accept_stat
            accepted[sample_index] = moved_step
            divergent[sample_index] = divergent_step
            for (site_index, site) in enumerate(sites)
                discrete_samples[site_index, sample_index] = Int(merged[site.address])
            end
        end
    end

    return GibbsChain(
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        logjoint_values,
        acceptance_stats,
        accepted,
        divergent,
        has_continuous ? driver.step_size : nuts_step_size,
        has_continuous ? copy(driver.inverse_mass_matrix) : Float64[],
        has_continuous && driver.metric_kind === :dense ? copy(driver.dense_metric.inverse_mass) : nothing,
        Any[site.address for site in sites],
        discrete_samples,
        discrete_accepted ./ total_iterations,
    )
end

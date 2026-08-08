# Posterior-draws interface (issue #337).
#
# Every inference result exposes its posterior draws in CONSTRAINED space
# through one accessor, `constrained_draws`, so the generic consumers
# (`predict`, `loo`/`psis_loo`/`waic` via `pointwise_loglikelihood`) accept any
# result type instead of a per-sampler whitelist. This file is included LAST in
# the inference tree because the interface union below needs every result type
# to be defined.

"""
    PosteriorDrawsResult

Union of every inference result type that implements
[`constrained_draws`](@ref): `HMCChains`, `GibbsChain`, `ADVIResult`,
`PathfinderResult`, `SVGDResult`, `ImportanceSamplingResult`, `SIRResult`,
`SMCResult`, `NestedSamplingResult`, `EllipticalSliceResult`, `LaplaceResult`,
and `MAPResult`. The generic `predict` and the four-argument
`pointwise_loglikelihood` / `waic` / `psis_loo` / `loo` forms dispatch on it,
so an unsupported argument fails with a `MethodError` instead of a downstream
field error.
"""
const PosteriorDrawsResult = Union{
    HMCChains,
    GibbsChain,
    ADVIResult,
    PathfinderResult,
    SVGDResult,
    ImportanceSamplingResult,
    SIRResult,
    SMCResult,
    NestedSamplingResult,
    EllipticalSliceResult,
    LaplaceResult,
    MAPResult,
}

# Default draw count for the results that GENERATE draws on demand (ADVI's
# fitted guide, the Laplace Gaussian) rather than storing a fixed set.
const _GENERATED_DRAWS_DEFAULT = 1000

# Stored-draw selection: `nothing` keeps every stored draw (the returned matrix
# may alias result-internal storage); an explicit count picks an evenly-spread
# subset, capped at the number available (same rule as `predict`).
_select_stored_draws(draws::AbstractMatrix, ::Nothing) = draws
function _select_stored_draws(draws::AbstractMatrix, num_draws::Int)
    indices = _even_draw_indices(size(draws, 2), num_draws)
    return draws[:, indices]
end

# Constrained-space display names from a result's conditioning triple.
function _constrained_draw_names(model::TeaModel, args, constraints::ChoiceMap)
    return _export_parameter_names(model, args, constraints, :constrained)
end

"""
    constrained_draws(result; num_draws=nothing, rng=Random.default_rng())
        -> (draws::AbstractMatrix{Float64}, names::Vector{String})

Uniform posterior-draws accessor implemented by every inference result (see
`PosteriorDrawsResult`): `draws` is a `num_params x num_draws` matrix
of CONSTRAINED-space parameter vectors (columns are draws) and `names` are the
per-parameter display names matching the row order (as in `parameter_names`).

`num_draws=nothing` (default) keeps every available draw for stored-draw
results and draws $(_GENERATED_DRAWS_DEFAULT) for generated-draw results
(`ADVIResult`, `LaplaceResult`); an explicit count selects an evenly-spread
subset (stored draws) or that many fresh draws (generated draws). `rng` drives
the weighted-result resampling and the generated draws; it is unused by
deterministic results.

Per-result semantics:
- `HMCChains`: the pooled post-warmup draws across chains.
- `GibbsChain`: the continuous block's draws (discrete sites live in
  `result.discrete_samples` / `discrete_ess`).
- `ADVIResult`: fresh draws from the fitted guide (`variational_samples`).
- `PathfinderResult`: the stored unconstrained draws mapped through the
  model's constraining transform.
- `SVGDResult`: the optimized particles.
- `ImportanceSamplingResult`, `SMCResult`, `NestedSamplingResult`: particles
  resampled in proportion to their normalized importance weights (systematic),
  so the returned columns are unweighted posterior draws.
- `SIRResult`: the already-resampled particles.
- `EllipticalSliceResult`: the stored draws (no model attached; names are
  generic `f[i]`).
- `LaplaceResult`: fresh Gaussian draws at the mode, constrained.
- `MAPResult`: the constrained mode as a single draw.

The returned matrix may alias result-internal storage; copy before mutating.
"""
function constrained_draws(
    chains::HMCChains;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    isempty(chains.chains) && throw(ArgumentError("constrained_draws requires at least one chain"))
    pooled =
        length(chains.chains) == 1 ? first(chains.chains).constrained_samples :
        reduce(hcat, (chain.constrained_samples for chain in chains.chains))
    return _select_stored_draws(pooled, num_draws),
    _constrained_draw_names(chains.model, chains.args, chains.constraints)
end

function constrained_draws(
    chain::GibbsChain;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    return _select_stored_draws(chain.constrained_samples, num_draws),
    _constrained_draw_names(chain.model, chain.args, chain.constraints)
end

function constrained_draws(
    result::ADVIResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    count = isnothing(num_draws) ? _GENERATED_DRAWS_DEFAULT : num_draws
    draws = variational_samples(result; num_samples=count, space=:constrained, rng=rng)
    return draws, _constrained_draw_names(result.model, result.args, result.constraints)
end

function constrained_draws(
    result::PathfinderResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    unconstrained = _select_stored_draws(result.draws, num_draws)
    layout = _conditioned_parameter_layout(result.model, result.constraints, result.args)
    constrained = Matrix{Float64}(undef, parametervaluecount(layout), size(unconstrained, 2))
    _signature_batched_transform_to_constrained!(
        constrained,
        result.model,
        unconstrained,
        result.args,
        result.constraints,
    )
    return constrained, _constrained_draw_names(result.model, result.args, result.constraints)
end

function constrained_draws(
    result::SVGDResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    return _select_stored_draws(result.constrained_particles, num_draws),
    _constrained_draw_names(result.model, result.args, result.constraints)
end

# Weighted particle populations: systematic resampling by the normalized
# importance weights turns them into (approximately) unweighted posterior
# draws, exactly as the predictive path has always done.
function _weighted_constrained_draws(
    particles::AbstractMatrix,
    normalized_weights::AbstractVector,
    num_draws::Union{Nothing,Int},
    rng::AbstractRNG,
)
    count = isnothing(num_draws) ? size(particles, 2) : num_draws
    count > 0 || throw(ArgumentError("constrained_draws requires num_draws > 0"))
    ancestors = _systematic_resample_indices(normalized_weights, count, rng)
    return particles[:, ancestors]
end

function constrained_draws(
    result::ImportanceSamplingResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    draws = _weighted_constrained_draws(
        result.constrained_particles, result.normalized_weights, num_draws, rng,
    )
    return draws, _constrained_draw_names(result.model, result.args, result.constraints)
end

function constrained_draws(
    result::SMCResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    return constrained_draws(result.importance; num_draws=num_draws, rng=rng)
end

# SIR already resampled its particles; keep them as-is (evenly subset when a
# count is requested) instead of re-resampling the importance population.
function constrained_draws(
    result::SIRResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    return _select_stored_draws(result.constrained_samples, num_draws),
    _constrained_draw_names(result.importance.model, result.importance.args, result.importance.constraints)
end

function constrained_draws(
    result::NestedSamplingResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    draws = _weighted_constrained_draws(
        result.constrained_samples, result.normalized_weights, num_draws, rng,
    )
    return draws, _constrained_draw_names(result.model, result.args, result.constraints)
end

# Elliptical slice sampling has no model attached (it targets a Gaussian prior
# plus a user log-likelihood), so the names are the generic latent coordinates.
function constrained_draws(
    result::EllipticalSliceResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    names = String[string("f[", index, "]") for index = 1:size(result.samples, 1)]
    return _select_stored_draws(result.samples, num_draws), names
end

function constrained_draws(
    result::LaplaceResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    count = isnothing(num_draws) ? _GENERATED_DRAWS_DEFAULT : num_draws
    count > 0 || throw(ArgumentError("constrained_draws requires num_draws > 0"))
    map_result = result.map
    unconstrained = rand(rng, result, count)
    layout = _conditioned_parameter_layout(map_result.model, map_result.constraints, map_result.args)
    constrained = Matrix{Float64}(undef, parametervaluecount(layout), count)
    _signature_batched_transform_to_constrained!(
        constrained,
        map_result.model,
        unconstrained,
        map_result.args,
        map_result.constraints,
    )
    return constrained, _constrained_draw_names(map_result.model, map_result.args, map_result.constraints)
end

function constrained_draws(
    result::MAPResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    draws = reshape(copy(result.constrained_mode), :, 1)
    return draws, _constrained_draw_names(result.model, result.args, result.constraints)
end

# --- GibbsChain diagnostics (issue #337) --------------------------------------
# The continuous block of a Gibbs run is a NUTS chain, so the HMC diagnostics
# apply verbatim; wrap it in a one-chain HMCChains with the per-draw stats the
# Gibbs sampler does not track (energies, tree depths) marked absent. Discrete
# sites keep their dedicated path (`discrete_ess`, `discrete_samples`).
function _gibbs_continuous_chains(chain::GibbsChain)
    num_samples = size(chain.constrained_samples, 2)
    continuous = HMCChain(
        :gibbs,
        chain.model,
        chain.args,
        chain.constraints,
        chain.unconstrained_samples,
        chain.constrained_samples,
        chain.logjoint_values,
        chain.acceptance_stats,
        fill(NaN, num_samples),  # energies: not tracked by gibbs
        fill(NaN, num_samples),  # energy errors: not tracked by gibbs
        chain.accepted,
        chain.divergent,
        chain.step_size,
        chain.mass_matrix,
        0,                        # num_leapfrog_steps (NUTS block)
        0,                        # max_tree_depth: not recorded
        zeros(Int, num_samples),  # tree depths: not recorded
        zeros(Int, num_samples),  # integration steps: not recorded
        NaN,                      # target_accept: not recorded on the result
        HMCMassAdaptationWindowSummary[],
        chain.dense_mass_matrix,
    )
    return HMCChains(chain.model, chain.args, chain.constraints, [continuous])
end

"""
    summarize(chain::GibbsChain; kwargs...) -> HMCSummary

Summarize the CONTINUOUS block of a Gibbs run (same keywords as the
`HMCChains` method). Discrete sites are not part of the continuous parameter
vector; inspect them via `discrete_ess(chain)` and `chain.discrete_samples`.
"""
summarize(chain::GibbsChain; kwargs...) = summarize(_gibbs_continuous_chains(chain); kwargs...)

rhat(chain::GibbsChain; kwargs...) = rhat(_gibbs_continuous_chains(chain); kwargs...)
ess(chain::GibbsChain; kwargs...) = ess(_gibbs_continuous_chains(chain); kwargs...)
posterior_array(chain::GibbsChain; kwargs...) = posterior_array(_gibbs_continuous_chains(chain); kwargs...)
parameter_names(chain::GibbsChain; kwargs...) = parameter_names(_gibbs_continuous_chains(chain); kwargs...)

# --- Generic consumers over the interface -------------------------------------

"""
    predict(model, args, result; num_draws=nothing, rng) -> PredictiveDraws
    predict(model, args, constraints, result; num_draws=nothing, rng) -> PredictiveDraws
    predict(model, args=(); num_draws=1000, rng) -> PredictiveDraws

Posterior predictive draws: for each posterior draw in `result` (any
`PosteriorDrawsResult` — `HMCChains`, `GibbsChain`, `ADVIResult`,
`PathfinderResult`, `SVGDResult`, `ImportanceSamplingResult`, `SIRResult`,
`SMCResult`, `NestedSamplingResult`, `EllipticalSliceResult`, `LaplaceResult`,
`MAPResult`), fix the latent parameters at that draw and re-sample the model's
remaining choices — the observation addresses. Draws come from
[`constrained_draws`](@ref), so weighted results are resampled in proportion
to their importance weights first and `num_draws`/`rng` follow its semantics
(`nothing` keeps every available draw). The form without a result runs the
prior instead (unconstrained `generate` per draw, keeping all addresses).
Returns `PredictiveDraws`; see also `prior_predictive`.

The four-argument form accepts the same conditioning triple as [`loo`](@ref) /
[`waic`](@ref) / [`pointwise_loglikelihood`](@ref), so the one argument list
`(model, args, constraints, result)` drives the whole posterior workflow.
`predict` does not need the observation values — they are exactly what it
re-samples, and the observed/latent split is already fixed by how `result` was
parameterized — so `constraints` only has to be the same choicemap the model
was conditioned on; its values are unused.
"""
function predict(
    model::TeaModel,
    args::Tuple,
    result::PosteriorDrawsResult;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    draws, _ = constrained_draws(result; num_draws=num_draws, rng=rng)
    size(draws, 2) > 0 || throw(ArgumentError("predict requires at least one posterior draw"))
    columns = (view(draws, :, index) for index in axes(draws, 2))
    return _predictive_from_param_columns(model, args, columns, rng)
end

# Arity alignment with `loo`/`waic`/`pointwise_loglikelihood` (issue #339): the
# `(model, args, constraints, result)` triple works verbatim. `constraints` is
# accepted for signature symmetry only — predict re-samples the observation
# addresses, whose split is fixed by `result`'s parameterization — so it
# forwards to the three-argument form (which keeps per-result refinements such
# as the GibbsChain discrete-site pinning via dispatch).
function predict(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    result::PosteriorDrawsResult;
    kwargs...,
)
    return predict(model, args, result; kwargs...)
end

# GibbsChain refinement: pin each predictive draw's DISCRETE-site values to the
# corresponding posterior sample (they are part of the posterior draw, so
# re-sampling them from the prior conditional would bias the predictive).
function predict(
    model::TeaModel,
    args::Tuple,
    chain::GibbsChain;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    total = size(chain.constrained_samples, 2)
    indices = isnothing(num_draws) ? collect(1:total) : _even_draw_indices(total, num_draws)
    sites = chain.discrete_addresses
    if isempty(sites)
        columns = (view(chain.constrained_samples, :, index) for index in indices)
        return _predictive_from_param_columns(model, args, columns, rng)
    end

    draws = ChoiceMap[]
    for draw_index in indices
        constraint_cm = parameterchoicemap(model, view(chain.constrained_samples, :, draw_index))
        for (site_index, address) in enumerate(sites)
            _pushchoice!(constraint_cm, address, chain.discrete_samples[site_index, draw_index])
        end
        trace, _ = generate(model, args, constraint_cm; rng=rng)
        draw = ChoiceMap()
        for entry in trace.choices
            address = first(entry)
            haskey(constraint_cm, address) || _pushchoice!(draw, address, last(entry))
        end
        push!(draws, draw)
    end
    return PredictiveDraws(draws)
end

function pointwise_loglikelihood(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    result::PosteriorDrawsResult;
    kwargs...,
)
    draws, _ = constrained_draws(result; kwargs...)
    return pointwise_loglikelihood(model, args, constraints, draws)
end

# GibbsChain refinement: condition each draw's pointwise walk on that draw's
# sampled DISCRETE-site values, so the per-observation terms are the
# conditional log-likelihoods p(y_i | theta_s, z_s). The discrete sites do NOT
# become observation columns: the output keeps one column per USER observation
# (an address present in `constraints`), in `observation_addresses` order.
function pointwise_loglikelihood(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    chain::GibbsChain;
    num_draws::Union{Nothing,Int}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    total = size(chain.constrained_samples, 2)
    indices = isnothing(num_draws) ? collect(1:total) : _even_draw_indices(total, num_draws)
    sites = chain.discrete_addresses
    if isempty(sites)
        return pointwise_loglikelihood(model, args, constraints, chain.constrained_samples[:, indices])
    end

    # Bool sites are stored as 0/1 Ints in `discrete_samples`; the scorers
    # accept the numeric encoding, so the Int values condition the walk as-is.
    merged = choicemap((first(entry), last(entry)) for entry in constraints.entries)
    for (site_index, address) in enumerate(sites)
        _pushchoice!(merged, address, chain.discrete_samples[site_index, first(indices)])
    end
    observed = observation_addresses(model, args, merged)
    obs_addresses = Any[address for address in observed if haskey(constraints, address)]
    column_of = Dict{Any,Int}()
    for (column, address) in enumerate(obs_addresses)
        column_of[address] = column
    end

    ll = Matrix{Float64}(undef, length(indices), length(obs_addresses))
    records = Pair{Any,Float64}[]
    for (s, draw_index) in enumerate(indices)
        for (site_index, address) in enumerate(sites)
            _pushchoice!(merged, address, chain.discrete_samples[site_index, draw_index])
        end
        empty!(records)
        _record_execution!(records, model, view(chain.constrained_samples, :, draw_index), args, merged)
        for (address, logdensity) in records
            column = get(column_of, address, 0)
            column == 0 && continue  # a discrete site's own log-density, not a user observation
            ll[s, column] = logdensity
        end
    end
    return ll
end

# Four-argument model-comparison wrappers: compute the pointwise log-likelihood
# from the result's constrained draws first. Typed on the interface union so a
# wrong argument dies with a MethodError instead of a field error (issue #337).
function waic(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    result::Union{PosteriorDrawsResult,AbstractMatrix};
    kwargs...,
)
    return waic(pointwise_loglikelihood(model, args, constraints, result; kwargs...))
end

function psis_loo(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    result::Union{PosteriorDrawsResult,AbstractMatrix};
    kwargs...,
)
    return psis_loo(pointwise_loglikelihood(model, args, constraints, result; kwargs...))
end

function loo(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    result::Union{PosteriorDrawsResult,AbstractMatrix};
    kwargs...,
)
    return psis_loo(pointwise_loglikelihood(model, args, constraints, result; kwargs...))
end

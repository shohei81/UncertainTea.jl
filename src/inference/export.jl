# Ecosystem interop: export HMCChains into the draw-major array layout used by
# ArviZ / MCMCChains, plus zero-dependency dictionary and (via a package
# extension) native MCMCChains.Chains construction.

# Display names for each parameter in the requested space, matching the
# per-component convention that `summarize` uses (see
# `_summary_parameter_entries` in diagnostics.jl): scalar slots keep their
# binding name (e.g. "sigma"), vector slots get a bracketed component index
# (e.g. "v[1]").
# Signature-aware naming (#95 PR-6): sampler output is sized by the conditioning
# signature of the chain's constraints, so its display names must come from that
# same layout, not the syntactic default `parameterlayout(model)`.
function _export_parameter_names(model::TeaModel, args, constraints::ChoiceMap, space::Symbol)
    return _export_parameter_names(_conditioned_parameter_layout(model, constraints, args), space)
end

function _export_parameter_names(model::TeaModel, space::Symbol)
    return _export_parameter_names(parameterlayout(model), space)
end

function _export_parameter_names(layout::ParameterLayout, space::Symbol)
    entries = _summary_parameter_entries(layout, space)
    num_params = length(entries)
    names = Vector{String}(undef, num_params)
    for slot in layout.slots
        indices = space === :constrained ? parametervalueindices(slot) : parameterindices(slot)
        component_count = length(indices)
        for (offset, parameter_index) in enumerate(indices)
            names[parameter_index] = component_count == 1 ?
                                     String(slot.binding) : string(slot.binding, "[", offset, "]")
        end
    end
    return names
end

"""
    parameter_names(chains::HMCChains; space=:constrained) -> Vector{String}

Return per-parameter display names in the requested `space` (`:constrained` or
`:unconstrained`), ordered to match the third dimension of [`posterior_array`](@ref).
"""
function parameter_names(chains::HMCChains; space::Symbol=:constrained)
    return _export_parameter_names(chains.model, chains.args, chains.constraints, space)
end

"""
    posterior_array(chains::HMCChains; space=:constrained) -> Array{Float64,3}

Return posterior draws shaped `(num_samples, num_chains, num_params)` following
the ArviZ / MCMCChains draw-major convention. `space` selects `:constrained`
(default) or `:unconstrained` samples.
"""
function posterior_array(chains::HMCChains; space::Symbol=:constrained)
    isempty(chains.chains) && throw(ArgumentError("posterior_array requires at least one chain"))
    first_samples = _diagnostic_space_samples(first(chains.chains), space)
    num_params, num_samples = size(first_samples)
    num_chains = length(chains.chains)
    result = Array{Float64,3}(undef, num_samples, num_chains, num_params)
    for (chain_index, chain) in enumerate(chains.chains)
        samples = _diagnostic_space_samples(chain, space)
        size(samples, 1) == num_params ||
            throw(DimensionMismatch("all chains must share the same parameter dimension"))
        size(samples, 2) == num_samples ||
            throw(DimensionMismatch("all chains must share the same number of samples"))
        for p = 1:num_params, s = 1:num_samples
            result[s, chain_index, p] = samples[p, s]
        end
    end
    return result
end

# Layout handling for `to_arviz_dict` (issue #339): every group matrix is built
# `(num_samples, num_chains)` internally; `:chain_draw` transposes it on the way
# out to match the Python `az.from_dict` `(chain, draw)` convention.
function _validate_arviz_layout(layout::Symbol)
    layout in (:draw_chain, :chain_draw) || throw(
        ArgumentError(
            "to_arviz_dict layout must be :draw_chain or :chain_draw, got $(repr(layout))",
        ),
    )
    return layout
end

_arviz_layout_matrix(matrix::AbstractMatrix, layout::Symbol) =
    layout === :chain_draw ? permutedims(matrix) : matrix

# Display name of an observation address, matching the flattened parameter
# naming convention: a single-part address keeps its name (e.g. "y"), a
# multi-part address gets bracketed trailing parts (e.g. `(:y, 3)` -> "y[3]").
_observation_display_name(address::Tuple) =
    length(address) == 1 ? string(address[1]) :
    string(address[1], "[", join(address[2:end], ","), "]")
_observation_display_name(address) = string(address)

"""
    to_arviz_dict(chains::HMCChains; layout=:draw_chain) -> Dict{String,Any}
    to_arviz_dict(model, args, constraints, chains::HMCChains; layout=:draw_chain)
        -> Dict{String,Any}

Return an ArviZ-style nested dictionary with:

- a `"posterior"` group: one matrix per constrained parameter. Vector-valued
  parameters are flattened to per-component keys following the same convention
  as [`parameter_names`](@ref) (e.g. a length-3 `v` becomes `"v[1]"`, `"v[2]"`,
  `"v[3]"`); re-assembling them into a single array with a coordinate
  dimension is future work.
- a `"sample_stats"` group: `"diverging"`, `"energy"`, `"tree_depth"`,
  `"acceptance_rate"`, `"lp"`, `"step_size"` (the chain's adapted step size,
  constant across its draws), and `"n_steps"` (per-draw leapfrog integration
  steps).
- an `"attrs"` entry recording `"layout"` and the producing package, so a
  round-trip through NPZ/JSON keeps the axis convention explicit.

The four-argument form (same conditioning triple as [`loo`](@ref) /
[`pointwise_loglikelihood`](@ref)) additionally emits a `"log_likelihood"`
group: one matrix per observation address (named like `"y[1]"`), computed via
[`pointwise_loglikelihood`](@ref) in [`observation_addresses`](@ref) order —
the group `az.loo` / `az.compare` need.

`layout` selects the matrix axis order for every group:

- `:draw_chain` (default): `(num_samples, num_chains)`, the Julia
  InferenceObjects convention.
- `:chain_draw`: `(num_chains, num_samples)`, the Python `az.from_dict`
  convention. Use this when handing the dictionary to Python ArviZ — with the
  default layout a 2-chain x 1000-draw run would silently round-trip as 1000
  chains x 2 draws.
"""
function to_arviz_dict(chains::HMCChains; layout::Symbol=:draw_chain)
    _validate_arviz_layout(layout)
    isempty(chains.chains) && throw(ArgumentError("to_arviz_dict requires at least one chain"))
    names = parameter_names(chains; space=:constrained)
    draws = posterior_array(chains; space=:constrained)
    num_samples, num_chains, num_params = size(draws)

    posterior = Dict{String,Any}()
    for p = 1:num_params
        posterior[names[p]] = _arviz_layout_matrix(Array{Float64,2}(draws[:, :, p]), layout)
    end

    diverging = Array{Bool,2}(undef, num_samples, num_chains)
    energy = Array{Float64,2}(undef, num_samples, num_chains)
    tree_depth = Array{Float64,2}(undef, num_samples, num_chains)
    acceptance_rate = Array{Float64,2}(undef, num_samples, num_chains)
    lp = Array{Float64,2}(undef, num_samples, num_chains)
    step_size = Array{Float64,2}(undef, num_samples, num_chains)
    n_steps = Array{Int,2}(undef, num_samples, num_chains)
    for (chain_index, chain) in enumerate(chains.chains)
        for s = 1:num_samples
            diverging[s, chain_index] = chain.divergent[s]
            energy[s, chain_index] = chain.energies[s]
            tree_depth[s, chain_index] = chain.tree_depths[s]
            acceptance_rate[s, chain_index] = chain.acceptance_stats[s]
            lp[s, chain_index] = chain.logjoint_values[s]
            step_size[s, chain_index] = chain.step_size
            n_steps[s, chain_index] = chain.integration_steps[s]
        end
    end

    sample_stats = Dict{String,Any}(
        "diverging" => _arviz_layout_matrix(diverging, layout),
        "energy" => _arviz_layout_matrix(energy, layout),
        "tree_depth" => _arviz_layout_matrix(tree_depth, layout),
        "acceptance_rate" => _arviz_layout_matrix(acceptance_rate, layout),
        "lp" => _arviz_layout_matrix(lp, layout),
        "step_size" => _arviz_layout_matrix(step_size, layout),
        "n_steps" => _arviz_layout_matrix(n_steps, layout),
    )

    attrs = Dict{String,Any}(
        "layout" => String(layout),
        "inference_library" => "UncertainTea",
        "inference_library_version" => string(pkgversion(UncertainTea)),
    )

    return Dict{String,Any}(
        "posterior" => posterior,
        "sample_stats" => sample_stats,
        "attrs" => attrs,
    )
end

function to_arviz_dict(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    chains::HMCChains;
    layout::Symbol=:draw_chain,
)
    result = to_arviz_dict(chains; layout=layout)
    num_samples = numsamples(chains)
    num_chains = nchains(chains)
    addresses = observation_addresses(model, args, constraints)
    # Rows are the pooled draws of `constrained_draws` (chain-major: chain 1's
    # draws, then chain 2's, ...), so a column reshapes straight into the
    # `(num_samples, num_chains)` posterior-group shape.
    ll = pointwise_loglikelihood(model, args, constraints, chains)
    size(ll, 1) == num_samples * num_chains || throw(
        DimensionMismatch(
            "pointwise log-likelihood has $(size(ll, 1)) draws, expected " *
            "$(num_samples * num_chains) ($(num_samples) samples x $(num_chains) chains)",
        ),
    )
    log_likelihood = Dict{String,Any}()
    for (column, address) in enumerate(addresses)
        matrix = reshape(ll[:, column], num_samples, num_chains)
        log_likelihood[_observation_display_name(address)] = _arviz_layout_matrix(matrix, layout)
    end
    result["log_likelihood"] = log_likelihood
    return result
end

# Canonical package-extension pattern: the core declares the function with no
# methods; the UncertainTeaMCMCChainsExt extension (loaded when MCMCChains.jl is
# present) adds the `::HMCChains` method. Calling without the extension loaded
# raises a MethodError, which is the intended "load MCMCChains.jl" signal.
"""
    to_mcmcchains(chains::HMCChains; space=:constrained)

Convert `chains` into an `MCMCChains.Chains` object with the standard
`iterations × parameters × chains` layout. Requires the optional `MCMCChains`
dependency to be loaded (which activates the package extension).

The `:parameters` section holds the posterior draws in the requested `space`
(named as in [`parameter_names`](@ref)); the `:internals` section holds the
per-draw sampler statistics `:lp`, `:diverging`, `:energy`, `:tree_depth`, and
`:acceptance_rate` (the same quantities as the `"sample_stats"` group of
[`to_arviz_dict`](@ref)).

!!! note "Name collisions with MCMCChains"
    `MCMCChains` also exports `summarize`, `ess`, and `rhat`, so after
    `using UncertainTea, MCMCChains` the unqualified names are ambiguous and
    raise an error. Call the UncertainTea versions qualified when both
    packages are loaded, e.g. `UncertainTea.summarize(chains)`,
    `UncertainTea.ess(chains)`, `UncertainTea.rhat(chains)` — or use the
    MCMCChains versions on the converted `Chains` object.
"""
function to_mcmcchains end

# Same package-extension pattern for host reverse-mode AD (issue #268, follow-up
# to RFC #263): the core declares `reverse_mode_gradient` method-less; the
# UncertainTeaEnzymeExt extension (loaded when Enzyme.jl is present) adds the
# implementation. Calling without Enzyme loaded raises a MethodError -- the
# intended "load Enzyme.jl" signal.
"""
    reverse_mode_gradient(f, x::AbstractVector) -> Vector
    reverse_mode_gradient(model::TeaModel, params, args=(), constraints=choicemap()) -> Vector

Reverse-mode gradient via Enzyme.jl. Requires the optional `Enzyme` dependency to
be loaded (which activates the package extension); without it these raise a
`MethodError` — the intended "load Enzyme" signal.

The first form differentiates a pure scalar objective `f` at `x` (`f` may close
over constant data, treated as non-differentiable) — e.g. a Gaussian-process
marginal likelihood as a function of its hyperparameters.

The second form is the reverse-mode analogue of `logjoint_gradient_unconstrained`:
`∇` of `model`'s unconstrained logjoint at `params`. It is `O(1)` in the parameter
count instead of the forward-mode path's `O(P)`, so it pays off on high-dimensional
non-analytic models (measured 18.9× at `P=100`, 107× at `P=800`; see
`bench/reverse_mode/`), returning a result identical to the forward-mode gradient.
It is available only for models on the type-stable generated-scorer path; a model
that falls back to the interpreter raises an `ArgumentError` (use the forward-mode
`logjoint_gradient_unconstrained` for those).

The batched sampler gradient path (`batched_logjoint_gradient_unconstrained`) still
runs forward-mode/analytic; wiring per-column reverse-mode into it is tracked in #268.

```julia
using UncertainTea, Enzyme
# pure objective
gp_nlml(h) = UncertainTea.logpdf(gaussianprocess(X, exp(h[1]), exp(h[2]), exp(h[3])), y)
g = reverse_mode_gradient(gp_nlml, [0.0, 0.0, -1.0])
# whole model
g = reverse_mode_gradient(model, theta, args, constraints)
```
"""
function reverse_mode_gradient end

# Value + gradient in one reverse pass, supplied by UncertainTeaEnzymeExt. Internal
# to the batched per-column reverse-mode path (issue #268, part A2); not exported.
function reverse_mode_value_and_gradient end

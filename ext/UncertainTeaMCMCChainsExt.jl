module UncertainTeaMCMCChainsExt

using UncertainTea
using MCMCChains

# NUTS sampler statistics exported alongside the draws, matching the
# `sample_stats` group of `to_arviz_dict`. Placed in the `:internals` section so
# `MCMCChains` summaries over `:parameters` are unaffected.
const _INTERNAL_STATS = (
    ("lp", :lp),
    ("diverging", :diverging),
    ("energy", :energy),
    ("tree_depth", :tree_depth),
    ("acceptance_rate", :acceptance_rate),
)

# Add the ::HMCChains method to the core `to_mcmcchains` function (which is
# declared method-less in UncertainTea). Uses only the stable
# `Chains(array, names, name_map)` constructor.
#
# Layout note: `posterior_array` follows the ArviZ draw-major convention
# `(num_samples, num_chains, num_params)`, while `MCMCChains.Chains` expects
# `iterations x parameters x chains` -- hence the (1, 3, 2) permutation.
function UncertainTea.to_mcmcchains(chains::UncertainTea.HMCChains; space::Symbol=:constrained)
    draws = UncertainTea.posterior_array(chains; space=space)
    num_samples, num_chains, num_params = size(draws)
    parameter_names = Symbol.(UncertainTea.parameter_names(chains; space=space))
    internal_names = Symbol[name for (_, name) in _INTERNAL_STATS]
    sample_stats = UncertainTea.to_arviz_dict(chains)["sample_stats"]

    data = Array{Float64,3}(undef, num_samples, num_params + length(internal_names), num_chains)
    for p = 1:num_params, c = 1:num_chains, s = 1:num_samples
        data[s, p, c] = draws[s, c, p]
    end
    for (offset, (key, _)) in enumerate(_INTERNAL_STATS)
        stat = sample_stats[key] # (num_samples, num_chains)
        for c = 1:num_chains, s = 1:num_samples
            data[s, num_params+offset, c] = Float64(stat[s, c])
        end
    end

    return MCMCChains.Chains(
        data,
        vcat(parameter_names, internal_names),
        Dict(:internals => internal_names),
    )
end

end # module

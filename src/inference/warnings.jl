# Sampler warning diagnostics: per-chain divergence counts, E-BFMI, treedepth
# saturation, and low-ESS / high-Rhat parameter lists. `SamplerWarnings` itself
# is defined in data_types.jl (so HMCSummary can carry it).

# Energy Bayesian fraction of missing information for one chain's energies.
# Returns NaN when there are fewer than two samples or the energy is constant.
function _ebfmi(energies::AbstractVector)
    num_samples = length(energies)
    num_samples < 2 && return NaN
    mean_energy = _sample_mean(energies)
    denominator = sum((energy - mean_energy)^2 for energy in energies)
    denominator == 0 && return NaN
    numerator = 0.0
    for index = 2:num_samples
        numerator += (energies[index] - energies[index-1])^2
    end
    return numerator / denominator
end

function _treedepth_hits(chain::HMCChain)
    chain.max_tree_depth > 0 || return 0
    isempty(chain.tree_depths) && return 0
    return count(==(chain.max_tree_depth), chain.tree_depths)
end

# Default threshold below which a chain's E-BFMI counts as a warning.
const _EBFMI_WARNING_THRESHOLD = 0.3

"""
    check_diagnostics(chains::HMCChains; space=:constrained, rhat_threshold=1.01, ess_threshold=100.0, ebfmi_threshold=0.3) -> SamplerWarnings

Collect the standard sampler health checks: per-chain divergence counts,
E-BFMI, treedepth-saturation counts, and the parameters whose R-hat exceeds
`rhat_threshold` or whose ESS falls below `ess_threshold` (the R-hat/ESS checks
need at least 4 samples; a single chain is checked via its split halves).
Returns a `SamplerWarnings`; query it with `has_warnings`.
"""
function check_diagnostics(
    chains::HMCChains;
    space::Symbol=:constrained,
    rhat_threshold::Real=1.01,
    ess_threshold::Real=100.0,
    ebfmi_threshold::Real=_EBFMI_WARNING_THRESHOLD,
)
    num_divergent = Int[count(identity, chain.divergent) for chain in chains.chains]
    ebfmi = Float64[_ebfmi(chain.energies) for chain in chains.chains]
    treedepth_hits = Int[_treedepth_hits(chain) for chain in chains.chains]

    low_ess_parameters = String[]
    high_rhat_parameters = String[]
    if length(chains) >= 1 && numsamples(chains) >= 4
        rhats = rhat(chains; space=space)
        ess_values = ess(chains; space=space)
        layout = _conditioned_parameter_layout(chains.model, chains.constraints, chains.args)
        entries = _summary_parameter_entries(layout, space)
        for (parameter_index, binding, address) in entries
            display_name = string(binding, " @ ", address)
            if isfinite(rhats[parameter_index]) && rhats[parameter_index] > rhat_threshold
                push!(high_rhat_parameters, display_name)
            elseif !isfinite(rhats[parameter_index])
                push!(high_rhat_parameters, display_name)
            end
            if !isfinite(ess_values[parameter_index]) || ess_values[parameter_index] < ess_threshold
                push!(low_ess_parameters, display_name)
            end
        end
    end

    return SamplerWarnings(
        num_divergent,
        ebfmi,
        treedepth_hits,
        low_ess_parameters,
        high_rhat_parameters,
        Float64(ebfmi_threshold),
    )
end

function has_warnings(warnings::SamplerWarnings)
    any(count -> count > 0, warnings.num_divergent) && return true
    any(value -> isfinite(value) && value < warnings.ebfmi_threshold, warnings.ebfmi) && return true
    any(count -> count > 0, warnings.treedepth_hits) && return true
    isempty(warnings.low_ess_parameters) || return true
    isempty(warnings.high_rhat_parameters) || return true
    return false
end

function _show_sampler_warnings(io::IO, warnings::SamplerWarnings; indent::AbstractString="  ")
    has_warnings(warnings) || return nothing
    println(io, indent, "warnings:")
    total_divergent = sum(warnings.num_divergent)
    if total_divergent > 0
        println(io, indent, "  divergences: ", total_divergent, " total ", warnings.num_divergent)
    end
    low_ebfmi = [value for value in warnings.ebfmi if isfinite(value) && value < warnings.ebfmi_threshold]
    if !isempty(low_ebfmi)
        println(io, indent, "  low E-BFMI (< ", warnings.ebfmi_threshold, "): ", warnings.ebfmi)
    end
    total_treedepth = sum(warnings.treedepth_hits)
    if total_treedepth > 0
        println(io, indent, "  treedepth saturation: ", total_treedepth, " total ", warnings.treedepth_hits)
    end
    if !isempty(warnings.low_ess_parameters)
        println(io, indent, "  low ESS parameters: ", warnings.low_ess_parameters)
    end
    if !isempty(warnings.high_rhat_parameters)
        println(io, indent, "  high Rhat parameters: ", warnings.high_rhat_parameters)
    end
    return nothing
end

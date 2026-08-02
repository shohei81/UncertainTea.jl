# Posterior-diagnostics extensions for `summarize` (see diagnostics.jl for the
# base machinery). These helpers add MCSE of the mean, tail-ESS, MCSE of
# quantiles (indicator/order-statistic method), and optional per-chain stats.

# Monte Carlo standard error of the posterior mean using the existing (bulk)
# effective sample size.
function _mcse_mean_value(sd::Real, ess_bulk::Real)
    (isfinite(ess_bulk) && ess_bulk > 0) || return NaN
    return sd * sqrt(1 / ess_bulk)
end

# Split-ESS of the indicator draws I(x <= threshold). Reuses the existing
# split-draw layout so the indicator transform is applied element-wise, keeping
# the chain/half ordering intact. Returns NaN for a degenerate (constant)
# indicator (all draws on one side of the threshold).
function _indicator_split_ess(chains::HMCChains, parameter_index::Int, space::Symbol, threshold::Real)
    split_draws = _split_chain_parameter_draws(chains, parameter_index, space)
    indicator = split_draws .<= threshold
    (all(indicator) || !any(indicator)) && return NaN
    return _split_ess(Float64.(indicator))
end

# Tail-ESS (Vehtari et al. 2021 style): min of the ESS of the lower- and
# upper-tail indicators evaluated at the pooled 5% / 95% quantiles.
function _summary_tail_ess(chains::HMCChains, parameter_index::Int, space::Symbol, q05::Real, q95::Real)
    ess_low = _indicator_split_ess(chains, parameter_index, space, q05)
    ess_high = _indicator_split_ess(chains, parameter_index, space, q95)
    return min(ess_low, ess_high)
end

# MCSE of each requested quantile via the practical order-statistic approach
# used by posterior/ArviZ: form the +/- one-SE interval q +/- sqrt(q(1-q)/ess_q)
# on the probability scale (ess_q = split-ESS of I(x <= theta_q)), read the two
# order statistics off the sorted pooled draws, and report half their spread as
# one standard error.
function _summary_mcse_quantiles(
    chains::HMCChains,
    parameter_index::Int,
    space::Symbol,
    sorted_draws::AbstractVector,
    quantile_probs::AbstractVector{Float64},
    quantile_values::AbstractVector{Float64},
)
    mcse_values = Vector{Float64}(undef, length(quantile_probs))
    for (index, probability) in enumerate(quantile_probs)
        theta = quantile_values[index]
        ess_q = _indicator_split_ess(chains, parameter_index, space, theta)
        if !isfinite(ess_q) || ess_q <= 0
            mcse_values[index] = NaN
            continue
        end
        spread = sqrt(probability * (1 - probability) / ess_q)
        lower_probability = clamp(probability - spread, 0.0, 1.0)
        upper_probability = clamp(probability + spread, 0.0, 1.0)
        lower_value = _quantile(sorted_draws, lower_probability)
        upper_value = _quantile(sorted_draws, upper_probability)
        mcse_values[index] = (upper_value - lower_value) / 2
    end
    return mcse_values
end

# Per-chain mean/sd of one parameter (used only when `per_chain=true`).
function _per_chain_parameter_statistics(chains::HMCChains, parameter_index::Int, space::Symbol)
    num_chains = length(chains)
    means = Vector{Float64}(undef, num_chains)
    sds = Vector{Float64}(undef, num_chains)
    for (chain_index, chain) in enumerate(chains.chains)
        samples = _diagnostic_space_samples(chain, space)
        chain_draws = view(samples, parameter_index, :)
        chain_mean = _sample_mean(chain_draws)
        means[chain_index] = chain_mean
        sds[chain_index] = _sample_sd(chain_draws, chain_mean)
    end
    return means, sds
end

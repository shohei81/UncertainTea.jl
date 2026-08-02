function _sampler_label(chains::HMCChains)
    isempty(chains.chains) && return "HMC"
    sampler = first(chains.chains).sampler
    return all(chain -> chain.sampler === sampler, chains.chains) ? uppercase(String(sampler)) : "MIXED"
end

function _summary_float(value::Real; digits::Int=4)
    if isnan(value)
        return "NaN"
    elseif !isfinite(value)
        return value > 0 ? "Inf" : "-Inf"
    end
    return string(round(Float64(value); digits=digits))
end

function _show_mass_adaptation_summary_line(io::IO, summary::HMCMassAdaptationSummary; indent::AbstractString="")
    print(
        io,
        indent,
        "window ",
        summary.window_index,
        " [",
        summary.iteration_start,
        ":",
        summary.iteration_end,
        "]",
        " updated=",
        summary.num_updated,
        "/",
        summary.chains,
        " eff=",
        _summary_float(summary.mean_effective_count; digits=2),
        " mass=",
        _summary_float(summary.mean_mass),
        " clip=",
        _summary_float(summary.mean_clip_scale_end; digits=2),
    )
end

function Base.show(io::IO, summary::HMCMassAdaptationWindowSummary)
    print(
        io,
        "HMCMassAdaptationWindowSummary(window=",
        summary.window_index,
        ", iterations=",
        summary.iteration_start,
        ":",
        summary.iteration_end,
        ", effective_count=",
        round(summary.effective_count; digits=2),
        ", updated=",
        summary.updated,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCMassAdaptationWindowSummary)
    println(io, "HMCMassAdaptationWindowSummary")
    println(io, "  window: ", summary.window_index)
    println(io, "  iterations: ", summary.iteration_start, ":", summary.iteration_end)
    println(io, "  window_length: ", summary.window_length)
    println(io, "  pooled_samples: ", summary.pooled_samples)
    println(io, "  effective_count: ", _summary_float(summary.effective_count; digits=2))
    println(io, "  mean_weight: ", _summary_float(summary.mean_weight; digits=3))
    println(
        io,
        "  clip_scale: ",
        _summary_float(summary.clip_scale_start; digits=2),
        " -> ",
        _summary_float(summary.clip_scale_end; digits=2),
    )
    println(io, "  updated: ", summary.updated)
    print(
        io,
        "  mass: mean=",
        _summary_float(summary.mass_mean),
        " min=",
        _summary_float(summary.mass_min),
        " max=",
        _summary_float(summary.mass_max),
    )
end

function Base.show(io::IO, summary::HMCMassAdaptationSummary)
    print(
        io,
        "HMCMassAdaptationSummary(window=",
        summary.window_index,
        ", chains=",
        summary.chains,
        ", effective_count=",
        round(summary.mean_effective_count; digits=2),
        ", updated=",
        summary.num_updated,
        "/",
        summary.chains,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCMassAdaptationSummary)
    println(io, "HMCMassAdaptationSummary")
    println(io, "  window: ", summary.window_index, " (", summary.iteration_start, ":", summary.iteration_end, ")")
    println(io, "  chains: ", summary.chains, " updated=", summary.num_updated, "/", summary.chains)
    println(io, "  window_length: ", summary.window_length)
    println(io, "  pooled_samples_mean: ", _summary_float(summary.mean_pooled_samples; digits=2))
    println(io, "  weight_sum_mean: ", _summary_float(summary.mean_weight_sum; digits=2))
    println(io, "  effective_count: mean=", _summary_float(summary.mean_effective_count; digits=2),
        " min=", _summary_float(summary.min_effective_count; digits=2),
        " max=", _summary_float(summary.max_effective_count; digits=2))
    println(io, "  mean_weight: ", _summary_float(summary.mean_weight; digits=3))
    println(io, "  clip_scale_end_mean: ", _summary_float(summary.mean_clip_scale_end; digits=2))
    print(
        io,
        "  mass: mean=",
        _summary_float(summary.mean_mass),
        " min=",
        _summary_float(summary.min_mass),
        " max=",
        _summary_float(summary.max_mass),
    )
end

function Base.show(io::IO, diagnostics::HMCDiagnosticsSummary)
    print(
        io,
        "HMCDiagnosticsSummary(acceptance_rate=",
        round(diagnostics.acceptance_rate; digits=3),
        ", divergence_rate=",
        round(diagnostics.divergence_rate; digits=3),
        ", mass_windows=",
        length(diagnostics.mass_adaptation_windows),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", diagnostics::HMCDiagnosticsSummary)
    println(io, "HMCDiagnosticsSummary")
    println(io, "  acceptance_rate: ", _summary_float(diagnostics.acceptance_rate; digits=3))
    println(io, "  divergence_rate: ", _summary_float(diagnostics.divergence_rate; digits=3))
    println(
        io,
        "  step_size: mean=",
        _summary_float(diagnostics.mean_step_size),
        " min=",
        _summary_float(isempty(diagnostics.step_sizes) ? 0.0 : minimum(diagnostics.step_sizes)),
        " max=",
        _summary_float(isempty(diagnostics.step_sizes) ? 0.0 : maximum(diagnostics.step_sizes)),
    )
    if isempty(diagnostics.mass_adaptation_windows)
        print(io, "  mass_adaptation_windows: none")
        return nothing
    end
    println(io, "  mass_adaptation_windows:")
    for summary in diagnostics.mass_adaptation_windows
        _show_mass_adaptation_summary_line(io, summary; indent="    ")
        println(io)
    end
    return nothing
end

function Base.show(io::IO, chain::HMCChain)
    print(io, "HMCChain(", lowercase(String(chain.sampler)), ", ", chain.model.name)
    print(io, ", samples=", length(chain))
    print(io, ", acceptance_rate=", round(acceptancerate(chain); digits=3))
    print(io, ", divergences=", count(identity, chain.divergent))
    print(io, ", step_size=", round(chain.step_size; digits=4))
    if chain.sampler === :nuts
        print(io, ", max_tree_depth=", chain.max_tree_depth)
    else
        print(io, ", num_leapfrog_steps=", chain.num_leapfrog_steps)
    end
    print(io, ", mass_windows=", length(chain.mass_adaptation_windows), ")")
end

function Base.show(io::IO, chains::HMCChains)
    print(
        io,
        "HMCChains(",
        lowercase(_sampler_label(chains)),
        ", ",
        chains.model.name,
        ", chains=",
        length(chains),
        ", samples=",
        numsamples(chains),
        ", acceptance_rate=",
        round(acceptancerate(chains); digits=3),
        ", divergences=",
        sum(count(identity, chain.divergent) for chain in chains.chains),
        ")",
    )
end

function Base.show(io::IO, summary::HMCSummary)
    print(
        io,
        "HMCSummary(",
        summary.model.name,
        ", space=",
        summary.space,
        ", parameters=",
        length(summary),
        ", acceptance_rate=",
        round(summary.diagnostics.acceptance_rate; digits=3),
        ", divergence_rate=",
        round(summary.diagnostics.divergence_rate; digits=3),
        ", mass_windows=",
        length(summary.diagnostics.mass_adaptation_windows),
        ", quantiles=",
        summary.quantile_probs,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", summary::HMCSummary)
    println(io, "HMCSummary(", summary.model.name, ")")
    println(io, "  space: ", summary.space)
    println(io, "  quantiles: ", summary.quantile_probs)
    println(io, "  parameters: ", length(summary))
    max_parameters = min(length(summary.parameters), 5)
    for parameter_index = 1:max_parameters
        parameter = summary.parameters[parameter_index]
        mid_quantile = parameter.quantiles[cld(length(parameter.quantiles), 2)]
        println(
            io,
            "    ",
            parameter.binding,
            " @ ",
            parameter.address,
            ": mean=",
            _summary_float(parameter.mean),
            " sd=",
            _summary_float(parameter.sd),
            " median=",
            _summary_float(mid_quantile),
            " rhat=",
            _summary_float(parameter.rhat; digits=3),
            " ess=",
            _summary_float(parameter.ess; digits=1),
        )
    end
    if length(summary.parameters) > max_parameters
        println(io, "    ... ", length(summary.parameters) - max_parameters, " more parameters")
    end
    println(io, "  diagnostics:")
    print(io, "    acceptance_rate: ", _summary_float(summary.diagnostics.acceptance_rate; digits=3))
    println(io)
    print(io, "    divergence_rate: ", _summary_float(summary.diagnostics.divergence_rate; digits=3))
    println(io)
    print(
        io,
        "    step_size: mean=",
        _summary_float(summary.diagnostics.mean_step_size),
        " min=",
        _summary_float(isempty(summary.diagnostics.step_sizes) ? 0.0 : minimum(summary.diagnostics.step_sizes)),
        " max=",
        _summary_float(isempty(summary.diagnostics.step_sizes) ? 0.0 : maximum(summary.diagnostics.step_sizes)),
    )
    println(io)
    if isempty(summary.diagnostics.mass_adaptation_windows)
        println(io, "    mass_adaptation_windows: none")
    else
        println(io, "    mass_adaptation_windows:")
        for window_summary in summary.diagnostics.mass_adaptation_windows
            _show_mass_adaptation_summary_line(io, window_summary; indent="      ")
            println(io)
        end
    end
    _show_sampler_warnings(io, summary.warnings; indent="  ")
    return nothing
end

function acceptancerate(chain::HMCChain)
    isempty(chain.acceptance_stats) && return 0.0
    return _sample_mean(chain.acceptance_stats)
end

function acceptancerate(chains::HMCChains)
    total_samples = sum(length(chain.acceptance_stats) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(sum(chain.acceptance_stats) for chain in chains.chains) / total_samples
end

function divergencerate(chain::HMCChain)
    isempty(chain.divergent) && return 0.0
    return count(identity, chain.divergent) / length(chain.divergent)
end

function divergencerate(chains::HMCChains)
    total_samples = sum(length(chain.divergent) for chain in chains.chains)
    total_samples == 0 && return 0.0
    return sum(count(identity, chain.divergent) for chain in chains.chains) / total_samples
end

function acceptancerate(summary::HMCSummary)
    return summary.diagnostics.acceptance_rate
end

function divergencerate(summary::HMCSummary)
    return summary.diagnostics.divergence_rate
end

function massadaptationwindows(chain::HMCChain)
    return chain.mass_adaptation_windows
end

function massadaptationwindows(chains::HMCChains)
    return [chain.mass_adaptation_windows for chain in chains.chains]
end

function massadaptationwindows(summary::HMCSummary)
    return summary.diagnostics.mass_adaptation_windows
end

function treedepths(chain::HMCChain)
    return chain.tree_depths
end

function treedepths(chains::HMCChains)
    return [chain.tree_depths for chain in chains.chains]
end

function integrationsteps(chain::HMCChain)
    return chain.integration_steps
end

function integrationsteps(chains::HMCChains)
    return [chain.integration_steps for chain in chains.chains]
end

function _diagnostic_space_samples(chain::HMCChain, space::Symbol)
    if space === :constrained
        return chain.constrained_samples
    elseif space === :unconstrained
        return chain.unconstrained_samples
    end

    throw(ArgumentError("diagnostic space must be :constrained or :unconstrained"))
end

function nchains(chains::HMCChains)
    return length(chains)
end

function numsamples(chains::HMCChains)
    isempty(chains.chains) && return 0
    return length(first(chains.chains))
end

function _summary_address(address::AddressSpec)
    parts = Any[part isa AddressLiteralPart ? part.value : part.value for part in address.parts]
    return length(parts) == 1 ? first(parts) : Tuple(parts)
end

function _summary_parameter_address(address::AddressSpec, component::Union{Nothing,Int})
    base = _summary_address(address)
    isnothing(component) && return base
    if base isa Tuple
        return (base..., component)
    end
    return (base, component)
end

function _summary_parameter_entries(layout::ParameterLayout, space::Symbol)
    entries = Tuple{Int,Symbol,Any}[]
    for slot in layout.slots
        indices = if space === :constrained
            parametervalueindices(slot)
        elseif space === :unconstrained
            parameterindices(slot)
        else
            throw(ArgumentError("diagnostic space must be :constrained or :unconstrained"))
        end
        component_count = length(indices)
        for (offset, parameter_index) in enumerate(indices)
            component = component_count == 1 ? nothing : offset
            push!(entries, (parameter_index, slot.binding, _summary_parameter_address(slot.address, component)))
        end
    end
    return entries
end

function _validate_hmc_diagnostics(chains::HMCChains, space::Symbol)
    # A single chain is allowed (issue #337): the split-chain machinery halves
    # each chain, so one chain still yields two split sequences for Rhat/ESS.
    length(chains) >= 1 || throw(ArgumentError("chain diagnostics require at least 1 chain"))
    num_samples = numsamples(chains)
    num_samples >= 4 || throw(ArgumentError("chain diagnostics require at least 4 samples per chain"))

    first_samples = _diagnostic_space_samples(first(chains.chains), space)
    num_params = size(first_samples, 1)
    for chain in chains.chains
        chain_samples = _diagnostic_space_samples(chain, space)
        size(chain_samples, 1) == num_params ||
            throw(DimensionMismatch("all chains must have the same parameter dimension"))
        size(chain_samples, 2) == num_samples ||
            throw(DimensionMismatch("all chains must have the same number of samples"))
    end

    return num_params, num_samples
end

function _validate_summary_quantiles(quantile_probs)
    isempty(quantile_probs) && throw(ArgumentError("summary quantiles must be non-empty"))
    probabilities = Float64[Float64(prob) for prob in quantile_probs]
    for prob in probabilities
        0.0 <= prob <= 1.0 || throw(ArgumentError("summary quantiles must lie in [0, 1]"))
    end
    return probabilities
end

function _mass_adaptation_diagnostics(chains::HMCChains)
    groups = Dict{NTuple{4,Int},Vector{HMCMassAdaptationWindowSummary}}()
    for chain in chains.chains
        for window in chain.mass_adaptation_windows
            key = (window.window_index, window.iteration_start, window.iteration_end, window.window_length)
            push!(get!(groups, key, HMCMassAdaptationWindowSummary[]), window)
        end
    end

    summaries = HMCMassAdaptationSummary[]
    for key in sort!(collect(keys(groups)); by=identity)
        windows = groups[key]
        push!(
            summaries,
            HMCMassAdaptationSummary(
                key[1],
                key[2],
                key[3],
                key[4],
                length(windows),
                count(window -> window.updated, windows),
                _sample_mean([window.pooled_samples for window in windows]),
                _sample_mean([window.weight_sum for window in windows]),
                _sample_mean([window.effective_count for window in windows]),
                minimum(window.effective_count for window in windows),
                maximum(window.effective_count for window in windows),
                _sample_mean([window.mean_weight for window in windows]),
                _sample_mean([window.clip_scale_end for window in windows]),
                _sample_mean([window.mass_mean for window in windows]),
                minimum(window.mass_min for window in windows),
                maximum(window.mass_max for window in windows),
            ),
        )
    end
    return summaries
end

function _diagnostics_summary(chains::HMCChains)
    step_sizes = Float64[chain.step_size for chain in chains.chains]
    return HMCDiagnosticsSummary(
        acceptancerate(chains),
        divergencerate(chains),
        step_sizes,
        isempty(step_sizes) ? 0.0 : _sample_mean(step_sizes),
        _mass_adaptation_diagnostics(chains),
    )
end

function _sample_mean(values::AbstractVector)
    return sum(values) / length(values)
end

function _sample_variance(values::AbstractVector, mean_value::Real=_sample_mean(values))
    length(values) > 1 || return 0.0
    return sum((value - mean_value)^2 for value in values) / (length(values) - 1)
end

function _sample_sd(values::AbstractVector, mean_value::Real=_sample_mean(values))
    return sqrt(max(_sample_variance(values, mean_value), 0.0))
end

function _pooled_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    pooled = Vector{Float64}(undef, length(chains) * num_samples)
    offset = 1
    for chain in chains.chains
        samples = _diagnostic_space_samples(chain, space)
        pooled[offset:(offset+num_samples-1)] = samples[parameter_index, :]
        offset += num_samples
    end
    return pooled
end

function _quantile(sorted_values::AbstractVector, probability::Float64)
    num_values = length(sorted_values)
    num_values == 0 && throw(ArgumentError("quantile requires at least one value"))
    num_values == 1 && return Float64(sorted_values[1])

    position = 1 + (num_values - 1) * probability
    lower = floor(Int, position)
    upper = ceil(Int, position)
    lower == upper && return Float64(sorted_values[lower])
    weight = position - lower
    return (1 - weight) * sorted_values[lower] + weight * sorted_values[upper]
end

function _quantiles(values::AbstractVector, probabilities::AbstractVector{Float64})
    sorted_values = sort(collect(values))
    return Float64[_quantile(sorted_values, probability) for probability in probabilities]
end

function _split_chain_parameter_draws(chains::HMCChains, parameter_index::Int, space::Symbol)
    _, num_samples = _validate_hmc_diagnostics(chains, space)
    split_samples = fld(num_samples, 2)
    even_samples = 2 * split_samples
    split_draws = Matrix{Float64}(undef, 2 * length(chains), split_samples)

    for (chain_index, chain) in enumerate(chains.chains)
        samples = _diagnostic_space_samples(chain, space)
        split_draws[2*chain_index-1, :] = samples[parameter_index, 1:split_samples]
        split_draws[2*chain_index, :] = samples[parameter_index, (split_samples+1):even_samples]
    end

    return split_draws
end

function _chain_draw_statistics(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means = Vector{Float64}(undef, num_chains)
    chain_variances = Vector{Float64}(undef, num_chains)
    for chain_index = 1:num_chains
        chain_draws = view(draws, chain_index, :)
        chain_means[chain_index] = _sample_mean(chain_draws)
        chain_variances[chain_index] = _sample_variance(chain_draws, chain_means[chain_index])
    end

    within_variance = _sample_mean(chain_variances)
    between_variance = num_samples > 1 ? num_samples * _sample_variance(chain_means) : 0.0
    var_plus = ((num_samples - 1) / num_samples) * within_variance + between_variance / num_samples
    return chain_means, chain_variances, within_variance, between_variance, var_plus
end

function _split_rhat(draws::AbstractMatrix)
    _, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    if within_variance == 0
        # All draws identical (stuck sampler): Rhat is undefined; NaN follows
        # the Stan/posterior convention so diagnostics flag it as a warning.
        return var_plus == 0 ? NaN : Inf
    end

    return sqrt(max(var_plus / within_variance, 1.0))
end

function _autocovariance(draws::AbstractVector, lag::Int, mean_value::Real)
    num_samples = length(draws)
    total = 0.0
    for index = 1:(num_samples-lag)
        total += (draws[index] - mean_value) * (draws[index+lag] - mean_value)
    end
    return total / num_samples
end

function _split_ess(draws::AbstractMatrix)
    num_chains, num_samples = size(draws)
    chain_means, _, within_variance, _, var_plus = _chain_draw_statistics(draws)
    total_draws = num_chains * num_samples

    if within_variance == 0 && var_plus == 0
        # All draws identical (stuck sampler): ESS is undefined; NaN follows
        # the Stan/posterior convention so diagnostics flag it as a warning.
        return NaN
    elseif var_plus <= 0
        return 0.0
    end

    pair_sums = Float64[]
    autocovariance_means = Vector{Float64}(undef, num_chains)
    for pair_start = 0:2:(num_samples-1)
        pair_sum = 0.0
        for lag = pair_start:min(pair_start+1, num_samples-1)
            for chain_index = 1:num_chains
                autocovariance_means[chain_index] = _autocovariance(view(draws, chain_index, :), lag, chain_means[chain_index])
            end
            mean_autocovariance = _sample_mean(autocovariance_means)
            rho_hat = lag == 0 ? 1.0 : 1 - (within_variance - mean_autocovariance) / var_plus
            pair_sum += min(rho_hat, 1.0)
        end

        pair_sum > 0 || break
        push!(pair_sums, pair_sum)
    end

    for index = 2:length(pair_sums)
        pair_sums[index] = min(pair_sums[index], pair_sums[index-1])
    end

    tau_hat = -1 + 2 * sum(pair_sums)
    tau_hat = max(tau_hat, 1.0)
    return min(Float64(total_draws), Float64(total_draws) / tau_hat)
end

# Rank-normal (inverse-normal) transform of the pooled draws (Vehtari et al.
# 2021): fractional ranks (r - 3/8)/(S + 1/4) mapped through the standard-normal
# quantile. Makes the split-Rhat robust to heavy tails and nonlinear scale.
function _rank_normal_transform(draws::AbstractMatrix)
    flat = vec(collect(draws))
    S = length(flat)
    order = sortperm(flat)
    ranks = Vector{Float64}(undef, S)
    for (position, index) in enumerate(order)
        ranks[index] = position
    end
    z = [sqrt(2.0) * erfinv(2 * ((r - 0.375) / (S + 0.25)) - 1) for r in ranks]
    return reshape(z, size(draws))
end

# Rank-normalized split-Rhat (Vehtari et al. 2021): the max of the bulk
# statistic (rank-normalized draws) and the tail-sensitive folded statistic
# (rank-normalized |x - median|).
function _rank_normalized_rhat(draws::AbstractMatrix)
    bulk = _split_rhat(_rank_normal_transform(draws))
    med = _quantile(sort(vec(collect(draws))), 0.5)
    folded = _split_rhat(_rank_normal_transform(abs.(draws .- med)))
    return max(bulk, folded)
end

"""
    rhat(chains::HMCChains; space=:constrained, method=:split) -> Vector{Float64}

Per-parameter split-R̂. `method=:split` (default) is the classical split-chain
statistic; `method=:rank` is the rank-normalized split-R̂ of Vehtari et al.
(2021) — the max of the bulk (rank-normalized) and folded (tail-sensitive)
statistics, robust to heavy tails and nonlinear scale, now the standard in
Stan/ArviZ.
"""
function rhat(chains::HMCChains; space::Symbol=:constrained, method::Symbol=:split)
    method in (:split, :rank) ||
        throw(ArgumentError("rhat method must be :split or :rank, got $(repr(method))"))
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index = 1:num_params
        draws = _split_chain_parameter_draws(chains, parameter_index, space)
        values[parameter_index] = method === :split ? _split_rhat(draws) : _rank_normalized_rhat(draws)
    end
    return values
end

"""
    ess(chains::HMCChains; space=:constrained) -> Vector{Float64}

Per-parameter (bulk) effective sample size from the split chains, using the
paired-autocorrelation estimator of Vehtari et al. (2021). `space` selects
`:constrained` (default) or `:unconstrained` draws. See also `rhat` and
`summarize` (which additionally reports tail ESS).
"""
function ess(chains::HMCChains; space::Symbol=:constrained)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    values = Vector{Float64}(undef, num_params)
    for parameter_index = 1:num_params
        values[parameter_index] = _split_ess(_split_chain_parameter_draws(chains, parameter_index, space))
    end
    return values
end

"""
    summarize(chains::HMCChains; space=:constrained, quantiles=(0.05, 0.5, 0.95), per_chain=false) -> HMCSummary

Summarize sampled `chains`: per-parameter posterior mean, standard deviation,
the requested `quantiles`, R-hat, bulk/tail effective sample size, and Monte
Carlo standard errors, together with run-level diagnostics and any sampler
warnings. `space` selects `:constrained` (default) or `:unconstrained` draws;
`per_chain=true` also reports per-chain means and standard deviations. See also
`rhat`, `ess`, and `check_diagnostics`.
"""
function summarize(chains::HMCChains; space::Symbol=:constrained, quantiles=(0.05, 0.5, 0.95), per_chain::Bool=false)
    num_params, _ = _validate_hmc_diagnostics(chains, space)
    quantile_probs = _validate_summary_quantiles(quantiles)
    rhats = rhat(chains; space=space)
    ess_values = ess(chains; space=space)
    diagnostics = _diagnostics_summary(chains)
    layout = _conditioned_parameter_layout(chains.model, chains.constraints)
    summary_entries = _summary_parameter_entries(layout, space)
    length(summary_entries) == num_params ||
        throw(DimensionMismatch("summary expected $num_params parameters in layout, got $(length(summary_entries))"))

    parameters = Vector{HMCParameterSummary}(undef, num_params)
    for (parameter_index, binding, address) in summary_entries
        draws = _pooled_parameter_draws(chains, parameter_index, space)
        sorted_draws = sort(draws)
        mean_value = _sample_mean(draws)
        sd_value = _sample_sd(draws, mean_value)
        ess_bulk = ess_values[parameter_index]
        quantile_values = Float64[_quantile(sorted_draws, probability) for probability in quantile_probs]
        q05 = _quantile(sorted_draws, 0.05)
        q95 = _quantile(sorted_draws, 0.95)
        per_chain_means = nothing
        per_chain_sds = nothing
        if per_chain
            per_chain_means, per_chain_sds = _per_chain_parameter_statistics(chains, parameter_index, space)
        end
        parameters[parameter_index] = HMCParameterSummary(
            parameter_index,
            binding,
            address,
            mean_value,
            sd_value,
            quantile_values,
            rhats[parameter_index],
            ess_bulk,
            _mcse_mean_value(sd_value, ess_bulk),
            _summary_tail_ess(chains, parameter_index, space, q05, q95),
            _summary_mcse_quantiles(chains, parameter_index, space, sorted_draws, quantile_probs, quantile_values),
            per_chain_means,
            per_chain_sds,
        )
    end

    warnings = check_diagnostics(chains; space=space)
    return HMCSummary(chains.model, space, quantile_probs, diagnostics, parameters, warnings)
end

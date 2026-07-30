# Simulation-based calibration (Talts et al. 2018): draw parameters from the
# prior, simulate data given them, run posterior inference, and rank the true
# parameter within the posterior draws. Under correct inference every rank
# statistic is uniform on 0:num_posterior_draws, so non-uniform ranks expose
# sampler bias that pointwise logpdf/gradient tests structurally cannot (e.g.
# a miscalibrated mass-matrix adaptation).

struct SBCResult
    parameter_names::Vector{String}
    # num_parameter_values x num_simulations; each entry in 0:num_posterior_draws
    ranks::Matrix{Int}
    num_posterior_draws::Int
    pvalues::Vector{Float64}
    warnings::Vector{String}
end

has_warnings(result::SBCResult) = !isempty(result.warnings)

function Base.show(io::IO, ::MIME"text/plain", result::SBCResult)
    println(
        io,
        "SBCResult: ",
        size(result.ranks, 2),
        " simulations x ",
        result.num_posterior_draws,
        " posterior draws",
    )
    for (name, pvalue) in zip(result.parameter_names, result.pvalues)
        if isnan(pvalue)
            println(io, "  ", name, ": rank-uniformity p = n/a (constant prior)")
        else
            println(io, "  ", name, ": rank-uniformity p = ", round(pvalue; sigdigits=3))
        end
    end
    if has_warnings(result)
        println(io, "  warnings:")
        for warning in result.warnings
            println(io, "    - ", warning)
        end
    else
        print(io, "  no warnings")
    end
    return nothing
end

_chisq_ccdf(dof::Int, x::Real) = gamma_inc(dof / 2, x / 2)[2]

_sbc_bin(rank::Int, num_bins::Int, num_posterior_draws::Int) =
    min(num_bins, fld(rank * num_bins, num_posterior_draws + 1) + 1)

# Chi-squared uniformity p-value for one parameter's rank statistics. The L+1
# possible ranks are binned equal-width, but when num_bins does not divide
# L+1 the bins hold different numbers of possible ranks, so each bin's
# expected count is proportional to the ranks it can receive (a uniform
# expected count would false-alarm on perfectly calibrated ranks).
function _sbc_uniformity_pvalue(ranks::AbstractVector{Int}, num_posterior_draws::Int, num_bins::Int)
    num_bins = min(num_bins, num_posterior_draws + 1)
    possible = zeros(Int, num_bins)
    for rank = 0:num_posterior_draws
        possible[_sbc_bin(rank, num_bins, num_posterior_draws)] += 1
    end
    counts = zeros(Int, num_bins)
    for rank in ranks
        counts[_sbc_bin(rank, num_bins, num_posterior_draws)] += 1
    end
    total = length(ranks)
    statistic = 0.0
    for bin = 1:num_bins
        expected = total * possible[bin] / (num_posterior_draws + 1)
        statistic += (counts[bin] - expected)^2 / expected
    end
    return _chisq_ccdf(num_bins - 1, statistic)
end

"""
    sbc(model, args=(); num_simulations, num_posterior_draws, num_warmup,
        thin=1, num_bins=..., warn_threshold=1e-3, rng=Random.default_rng(),
        observation_addresses=nothing, nuts_kwargs...) -> SBCResult

Simulation-based calibration of NUTS on `model`: for each simulation, draw
parameters and data jointly from the prior, condition on the simulated data,
sample `num_posterior_draws * thin` posterior draws (keeping every `thin`-th
to reduce autocorrelation), and record the rank of the true parameter within
the kept draws. Correct inference makes each rank uniform on
`0:num_posterior_draws`; per-parameter chi-squared uniformity p-values below
`warn_threshold` produce warnings (see `has_warnings`). A coordinate with a
point-mass prior (e.g. the unit diagonal of an `lkjcholesky` Cholesky factor)
is structurally constant and cannot be calibrated, so it is reported as
`p = NaN` and never warns.

`observation_addresses` defaults to every choice address in a prior trace
that is not a latent parameter slot. `sampler=:gibbs` runs [`gibbs`](@ref)
instead of [`nuts`](@ref) and requires explicit `observation_addresses` —
under the default every discrete choice would be conditioned as data,
leaving no Gibbs sites free. `sampler=:chees` runs [`batched_chees`](@ref)
and requires the `num_chains` keyword (ChEES tunes its trajectory length from
the cross-chain ensemble); chain 1's draws are ranked. `sampler=:meads` runs
[`batched_meads`](@ref) and likewise requires `num_chains` (>= `2*num_folds`);
chain 1's draws are ranked. Remaining keyword
arguments are forwarded to the chosen sampler. Runtime scales with
`num_simulations * (num_warmup + num_posterior_draws * thin)`; keep the fast
suite variant small and use `bench/sbc_validation.jl` for release-grade runs.
"""
function sbc(
    model::TeaModel,
    args::Tuple=();
    num_simulations::Int,
    num_posterior_draws::Int,
    num_warmup::Int,
    thin::Int=1,
    num_bins::Int=max(2, min(20, cld(num_simulations, 5))),
    warn_threshold::Real=1e-3,
    rng::AbstractRNG=Random.default_rng(),
    observation_addresses::Union{Nothing,AbstractVector}=nothing,
    sampler::Symbol=:nuts,
    nuts_kwargs...,
)
    sampler in (:nuts, :gibbs, :chees, :meads) ||
        throw(ArgumentError("sbc sampler must be :nuts, :gibbs, :chees, or :meads, got :$sampler"))
    # the default observation set is EVERY non-slot choice, which would
    # condition all discrete latents as data and leave no Gibbs sites; SBC
    # cannot guess which discrete choices are data, so the caller must say
    sampler === :gibbs && isnothing(observation_addresses) &&
        throw(
            ArgumentError(
                "sbc with sampler=:gibbs requires explicit observation_addresses (the default " *
                "observes every non-slot choice, leaving no discrete Gibbs sites free)",
            ),
        )
    num_simulations > 0 || throw(ArgumentError("sbc requires num_simulations > 0"))
    num_posterior_draws > 0 || throw(ArgumentError("sbc requires num_posterior_draws > 0"))
    thin > 0 || throw(ArgumentError("sbc requires thin > 0"))
    num_bins > 1 || throw(ArgumentError("sbc requires num_bins > 1"))

    # The posterior is over the SIGNATURE latents of the conditioning `data`
    # (#95 PR-6): the ranked truth vector and the sampler's constrained draws
    # must use that same layout, not the syntactic default. `num_values` is
    # fixed across simulations (the observed address set is constant), so it is
    # resolved from the first simulation's `data` and the ranks matrix is sized
    # then.
    parametervaluecount(parameterlayout(model)) > 0 ||
        throw(ArgumentError("sbc requires a model with at least one latent parameter"))
    data_addresses = isnothing(observation_addresses) ? nothing : collect(Any, observation_addresses)

    ranks = Matrix{Int}(undef, 0, 0)
    signature_layout = nothing
    num_values = 0
    # Track each signature coordinate's prior spread across simulations: a
    # structurally constant coordinate (a point-mass prior, e.g. the unit
    # diagonal `L[1,1] == 1` of an lkjcholesky Cholesky factor, or a derived
    # packed entry that never varies) carries no calibration information -- its
    # rank is degenerately pinned, so a uniformity test would always false-alarm.
    # Such coordinates are excluded from the p-values/warnings below (issue #226).
    truth_lo = Float64[]
    truth_hi = Float64[]
    for simulation = 1:num_simulations
        prior_trace, _ = generate(model, args, choicemap(); rng=rng)
        if isnothing(data_addresses)
            # observations = every trace address that is not a default latent slot
            latent_map = parameterchoicemap(model, parameter_vector(prior_trace))
            data_addresses = Any[
                first(entry) for entry in prior_trace.choices.entries if !haskey(latent_map, first(entry))
            ]
            isempty(data_addresses) &&
                throw(ArgumentError("sbc requires at least one observation address to condition on"))
        end
        data = choicemap((address, prior_trace[address]) for address in data_addresses)
        if isnothing(signature_layout)
            signature_layout = _conditioned_parameter_layout(model, data)
            num_values = parametervaluecount(signature_layout)
            num_values > 0 ||
                throw(ArgumentError("sbc requires at least one free latent after conditioning on the observations"))
            ranks = Matrix{Int}(undef, num_values, num_simulations)
            truth_lo = fill(Inf, num_values)
            truth_hi = fill(-Inf, num_values)
        end
        # truth = the conditioned latents' values read from the prior trace, in
        # the signature layout that the sampler's constrained_samples follow.
        truth = Vector{Float64}(undef, num_values)
        for slot in signature_layout.slots
            _write_slot_value!(truth, slot, prior_trace[_static_address(slot.address)])
        end
        for value_index = 1:num_values
            truth_lo[value_index] = min(truth_lo[value_index], truth[value_index])
            truth_hi[value_index] = max(truth_hi[value_index], truth[value_index])
        end
        chain = if sampler === :gibbs
            gibbs(
                model,
                args,
                data;
                num_samples=num_posterior_draws * thin,
                num_warmup=num_warmup,
                rng=rng,
                nuts_kwargs...,
            )
        elseif sampler === :chees
            # ChEES needs many chains for its cross-chain trajectory-length
            # adaptation; all chains target the SAME conditioned posterior, so
            # chain 1's draws are a valid posterior sample for the rank statistic
            # (the ensemble only aids adaptation). `num_chains` is a required kwarg
            # of `batched_chees` and must be passed through `sbc`.
            batched = batched_chees(
                model,
                args,
                data;
                num_samples=num_posterior_draws * thin,
                num_warmup=num_warmup,
                rng=rng,
                nuts_kwargs...,
            )
            first(batched.chains)
        elseif sampler === :meads
            # MEADS tunes step size/damping/mass from the cross-fold ensemble; like
            # ChEES, all chains target the SAME conditioned posterior, so chain 1's
            # draws are a valid posterior sample for the rank statistic. `num_chains`
            # (>= 2*num_folds) is a required kwarg of `batched_meads`.
            batched = batched_meads(
                model,
                args,
                data;
                num_samples=num_posterior_draws * thin,
                num_warmup=num_warmup,
                rng=rng,
                nuts_kwargs...,
            )
            first(batched.chains)
        else
            nuts(
                model,
                args,
                data;
                num_samples=num_posterior_draws * thin,
                num_warmup=num_warmup,
                rng=rng,
                nuts_kwargs...,
            )
        end
        draws = view(chain.constrained_samples, :, thin:thin:(num_posterior_draws*thin))
        for value_index = 1:num_values
            ranks[value_index, simulation] = count(<(truth[value_index]), view(draws, value_index, :))
        end
    end

    names = _export_parameter_names(signature_layout, :constrained)
    # A coordinate whose prior never moved it across simulations is a structural
    # constant (point-mass prior); report NaN and skip its warning rather than
    # false-alarm on the degenerate rank statistic (issue #226).
    is_constant = truth_hi .== truth_lo
    pvalues = [
        is_constant[value_index] ? NaN :
        _sbc_uniformity_pvalue(view(ranks, value_index, :), num_posterior_draws, num_bins) for
        value_index = 1:num_values
    ]
    warnings = String[]
    for (name, pvalue) in zip(names, pvalues)
        (isnan(pvalue) || pvalue >= warn_threshold) || push!(
            warnings,
            "rank distribution for $name deviates from uniform (p = $(round(pvalue; sigdigits=3)))",
        )
    end
    return SBCResult(names, ranks, num_posterior_draws, pvalues, warnings)
end

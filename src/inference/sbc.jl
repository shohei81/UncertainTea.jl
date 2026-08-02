# Simulation-based calibration (Talts et al. 2018): draw parameters from the
# prior, simulate data given them, run posterior inference, and rank the true
# parameter within the posterior draws. Under correct inference every rank
# statistic is uniform on 0:num_samples, so non-uniform ranks expose
# sampler bias that pointwise logpdf/gradient tests structurally cannot (e.g.
# a miscalibrated mass-matrix adaptation).

struct SBCResult
    parameter_names::Vector{String}
    # num_parameter_values x num_simulations; each entry in 0:num_samples
    ranks::Matrix{Int}
    num_samples::Int
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
        result.num_samples,
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

_sbc_bin(rank::Int, num_bins::Int, num_samples::Int) =
    min(num_bins, fld(rank * num_bins, num_samples + 1) + 1)

# Chi-squared uniformity p-value for one parameter's rank statistics. The L+1
# possible ranks are binned equal-width, but when num_bins does not divide
# L+1 the bins hold different numbers of possible ranks, so each bin's
# expected count is proportional to the ranks it can receive (a uniform
# expected count would false-alarm on perfectly calibrated ranks).
function _sbc_uniformity_pvalue(ranks::AbstractVector{Int}, num_samples::Int, num_bins::Int)
    num_bins = min(num_bins, num_samples + 1)
    possible = zeros(Int, num_bins)
    for rank = 0:num_samples
        possible[_sbc_bin(rank, num_bins, num_samples)] += 1
    end
    counts = zeros(Int, num_bins)
    for rank in ranks
        counts[_sbc_bin(rank, num_bins, num_samples)] += 1
    end
    total = length(ranks)
    statistic = 0.0
    for bin = 1:num_bins
        expected = total * possible[bin] / (num_samples + 1)
        statistic += (counts[bin] - expected)^2 / expected
    end
    return _chisq_ccdf(num_bins - 1, statistic)
end

"""
    sbc(model, args=(); num_simulations, num_samples, num_warmup,
        thin=1, num_bins=..., warn_threshold=1e-3, rng=Random.default_rng(),
        observation_addresses=nothing, nuts_kwargs...) -> SBCResult

Simulation-based calibration of NUTS on `model`: for each simulation, draw
parameters and data jointly from the prior, condition on the simulated data,
sample `num_samples * thin` posterior draws (keeping every `thin`-th
to reduce autocorrelation), and record the rank of the true parameter within
the kept draws. Correct inference makes each rank uniform on
`0:num_samples`; per-parameter chi-squared uniformity p-values below
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
chain 1's draws are ranked. `sampler=:batched_nuts` runs [`batched_nuts`](@ref)
(requires `num_chains`; forward `tree_strategy=:masked`/`:persistent`,
`backend`, and `precision` to calibrate the device tree kernels); chain 1's
draws are ranked. Remaining keyword
arguments are forwarded to the chosen sampler. Runtime scales with
`num_simulations * (num_warmup + num_samples * thin)`; keep the fast
suite variant small and use `bench/sbc_validation.jl` for release-grade runs.

`execution=:batched` (default `:serial`) runs all `num_simulations`
replications as the chains of a SINGLE [`batched_nuts`](@ref) call
(`num_chains = num_simulations`, each chain conditioned on its own
replication's data), instead of one serial sampler run per replication. It
supports `sampler=:nuts`/`:batched_nuts` only, forwards `backend`/`precision`/
`tree_strategy` (so the whole study can be one device run), forces per-chain
step-size adaptation (the posteriors are heterogeneous), and must not be given
`num_chains`. The rank statistic is unchanged.
"""
function sbc(
    model::TeaModel,
    args::Tuple=();
    num_simulations::Int,
    num_samples::Int,
    num_warmup::Int,
    thin::Int=1,
    num_bins::Int=max(2, min(20, cld(num_simulations, 5))),
    warn_threshold::Real=1e-3,
    rng::AbstractRNG=Random.default_rng(),
    observation_addresses::Union{Nothing,AbstractVector}=nothing,
    sampler::Symbol=:nuts,
    execution::Symbol=:serial,
    nuts_kwargs...,
)
    sampler in (:nuts, :gibbs, :chees, :meads, :batched_nuts) || throw(
        ArgumentError(
            "sbc sampler must be :nuts, :gibbs, :chees, :meads, or :batched_nuts, got :$sampler",
        ),
    )
    execution in (:serial, :batched) ||
        throw(ArgumentError("sbc execution must be :serial or :batched, got :$execution"))
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
    num_samples > 0 || throw(ArgumentError("sbc requires num_samples > 0"))
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

    if execution === :batched
        return _sbc_batched(
            model,
            args,
            data_addresses;
            num_simulations=num_simulations,
            num_samples=num_samples,
            num_warmup=num_warmup,
            thin=thin,
            num_bins=num_bins,
            warn_threshold=warn_threshold,
            rng=rng,
            sampler=sampler,
            nuts_kwargs...,
        )
    end

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
                num_samples=num_samples * thin,
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
                num_samples=num_samples * thin,
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
                num_samples=num_samples * thin,
                num_warmup=num_warmup,
                rng=rng,
                nuts_kwargs...,
            )
            first(batched.chains)
        elseif sampler === :batched_nuts
            # batched_nuts runs many independent chains against the SAME
            # conditioned posterior; chain 1's draws are a valid posterior sample
            # for the rank statistic. This calibrates the masked and persistent
            # device tree kernels (forwarded via `tree_strategy` in nuts_kwargs),
            # which are only statistically -- not bitwise -- equivalent to the host
            # path (on-device RNG, Float32 transcendental drift) and get no other
            # rank-calibration gate (issue #225). `num_chains` is a required kwarg
            # of `batched_nuts`; `backend`/`precision`/`tree_strategy` pass through.
            batched = batched_nuts(
                model,
                args,
                data;
                num_samples=num_samples * thin,
                num_warmup=num_warmup,
                rng=rng,
                nuts_kwargs...,
            )
            first(batched.chains)
        else
            # `nuts` returns a one-chain HMCChains (issue #337); the rank
            # statistic reads the single chain's draws.
            only(
                nuts(
                    model,
                    args,
                    data;
                    num_samples=num_samples * thin,
                    num_warmup=num_warmup,
                    rng=rng,
                    nuts_kwargs...,
                ).chains,
            )
        end
        draws = view(chain.constrained_samples, :, thin:thin:(num_samples*thin))
        for value_index = 1:num_values
            ranks[value_index, simulation] = count(<(truth[value_index]), view(draws, value_index, :))
        end
    end

    return _sbc_finalize(
        signature_layout, ranks, truth_lo, truth_hi, num_samples, num_bins, warn_threshold,
    )
end

# Turn the ranks matrix + per-coordinate prior spread into an SBCResult: p-values
# with structurally-constant (point-mass prior) coordinates reported as NaN and
# excluded from the warnings (issue #226). Shared by the serial and batched
# (issue #222) execution paths; pure in its inputs (no RNG), so lifting it out of
# the serial loop does not change any existing result.
function _sbc_finalize(
    signature_layout,
    ranks::AbstractMatrix{Int},
    truth_lo::AbstractVector,
    truth_hi::AbstractVector,
    num_samples::Int,
    num_bins::Int,
    warn_threshold::Real,
)
    names = _export_parameter_names(signature_layout, :constrained)
    is_constant = truth_hi .== truth_lo
    pvalues = [
        is_constant[value_index] ? NaN :
        _sbc_uniformity_pvalue(view(ranks, value_index, :), num_samples, num_bins) for
        value_index = 1:size(ranks, 1)
    ]
    warnings = String[]
    for (name, pvalue) in zip(names, pvalues)
        (isnan(pvalue) || pvalue >= warn_threshold) || push!(
            warnings,
            "rank distribution for $name deviates from uniform (p = $(round(pvalue; sigdigits=3)))",
        )
    end
    return SBCResult(names, ranks, num_samples, pvalues, warnings)
end

# issue #222: batched execution. SBC is embarrassingly parallel across
# replications (independent datasets -> independent posteriors), so instead of
# `num_simulations` serial single-chain runs, draw every replication's prior +
# data up front and run ONE `batched_nuts` with `num_chains = num_simulations`,
# each chain conditioned on its own replication's data via the per-chain
# constraint-vector path. Each chain targets a DIFFERENT posterior, so per-chain
# step-size adaptation is forced; a device `backend`'s shared diagonal mass is a
# caveat that rank uniformity itself tests (if it biases ranks, the leg fails
# loudly). Ranks/binning/p-values are identical to the serial path.
function _sbc_batched(
    model::TeaModel,
    args::Tuple,
    data_addresses;
    num_simulations::Int,
    num_samples::Int,
    num_warmup::Int,
    thin::Int,
    num_bins::Int,
    warn_threshold::Real,
    rng::AbstractRNG,
    sampler::Symbol,
    nuts_kwargs...,
)
    sampler in (:nuts, :batched_nuts) || throw(
        ArgumentError(
            "sbc execution=:batched runs one batched_nuts (each replication is a chain), so it " *
            "supports only sampler=:nuts or :batched_nuts, got :$sampler",
        ),
    )
    haskey(nuts_kwargs, :num_chains) && throw(
        ArgumentError(
            "sbc execution=:batched sets num_chains = num_simulations internally; do not pass num_chains",
        ),
    )

    # Draw every replication's prior trace + data up front, resolving the
    # observed-address set and signature layout from the first replication (the
    # address set is constant across replications).
    signature_layout = nothing
    num_values = 0
    data_vector = Vector{ChoiceMap}(undef, num_simulations)
    truth = Matrix{Float64}(undef, 0, 0)
    for simulation = 1:num_simulations
        prior_trace, _ = generate(model, args, choicemap(); rng=rng)
        if isnothing(data_addresses)
            latent_map = parameterchoicemap(model, parameter_vector(prior_trace))
            data_addresses = Any[
                first(entry) for entry in prior_trace.choices.entries if !haskey(latent_map, first(entry))
            ]
            isempty(data_addresses) &&
                throw(ArgumentError("sbc requires at least one observation address to condition on"))
        end
        data = choicemap((address, prior_trace[address]) for address in data_addresses)
        data_vector[simulation] = data
        if isnothing(signature_layout)
            signature_layout = _conditioned_parameter_layout(model, data)
            num_values = parametervaluecount(signature_layout)
            num_values > 0 ||
                throw(ArgumentError("sbc requires at least one free latent after conditioning on the observations"))
            truth = Matrix{Float64}(undef, num_values, num_simulations)
        end
        column = view(truth, :, simulation)
        for slot in signature_layout.slots
            _write_slot_value!(column, slot, prior_trace[_static_address(slot.address)])
        end
    end

    # One batched run: every replication is a chain conditioned on its own data.
    # Per-chain adaptation is correct because the posteriors are heterogeneous.
    chains = batched_nuts(
        model,
        args,
        data_vector;
        num_chains=num_simulations,
        num_samples=num_samples * thin,
        num_warmup=num_warmup,
        per_chain_adaptation=true,
        rng=rng,
        nuts_kwargs...,
    )

    ranks = Matrix{Int}(undef, num_values, num_simulations)
    for simulation = 1:num_simulations
        chain = chains.chains[simulation]
        draws = view(chain.constrained_samples, :, thin:thin:(num_samples*thin))
        for value_index = 1:num_values
            ranks[value_index, simulation] =
                count(<(truth[value_index, simulation]), view(draws, value_index, :))
        end
    end

    truth_lo = vec(minimum(truth; dims=2))
    truth_hi = vec(maximum(truth; dims=2))
    return _sbc_finalize(
        signature_layout, ranks, truth_lo, truth_hi, num_samples, num_bins, warn_threshold,
    )
end

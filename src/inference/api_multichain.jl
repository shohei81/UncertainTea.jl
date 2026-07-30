# Multi-chain drivers (issue #136): chains run in parallel across Julia
# threads when more than one is available. Per-chain seeds are pre-drawn from
# the caller's `rng` BEFORE any chain runs and every chain samples from its own
# `MersenneTwister(seed)`, so thread scheduling cannot change results -- each
# chain's draws are bitwise identical to a sequential run with the same caller
# rng. Each per-chain `hmc`/`nuts` call builds its own gradient cache, warmup
# driver, and sample buffers; the compiled signature plan is resolved once up
# front by `_conditioned_parameter_layout` and the plan memo itself is
# lock-protected (`_PLAN_MEMO_LOCK`, src/evaluator.jl), so the only shared
# mutable state left is the user `callback`. Callbacks are serialized behind a
# per-call lock: at most one callback invocation runs at a time, but
# invocations from different chains may interleave in any order.

# Run `run_chain(chain_index)` for every chain, writing into `chains`.
# Exceptions are captured per chain and the first (by chain index) is rethrown
# with its original type -- callers and tests match on ArgumentError /
# DimensionMismatch, so a TaskFailedException wrapper must not escape. The
# rethrow carries the original exception but not its original backtrace.
function _run_chains!(run_chain::F, chains::Vector{HMCChain}) where {F}
    num_chains = length(chains)
    if Threads.nthreads() == 1 || num_chains == 1
        for chain_index = 1:num_chains
            chains[chain_index] = run_chain(chain_index)
        end
        return chains
    end
    errors = Vector{Any}(nothing, num_chains)
    Threads.@threads for chain_index = 1:num_chains
        try
            chains[chain_index] = run_chain(chain_index)
        catch err
            errors[chain_index] = err
        end
    end
    for err in errors
        isnothing(err) || throw(err)
    end
    return chains
end

# Wrap the user callback for one chain: tag the info NamedTuple with the chain
# index and serialize invocations across chains behind `callback_lock`.
function _chain_progress_callback(callback, callback_lock::ReentrantLock, chain_index::Int)
    isnothing(callback) && return nothing
    return function (info)
        lock(callback_lock) do
            callback(merge(info, (chain=chain_index,)))
        end
    end
end

"""
    hmc_chains(model, args=(), constraints=choicemap(); num_chains, num_samples, kwargs...) -> HMCChains

Run `num_chains` independent [`hmc`](@ref) chains (each with its own seed and
warmup) and collect them into an [`HMCChains`](@ref) result suitable for
[`summarize`](@ref), `rhat`, and `ess`. Accepts the same per-chain keyword
arguments as [`hmc`](@ref).
"""
function hmc_chains(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    metric::Symbol=:diag,
    callback=nothing,
    callback_every::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    _validate_hmc_chains_arguments(num_chains)
    layout = _conditioned_parameter_layout(model, constraints)
    num_params = parametercount(layout)
    constrained_num_params = parametervaluecount(layout)
    seeds = rand(rng, UInt, num_chains)
    chains = Vector{HMCChain}(undef, num_chains)
    callback_lock = ReentrantLock()

    _run_chains!(chains) do chain_index
        chain_rng = MersenneTwister(seeds[chain_index])
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, constrained_num_params, num_chains)
        hmc(
            model,
            args,
            constraints;
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            initial_params=chain_initial_params,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            find_reasonable_step_size=find_reasonable_step_size,
            divergence_threshold=divergence_threshold,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            metric=metric,
            callback=_chain_progress_callback(callback, callback_lock, chain_index),
            callback_every=callback_every,
            rng=chain_rng,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

"""
    nuts_chains(model, args=(), constraints=choicemap(); num_chains, num_samples, kwargs...) -> HMCChains

Run `num_chains` independent [`nuts`](@ref) chains (each with its own seed and
warmup) and collect them into an [`HMCChains`](@ref) result suitable for
[`summarize`](@ref), `rhat`, and `ess`. Accepts the same per-chain keyword
arguments as [`nuts`](@ref). For many parallel chains over a dense layout, see
[`batched_nuts`](@ref).
"""
function nuts_chains(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    max_tree_depth::Int=10,
    initial_params=nothing,
    target_accept::Real=0.8,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    max_delta_energy::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    metric::Symbol=:diag,
    callback=nothing,
    callback_every::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    _validate_hmc_chains_arguments(num_chains, "NUTS")
    layout = _conditioned_parameter_layout(model, constraints)
    num_params = parametercount(layout)
    constrained_num_params = parametervaluecount(layout)
    seeds = rand(rng, UInt, num_chains)
    chains = Vector{HMCChain}(undef, num_chains)
    callback_lock = ReentrantLock()

    _run_chains!(chains) do chain_index
        chain_rng = MersenneTwister(seeds[chain_index])
        chain_initial_params = _chain_initial_params(initial_params, chain_index, num_params, constrained_num_params, num_chains)
        nuts(
            model,
            args,
            constraints;
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            initial_params=chain_initial_params,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            find_reasonable_step_size=find_reasonable_step_size,
            max_delta_energy=max_delta_energy,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            metric=metric,
            callback=_chain_progress_callback(callback, callback_lock, chain_index),
            callback_every=callback_every,
            rng=chain_rng,
        )
    end

    return HMCChains(model, args, constraints, chains)
end

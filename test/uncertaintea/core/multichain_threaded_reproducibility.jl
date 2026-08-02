# Threaded multi-chain drivers (issue #136): hmc_chains/nuts_chains run chains
# across Julia threads. Per-chain seeds are pre-drawn from the caller rng, so a
# threaded run must be bitwise identical, chain by chain, to running the
# single-chain sampler with the reconstructed per-chain rngs. These checks are
# scheduling-independent: with JULIA_NUM_THREADS=1 the driver degrades to the
# sequential loop and the same equalities hold, so the suite passes at any
# thread count (CI may run single-threaded; run locally with -t >1 to exercise
# the parallel path).
@testset "threaded multichain reproducibility" begin
    multichain_kwargs =
        (num_chains=4, num_samples=40, num_warmup=25, step_size=0.2, target_accept=0.8)

    # (a) bitwise per-chain reproducibility against the single-chain samplers:
    # reconstruct the pre-drawn seeds exactly as the driver does.
    reference_seeds = rand(MersenneTwister(136), UInt, multichain_kwargs.num_chains)

    threaded_nuts = nuts_chains(
        iid_model,
        (length(ys),),
        repeated;
        multichain_kwargs...,
        max_tree_depth=6,
        rng=MersenneTwister(136),
    )
    for (chain_index, chain) in enumerate(threaded_nuts.chains)
        reference = first(
            nuts(
                iid_model,
                (length(ys),),
                repeated;
                num_samples=multichain_kwargs.num_samples,
                num_warmup=multichain_kwargs.num_warmup,
                step_size=multichain_kwargs.step_size,
                target_accept=multichain_kwargs.target_accept,
                max_tree_depth=6,
                rng=MersenneTwister(reference_seeds[chain_index]),
            ),
        )
        @test chain.unconstrained_samples == reference.unconstrained_samples
        @test chain.constrained_samples == reference.constrained_samples
        @test chain.logjoint_values == reference.logjoint_values
        @test chain.accepted == reference.accepted
        @test chain.step_size == reference.step_size
        @test chain.mass_matrix == reference.mass_matrix
    end

    threaded_hmc = hmc_chains(
        gaussian_mean,
        (),
        constraints;
        multichain_kwargs...,
        num_leapfrog_steps=6,
        rng=MersenneTwister(136),
    )
    for (chain_index, chain) in enumerate(threaded_hmc.chains)
        reference = first(
            hmc(
                gaussian_mean,
                (),
                constraints;
                num_samples=multichain_kwargs.num_samples,
                num_warmup=multichain_kwargs.num_warmup,
                step_size=multichain_kwargs.step_size,
                target_accept=multichain_kwargs.target_accept,
                num_leapfrog_steps=6,
                rng=MersenneTwister(reference_seeds[chain_index]),
            ),
        )
        @test chain.unconstrained_samples == reference.unconstrained_samples
        @test chain.constrained_samples == reference.constrained_samples
        @test chain.logjoint_values == reference.logjoint_values
        @test chain.accepted == reference.accepted
        @test chain.step_size == reference.step_size
        @test chain.mass_matrix == reference.mass_matrix
    end

    # (b) determinism across two threaded runs with the same caller rng seed.
    replay_nuts = nuts_chains(
        iid_model,
        (length(ys),),
        repeated;
        multichain_kwargs...,
        max_tree_depth=6,
        rng=MersenneTwister(136),
    )
    for (chain, replay) in zip(threaded_nuts.chains, replay_nuts.chains)
        @test chain.unconstrained_samples == replay.unconstrained_samples
        @test chain.logjoint_values == replay.logjoint_values
    end
    replay_hmc = hmc_chains(
        gaussian_mean,
        (),
        constraints;
        multichain_kwargs...,
        num_leapfrog_steps=6,
        rng=MersenneTwister(136),
    )
    for (chain, replay) in zip(threaded_hmc.chains, replay_hmc.chains)
        @test chain.unconstrained_samples == replay.unconstrained_samples
        @test chain.logjoint_values == replay.logjoint_values
    end

    # (c) callbacks: serialized behind a lock, tagged with the chain index, and
    # never consuming the chain rng, so a callback run replays the plain run.
    # The un-synchronized push! below is safe exactly because the driver holds
    # the callback lock around every invocation.
    callback_records = NamedTuple[]
    callback_nuts = nuts_chains(
        iid_model,
        (length(ys),),
        repeated;
        multichain_kwargs...,
        max_tree_depth=6,
        callback=info -> push!(callback_records, info),
        callback_every=10,
        rng=MersenneTwister(136),
    )
    @test Set(info.chain for info in callback_records) ==
          Set(1:multichain_kwargs.num_chains)
    @test all(info.phase in (:warmup, :sample) for info in callback_records)
    for (chain, plain) in zip(callback_nuts.chains, threaded_nuts.chains)
        @test chain.unconstrained_samples == plain.unconstrained_samples
    end

    # Exceptions raised inside a chain keep their original type (not a
    # TaskFailedException wrapper), whether the loop is threaded or not.
    @test_throws DimensionMismatch nuts_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=4,
        num_samples=10,
        initial_params=zeros(1, 2),
    )
end

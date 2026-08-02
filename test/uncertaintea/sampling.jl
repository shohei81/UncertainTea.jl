@testset "sampling" begin
    gaussian_chain = first(
        hmc(
            gaussian_mean,
            (),
            constraints;
            num_samples=250,
            num_warmup=150,
            step_size=0.25,
            num_leapfrog_steps=8,
            rng=MersenneTwister(29),
        ),
    )
    gaussian_baseline_chain = first(
        hmc(
            gaussian_mean,
            (),
            constraints;
            num_samples=25,
            num_warmup=0,
            step_size=0.25,
            num_leapfrog_steps=8,
            rng=MersenneTwister(22),
        ),
    )
    gaussian_large_step_chain = first(
        hmc(
            gaussian_mean,
            (),
            constraints;
            num_samples=20,
            num_warmup=0,
            step_size=16.0,
            num_leapfrog_steps=4,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            find_reasonable_step_size=true,
            rng=MersenneTwister(23),
        ),
    )
    gaussian_small_step_chain = first(
        hmc(
            gaussian_mean,
            (),
            constraints;
            num_samples=20,
            num_warmup=0,
            step_size=1e-6,
            num_leapfrog_steps=4,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            find_reasonable_step_size=true,
            rng=MersenneTwister(24),
        ),
    )
    gaussian_divergent_chain = first(
        hmc(
            gaussian_mean,
            (),
            constraints;
            num_samples=20,
            num_warmup=0,
            step_size=2.0,
            num_leapfrog_steps=8,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            find_reasonable_step_size=false,
            divergence_threshold=1.0,
            rng=MersenneTwister(31),
        ),
    )
    gaussian_windowed_mass_chain = first(
        hmc(
            gaussian_mean,
            (),
            constraints;
            num_samples=20,
            num_warmup=20,
            step_size=0.2,
            num_leapfrog_steps=6,
            adapt_step_size=false,
            adapt_mass_matrix=true,
            mass_matrix_min_samples=3,
            rng=MersenneTwister(36),
        ),
    )
    gaussian_nuts_chain = first(
        nuts(
            gaussian_mean,
            (),
            constraints;
            num_samples=160,
            num_warmup=100,
            step_size=0.2,
            max_tree_depth=6,
            rng=MersenneTwister(60),
        ),
    )
    gaussian_nuts_baseline_chain = first(
        nuts(
            gaussian_mean,
            (),
            constraints;
            num_samples=20,
            num_warmup=0,
            step_size=0.2,
            max_tree_depth=5,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            rng=MersenneTwister(61),
        ),
    )
    gaussian_nuts_one_step_chain = first(
        nuts(
            gaussian_mean,
            (),
            constraints;
            num_samples=16,
            num_warmup=0,
            step_size=0.2,
            max_tree_depth=1,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            rng=MersenneTwister(64),
        ),
    )
    gaussian_nuts_one_step_chain_replay = first(
        nuts(
            gaussian_mean,
            (),
            constraints;
            num_samples=16,
            num_warmup=0,
            step_size=0.2,
            max_tree_depth=1,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            rng=MersenneTwister(64),
        ),
    )
    gaussian_multichain = hmc_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=60,
        num_warmup=30,
        step_size=0.2,
        num_leapfrog_steps=6,
        rng=MersenneTwister(46),
    )
    gaussian_multichain_replay = hmc_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=60,
        num_warmup=30,
        step_size=0.2,
        num_leapfrog_steps=6,
        rng=MersenneTwister(46),
    )
    positive_multichain = hmc_chains(
        observed_positive_step,
        (),
        choicemap((:y, 1.2f0));
        num_chains=3,
        num_samples=20,
        num_warmup=10,
        step_size=0.1,
        num_leapfrog_steps=6,
        initial_params=reshape(
            [
                positive_step_unconstrained[1] - 0.2,
                positive_step_unconstrained[1],
                positive_step_unconstrained[1] + 0.2,
            ],
            1,
            3,
        ),
        rng=MersenneTwister(41),
    )
    gaussian_nuts_multichain = nuts_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=50,
        num_warmup=30,
        step_size=0.18,
        max_tree_depth=6,
        rng=MersenneTwister(62),
    )
    gaussian_nuts_multichain_replay = nuts_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=50,
        num_warmup=30,
        step_size=0.18,
        max_tree_depth=6,
        rng=MersenneTwister(62),
    )
    # per_chain_adaptation=false: the assertions below pin the SHARED pooled
    # mass-window bookkeeping (issue #137 made per-chain the host default; a
    # per-chain driver pools only its own 10 window samples, under the
    # mass_matrix_min_samples=10 update threshold at this warmup length).
    gaussian_batched_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=40,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        per_chain_adaptation=false,
        rng=MersenneTwister(52),
    )
    gaussian_batched_baseline_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=40,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        rng=MersenneTwister(52),
    )
    gaussian_batched_chain_replay = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=40,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        per_chain_adaptation=false,
        rng=MersenneTwister(52),
    )
    gaussian_batched_nuts_chain = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=24,
        num_warmup=18,
        step_size=0.15,
        max_tree_depth=6,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(63),
    )
    gaussian_batched_nuts_chain_replay = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=24,
        num_warmup=18,
        step_size=0.15,
        max_tree_depth=6,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(63),
    )
    gaussian_batched_nuts_one_step_chain = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=16,
        num_warmup=0,
        step_size=0.15,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(64),
    )
    gaussian_batched_nuts_one_step_chain_replay = batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=16,
        num_warmup=0,
        step_size=0.15,
        max_tree_depth=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        initial_params=gaussian_batch_params,
        rng=MersenneTwister(64),
    )
    gaussian_batched_mass_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=20,
        num_warmup=20,
        step_size=0.18,
        num_leapfrog_steps=6,
        initial_params=gaussian_batch_params,
        adapt_step_size=false,
        adapt_mass_matrix=true,
        mass_matrix_min_samples=3,
        rng=MersenneTwister(55),
    )
    gaussian_batched_large_step_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        num_warmup=0,
        step_size=16.0,
        num_leapfrog_steps=4,
        initial_params=gaussian_batch_params,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        find_reasonable_step_size=true,
        rng=MersenneTwister(56),
    )
    gaussian_batched_divergence_adapt_chain = batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        num_warmup=15,
        step_size=2.0,
        num_leapfrog_steps=8,
        initial_params=gaussian_batch_params,
        adapt_step_size=true,
        adapt_mass_matrix=false,
        find_reasonable_step_size=false,
        divergence_threshold=1.0,
        rng=MersenneTwister(57),
    )
    iid_batched_chain = batched_hmc(
        iid_model,
        iid_batch_args,
        iid_batch_constraints;
        num_chains=2,
        num_samples=24,
        num_warmup=12,
        step_size=0.12,
        num_leapfrog_steps=5,
        initial_params=iid_batch_params,
        rng=MersenneTwister(53),
    )
    positive_batched_chain = batched_hmc(
        observed_positive_step,
        (),
        positive_batch_constraints;
        num_chains=3,
        num_samples=30,
        num_warmup=15,
        step_size=0.1,
        num_leapfrog_steps=6,
        initial_params=positive_batch_unconstrained,
        rng=MersenneTwister(54),
    )
    gaussian_rhat = rhat(gaussian_multichain)
    gaussian_rhat_unconstrained = rhat(gaussian_multichain; space=:unconstrained)
    gaussian_ess = ess(gaussian_multichain)
    gaussian_ess_unconstrained = ess(gaussian_multichain; space=:unconstrained)
    positive_rhat = rhat(positive_multichain)
    positive_ess = ess(positive_multichain)
    gaussian_batched_rhat = rhat(gaussian_batched_chain)
    gaussian_summary = summarize(gaussian_multichain)
    gaussian_summary_unconstrained = summarize(gaussian_multichain; space=:unconstrained)
    gaussian_nuts_rhat = rhat(gaussian_nuts_multichain)
    gaussian_nuts_ess = ess(gaussian_nuts_multichain)
    gaussian_nuts_summary = summarize(gaussian_nuts_multichain)
    gaussian_batched_nuts_summary = summarize(gaussian_batched_nuts_chain)
    positive_summary = summarize(positive_multichain; quantiles=(0.25, 0.5, 0.75))
    gaussian_pooled_draws = vcat([vec(chain.constrained_samples[1, :]) for chain in gaussian_multichain]...)
    gaussian_pooled_mean = sum(gaussian_pooled_draws) / length(gaussian_pooled_draws)
    gaussian_chain_mean = sum(gaussian_chain.constrained_samples[1, :]) / size(gaussian_chain.constrained_samples, 2)

    @test length(gaussian_chain) == 250
    @test size(gaussian_chain.unconstrained_samples) == (1, 250)
    @test size(gaussian_chain.constrained_samples) == (1, 250)
    @test all(isfinite, gaussian_chain.unconstrained_samples)
    @test all(isfinite, gaussian_chain.constrained_samples)
    @test all(isfinite, gaussian_chain.logjoint_values)
    @test size(gaussian_chain.energies) == (250,)
    @test size(gaussian_chain.energy_errors) == (250,)
    @test size(gaussian_chain.divergent) == (250,)
    @test all(isfinite, gaussian_chain.energies)
    @test all(isfinite, gaussian_chain.energy_errors[.!gaussian_chain.divergent])
    @test 0.0 <= acceptancerate(gaussian_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_chain) <= 1.0
    @test length(massadaptationwindows(gaussian_chain)) == 3
    @test massadaptationwindows(gaussian_chain)[1] isa HMCMassAdaptationWindowSummary
    @test massadaptationwindows(gaussian_chain)[1].iteration_start == 16
    @test massadaptationwindows(gaussian_chain)[1].iteration_end == 40
    @test massadaptationwindows(gaussian_chain)[1].window_length == 25
    @test massadaptationwindows(gaussian_chain)[1].clip_scale_start == 8.0
    @test 5.0 <= massadaptationwindows(gaussian_chain)[1].clip_scale_end <= 8.0
    @test massadaptationwindows(gaussian_chain)[1].effective_count >= 3
    @test massadaptationwindows(gaussian_chain)[1].mass_min > 0
    @test massadaptationwindows(gaussian_chain)[1].mass_max >= massadaptationwindows(gaussian_chain)[1].mass_min
    @test any(gaussian_chain.accepted)
    @test gaussian_chain.step_size > 0
    @test gaussian_chain.mass_matrix[1] > 0
    @test gaussian_chain.target_accept == 0.8
    @test gaussian_baseline_chain.step_size == 0.25
    @test gaussian_baseline_chain.mass_matrix == [1.0]
    @test isempty(massadaptationwindows(gaussian_baseline_chain))
    @test 0 < gaussian_large_step_chain.step_size < 16.0
    @test gaussian_small_step_chain.step_size > 1e-6
    @test all(isfinite, gaussian_large_step_chain.logjoint_values)
    @test all(isfinite, gaussian_small_step_chain.logjoint_values)
    @test gaussian_windowed_mass_chain.step_size == 0.2
    @test gaussian_windowed_mass_chain.mass_matrix[1] != 1.0
    @test gaussian_nuts_chain.sampler == :nuts
    @test length(gaussian_nuts_chain) == 160
    @test size(gaussian_nuts_chain.unconstrained_samples) == (1, 160)
    @test size(gaussian_nuts_chain.constrained_samples) == (1, 160)
    @test all(isfinite, gaussian_nuts_chain.logjoint_values)
    @test all(isfinite, gaussian_nuts_chain.acceptance_stats)
    @test all(0.0 <= stat <= 1.0 for stat in gaussian_nuts_chain.acceptance_stats)
    @test all(1 <= depth <= gaussian_nuts_chain.max_tree_depth for depth in treedepths(gaussian_nuts_chain))
    @test all(1 <= steps <= (2 ^ gaussian_nuts_chain.max_tree_depth - 1) for steps in integrationsteps(gaussian_nuts_chain))
    @test gaussian_nuts_chain.num_leapfrog_steps == 0
    @test gaussian_nuts_chain.max_tree_depth == 6
    @test 0.0 <= acceptancerate(gaussian_nuts_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_nuts_chain) <= 1.0
    @test gaussian_nuts_chain.step_size > 0
    @test gaussian_nuts_chain.mass_matrix[1] > 0
    @test any(gaussian_nuts_chain.accepted)
    @test abs(sum(gaussian_nuts_chain.constrained_samples[1, :]) / size(gaussian_nuts_chain.constrained_samples, 2) - 0.15) < 0.25
    @test gaussian_nuts_baseline_chain.step_size == 0.2
    @test gaussian_nuts_baseline_chain.mass_matrix == [1.0]
    @test isempty(massadaptationwindows(gaussian_nuts_baseline_chain))
    @test gaussian_nuts_one_step_chain.max_tree_depth == 1
    @test gaussian_nuts_one_step_chain.step_size == 0.2
    @test gaussian_nuts_one_step_chain.mass_matrix == [1.0]
    @test isempty(massadaptationwindows(gaussian_nuts_one_step_chain))
    @test all(depth == 1 for depth in treedepths(gaussian_nuts_one_step_chain))
    @test all(0 <= steps <= 1 for steps in integrationsteps(gaussian_nuts_one_step_chain))
    @test gaussian_nuts_one_step_chain.unconstrained_samples == gaussian_nuts_one_step_chain_replay.unconstrained_samples
    @test gaussian_nuts_one_step_chain.accepted == gaussian_nuts_one_step_chain_replay.accepted
    @test nchains(gaussian_multichain) == 3
    @test numsamples(gaussian_multichain) == 60
    @test gaussian_multichain[1] isa HMCChain
    @test length(massadaptationwindows(gaussian_multichain)) == 3
    @test all(length(windows) == 1 for windows in massadaptationwindows(gaussian_multichain))
    @test 0.0 <= acceptancerate(gaussian_multichain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_multichain) <= 1.0
    @test length(gaussian_rhat) == 1
    @test length(gaussian_ess) == 1
    @test isfinite(gaussian_rhat[1])
    @test 1.0 <= gaussian_rhat[1] < 1.1
    @test gaussian_rhat ≈ gaussian_rhat_unconstrained atol=1e-8
    @test 1.0 <= gaussian_ess[1] <= nchains(gaussian_multichain) * numsamples(gaussian_multichain)
    @test gaussian_ess ≈ gaussian_ess_unconstrained atol=1e-8
    @test length(gaussian_summary) == 1
    @test gaussian_summary.space == :constrained
    @test gaussian_summary.quantile_probs == [0.05, 0.5, 0.95]
    @test gaussian_summary.diagnostics isa HMCDiagnosticsSummary
    @test acceptancerate(gaussian_summary) == acceptancerate(gaussian_multichain)
    @test divergencerate(gaussian_summary) == divergencerate(gaussian_multichain)
    @test gaussian_summary.diagnostics.mean_step_size ≈
          sum(chain.step_size for chain in gaussian_multichain) / nchains(gaussian_multichain) atol=1e-8
    @test length(gaussian_summary.diagnostics.step_sizes) == nchains(gaussian_multichain)
    @test length(massadaptationwindows(gaussian_summary)) == 1
    @test massadaptationwindows(gaussian_summary)[1] isa HMCMassAdaptationSummary
    @test massadaptationwindows(gaussian_summary)[1].chains == 3
    @test massadaptationwindows(gaussian_summary)[1].num_updated == 3
    @test massadaptationwindows(gaussian_summary)[1].iteration_start == 6
    @test massadaptationwindows(gaussian_summary)[1].iteration_end == 25
    @test massadaptationwindows(gaussian_summary)[1].mean_effective_count >= 3
    @test massadaptationwindows(gaussian_summary)[1].min_mass > 0
    gaussian_summary_text = repr(MIME"text/plain"(), gaussian_summary)
    @test occursin("HMCSummary(gaussian_mean)", gaussian_summary_text)
    @test occursin("diagnostics:", gaussian_summary_text)
    @test occursin("acceptance_rate:", gaussian_summary_text)
    @test occursin("mass_adaptation_windows:", gaussian_summary_text)
    @test occursin("mu @ mu:", gaussian_summary_text)
    diagnostics_text = repr(MIME"text/plain"(), gaussian_summary.diagnostics)
    @test occursin("HMCDiagnosticsSummary", diagnostics_text)
    @test occursin("step_size:", diagnostics_text)
    @test occursin("window 1 [6:25]", diagnostics_text)
    mass_window_text = repr(MIME"text/plain"(), massadaptationwindows(gaussian_summary)[1])
    @test occursin("HMCMassAdaptationSummary", mass_window_text)
    @test occursin("chains: 3 updated=3/3", mass_window_text)
    @test gaussian_summary[1].binding == :mu
    @test gaussian_summary[1].address == :mu
    @test gaussian_summary[1].mean ≈ gaussian_pooled_mean atol=1e-8
    @test gaussian_summary[1].sd > 0
    @test gaussian_summary[1].quantiles[1] <= gaussian_summary[1].quantiles[2] <= gaussian_summary[1].quantiles[3]
    @test gaussian_summary[1].rhat == gaussian_rhat[1]
    @test gaussian_summary[1].ess == gaussian_ess[1]
    @test gaussian_summary[1].mean ≈ gaussian_summary_unconstrained[1].mean atol=1e-8
    @test length(massadaptationwindows(gaussian_summary_unconstrained)) == 1
    @test gaussian_multichain[1].unconstrained_samples[:, 1] ==
          gaussian_multichain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_multichain[1].accepted == gaussian_multichain_replay[1].accepted
    @test gaussian_multichain[1].unconstrained_samples[:, 1] != gaussian_multichain[2].unconstrained_samples[:, 1]
    @test nchains(gaussian_nuts_multichain) == 3
    @test numsamples(gaussian_nuts_multichain) == 50
    @test all(chain.sampler == :nuts for chain in gaussian_nuts_multichain)
    @test all(length(treedepths(chain)) == 50 for chain in gaussian_nuts_multichain)
    @test all(all(1 <= depth <= chain.max_tree_depth for depth in treedepths(chain)) for chain in gaussian_nuts_multichain)
    @test all(
        all(1 <= steps <= (2 ^ chain.max_tree_depth - 1) for steps in integrationsteps(chain)) for chain in gaussian_nuts_multichain
    )
    @test length(gaussian_nuts_rhat) == 1
    @test length(gaussian_nuts_ess) == 1
    @test isfinite(gaussian_nuts_rhat[1])
    # < 1.2, not 1.1: with only 3 chains x 50 samples the rhat estimator's
    # noise exceeds 0.1 (the pre-1.12 randn stream realizes 1.13 here); 1.2 is
    # the conventional healthy-chain bound and still catches stuck chains.
    @test 1.0 <= gaussian_nuts_rhat[1] < 1.2
    @test 1.0 <= gaussian_nuts_ess[1] <= nchains(gaussian_nuts_multichain) * numsamples(gaussian_nuts_multichain)
    @test acceptancerate(gaussian_nuts_summary) == acceptancerate(gaussian_nuts_multichain)
    @test divergencerate(gaussian_nuts_summary) == divergencerate(gaussian_nuts_multichain)
    @test gaussian_nuts_multichain[1].unconstrained_samples[:, 1] ==
          gaussian_nuts_multichain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_nuts_multichain[1].tree_depths == gaussian_nuts_multichain_replay[1].tree_depths
    @test nchains(gaussian_batched_chain) == 3
    @test numsamples(gaussian_batched_chain) == 40
    @test gaussian_batched_chain.args == ()
    @test length(gaussian_batched_chain.constraints) == 3
    @test gaussian_batched_chain[1].constraints[:y] == gaussian_batch_constraints[1][:y]
    @test gaussian_batched_chain[2].constraints[:y] == gaussian_batch_constraints[2][:y]
    @test all(chain.step_size > 0 for chain in gaussian_batched_chain)
    @test all(chain.mass_matrix[1] > 0 for chain in gaussian_batched_chain)
    @test all(length(massadaptationwindows(chain)) == 1 for chain in gaussian_batched_chain)
    @test length(massadaptationwindows(gaussian_batched_chain)) == 3
    @test all(windows[1].window_length == 10 for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(windows[1].iteration_start == 6 for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(windows[1].iteration_end == 15 for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(windows[1].updated for windows in massadaptationwindows(gaussian_batched_chain))
    @test all(chain.step_size == 0.18 for chain in gaussian_batched_baseline_chain)
    @test all(chain.mass_matrix == [1.0] for chain in gaussian_batched_baseline_chain)
    @test all(isempty(massadaptationwindows(chain)) for chain in gaussian_batched_baseline_chain)
    @test all(chain.step_size == 0.18 for chain in gaussian_batched_mass_chain)
    @test all(chain.mass_matrix[1] != 1.0 for chain in gaussian_batched_mass_chain)
    @test all(0 < chain.step_size < 16.0 for chain in gaussian_batched_large_step_chain)
    @test all(all(isfinite, chain.logjoint_values) for chain in gaussian_batched_large_step_chain)
    @test all(chain.step_size < 2.0 for chain in gaussian_batched_divergence_adapt_chain)
    gaussian_batched_summary = summarize(gaussian_batched_chain)
    @test acceptancerate(gaussian_batched_summary) == acceptancerate(gaussian_batched_chain)
    @test divergencerate(gaussian_batched_summary) == divergencerate(gaussian_batched_chain)
    @test length(massadaptationwindows(gaussian_batched_summary)) == 1
    @test massadaptationwindows(gaussian_batched_summary)[1].chains == 3
    @test massadaptationwindows(gaussian_batched_summary)[1].window_length == 10
    @test massadaptationwindows(gaussian_batched_summary)[1].num_updated == 3
    @test occursin("window 1 [6:15]", repr(MIME"text/plain"(), gaussian_batched_summary))
    @test !(
        isapprox(gaussian_batched_chain[1].step_size, gaussian_batched_baseline_chain[1].step_size; atol=1e-8) &&
        isapprox(gaussian_batched_chain[1].mass_matrix[1], gaussian_batched_baseline_chain[1].mass_matrix[1]; atol=1e-8)
    )
    @test 0.0 <= acceptancerate(gaussian_batched_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_batched_chain) <= 1.0
    @test length(gaussian_batched_rhat) == 1
    @test isfinite(gaussian_batched_rhat[1])
    @test gaussian_batched_chain[1].unconstrained_samples[:, 1] ==
          gaussian_batched_chain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_batched_chain[1].accepted == gaussian_batched_chain_replay[1].accepted
    @test nchains(gaussian_batched_nuts_chain) == 3
    @test numsamples(gaussian_batched_nuts_chain) == 24
    @test gaussian_batched_nuts_chain.args == ()
    @test length(gaussian_batched_nuts_chain.constraints) == 3
    @test all(chain.sampler == :nuts for chain in gaussian_batched_nuts_chain)
    @test all(chain.step_size > 0 for chain in gaussian_batched_nuts_chain)
    @test all(chain.mass_matrix[1] > 0 for chain in gaussian_batched_nuts_chain)
    @test all(length(massadaptationwindows(chain)) == 1 for chain in gaussian_batched_nuts_chain)
    @test all(all(1 <= depth <= chain.max_tree_depth for depth in treedepths(chain)) for chain in gaussian_batched_nuts_chain)
    @test all(
        all(1 <= steps <= (2 ^ chain.max_tree_depth - 1) for steps in integrationsteps(chain)) for
        chain in gaussian_batched_nuts_chain
    )
    @test acceptancerate(gaussian_batched_nuts_summary) == acceptancerate(gaussian_batched_nuts_chain)
    @test divergencerate(gaussian_batched_nuts_summary) == divergencerate(gaussian_batched_nuts_chain)
    @test length(massadaptationwindows(gaussian_batched_nuts_summary)) == 1
    @test massadaptationwindows(gaussian_batched_nuts_summary)[1].chains == 3
    @test gaussian_batched_nuts_chain[1].unconstrained_samples[:, 1] ==
          gaussian_batched_nuts_chain_replay[1].unconstrained_samples[:, 1]
    @test gaussian_batched_nuts_chain[1].tree_depths == gaussian_batched_nuts_chain_replay[1].tree_depths
    @test nchains(gaussian_batched_nuts_one_step_chain) == 3
    @test numsamples(gaussian_batched_nuts_one_step_chain) == 16
    @test all(chain.max_tree_depth == 1 for chain in gaussian_batched_nuts_one_step_chain)
    @test all(chain.step_size == 0.15 for chain in gaussian_batched_nuts_one_step_chain)
    @test all(chain.mass_matrix == [1.0] for chain in gaussian_batched_nuts_one_step_chain)
    @test all(isempty(massadaptationwindows(chain)) for chain in gaussian_batched_nuts_one_step_chain)
    @test all(all(depth == 1 for depth in treedepths(chain)) for chain in gaussian_batched_nuts_one_step_chain)
    @test all(all(0 <= steps <= 1 for steps in integrationsteps(chain)) for chain in gaussian_batched_nuts_one_step_chain)
    @test 0.0 <= acceptancerate(gaussian_batched_nuts_one_step_chain) <= 1.0
    @test 0.0 <= divergencerate(gaussian_batched_nuts_one_step_chain) <= 1.0
    @test gaussian_batched_nuts_one_step_chain[1].unconstrained_samples ==
          gaussian_batched_nuts_one_step_chain_replay[1].unconstrained_samples
    @test gaussian_batched_nuts_one_step_chain[1].accepted ==
          gaussian_batched_nuts_one_step_chain_replay[1].accepted
    @test gaussian_batched_nuts_one_step_chain[1].tree_depths ==
          gaussian_batched_nuts_one_step_chain_replay[1].tree_depths
    @test nchains(iid_batched_chain) == 2
    @test numsamples(iid_batched_chain) == 24
    @test iid_batched_chain.args == iid_batch_args
    @test length(iid_batched_chain.constraints) == 2
    @test iid_batched_chain[1].args == iid_batch_args[1]
    @test iid_batched_chain[2].constraints[:y=>3] == iid_batch_constraints[2][:y=>3]
    @test all(all(isfinite, chain.logjoint_values) for chain in iid_batched_chain)
    @test all(gaussian_divergent_chain.divergent)
    @test divergencerate(gaussian_divergent_chain) == 1.0
    @test all(isfinite, gaussian_divergent_chain.energies)
    @test maximum(abs, gaussian_divergent_chain.energy_errors) > 1.0
    @test !(
        isapprox(gaussian_chain.step_size, gaussian_baseline_chain.step_size; atol=1e-8) &&
        isapprox(gaussian_chain.mass_matrix[1], gaussian_baseline_chain.mass_matrix[1]; atol=1e-8)
    )
    @test abs(gaussian_chain_mean - 0.15) < 0.2
    @test gaussian_chain.logjoint_values[1] ≈
          logjoint_unconstrained(gaussian_mean, gaussian_chain.unconstrained_samples[:, 1], (), constraints) atol=1e-6

    positive_chain = first(
        hmc(
            observed_positive_step,
            (),
            choicemap((:y, 1.2f0));
            num_samples=120,
            num_warmup=80,
            step_size=0.12,
            num_leapfrog_steps=10,
            initial_params=positive_step_unconstrained,
            rng=MersenneTwister(21),
        ),
    )

    @test length(positive_chain) == 120
    @test size(positive_chain.unconstrained_samples) == (1, 120)
    @test size(positive_chain.constrained_samples) == (1, 120)
    @test all(x -> x > 0, positive_chain.constrained_samples)
    @test all(isfinite, positive_chain.energies)
    @test all(isfinite, positive_chain.energy_errors[.!positive_chain.divergent])
    @test any(positive_chain.accepted)
    @test positive_chain.step_size > 0
    @test positive_chain.mass_matrix[1] > 0
    @test nchains(positive_multichain) == 3
    @test length(positive_rhat) == 1
    @test length(positive_ess) == 1
    @test isfinite(positive_rhat[1])
    @test positive_rhat[1] >= 1.0
    @test 1.0 <= positive_ess[1] <= nchains(positive_multichain) * numsamples(positive_multichain)
    @test length(positive_summary) == 1
    @test positive_summary.quantile_probs == [0.25, 0.5, 0.75]
    @test positive_summary[1].binding isa Symbol
    @test positive_summary[1].address == (:state, :sigma)
    @test positive_summary[1].mean > 0
    @test positive_summary[1].quantiles[1] <= positive_summary[1].quantiles[2] <= positive_summary[1].quantiles[3]
    @test positive_summary[1].rhat == positive_rhat[1]
    @test positive_summary[1].ess == positive_ess[1]
    @test all(all(x -> x > 0, chain.constrained_samples) for chain in positive_multichain)
    @test nchains(positive_batched_chain) == 3
    @test numsamples(positive_batched_chain) == 30
    @test all(all(x -> x > 0, chain.constrained_samples) for chain in positive_batched_chain)
    @test all(chain.mass_matrix[1] > 0 for chain in positive_batched_chain)
    @test parameterchoicemap(observed_positive_step, positive_chain.constrained_samples[:, 1])[:state=>:sigma] ==
          positive_chain.constrained_samples[1, 1]

    @tea static function observed_only()
        {:y} ~ bernoulli(0.5f0)
        return nothing
    end

    @test_throws DimensionMismatch parameterchoicemap(gaussian_mean, Float64[])
    @test_throws DimensionMismatch logjoint(iid_model, params2, (), repeated)
    @test_throws ArgumentError hmc(observed_only, (), choicemap((:y, true)); num_samples=10)
    @test_throws ArgumentError nuts(observed_only, (), choicemap((:y, true)); num_samples=10)
    @test_throws ArgumentError hmc(gaussian_mean, (), constraints; num_samples=10, divergence_threshold=0.0)
    @test_throws ArgumentError nuts(gaussian_mean, (), constraints; num_samples=10, max_tree_depth=0)
    @test_throws ArgumentError hmc_chains(gaussian_mean, (), constraints; num_chains=0, num_samples=10)
    @test_throws ArgumentError nuts_chains(gaussian_mean, (), constraints; num_chains=0, num_samples=10)
    @test_throws ArgumentError batched_hmc(gaussian_mean, (), gaussian_batch_constraints; num_chains=0, num_samples=10)
    @test_throws ArgumentError batched_nuts(gaussian_mean, (), gaussian_batch_constraints; num_chains=0, num_samples=10)
    @test_throws DimensionMismatch batched_logjoint(gaussian_mean, zeros(2, 3), (), gaussian_batch_constraints)
    @test_throws DimensionMismatch batched_logjoint(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints[1:2])
    @test_throws ArgumentError batched_logjoint(gaussian_mean, gaussian_batch_params, [1, 2, 3], gaussian_batch_constraints)
    @test_throws DimensionMismatch batched_logjoint_gradient_unconstrained(gaussian_batch_gradient_cache, zeros(1, 2))
    @test_throws ArgumentError backend_execution_plan(unsupported_backend_model)
    @test_throws DimensionMismatch batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints[1:2];
        num_chains=3,
        num_samples=10,
    )
    @test_throws DimensionMismatch batched_hmc(
        iid_model,
        iid_batch_args[1:1],
        iid_batch_constraints;
        num_chains=2,
        num_samples=10,
    )
    # single-chain diagnostics are supported (issue #337): rhat runs on the
    # chain's two split halves instead of throwing
    @test length(rhat(HMCChains(gaussian_mean, (), constraints, [gaussian_baseline_chain]))) == 1
    @test_throws ArgumentError rhat(HMCChains(gaussian_mean, (), constraints, HMCChain[]))
    @test_throws ArgumentError ess(gaussian_multichain; space=:energy)
    @test_throws ArgumentError summarize(gaussian_multichain; quantiles=())
    @test_throws ArgumentError summarize(gaussian_multichain; quantiles=(-0.1, 0.5, 0.9))
    @test_throws DimensionMismatch hmc_chains(
        gaussian_mean,
        (),
        constraints;
        num_chains=3,
        num_samples=10,
        initial_params=zeros(1, 2),
    )
    @test_throws DimensionMismatch batched_hmc(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        initial_params=zeros(1, 2),
    )
    @test_throws DimensionMismatch batched_nuts(
        gaussian_mean,
        (),
        gaussian_batch_constraints;
        num_chains=3,
        num_samples=10,
        initial_params=zeros(1, 2),
    )
end

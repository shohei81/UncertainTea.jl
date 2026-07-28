# PR 39 / issue #137 / issue #158: pooled-mass + per-chain-step warmup adaptation
# for the batched samplers.
# Contract: batched_hmc / batched_nuts accept per_chain_adaptation. Since issue
# #137 the HOST default is per-chain step; since issue #158 `per_chain_adaptation`
# means pooled mass + per-chain step on BOTH host and device (the retired host
# per-chain-MASS mode is gone). An unset value resolves to true.
# When false the run is bitwise identical to the shared-driver path; when true a
# SINGLE diagonal inverse mass matrix is pooled across all chains (one
# running-variance state fed every chain's draws) while each chain owns its own
# dual-averaged step size. The batched integrators / momentum sampling /
# Hamiltonians carry the per-chain step vector against the shared pooled mass. The
# progress callback reports the mean of the per-chain step sizes.

@testset "per_chain_warmup_batched" begin
    @tea static function pca_gaussian_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
        return mu
    end

    pca_gaussian_constraints = choicemap((:y, 0.3f0))

    # Fixed-seed regression: per_chain_adaptation=false reproduces the shared
    # driver's adapted step size / mass matrix exactly (values captured from the
    # code prior to this change; these match the warmup_driver_regression shared-driver regression).
    pca_reg = batched_nuts(
        pca_gaussian_model,
        (),
        pca_gaussian_constraints;
        num_chains=3,
        num_samples=20,
        num_warmup=30,
        per_chain_adaptation=false,
        rng=MersenneTwister(404),
    )
    for pca_reg_chain in pca_reg.chains
        # Re-pinned after the issue #159 biased progressive merge (the
        # doubling-merge swap is now Stan's min(1, w_subtree/w_continuation),
        # shifting the warmup trajectory under the same RNG draws); still
        # matches the warmup_driver_regression shared-driver regression.
        # (Previous re-pin: issue #93/#81 fixes.)
        if adaptation_pins_exact
            @test pca_reg_chain.step_size ≈ 1.2243640004920302 atol = 1e-12
        end
        @test length(pca_reg_chain.mass_matrix) == 1
        if adaptation_pins_exact
            @test pca_reg_chain.mass_matrix[1] ≈ 0.3968463881646532 atol = 1e-12
        end
        # Version-independent: per_chain_adaptation=false means the shared
        # driver adapts once for the whole batch.
        @test pca_reg_chain.step_size == pca_reg.chains[1].step_size
        @test pca_reg_chain.mass_matrix == pca_reg.chains[1].mass_matrix
    end

    # Per-chain divergence: two chains facing different observation scales via a
    # per-chain args vector. sigma=0.1 yields a tight posterior on mu; sigma=10.0
    # yields a wide one. _validate_batched_args accepts a vector of tuples.
    @tea static function pca_scale_model(sigma_arg)
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, sigma_arg)
        return mu
    end

    pca_scale_constraints = choicemap((:y, 0.0f0))
    pca_scale_args = [(0.1f0,), (10.0f0,)]

    pca_scale = batched_nuts(
        pca_scale_model,
        pca_scale_args,
        pca_scale_constraints;
        num_chains=2,
        num_samples=50,
        num_warmup=150,
        per_chain_adaptation=true,
        rng=MersenneTwister(2024),
    )
    pca_step_tight = pca_scale.chains[1].step_size
    pca_step_wide = pca_scale.chains[2].step_size
    pca_mass_tight = pca_scale.chains[1].mass_matrix[1]
    pca_mass_wide = pca_scale.chains[2].mass_matrix[1]
    # Adapted step sizes must differ by at least 2x across the two scales: the
    # per-chain step is the operative per-chain property (and the #137 stranding
    # fix) that the pooled-mass / per-chain-step host default (issue #158) keeps.
    @test max(pca_step_tight, pca_step_wide) / min(pca_step_tight, pca_step_wide) >= 2.0
    # Under the pooled-mass default the diagonal mass is shared across all chains
    # (one running-variance state fed every chain's draws), so the two chains carry
    # the SAME mass entry -- superseding the retired per-chain-mass mode where the
    # wider posterior carried the larger entry.
    @test pca_mass_wide == pca_mass_tight

    # Statistical sanity: per-chain mode on the shared gaussian recovers the
    # posterior mean (0.15) and mixes (rhat < 1.2). Uses 4 chains x 500 draws:
    # a 2x200 run estimates the mean at only ~1 sigma, so the metric-aware
    # U-turn / invalid-subtree fixes (which shifted the seeded trajectories)
    # tipped it past atol on the 1.10 CI entry; the larger budget recovers the
    # mean within tolerance on both the 1.10 and latest matrix entries.
    pca_stat = batched_nuts(
        pca_gaussian_model,
        (),
        pca_gaussian_constraints;
        num_chains=4,
        num_samples=500,
        num_warmup=300,
        per_chain_adaptation=true,
        rng=MersenneTwister(20260705),
    )
    pca_stat_summary = summarize(pca_stat)
    @test pca_stat_summary[1].mean ≈ 0.15 atol = 0.1
    @test pca_stat_summary[1].rhat < 1.2

    # batched_hmc per-chain smoke: adapts distinct per-chain step sizes too.
    pca_hmc = batched_hmc(
        pca_scale_model,
        pca_scale_args,
        pca_scale_constraints;
        num_chains=2,
        num_samples=50,
        num_warmup=150,
        per_chain_adaptation=true,
        rng=MersenneTwister(99),
    )
    @test pca_hmc.chains[1].step_size != pca_hmc.chains[2].step_size
    # Pooled mass: both chains share the diagonal mass entry under the issue #158
    # host default (per-chain step, pooled mass).
    @test pca_hmc.chains[2].mass_matrix[1] == pca_hmc.chains[1].mass_matrix[1]

    # Callback still fires in per-chain mode with the documented phase fields; the
    # reported step_size is the mean of the per-chain step sizes.
    pca_cb_events = NamedTuple[]
    batched_nuts(
        pca_gaussian_model,
        (),
        pca_gaussian_constraints;
        num_chains=2,
        num_samples=20,
        num_warmup=20,
        per_chain_adaptation=true,
        callback=info -> push!(pca_cb_events, info),
        callback_every=10,
        rng=MersenneTwister(313),
    )
    @test !isempty(pca_cb_events)
    @test all(e -> e.phase === :warmup || e.phase === :sample, pca_cb_events)
    @test all(e -> haskey(e, :step_size) && haskey(e, :divergences), pca_cb_events)
    @test all(e -> isfinite(e.step_size), pca_cb_events)
end

# Issue #137: with prior-draw initialization and SHARED step-size adaptation,
# the chains whose initial `s` sits in the high-curvature small-scale region
# never get a usable step size and diverge on every iteration (~5-7% overall
# divergence, a handful of fully stranded chains at 64 chains). Per-chain
# adaptation -- the host default since #137 -- recovers all of them. The model
# is the gauss shape from the #121 benchmark (mu ~ normal, s ~ gamma, loop
# observations).
@testset "pca_default_per_chain_unstrands_gauss" begin
    @tea static function pca_strand_gauss_model(n)
        mu ~ normal(0.0, 1.0)
        s ~ gamma(2.0, 1.0)
        for i = 1:n
            {:y => i} ~ normal(mu, s)
        end
        return mu
    end

    pca_strand_n = 200
    pca_strand_data_rng = MersenneTwister(42)
    pca_strand_y = 0.5 .+ 1.2 .* randn(pca_strand_data_rng, pca_strand_n)
    pca_strand_cm = choicemap(((:y => i, pca_strand_y[i]) for i = 1:pca_strand_n)...)

    pca_strand_chains = batched_nuts(
        pca_strand_gauss_model,
        (pca_strand_n,),
        pca_strand_cm;
        num_chains=64,
        num_samples=200,
        num_warmup=200,
        tree_strategy=:hybrid,
        rng=MersenneTwister(11),
    )
    pca_strand_rates = [sum(pca_chain.divergent) / length(pca_chain.divergent) for pca_chain in pca_strand_chains]
    pca_strand_total = sum(sum(pca_chain.divergent) for pca_chain in pca_strand_chains)
    # On the shared-adaptation default this seed realizes ~6% divergence with
    # 4/64 chains stranded above 50%; per-chain adaptation gives 0.0%.
    @test pca_strand_total / (64 * 200) < 0.01
    @test all(<(0.5), pca_strand_rates)
end

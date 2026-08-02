# PR 38: MAP estimation and Laplace approximation.
# Contract: map_estimate maximizes logjoint_unconstrained via a self-contained
# L-BFGS with backtracking Armijo line search; laplace_approximation forms the
# Gaussian covariance inv(-H) at the mode and can draw unconstrained samples.

# Exact Gaussian: mu ~ N(0,1); y | mu ~ N(mu,1) with y=0.3 has posterior mode
# mu = y/2 = 0.15 and posterior variance 0.5 in the (already unconstrained)
# latent space.
@testset "map_laplace_approximation" begin
    @tea static function map_gaussian_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(mu, 1.0f0)
    end

    map_gaussian_constraints = choicemap((:y, 0.3f0))
    # Seed the default random init: map_estimate's start point is drawn from the
    # passed rng, so a fixed seed makes the `converged`/`gradient_norm` assertions
    # deterministic across processes instead of depending on the unseeded global
    # RNG (issue #124). The mode values below are start-independent (convex).
    map_gaussian_result = map_estimate(map_gaussian_model, (), map_gaussian_constraints; rng=MersenneTwister(1))
    @test map_gaussian_result.converged
    @test map_gaussian_result.unconstrained_mode ≈ [0.15] atol=1e-6
    @test map_gaussian_result.constrained_mode ≈ [0.15] atol=1e-6
    @test map_gaussian_result.logjoint ≈ logjoint_unconstrained(
        map_gaussian_model,
        map_gaussian_result.unconstrained_mode,
        (),
        map_gaussian_constraints,
    ) atol=1e-10

    map_gaussian_laplace = laplace_approximation(map_gaussian_model, (), map_gaussian_constraints; rng=MersenneTwister(2))
    @test map_gaussian_laplace.covariance ≈ [0.5;;] atol=1e-6
    @test map_gaussian_laplace.map.unconstrained_mode ≈ [0.15] atol=1e-6

    # Transformed latent: s ~ LogNormal(0, 0.5); y | s ~ N(0, s). The latent is
    # sampled in unconstrained (log) space; MAP should converge there and yield a
    # strictly positive constrained mode.
    @tea static function map_lognormal_model()
        s ~ lognormal(0.0f0, 0.5f0)
        {:y} ~ normal(0.0f0, s)
    end

    map_lognormal_constraints = choicemap((:y, 1.2f0))
    # Seeded init (issue #124): makes `converged`/`gradient_norm` deterministic.
    map_lognormal_result = map_estimate(map_lognormal_model, (), map_lognormal_constraints; rng=MersenneTwister(3))
    @test map_lognormal_result.converged
    @test map_lognormal_result.gradient_norm < 1e-6
    @test map_lognormal_result.constrained_mode[1] > 0

    # Multi-parameter convergence from a poor initialization.
    @tea static function map_two_latent_model()
        a ~ normal(0.0f0, 1.0f0)
        b ~ normal(0.0f0, 1.0f0)
        {:y1} ~ normal(a, 1.0f0)
        {:y2} ~ normal(b, 1.0f0)
    end

    map_two_latent_constraints = choicemap((:y1, 2.0f0), (:y2, -1.0f0))
    map_two_latent_result = map_estimate(
        map_two_latent_model,
        (),
        map_two_latent_constraints;
        initial_params=[5.0, -5.0],
    )
    @test map_two_latent_result.converged
    @test map_two_latent_result.gradient_norm < 1e-6
    @test map_two_latent_result.unconstrained_mode ≈ [1.0, -0.5] atol=1e-6

    # Laplace rand: seeded draws from the exact-Gaussian case reproduce the
    # posterior mean (0.15) and variance (0.5).
    map_rand_draws = rand(MersenneTwister(20240705), map_gaussian_laplace, 5000)
    @test size(map_rand_draws) == (1, 5000)
    map_rand_mean = sum(map_rand_draws) / length(map_rand_draws)
    map_rand_var =
        sum((map_rand_draws .- map_rand_mean) .^ 2) / (length(map_rand_draws) - 1)
    @test map_rand_mean ≈ 0.15 atol=0.05
    @test map_rand_var ≈ 0.5 rtol=0.2

    # End-to-end: use the MAP mode as the NUTS initialization and confirm sampling
    # produces finite draws.
    map_nuts_chain = nuts(
        map_gaussian_model,
        (),
        map_gaussian_constraints;
        num_samples=50,
        num_warmup=50,
        initial_params=map_gaussian_result.unconstrained_mode,
        rng=MersenneTwister(7),
    )
    @test all(isfinite, map_nuts_chain.unconstrained_samples)
    @test all(isfinite, map_nuts_chain.constrained_samples)
    @test size(map_nuts_chain.unconstrained_samples, 2) == 50

    # ------------------------------------------------------------------
    # PR: ecosystem-interop chain export (zero-dep core + MCMCChains ext).
    # ------------------------------------------------------------------
    @tea static function export_two_latent_model()
        a ~ normal(0.0f0, 1.0f0)
        b ~ normal(0.0f0, 1.0f0)
        {:y1} ~ normal(a, 1.0f0)
        {:y2} ~ normal(b, 1.0f0)
    end

    export_constraints = choicemap((:y1, 2.0f0), (:y2, -1.0f0))
    export_chains = hmc_chains(
        export_two_latent_model,
        (),
        export_constraints;
        num_chains=2,
        num_samples=30,
        num_warmup=20,
        rng=MersenneTwister(4242),
    )
    export_num_params = parametervaluecount(parameterlayout(export_two_latent_model))

    export_arr = posterior_array(export_chains)
    @test size(export_arr) == (30, 2, export_num_params)
    # Draw-major layout: arr[s, c, p] == chains[c].constrained_samples[p, s].
    @test export_arr[1, 1, 1] == export_chains.chains[1].constrained_samples[1, 1]
    @test export_arr[30, 2, export_num_params] ==
          export_chains.chains[2].constrained_samples[export_num_params, 30]
    @test export_arr[15, 2, 1] == export_chains.chains[2].constrained_samples[1, 15]

    export_names = parameter_names(export_chains)
    @test length(export_names) == export_num_params
    @test eltype(export_names) == String

    export_unconstrained = posterior_array(export_chains; space=:unconstrained)
    @test size(export_unconstrained) == (30, 2, export_num_params)
    @test export_unconstrained[7, 1, 1] ==
          export_chains.chains[1].unconstrained_samples[1, 7]

    export_dict = to_arviz_dict(export_chains)
    @test Set(keys(export_dict)) == Set(["posterior", "sample_stats"])
    export_posterior = export_dict["posterior"]
    @test Set(keys(export_posterior)) == Set(export_names)
    @test size(export_posterior[export_names[1]]) == (30, 2)
    export_stats = export_dict["sample_stats"]
    @test Set(keys(export_stats)) ==
          Set(["diverging", "energy", "tree_depth", "acceptance_rate", "lp"])
    @test size(export_stats["diverging"]) == (30, 2)
    @test export_stats["diverging"] isa Array{Bool,2}
    @test export_stats["lp"][3, 1] == export_chains.chains[1].logjoint_values[3]
    @test export_stats["lp"][10, 2] == export_chains.chains[2].logjoint_values[10]

    # The core declares `to_mcmcchains` method-less; the UncertainTeaMCMCChainsExt
    # extension adds the ::HMCChains method once MCMCChains is loaded (which
    # mcmcchains_export.jl does, so the method count here depends on include
    # order — the functional coverage lives in that file).
    @test to_mcmcchains isa Function

    # ------------------------------------------------------------------
    # PR: progress callbacks (zero-overhead when nothing; RNG untouched).
    # ------------------------------------------------------------------
    cb_events = NamedTuple[]
    cb_collector = info -> push!(cb_events, info)
    cb_chain = nuts(
        export_two_latent_model,
        (),
        export_constraints;
        num_samples=20,
        num_warmup=20,
        callback=cb_collector,
        callback_every=10,
        rng=MersenneTwister(99),
    )
    # Warmup fires at 10, 20; sampling fires at 10, 20 -> exactly 4 invocations.
    @test length(cb_events) == 4
    @test [e.phase for e in cb_events] == [:warmup, :warmup, :sample, :sample]
    @test [e.iteration for e in cb_events] == [10, 20, 10, 20]
    @test cb_events[end].iteration == cb_events[end].total
    @test cb_events[end].total == 20
    @test all(e -> haskey(e, :step_size) && haskey(e, :divergences), cb_events)

    # Callbacks must not consume the RNG: same seed => bitwise-identical samples.
    cb_baseline = nuts(
        export_two_latent_model,
        (),
        export_constraints;
        num_samples=20,
        num_warmup=20,
        callback=nothing,
        rng=MersenneTwister(99),
    )
    @test cb_chain.unconstrained_samples == cb_baseline.unconstrained_samples

    # batched_smc invokes the callback per stage with phase :stage.
    cb_smc_events = NamedTuple[]
    batched_smc(
        map_gaussian_model,
        (),
        map_gaussian_constraints;
        num_particles=32,
        callback=info -> push!(cb_smc_events, info),
        rng=MersenneTwister(11),
    )
    @test length(cb_smc_events) >= 1
    @test all(e -> e.phase === :stage, cb_smc_events)

    # batched_advi invokes the callback per step with phase :step.
    cb_advi_events = NamedTuple[]
    batched_advi(
        map_gaussian_model,
        (),
        map_gaussian_constraints;
        num_iterations=25,
        callback=info -> push!(cb_advi_events, info),
        callback_every=10,
        rng=MersenneTwister(12),
    )
    @test length(cb_advi_events) >= 1
    @test all(e -> e.phase === :step, cb_advi_events)
    @test cb_advi_events[end].iteration == 25
end

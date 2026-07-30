# Seeding the warmup INITIAL inverse-mass matrix from the Pathfinder covariance
# diagonal (issue #162, part 3; Zhang, Carpenter, Gelman, Vehtari, JMLR 2022).
# In HMC/NUTS the inverse-mass matrix IS the position-covariance estimate, so
# when `batched_nuts` is initialized from a PathfinderResult the initial diagonal
# metric is seeded from `diag(pf.covariance)` (clamped) instead of `ones`.
#
# The observation hook is `num_warmup=0`: with no warmup, mass adaptation never
# runs, so the returned chain's `mass_matrix` is exactly the seed the driver
# started from.

# Ill-conditioned diagonal Gaussian: two independent latents observed at very
# different noise scales, so the analytic posterior variances (hence the
# Pathfinder covariance diagonal) are clearly non-unit and clearly unequal.
#   a | ya:  var = 1/(1/10^2 + 1/0.5^2) = 1/4.01   ~= 0.2494,  sd ~= 0.4994
#   b | yb:  var = 1/(1/10^2 + 1/3.0^2) = 1/0.1211 ~= 8.2569,  sd ~= 2.8735
@tea static function pf_illcond_model()
    a ~ normal(0.0, 10.0)
    b ~ normal(0.0, 10.0)
    {:ya} ~ normal(a, 0.5)
    {:yb} ~ normal(b, 3.0)
    return a
end

# A different model (and different dimension), for the wrong-model checks.
@tea static function pf_seed_other_model()
    m ~ normal(0.0, 1.0)
    {:z} ~ normal(m, 1.0)
    return m
end

@testset "pathfinder_mass_seed" begin
    illcond_constraints = choicemap((:ya, 1.0), (:yb, 2.0))
    # analytic posterior (see model comment)
    a_var = 1 / (1 / 100 + 1 / 0.25)
    b_var = 1 / (1 / 100 + 1 / 9.0)
    a_mean = a_var * (1.0 / 0.25)
    b_mean = b_var * (2.0 / 9.0)

    reg = 1e-3

    @testset "driver_kwarg" begin
        # WarmupDriver: default omitted -> ones; explicit -> that vector verbatim.
        default_driver = UncertainTea.WarmupDriver(
            3, 10, 0.1, 0.8;
            adapt_step_size=true, adapt_mass_matrix=true,
            mass_matrix_regularization=reg, mass_matrix_min_samples=10,
        )
        @test default_driver.inverse_mass_matrix == ones(3)

        seeded_driver = UncertainTea.WarmupDriver(
            3, 10, 0.1, 0.8;
            adapt_step_size=true, adapt_mass_matrix=true,
            mass_matrix_regularization=reg, mass_matrix_min_samples=10,
            initial_inverse_mass_matrix=[2.0, 3.0, 4.0],
        )
        @test seeded_driver.inverse_mass_matrix == [2.0, 3.0, 4.0]

        # PooledMassPerChainStepDriver: the seed lands on its inner shared mass.
        default_pooled = UncertainTea.PooledMassPerChainStepDriver(
            3, 10, [0.1, 0.1], 0.8;
            adapt_step_size=true, adapt_mass_matrix=true,
            mass_matrix_regularization=reg, mass_matrix_min_samples=10,
        )
        @test default_pooled.mass.inverse_mass_matrix == ones(3)

        seeded_pooled = UncertainTea.PooledMassPerChainStepDriver(
            3, 10, [0.1, 0.1], 0.8;
            adapt_step_size=true, adapt_mass_matrix=true,
            mass_matrix_regularization=reg, mass_matrix_min_samples=10,
            initial_inverse_mass_matrix=[2.0, 3.0, 4.0],
        )
        @test seeded_pooled.mass.inverse_mass_matrix == [2.0, 3.0, 4.0]

        # Validation: wrong length, negative, and non-finite all rejected.
        @test_throws ArgumentError UncertainTea.WarmupDriver(
            3, 10, 0.1, 0.8;
            adapt_step_size=true, adapt_mass_matrix=true,
            mass_matrix_regularization=reg, mass_matrix_min_samples=10,
            initial_inverse_mass_matrix=[1.0, 2.0],
        )
        @test_throws ArgumentError UncertainTea.WarmupDriver(
            3, 10, 0.1, 0.8;
            adapt_step_size=true, adapt_mass_matrix=true,
            mass_matrix_regularization=reg, mass_matrix_min_samples=10,
            initial_inverse_mass_matrix=[1.0, -2.0, 3.0],
        )
        @test_throws ArgumentError UncertainTea.WarmupDriver(
            3, 10, 0.1, 0.8;
            adapt_step_size=true, adapt_mass_matrix=true,
            mass_matrix_regularization=reg, mass_matrix_min_samples=10,
            initial_inverse_mass_matrix=[1.0, Inf, 3.0],
        )
    end

    @testset "seed_helper" begin
        pf = pathfinder(pf_illcond_model, (), illcond_constraints; num_draws=200, rng=MersenneTwister(101))
        num_params = length(pf.location)
        @test num_params == 2

        seed = UncertainTea._pathfinder_inverse_mass_seed(pf, pf_illcond_model, num_params, reg)
        expected = [max(pf.covariance[i, i], reg) for i = 1:num_params]
        @test seed == expected
        # the covariance diagonal is genuinely non-unit and unequal here
        @test maximum(abs.(seed .- 1.0)) > 0.5
        @test seed[2] > 3.0

        # clamping: negative -> regularization, non-finite -> 1.0
        bad = UncertainTea.PathfinderResult(
            pf.model, pf.args, pf.constraints, pf.location,
            [-2.0 0.0; 0.0 NaN],
            pf.cholesky_lower, pf.draws, pf.elbo, pf.elbo_history,
            pf.num_paths, pf.best_path, pf.converged,
        )
        bad_seed = UncertainTea._pathfinder_inverse_mass_seed(bad, pf.model, 2, reg)
        @test bad_seed[1] == reg
        @test bad_seed[2] == 1.0

        # dimension mismatch throws
        @test_throws ArgumentError UncertainTea._pathfinder_inverse_mass_seed(pf, pf_illcond_model, num_params + 1, reg)
        # wrong model throws
        other = pathfinder(pf_seed_other_model, (), choicemap((:z, 1.0)); rng=MersenneTwister(7))
        @test_throws ArgumentError UncertainTea._pathfinder_inverse_mass_seed(other, pf_illcond_model, num_params, reg)
    end

    @testset "seeding_wired_num_warmup_0" begin
        pf = pathfinder(pf_illcond_model, (), illcond_constraints; num_draws=200, rng=MersenneTwister(102))
        expected = [max(pf.covariance[i, i], reg) for i = 1:length(pf.location)]

        # num_warmup=0: adaptation never overwrites the seed, so the reported
        # mass_matrix is exactly the Pathfinder-seeded initial inverse mass.
        seeded = batched_nuts(
            pf_illcond_model, (), illcond_constraints;
            num_chains=3, num_samples=5, num_warmup=0,
            initial_params=pf, mass_matrix_regularization=reg,
            rng=MersenneTwister(103),
        )
        for chain in seeded.chains
            @test chain.mass_matrix == expected
        end
        @test maximum(abs.(expected .- 1.0)) > 0.5

        # regression guard: default (non-Pathfinder) init keeps the ones metric.
        default = batched_nuts(
            pf_illcond_model, (), illcond_constraints;
            num_chains=3, num_samples=5, num_warmup=0,
            mass_matrix_regularization=reg, rng=MersenneTwister(104),
        )
        for chain in default.chains
            @test chain.mass_matrix == ones(2)
        end
    end

    @testset "wrong_model_rejected" begin
        other = pathfinder(pf_seed_other_model, (), choicemap((:z, 1.0)); rng=MersenneTwister(8))
        @test_throws ArgumentError batched_nuts(
            pf_illcond_model, (), illcond_constraints;
            num_chains=2, num_samples=4, num_warmup=0, initial_params=other,
            rng=MersenneTwister(9),
        )
    end

    @testset "recovery" begin
        pf = pathfinder(pf_illcond_model, (), illcond_constraints; num_draws=200, rng=MersenneTwister(105))
        chains = batched_nuts(
            pf_illcond_model, (), illcond_constraints;
            num_chains=4, num_samples=1500, num_warmup=800,
            initial_params=pf, rng=MersenneTwister(106),
        )
        names = parameter_names(chains)
        draws = posterior_array(chains)  # (num_samples, num_chains, num_params)
        stats = Dict{Symbol,Tuple{Float64,Float64}}()
        for (p, name) in enumerate(names)
            col = vec(draws[:, :, p])
            m = sum(col) / length(col)
            s = sqrt(sum((col .- m) .^ 2) / (length(col) - 1))
            stats[Symbol(name)] = (m, s)
        end
        rhats = rhat(chains)

        @test isapprox(stats[:a][1], a_mean; atol=0.15)
        @test isapprox(stats[:a][2], sqrt(a_var); rtol=0.2)
        @test isapprox(stats[:b][1], b_mean; atol=0.3)
        @test isapprox(stats[:b][2], sqrt(b_var); rtol=0.2)
        @test maximum(rhats) < 1.05
        @test divergencerate(chains) < 0.02
    end
end

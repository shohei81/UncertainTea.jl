# Pathfinder variational initialization (issue #15). The correlated conjugate
# model has an analytic Gaussian posterior, so the best Pathfinder Gaussian
# must recover its mean and covariance almost exactly; the funnel model shows
# the initialization value (draws land in the typical set instead of the
# prior tails).

@tea static function pf_corr_model()
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ normal(a + b, 0.5)
    return a
end

@tea static function pf_funnel_model()
    v ~ normal(0.0, 3.0)
    x ~ normal(0.0, exp(v / 2))
    {:y} ~ normal(x, 0.3)
    return v
end

@tea static function pf_leaky_support_model()
    lo ~ normal(0.0, 1.0)
    {:y} ~ truncatednormal(0.0, 1.0, lo, Inf)
    return lo
end

@testset "pathfinder_init" begin
    pf_corr_constraints = choicemap((:y, 1.0))
    pf_corr_cov = [5.0 -4.0; -4.0 5.0] ./ 9.0
    pf_corr_mean = pf_corr_cov * [4.0, 4.0]

    @testset "pf_gaussian_recovery" begin
        result = pathfinder(pf_corr_model, (), pf_corr_constraints; num_draws=400, rng=MersenneTwister(11))
        @test result.converged
        @test result.num_paths == 1
        @test maximum(abs.(result.location .- pf_corr_mean)) < 1e-3
        @test maximum(abs.(result.covariance .- pf_corr_cov)) < 0.05
        @test isfinite(result.elbo)
        @test size(result.draws) == (2, 400)
        draw_mean = vec(sum(result.draws; dims=2)) ./ 400
        @test maximum(abs.(draw_mean .- pf_corr_mean)) < 0.1
        shown = repr(result)
        @test occursin("PathfinderResult", shown)
    end

    @testset "pf_multipath" begin
        result = pathfinder(
            pf_corr_model,
            (),
            pf_corr_constraints;
            num_paths=4,
            num_draws=300,
            rng=MersenneTwister(12),
        )
        @test result.num_paths == 4
        @test 1 <= result.best_path <= 4
        draw_mean = vec(sum(result.draws; dims=2)) ./ 300
        @test maximum(abs.(draw_mean .- pf_corr_mean)) < 0.15
    end

    @testset "pf_funnel_initialization_quality" begin
        funnel_constraints = choicemap((:y, 0.5))
        result = pathfinder(
            pf_funnel_model,
            (),
            funnel_constraints;
            num_paths=4,
            num_draws=200,
            rng=MersenneTwister(77),
        )
        logjoint_at = theta -> logjoint_unconstrained(pf_funnel_model, theta, (), funnel_constraints)
        rng = MersenneTwister(5)
        prior_mean_logjoint =
            sum(
                logjoint_at(UncertainTea._initial_hmc_position(pf_funnel_model, (), funnel_constraints, nothing, rng)) for _ = 1:50
            ) / 50
        pathfinder_mean_logjoint = sum(logjoint_at(result.draws[:, j]) for j = 1:200) / 200
        # pathfinder draws land in the typical set; prior draws start far out
        @test pathfinder_mean_logjoint > prior_mean_logjoint + 10.0

        chain = first(
            nuts(
                pf_funnel_model,
                (),
                funnel_constraints;
                num_samples=60,
                num_warmup=40,
                initial_params=result,
                rng=MersenneTwister(2),
            ),
        )
        @test all(isfinite, chain.constrained_samples)

        batched = batched_nuts(
            pf_funnel_model,
            (),
            funnel_constraints;
            num_chains=3,
            num_samples=20,
            num_warmup=20,
            initial_params=result,
            rng=MersenneTwister(3),
        )
        @test length(batched.chains) == 3
        @test all(all(isfinite, chain.constrained_samples) for chain in batched.chains)
    end

    @testset "pf_finite_draws_on_leaky_support" begin
        # the log joint is -Inf whenever lo > y, so every candidate Gaussian
        # carries invalid mass; all returned draws must still be usable inits
        leaky_constraints = choicemap((:y, 0.5))
        logjoint_at = theta -> logjoint_unconstrained(pf_leaky_support_model, theta, (), leaky_constraints)
        single = pathfinder(pf_leaky_support_model, (), leaky_constraints; num_draws=80, rng=MersenneTwister(21))
        @test all(isfinite(logjoint_at(single.draws[:, j])) for j = 1:80)
        multi = pathfinder(
            pf_leaky_support_model,
            (),
            leaky_constraints;
            num_paths=3,
            num_draws=80,
            rng=MersenneTwister(22),
        )
        @test all(isfinite(logjoint_at(multi.draws[:, j])) for j = 1:80)
    end

    @testset "pf_validation" begin
        @test_throws ArgumentError pathfinder(pf_corr_model, (), pf_corr_constraints; num_paths=0)
        @test_throws ArgumentError pathfinder(pf_corr_model, (), pf_corr_constraints; num_draws=0)

        other_model_result = pathfinder(pf_funnel_model, (), choicemap((:y, 0.5)); rng=MersenneTwister(1))
        @test_throws ArgumentError nuts(
            pf_corr_model,
            (),
            pf_corr_constraints;
            num_samples=4,
            initial_params=other_model_result,
        )
    end
end

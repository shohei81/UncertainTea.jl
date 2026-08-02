# Full-rank and low-rank ADVI guides (issue #14). The correlated-posterior
# model has the analytic posterior covariance [5 -4; -4 5]/9 (correlation
# -0.8), which mean-field structurally cannot represent, a rank-1-plus-
# diagonal factorization represents exactly, and full-rank recovers.

using KernelAbstractions: CPU as ADVIStructuredCPU

@tea static function advi_corr_model()
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ normal(a + b, 0.5)
    return a
end

@tea static function advi_scale_corr_model()
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    {:y} ~ normal(mu, sigma)
    return mu
end

@testset "advi_structured_guides" begin
    advi_corr_target = [5.0 -4.0; -4.0 5.0] ./ 9.0
    advi_corr_constraints = choicemap((:y, 1.0))

    advi_fit =
        (guide; kwargs...) -> batched_advi(
            advi_corr_model,
            (),
            advi_corr_constraints;
            num_iterations=2000,
            num_particles=64,
            learning_rate=0.02,
            guide=guide,
            rng=MersenneTwister(5),
            kwargs...,
        )

    @testset "advi_meanfield_cannot_represent_correlation" begin
        result = advi_fit(:meanfield)
        @test result.guide === :meanfield
        @test result.scale_factor === nothing
        covariance = variational_covariance(result)
        @test covariance[1, 2] == 0.0
        @test maximum(abs.(covariance .- advi_corr_target)) > 0.3
    end

    @testset "advi_fullrank_recovers_covariance" begin
        result = advi_fit(:fullrank)
        @test result.guide === :fullrank
        @test size(result.scale_factor) == (2, 2)
        covariance = variational_covariance(result)
        @test maximum(abs.(covariance .- advi_corr_target)) < 0.12
        @test covariance[1, 2] < -0.3
        @test all(isfinite, result.elbo_history)

        # samples drawn from the fitted guide match its reported covariance
        samples = variational_samples(result; num_samples=6000, space=:unconstrained, rng=MersenneTwister(9))
        centered = samples .- sum(samples; dims=2) ./ size(samples, 2)
        empirical = centered * transpose(centered) ./ (size(samples, 2) - 1)
        @test maximum(abs.(empirical .- covariance)) < 0.05
    end

    @testset "advi_lowrank_recovers_covariance" begin
        result = advi_fit(:lowrank; lowrank_rank=1)
        @test result.guide === :lowrank
        @test size(result.scale_factor) == (2, 1)
        covariance = variational_covariance(result)
        @test maximum(abs.(covariance .- advi_corr_target)) < 0.08
        @test covariance[1, 2] < -0.3
    end

    @testset "advi_fullrank_transformed_parameters" begin
        result = batched_advi(
            advi_scale_corr_model,
            (),
            choicemap((:y, 0.8));
            num_iterations=400,
            num_particles=32,
            learning_rate=0.05,
            guide=:fullrank,
            rng=MersenneTwister(3),
        )
        @test all(isfinite, result.elbo_history)
        constrained = variational_samples(result; num_samples=64, rng=MersenneTwister(4))
        @test all(constrained[2, :] .> 0)
    end

    @testset "advi_guide_validation" begin
        @test_throws ArgumentError batched_advi(
            advi_corr_model,
            (),
            advi_corr_constraints;
            num_iterations=4,
            guide=:bogus,
        )
        @test_throws ArgumentError batched_advi(
            advi_corr_model,
            (),
            advi_corr_constraints;
            num_iterations=4,
            guide=:lowrank,
            lowrank_rank=0,
        )
        @test_throws ArgumentError batched_advi(
            advi_corr_model,
            (),
            advi_corr_constraints;
            num_iterations=4,
            guide=:lowrank,
            lowrank_rank=3,
        )
        @test_throws ArgumentError batched_advi(
            advi_corr_model,
            (),
            advi_corr_constraints;
            num_iterations=4,
            guide=:fullrank,
            backend=ADVIStructuredCPU(),
        )
    end
end

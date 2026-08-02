@testset "Batched ADVI and Particle Methods" begin
    @tea static function mvnormal_vi_model()
        state ~ mvnormal([0.0f0, 1.0f0], [1.5f0, 0.8f0])
        return state
    end

    @tea static function dirichlet_vi_model()
        weights ~ dirichlet([2.0f0, 3.0f0, 4.0f0])
        return weights
    end

    mvnormal_advi = batched_advi(
        mvnormal_vi_model,
        (),
        choicemap();
        num_iterations=60,
        num_particles=24,
        learning_rate=0.05,
        initial_params=Float64[-1.0, -1.0],
        initial_log_scale=fill(-0.3, 2),
        rng=MersenneTwister(200),
    )
    dirichlet_advi = batched_advi(
        dirichlet_vi_model,
        (),
        choicemap();
        num_iterations=60,
        num_particles=24,
        learning_rate=0.03,
        initial_params=Float64[0.2, 0.3, 0.5],
        initial_log_scale=fill(-0.7, 2),
        rng=MersenneTwister(201),
    )

    @test mvnormal_advi.gradient_backend == :backend_native
    @test dirichlet_advi.gradient_backend == :backend_native
    @test length(mvnormal_advi.elbo_history) == 60
    @test length(dirichlet_advi.elbo_history) == 60
    @test all(isfinite, mvnormal_advi.elbo_history)
    @test all(isfinite, dirichlet_advi.elbo_history)
    @test all(isfinite, mvnormal_advi.gradient_norm_history)
    @test all(isfinite, dirichlet_advi.gradient_norm_history)
    @test mvnormal_advi.best_elbo >= maximum(mvnormal_advi.elbo_history) - 1e-10
    @test dirichlet_advi.best_elbo >= maximum(dirichlet_advi.elbo_history) - 1e-10

    mvnormal_vi_mean = variational_mean(mvnormal_advi; space=:unconstrained)
    mvnormal_vi_samples = variational_samples(
        mvnormal_advi;
        num_samples=8,
        space=:unconstrained,
        rng=MersenneTwister(202),
    )
    dirichlet_vi_mean = variational_mean(dirichlet_advi; space=:constrained)
    dirichlet_vi_samples = variational_samples(
        dirichlet_advi;
        num_samples=8,
        space=:constrained,
        rng=MersenneTwister(203),
    )

    @test mvnormal_vi_mean ≈ [0.0, 1.0] atol=0.8
    @test size(mvnormal_vi_samples) == (2, 8)
    @test size(dirichlet_vi_samples) == (3, 8)
    @test all(>(0.0), dirichlet_vi_mean)
    @test sum(dirichlet_vi_mean) ≈ 1.0 atol=1e-6
    for sample_index = 1:size(dirichlet_vi_samples, 2)
        @test all(>(0.0), dirichlet_vi_samples[:, sample_index])
        @test sum(dirichlet_vi_samples[:, sample_index]) ≈ 1.0 atol=1e-6
    end

    mvnormal_importance = batched_importance_sampling(
        mvnormal_vi_model,
        (),
        choicemap();
        num_particles=16,
        proposal_loc=Float64[0.0, 1.0],
        proposal_log_scale=log.([1.5, 0.8]),
        rng=MersenneTwister(204),
    )

    @test mvnormal_importance.evaluation_backend == :backend_native
    @test size(mvnormal_importance.unconstrained_particles) == (2, 16)
    @test size(mvnormal_importance.constrained_particles) == (2, 16)
    @test mvnormal_importance.constrained_particles ≈ mvnormal_importance.unconstrained_particles atol=1e-8
    @test maximum(abs.(mvnormal_importance.logweights .- first(mvnormal_importance.logweights))) <= 5e-8
    @test mvnormal_importance.log_evidence_estimate ≈ 0.0 atol=1e-8
    @test sum(mvnormal_importance.normalized_weights) ≈ 1.0 atol=1e-8
    @test ess(mvnormal_importance) ≈ 16.0 atol=1e-8

    dirichlet_sir = batched_sir(
        dirichlet_vi_model,
        (),
        choicemap();
        num_particles=24,
        num_samples=10,
        proposal_loc=Float64[0.2, 0.3, 0.5],
        proposal_log_scale=fill(-0.5, 2),
        rng=MersenneTwister(205),
    )
    dirichlet_smc = batched_smc(
        dirichlet_vi_model,
        (),
        choicemap();
        num_particles=18,
        proposal_loc=Float64[0.2, 0.3, 0.5],
        proposal_log_scale=fill(-0.4, 2),
        rng=MersenneTwister(206),
    )

    @test dirichlet_sir isa SIRResult
    @test dirichlet_smc isa SMCResult
    @test dirichlet_sir.importance.evaluation_backend == :backend_native
    @test dirichlet_smc.importance.evaluation_backend == :backend_native
    @test numsamples(dirichlet_sir) == 10
    @test numsamples(dirichlet_smc) == 18
    @test numstages(dirichlet_smc) >= 1
    @test size(dirichlet_sir.unconstrained_samples) == (2, 10)
    @test size(dirichlet_sir.constrained_samples) == (3, 10)
    @test size(dirichlet_smc.importance.unconstrained_particles) == (2, 18)
    @test size(dirichlet_smc.importance.constrained_particles) == (3, 18)
    @test all(1 .<= dirichlet_sir.ancestors .<= 24)
    @test all(stage.beta_start <= stage.beta_end && stage.beta_end <= 1.0 for stage in dirichlet_smc.stages)
    @test last(dirichlet_smc.stages).beta_end ≈ 1.0 atol=1e-6
    @test 0.0 < ess(dirichlet_sir.importance) <= 24.0
    @test 0.0 < ess(dirichlet_smc) <= 18.0
    for sample_index = 1:size(dirichlet_sir.constrained_samples, 2)
        @test all(>(0.0), dirichlet_sir.constrained_samples[:, sample_index])
        @test sum(dirichlet_sir.constrained_samples[:, sample_index]) ≈ 1.0 atol=1e-6
    end
    for sample_index = 1:size(dirichlet_smc.importance.constrained_particles, 2)
        @test all(>(0.0), dirichlet_smc.importance.constrained_particles[:, sample_index])
        @test sum(dirichlet_smc.importance.constrained_particles[:, sample_index]) ≈ 1.0 atol=1e-6
    end
end

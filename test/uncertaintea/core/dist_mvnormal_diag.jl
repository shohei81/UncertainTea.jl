@testset "dist_mvnormal_diag" begin
    @test UncertainTea.logpdf(
        mvnormal([0.0, 1.0], [1.5, 0.8]),
        [0.2, 1.3],
    ) ≈ (
        UncertainTea.logpdf(normal(0.0, 1.5), 0.2) +
        UncertainTea.logpdf(normal(1.0, 0.8), 1.3)
    ) atol=1e-8

    @tea static function mvnormal_latent_model()
        state ~ mvnormal([0.0f0, 1.0f0], [1.5f0, 0.8f0])
        return state
    end

    mvnormal_trace, _ = generate(mvnormal_latent_model, (), choicemap(); rng=MersenneTwister(180))
    mvnormal_spec = modelspec(mvnormal_latent_model)
    mvnormal_plan = executionplan(mvnormal_latent_model)
    mvnormal_backend_report = backend_report(mvnormal_latent_model)
    mvnormal_params = parameter_vector(mvnormal_trace)
    mvnormal_unconstrained = transform_to_unconstrained(mvnormal_trace)
    mvnormal_reconstrained = transform_to_constrained(mvnormal_latent_model, mvnormal_unconstrained)
    mvnormal_choicemap = parameterchoicemap(mvnormal_latent_model, mvnormal_params)

    @test mvnormal_spec.choices[1].rhs.family == :mvnormal
    @test mvnormal_spec.parameter_layout.slots[1].transform isa VectorIdentityTransform
    @test mvnormal_spec.parameter_layout.slots[1].dimension == 2
    @test mvnormal_spec.parameter_layout.slots[1].value_length == 2
    @test mvnormal_plan.steps[1].parameter_slot == 1
    @test parametercount(parameterlayout(mvnormal_latent_model)) == 2
    @test parametervaluecount(parameterlayout(mvnormal_latent_model)) == 2
    @test length(mvnormal_unconstrained) == 2
    @test length(mvnormal_params) == 2
    @test mvnormal_unconstrained ≈ mvnormal_params atol=1e-8
    @test mvnormal_reconstrained ≈ mvnormal_params atol=1e-8
    @test mvnormal_choicemap[:state] ≈ mvnormal_params atol=1e-8
    @test logjoint(mvnormal_latent_model, mvnormal_params) ≈
          assess(mvnormal_latent_model, (), choicemap((:state, mvnormal_trace[:state]))) atol=1e-6
    @test mvnormal_backend_report.supported
    @test isempty(mvnormal_backend_report.issues)
    @test backend_execution_plan(mvnormal_latent_model).steps[1] isa UncertainTea.BackendMvNormalChoicePlanStep

    mvnormal_batch_params = hcat(
        mvnormal_unconstrained,
        mvnormal_unconstrained .+ Float64[0.2, -0.1],
        mvnormal_unconstrained .+ Float64[-0.15, 0.05],
    )
    mvnormal_batch_values = batched_logjoint_unconstrained(mvnormal_latent_model, mvnormal_batch_params, (), choicemap())
    mvnormal_batch_gradient = batched_logjoint_gradient_unconstrained(mvnormal_latent_model, mvnormal_batch_params, (), choicemap())
    mvnormal_batch_cache = BatchedLogjointGradientCache(mvnormal_latent_model, mvnormal_batch_params, (), choicemap())

    @test mvnormal_batch_values ≈ [
        logjoint_unconstrained(mvnormal_latent_model, mvnormal_batch_params[:, index], (), choicemap()) for index = 1:3
    ] atol=1e-8
    @test mvnormal_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(mvnormal_latent_model, mvnormal_batch_params[:, index], (), choicemap()) for index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(mvnormal_batch_cache.backend_cache)
    @test isnothing(mvnormal_batch_cache.flat_cache)
    @test isempty(mvnormal_batch_cache.column_caches)

    mvnormal_chain = first(
        hmc(
            mvnormal_latent_model,
            (),
            choicemap();
            num_samples=6,
            num_warmup=0,
            step_size=0.05,
            num_leapfrog_steps=2,
            initial_params=mvnormal_params,
            rng=MersenneTwister(181),
        ),
    )

    @test size(mvnormal_chain.unconstrained_samples) == (2, 6)
    @test size(mvnormal_chain.constrained_samples) == (2, 6)

    mvnormal_chains = hmc_chains(
        mvnormal_latent_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=4,
        num_warmup=0,
        step_size=0.05,
        num_leapfrog_steps=2,
        rng=MersenneTwister(182),
    )
    constrained_summary = summarize(mvnormal_chains; space=:constrained)
    unconstrained_summary = summarize(mvnormal_chains; space=:unconstrained)

    @test length(constrained_summary.parameters) == 2
    @test length(unconstrained_summary.parameters) == 2
    @test constrained_summary.parameters[1].address == (:state, 1)
    @test constrained_summary.parameters[2].address == (:state, 2)
    @test unconstrained_summary.parameters[1].address == (:state, 1)
    @test unconstrained_summary.parameters[2].address == (:state, 2)
end

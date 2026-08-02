@testset "dist_dirichlet" begin
    @test UncertainTea.logpdf(dirichlet([2.0, 3.0, 4.0]), [0.2, 0.3, 0.5]) ≈
          (
        UncertainTea.loggamma(9.0) - UncertainTea.loggamma(2.0) - UncertainTea.loggamma(3.0) - UncertainTea.loggamma(4.0) +
        log(0.2) + 2 * log(0.3) + 3 * log(0.5)
    ) atol=1e-8

    @tea static function dirichlet_latent_model()
        weights ~ dirichlet([2.0f0, 3.0f0, 4.0f0])
        return weights
    end

    dirichlet_trace, _ = generate(dirichlet_latent_model, (), choicemap(); rng=MersenneTwister(170))
    dirichlet_spec = modelspec(dirichlet_latent_model)
    dirichlet_plan = executionplan(dirichlet_latent_model)
    dirichlet_backend_report = backend_report(dirichlet_latent_model)
    dirichlet_params = parameter_vector(dirichlet_trace)
    dirichlet_unconstrained = transform_to_unconstrained(dirichlet_trace)
    dirichlet_reconstrained = transform_to_constrained(dirichlet_latent_model, dirichlet_unconstrained)
    dirichlet_choicemap = parameterchoicemap(dirichlet_latent_model, dirichlet_params)

    @test dirichlet_spec.choices[1].rhs.family == :dirichlet
    @test dirichlet_spec.parameter_layout.slots[1].transform isa SimplexTransform
    @test dirichlet_spec.parameter_layout.slots[1].dimension == 2
    @test dirichlet_spec.parameter_layout.slots[1].value_length == 3
    @test dirichlet_plan.steps[1].parameter_slot == 1
    @test parametercount(parameterlayout(dirichlet_latent_model)) == 2
    @test parametervaluecount(parameterlayout(dirichlet_latent_model)) == 3
    @test length(dirichlet_unconstrained) == 2
    @test length(dirichlet_params) == 3
    @test all(>(0.0), dirichlet_params)
    @test sum(dirichlet_params) ≈ 1.0 atol=1e-6
    @test dirichlet_reconstrained ≈ dirichlet_params atol=1e-6
    @test dirichlet_choicemap[:weights] ≈ dirichlet_params atol=1e-6
    @test logjoint(dirichlet_latent_model, dirichlet_params) ≈
          assess(dirichlet_latent_model, (), choicemap((:weights, dirichlet_trace[:weights]))) atol=1e-6
    @test dirichlet_backend_report.supported
    @test isempty(dirichlet_backend_report.issues)
    @test backend_execution_plan(dirichlet_latent_model).steps[1] isa UncertainTea.BackendDirichletChoicePlanStep

    dirichlet_batch_params = hcat(
        dirichlet_unconstrained,
        dirichlet_unconstrained .+ Float64[0.2, -0.1],
        dirichlet_unconstrained .+ Float64[-0.15, 0.05],
    )
    dirichlet_batch_values = batched_logjoint_unconstrained(dirichlet_latent_model, dirichlet_batch_params, (), choicemap())
    dirichlet_batch_gradient =
        batched_logjoint_gradient_unconstrained(dirichlet_latent_model, dirichlet_batch_params, (), choicemap())
    dirichlet_batch_cache = BatchedLogjointGradientCache(dirichlet_latent_model, dirichlet_batch_params, (), choicemap())

    @test dirichlet_batch_values ≈ [
        logjoint_unconstrained(dirichlet_latent_model, dirichlet_batch_params[:, index], (), choicemap()) for index = 1:3
    ] atol=2e-6
    @test dirichlet_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(dirichlet_latent_model, dirichlet_batch_params[:, index], (), choicemap()) for
            index = 1:3
        ]...,
    ) atol=2e-6
    @test !isnothing(dirichlet_batch_cache.backend_cache)
    @test isnothing(dirichlet_batch_cache.flat_cache)
    @test isempty(dirichlet_batch_cache.column_caches)

    dirichlet_chain = first(
        hmc(
            dirichlet_latent_model,
            (),
            choicemap();
            num_samples=6,
            num_warmup=0,
            step_size=0.05,
            num_leapfrog_steps=2,
            initial_params=dirichlet_params,
            rng=MersenneTwister(171),
        ),
    )

    @test size(dirichlet_chain.unconstrained_samples) == (2, 6)
    @test size(dirichlet_chain.constrained_samples) == (3, 6)
    for sample_index = 1:6
        @test all(>(0.0), dirichlet_chain.constrained_samples[:, sample_index])
        @test sum(dirichlet_chain.constrained_samples[:, sample_index]) ≈ 1.0 atol=1e-6
    end

    dirichlet_chains = hmc_chains(
        dirichlet_latent_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=4,
        num_warmup=0,
        step_size=0.05,
        num_leapfrog_steps=2,
        rng=MersenneTwister(172),
    )
    constrained_summary = summarize(dirichlet_chains; space=:constrained)
    unconstrained_summary = summarize(dirichlet_chains; space=:unconstrained)

    @test length(constrained_summary.parameters) == 3
    @test length(unconstrained_summary.parameters) == 2
    @test constrained_summary.parameters[1].address == (:weights, 1)
    @test constrained_summary.parameters[3].address == (:weights, 3)
    @test unconstrained_summary.parameters[1].address == (:weights, 1)
    @test unconstrained_summary.parameters[2].address == (:weights, 2)
end

# --- small-alpha sampling: no exact-zero components at HMC init (issue #101) --
# For shape < 1 the Marsaglia-Tsang boost gamma(shape+1) * u^(1/shape) used to
# underflow to exactly 0.0, and the strictly-positive simplex transform then
# crashed HMC initialization seed-dependently.
@testset "dirichlet_small_alpha_init" begin
    @tea function dirichlet_sa_model()
        w ~ dirichlet(0.01, 0.01, 1.0)
    end
    # The exact failing seed from the issue.
    dirichlet_sa_chain =
        first(hmc(dirichlet_sa_model, (); num_samples=5, num_warmup=5, rng=MersenneTwister(2795)))
    @test dirichlet_sa_chain isa HMCChain

    # Sweep: small-alpha draws are strictly positive and survive the simplex
    # round trip.
    dirichlet_sa_rng = MersenneTwister(4920)
    dirichlet_sa_dist = dirichlet(0.01, 0.01, 1.0)
    dirichlet_sa_t = SimplexTransform(3)
    dirichlet_sa_zeros = 0
    dirichlet_sa_nonfinite = 0
    for _ = 1:20_000
        dirichlet_sa_w = rand(dirichlet_sa_rng, dirichlet_sa_dist)
        dirichlet_sa_zeros += count(iszero, dirichlet_sa_w)
        dirichlet_sa_u = UncertainTea.to_unconstrained(dirichlet_sa_t, dirichlet_sa_w)
        dirichlet_sa_nonfinite += count(!isfinite, dirichlet_sa_u)
    end
    @test dirichlet_sa_zeros == 0
    @test dirichlet_sa_nonfinite == 0

    # Moderate-alpha moments are untouched by the log-space boost (the
    # shape >= 1 sampler path is unchanged): alpha = (2, 3, 5) has mean
    # (0.2, 0.3, 0.5) and var[1] = 2*8/(100*11).
    dirichlet_sa_rng2 = MersenneTwister(4921)
    dirichlet_sa_dmod = dirichlet(2.0, 3.0, 5.0)
    dirichlet_sa_n = 50_000
    dirichlet_sa_mean = zeros(3)
    dirichlet_sa_sq1 = 0.0
    for _ = 1:dirichlet_sa_n
        dirichlet_sa_draw = rand(dirichlet_sa_rng2, dirichlet_sa_dmod)
        dirichlet_sa_mean .+= dirichlet_sa_draw
        dirichlet_sa_sq1 += dirichlet_sa_draw[1]^2
    end
    dirichlet_sa_mean ./= dirichlet_sa_n
    @test maximum(abs.(dirichlet_sa_mean .- [0.2, 0.3, 0.5])) < 0.01
    @test abs(dirichlet_sa_sq1 / dirichlet_sa_n - dirichlet_sa_mean[1]^2 - 16 / 1100) < 0.002

    # If a zero component still reaches the simplex transform, the error names
    # the offending component instead of a bare message.
    dirichlet_sa_err = try
        UncertainTea.to_unconstrained(SimplexTransform(3), [0.5, 0.0, 0.5])
        nothing
    catch dirichlet_sa_caught
        dirichlet_sa_caught
    end
    @test dirichlet_sa_err isa ArgumentError
    @test occursin("component 2 of 3", dirichlet_sa_err.msg)
end

# PR 43: finite marginalized mixture distributions.
# Contract: `mixture(weights, components...)` marginalizes a finite mixture of
# AbstractTeaDistribution components with
#   logpdf(mix, x) = logsumexp_k( log(w_k) + logpdf(component_k, x) ).
# Weights are validated (nonnegative, sum ≈ 1 within 1e-8, one per component,
# at least one component) and may be a literal tuple/vector or any runtime
# expression (e.g. a latent dirichlet simplex). Mixtures work as observations
# for any component families; as latents (parameter slots) they require every
# component to be a real-line location-scale family (normal, laplace, studentt)
# so an IdentityTransform is exact — other component families raise an
# ArgumentError at macro expansion. Mixtures are CPU-reference only and are
# honestly reported unsupported by the GPU backend, but models still run through
# the compiled logjoint and the batched ForwardDiff fallback.
mix_mean(xs) = sum(xs) / length(xs)
mix_std(xs) = sqrt(sum(abs2, xs .- mix_mean(xs)) / (length(xs) - 1))

# --- logpdf against a manual logsumexp reference -------------------------
@testset "mix_logpdf_reference" begin
    mix_d = mixture((0.2, 0.3, 0.5), normal(-1.0, 1.0), normal(0.0, 2.0), normal(3.0, 0.5))
    mix_comps = (normal(-1.0, 1.0), normal(0.0, 2.0), normal(3.0, 0.5))
    mix_w = (0.2, 0.3, 0.5)
    for mix_x in (-2.0, 0.0, 1.5, 4.0, 6.0)
        mix_manual = log(sum(
            mix_w[k] * exp(UncertainTea.logpdf(mix_comps[k], mix_x)) for k = 1:3
        ))
        @test UncertainTea.logpdf(mix_d, mix_x) ≈ mix_manual atol = 1e-12
    end

    # A zero-weight component drops out of the mixture entirely.
    mix_dz = mixture((0.0, 1.0), normal(-5.0, 0.1), normal(2.0, 1.0))
    for mix_x in (-1.0, 2.0, 3.5)
        @test UncertainTea.logpdf(mix_dz, mix_x) ≈
              UncertainTea.logpdf(normal(2.0, 1.0), mix_x) atol = 1e-12
    end
end

# --- weight validation ---------------------------------------------------
@testset "mix_weight_validation" begin
    @test_throws ArgumentError mixture((-0.1, 1.1), normal(0.0, 1.0), normal(1.0, 1.0))
    @test_throws ArgumentError mixture((0.3, 0.3), normal(0.0, 1.0), normal(1.0, 1.0))
    @test_throws ArgumentError mixture((0.3, 0.3, 0.4), normal(0.0, 1.0), normal(1.0, 1.0))
    # Within-tolerance sums are accepted; grossly off sums are not.
    @test mixture((0.5 + 1e-10, 0.5 - 1e-10), normal(0.0, 1.0), normal(1.0, 1.0)) isa
          UncertainTea.MixtureDist
end

# --- sampling matches the mixing proportions and mean --------------------
@testset "mix_rand" begin
    mix_rng = MersenneTwister(20240703)
    mix_dr = mixture((0.3, 0.7), normal(-2.0, 0.3), normal(2.0, 0.3))
    mix_draws = [rand(mix_rng, mix_dr) for _ = 1:30000]
    # Components are cleanly separated around 0, so the sign identifies them.
    mix_frac_low = count(<(0.0), mix_draws) / 30000
    @test isapprox(mix_frac_low, 0.3; atol=0.02)
    @test isapprox(mix_mean(mix_draws), 0.3 * (-2.0) + 0.7 * 2.0; atol=0.05)
end

# --- observation end-to-end ----------------------------------------------
@testset "mix_observation_end_to_end" begin
    @tea static function mix_obs_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ mixture((0.5f0, 0.5f0), normal(mu - 2.0f0, 0.5f0), normal(mu + 2.0f0, 0.5f0))
    end

    # A single latent normal slot; the mixture observation carries no slot.
    mix_layout = parameterlayout(mix_obs_model)
    @test parametercount(mix_layout) == 1
    @test parametervaluecount(mix_layout) == 1
    @test mix_layout.slots[1].transform isa IdentityTransform

    # generate/logjoint agreement over the full joint: `mu` is scored as a
    # latent (from the parameter vector) and only the observation `:y` is
    # conditioned on (constraining `:mu` would make it an observation --
    # docs/constraint-driven-conditioning.md, issue #95).
    mix_full = choicemap((:mu, 0.4f0), (:y, 1.3f0))
    mix_trace, _ = generate(mix_obs_model, (), mix_full; rng=MersenneTwister(11))
    mix_pv = parameter_vector(mix_trace)
    @test mix_trace.log_weight ≈ logjoint(mix_obs_model, mix_pv, (), choicemap((:y, 1.3f0))) atol = 1e-6

    # Batched logjoint (ForwardDiff fallback) matches the per-column value.
    mix_obs = choicemap((:y, 1.3f0))
    mix_u = transform_to_unconstrained(mix_obs_model, [0.4])
    mix_params = hcat(mix_u, mix_u .+ 0.6, mix_u .- 0.8)
    mix_batched = batched_logjoint_unconstrained(mix_obs_model, mix_params, (), mix_obs)
    mix_percol = [
        logjoint_unconstrained(mix_obs_model, mix_params[:, i], (), mix_obs)
        for i = 1:size(mix_params, 2)
    ]
    @test mix_batched ≈ mix_percol atol = 1e-6

    # NUTS produces sane draws on the mixture observation model. The mu
    # posterior is genuinely bimodal -- with y = 1.3 the two components imply
    # posterior modes at 0.8*(y-2) = -0.56 (~98% of the mass) and
    # 0.8*(y+2) = 2.64 (~2%), each with sd ~ 0.45 -- and whether a given chain
    # gets trapped in the far basin during the large-step warmup phase is
    # seed/platform roulette (rhat-threshold re-tunes chased this across the
    # issue #93 and #159 trajectory changes without converging). Assert
    # mode-aware draw-level properties instead, which hold for ANY allocation
    # of chains to modes: every draw lies in one of the two mode
    # neighborhoods (+-1.5 ~ 3.3 posterior sds, disjoint windows), the
    # dominant mode is found and sampled, and the draws are not degenerate.
    mix_chains = nuts_chains(
        mix_obs_model,
        (),
        mix_obs;
        num_chains=4,
        num_samples=400,
        num_warmup=300,
        rng=MersenneTwister(13),
    )
    @test all(isfinite, rhat(mix_chains))
    mix_pooled_draws = vcat(
        (vec(mix_chain.constrained_samples[1, :]) for mix_chain in mix_chains.chains)...,
    )
    mix_low_frac = count(d -> abs(d - (-0.56)) < 1.5, mix_pooled_draws) / length(mix_pooled_draws)
    mix_high_frac = count(d -> abs(d - 2.64) < 1.5, mix_pooled_draws) / length(mix_pooled_draws)
    @test mix_low_frac + mix_high_frac > 0.99
    @test mix_low_frac > 0.15
    @test mix_std(mix_pooled_draws) > 0.2
end

# --- latent mixture prior concentrates the posterior ---------------------
@testset "mix_latent_end_to_end" begin
    @tea static function mix_latent_model()
        x ~ mixture((0.5f0, 0.5f0), normal(-2.0f0, 0.5f0), normal(2.0f0, 0.5f0))
        {:y} ~ normal(x, 0.5f0)
    end

    # The bimodal-prior latent earns an identity parameter slot.
    mix_latent_layout = parameterlayout(mix_latent_model)
    @test parametercount(mix_latent_layout) == 1
    @test mix_latent_layout.slots[1].transform isa IdentityTransform

    # Observing y = 2 should pull the posterior toward the +2 mode.
    mix_latent_obs = choicemap((:y, 2.0f0))
    mix_latent_chain = first(nuts(
        mix_latent_model,
        (),
        mix_latent_obs;
        num_samples=150,
        num_warmup=150,
        rng=MersenneTwister(2024),
    ))
    @test size(mix_latent_chain.constrained_samples, 2) == 150
    @test all(isfinite, mix_latent_chain.constrained_samples)
    @test mix_mean(vec(mix_latent_chain.constrained_samples)) > 1.0
end

# --- dirichlet-weight mixture observation --------------------------------
@testset "mix_dirichlet_weights" begin
    @tea static function mix_dirichlet_model()
        w ~ dirichlet(2.0f0, 2.0f0)
        {:y} ~ mixture(w, normal(-1.0f0, 1.0f0), normal(1.0f0, 1.0f0))
    end

    # The dirichlet latent supplies the runtime weight vector; the mixture
    # observation carries no slot of its own.
    mix_dir_layout = parameterlayout(mix_dirichlet_model)
    @test parametercount(mix_dir_layout) == 1
    @test parametervaluecount(mix_dir_layout) == 2
    @test mix_dir_layout.slots[1].transform isa SimplexTransform

    # generate/logjoint agreement over the full joint: the dirichlet `w` is a
    # latent (from the parameter vector) and only the observation `:y` is
    # conditioned on (docs/constraint-driven-conditioning.md, issue #95).
    mix_dir_full = choicemap((:w, [0.4, 0.6]), (:y, 0.7f0))
    mix_dir_trace, _ = generate(mix_dirichlet_model, (), mix_dir_full; rng=MersenneTwister(21))
    mix_dir_pv = parameter_vector(mix_dir_trace)
    @test mix_dir_trace.log_weight ≈
          logjoint(mix_dirichlet_model, mix_dir_pv, (), choicemap((:y, 0.7f0))) atol = 1e-6

    # NUTS runs finite with the weights flowing in as a differentiable simplex.
    mix_dir_obs = choicemap((:y, 0.7f0))
    mix_dir_chain = first(nuts(
        mix_dirichlet_model,
        (),
        mix_dir_obs;
        num_samples=100,
        num_warmup=100,
        rng=MersenneTwister(23),
    ))
    @test size(mix_dir_chain.constrained_samples, 2) == 100
    @test all(isfinite, mix_dir_chain.constrained_samples)
    for sample_index = 1:100
        @test sum(mix_dir_chain.constrained_samples[:, sample_index]) ≈ 1.0 atol = 1e-6
    end
end

# --- latent mixtures with non-real components are rejected ---------------
@testset "mix_latent_component_rejection" begin
    # Rejection surfaces while resolving the latent's parameter transform.
    @test_throws ArgumentError UncertainTea._supported_distribution_family(
        :(mixture((0.5f0, 0.5f0), gamma(2.0f0, 1.0f0), normal(0.0f0, 1.0f0))),
    )

    # ... and at macro-expansion of a model that declares such a latent.
    mix_bad_model = :(@tea static function mix_bad()
        x ~ mixture((0.5f0, 0.5f0), gamma(2.0f0, 1.0f0), normal(0.0f0, 1.0f0))
        {:o} ~ normal(x, 1.0)
    end)
    @test_throws ArgumentError macroexpand(@__MODULE__, mix_bad_model)

    # Real-line component mixtures remain eligible as latents.
    @test UncertainTea._supported_distribution_family(
        :(mixture((0.5f0, 0.5f0), normal(-2.0f0, 0.5f0), studentt(5.0f0, 2.0f0, 0.5f0))),
    ) === :mixture
end

# --- backend now natively supports all-normal-component mixtures (PR 50) --
@testset "mix_backend_report" begin
    @tea static function mix_report_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ mixture((0.5f0, 0.5f0), normal(mu - 2.0f0, 0.5f0), normal(mu + 2.0f0, 0.5f0))
    end
    mix_report = backend_report(mix_report_model)
    @test mix_report.supported == true
    @test backend_execution_plan(mix_report_model).steps[2] isa
          UncertainTea.BackendMixtureNormalChoicePlanStep

    # A mixture with a non-normal component stays on the fallback.
    @tea static function mix_report_laplace_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ mixture((0.5f0, 0.5f0), normal(mu, 0.5f0), laplace(mu, 0.5f0))
    end
    mix_report_laplace = backend_report(mix_report_laplace_model)
    @test mix_report_laplace.supported == false
    @test any(issue -> occursin("mixture", issue), mix_report_laplace.issues)
end

# Issue #75: docs/dsl.md allows both tuple and vector mixture weights; the
# compiled evaluator materializes a vector literal as Vector{Any}, which the
# builder now narrows before constructing MixtureDist.
@testset "mix_vector_literal_weights" begin
    @tea static function mix_vec_weights_model()
        {:y} ~ mixture([0.5, 0.5], normal(-1.0, 1.0), normal(1.0, 1.0))
    end
    @tea static function mix_tuple_weights_model()
        {:y} ~ mixture((0.5, 0.5), normal(-1.0, 1.0), normal(1.0, 1.0))
    end
    mix_vec_cm = choicemap((:y, 0.0))
    mix_tuple_value = logjoint(mix_tuple_weights_model, Float64[], (), mix_vec_cm)
    @test isfinite(mix_tuple_value)
    @test logjoint(mix_vec_weights_model, Float64[], (), mix_vec_cm) ≈ mix_tuple_value atol = 1e-12
    @test batched_logjoint(mix_vec_weights_model, zeros(0, 1), (), mix_vec_cm)[1] ≈
          mix_tuple_value atol = 1e-12
    # the builder narrows Vector{Any} weights (the compiled vector-literal form)
    mix_any_weights = mixture(Any[0.5, 0.5], normal(-1.0, 1.0), normal(1.0, 1.0))
    @test mix_any_weights.weights isa Vector{Float64}
end

# Issue #76: the weights-sum-to-1 validation holds on every execution path; the
# device mirror degrades to -Inf instead of throwing (exception-free contract).
@testset "mix_weight_validation_paths" begin
    @tea static function mix_bad_weights_model()
        {:y} ~ mixture((0.2, 0.2), normal(-1.0, 1.0), normal(1.0, 1.0))
    end
    mix_bad_cm = choicemap((:y, 0.0))
    @test_throws ArgumentError logjoint(mix_bad_weights_model, Float64[], (), mix_bad_cm)
    @test_throws ArgumentError batched_logjoint(mix_bad_weights_model, zeros(0, 1), (), mix_bad_cm)
    @test_throws ArgumentError UncertainTea._backend_mixture_normal_logpdf(
        (0.2, 0.2), (-1.0, 1.0), (1.0, 1.0), 0.0,
    )
    @test_throws ArgumentError UncertainTea._backend_mixture_normal_logpdf(
        (1.5, -0.5), (-1.0, 1.0), (1.0, 1.0), 0.0,
    )
    @test UncertainTea._device_mixture_normal_logpdf((0.2, 0.2), (-1.0, 1.0), (1.0, 1.0), 0.0) == -Inf
    @test UncertainTea._device_mixture_normal_logpdf((1.5, -0.5), (-1.0, 1.0), (1.0, 1.0), 0.0) == -Inf
    @test UncertainTea._device_mixture_normal_logpdf((0.5, 0.5), (-1.0, 1.0), (1.0, 1.0), 0.0) ≈
          UncertainTea._backend_mixture_normal_logpdf((0.5, 0.5), (-1.0, 1.0), (1.0, 1.0), 0.0) atol = 1e-12
end

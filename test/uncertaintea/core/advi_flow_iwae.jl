# ADVI extensions (issue #235): the affine-coupling normalizing-flow guide
# (guide=:flow) and the importance-weighted objective (elbo=:iwae).
#
# The correlated-Gaussian target has posterior covariance [5 -4; -4 5]/9
# (correlation -0.8) and marginal means E[a]=E[b]=4/9, which mean-field cannot
# represent. The banana target is non-Gaussian, so the flow reaches a strictly
# tighter ELBO than either Gaussian guide. Local mean/cov helpers only (no
# `using Statistics`, per CI hygiene).

@tea static function advi_flow_corr_model()
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ normal(a + b, 0.5)
    return a
end

@tea static function advi_flow_banana_model()
    x1 ~ normal(0.0, 1.0)
    x2 ~ normal(0.0, 1.0)
    {:y} ~ normal(x2 + 0.5 * x1 * x1, 0.5)
    return x1
end

# Mildly non-Gaussian target on which the standard mean-field ELBO is loose, so
# the importance-weighted bound tightens as K grows.
@tea static function advi_iwae_funnel_model()
    v ~ normal(0.0, 1.5)
    x ~ normal(0.0, 1.0)
    {:y} ~ normal(x * exp(0.5 * v), 0.5)
    return v
end

@testset "advi_flow_iwae" begin
    _advi_mean(x) = sum(x) / length(x)
    function _advi_cov(samples)
        n = size(samples, 2)
        column_mean = sum(samples; dims=2) ./ n
        centered = samples .- column_mean
        return centered * transpose(centered) ./ (n - 1)
    end
    _advi_tail(history) = _advi_mean(history[(end-min(200, length(history)-1)):end])

    corr_target = [5.0 -4.0; -4.0 5.0] ./ 9.0
    corr_constraints = choicemap((:y, 1.0))
    banana_constraints = choicemap((:y, 1.0))
    funnel_constraints = choicemap((:y, 1.2))

    @testset "flow_recovers_correlation_meanfield_cannot" begin
        flow = batched_advi(
            advi_flow_corr_model,
            (),
            corr_constraints;
            num_iterations=2000,
            num_particles=48,
            learning_rate=0.025,
            guide=:flow,
            num_flow_layers=6,
            rng=MersenneTwister(5),
        )
        @test flow.guide === :flow
        @test flow.flow !== nothing
        @test flow.best_flow !== nothing
        @test all(isfinite, flow.elbo_history)

        flow_cov = variational_covariance(flow)
        @test size(flow_cov) == (2, 2)
        # captures the negative correlation the mean-field guide sends to zero
        @test flow_cov[1, 2] < -0.3
        @test maximum(abs.(flow_cov .- corr_target)) < 0.15

        meanfield = batched_advi(
            advi_flow_corr_model,
            (),
            corr_constraints;
            num_iterations=2000,
            num_particles=48,
            learning_rate=0.025,
            guide=:meanfield,
            rng=MersenneTwister(5),
        )
        meanfield_cov = variational_covariance(meanfield)
        @test meanfield_cov[1, 2] == 0.0
        # flow is strictly closer to the true correlated covariance
        @test maximum(abs.(flow_cov .- corr_target)) <
              maximum(abs.(meanfield_cov .- corr_target))

        # sampled empirical covariance agrees with the reported (sampled) one
        samples = variational_samples(flow; num_samples=6000, space=:unconstrained, rng=MersenneTwister(9))
        empirical = _advi_cov(samples)
        @test empirical[1, 2] < -0.2
    end

    @testset "flow_beats_gaussian_guides_on_nongaussian_target" begin
        tails = Dict{Symbol,Float64}()
        for guide in (:meanfield, :fullrank, :flow)
            result = batched_advi(
                advi_flow_banana_model,
                (),
                banana_constraints;
                num_iterations=2000,
                num_particles=48,
                learning_rate=0.02,
                guide=guide,
                num_flow_layers=6,
                rng=MersenneTwister(11),
            )
            @test all(isfinite, result.standard_elbo_history)
            tails[guide] = _advi_tail(result.standard_elbo_history)
        end
        # the flow's ELBO is strictly higher (tighter) than both Gaussian guides
        @test tails[:flow] > tails[:meanfield] + 0.1
        @test tails[:flow] > tails[:fullrank] + 0.1
    end

    @testset "iwae_tightens_the_bound" begin
        bounds = Float64[]
        for K in (1, 2, 4)
            result = batched_advi(
                advi_iwae_funnel_model,
                (),
                funnel_constraints;
                num_iterations=1500,
                num_particles=32,
                learning_rate=0.02,
                guide=:meanfield,
                elbo=:iwae,
                iwae_samples=K,
                rng=MersenneTwister(7),
            )
            @test result.elbo === :iwae
            @test result.iwae_samples == K
            @test length(result.standard_elbo_history) == length(result.elbo_history)
            push!(bounds, _advi_tail(result.elbo_history))
        end
        # L_1 < L_2 < L_4: strictly tighter bound with more importance samples
        @test bounds[1] < bounds[2]
        @test bounds[2] < bounds[3]
    end

    @testset "iwae_recovers_posterior_mean" begin
        # meanfield captures the (symmetric) posterior means E[a]=E[b]=4/9;
        # the IWAE objective must still recover them.
        result = batched_advi(
            advi_flow_corr_model,
            (),
            corr_constraints;
            num_iterations=1200,
            num_particles=32,
            learning_rate=0.03,
            guide=:meanfield,
            elbo=:iwae,
            iwae_samples=4,
            rng=MersenneTwister(7),
        )
        mean_unconstrained = variational_mean(result; space=:unconstrained)
        @test maximum(abs.(mean_unconstrained .- (4.0 / 9.0))) < 0.15
    end

    @testset "iwae_fullrank_supported" begin
        result = batched_advi(
            advi_flow_corr_model,
            (),
            corr_constraints;
            num_iterations=1500,
            num_particles=32,
            learning_rate=0.02,
            guide=:fullrank,
            elbo=:iwae,
            iwae_samples=4,
            rng=MersenneTwister(5),
        )
        @test result.guide === :fullrank
        @test result.elbo === :iwae
        @test all(isfinite, result.elbo_history)
        covariance = variational_covariance(result)
        # the full-rank IWAE fit still captures the negative correlation
        @test covariance[1, 2] < 0.0
    end

    @testset "determinism_under_seed" begin
        flow_a = batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=200, num_particles=32, guide=:flow, rng=MersenneTwister(3),
        )
        flow_b = batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=200, num_particles=32, guide=:flow, rng=MersenneTwister(3),
        )
        @test flow_a.elbo_history == flow_b.elbo_history
        @test flow_a.best_flow.params == flow_b.best_flow.params

        iwae_a = batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=200, num_particles=32, guide=:meanfield, elbo=:iwae,
            iwae_samples=4, rng=MersenneTwister(3),
        )
        iwae_b = batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=200, num_particles=32, guide=:meanfield, elbo=:iwae,
            iwae_samples=4, rng=MersenneTwister(3),
        )
        @test iwae_a.elbo_history == iwae_b.elbo_history
    end

    @testset "standard_paths_unchanged" begin
        # The standard-ELBO defaults keep the original struct fields and record
        # standard_elbo_history identical to elbo_history.
        result = batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=100, num_particles=32, guide=:meanfield, rng=MersenneTwister(3),
        )
        @test result.elbo === :standard
        @test result.iwae_samples == 1
        @test result.flow === nothing
        @test result.standard_elbo_history == result.elbo_history
    end

    @testset "extension_argument_validation" begin
        @test_throws ArgumentError batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=4, guide=:meanfield, elbo=:bogus,
        )
        # IWAE unsupported for the redundant lowrank reparameterization
        @test_throws ArgumentError batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=4, guide=:lowrank, elbo=:iwae,
        )
        # iwae_samples must divide num_particles
        @test_throws ArgumentError batched_advi(
            advi_flow_corr_model, (), corr_constraints;
            num_iterations=4, num_particles=32, guide=:meanfield, elbo=:iwae, iwae_samples=5,
        )
        # the flow guide needs at least two latent dimensions
        @test_throws ArgumentError batched_advi(
            gaussian_mean, (), constraints;
            num_iterations=4, guide=:flow,
        )
    end
end

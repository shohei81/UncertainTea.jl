# Simulation-based calibration harness (issue #18). Fast seeded smoke runs
# only -- release-grade validation lives in bench/sbc_validation.jl.

@tea static function sbc_conjugate_model()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

@tea static function sbc_scale_model()
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    {:y} ~ normal(mu, sigma)
    return mu
end

# issue #226: exercise the constrained-transform Jacobian layer with rank
# calibration -- the bounded (Logit), simplex (stick-breaking), and correlation
# (Cholesky tanh) transforms had no SBC coverage, and a systematically wrong
# log-Jacobian biases the posterior in a way pointwise logpdf/gradient tests
# (which share the transform code with the oracle) cannot see.

# Bounded: beta latent through the Logit transform, bernoulli observations.
@tea static function sbc_bounded_model(n)
    p ~ beta(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ bernoulli(p)
    end
    return p
end

# Simplex: dirichlet latent through the stick-breaking transform, categorical
# observations.
@tea static function sbc_simplex_model(n)
    theta ~ dirichlet([2.0, 3.0, 4.0])
    for i = 1:n
        {:c => i} ~ categorical(theta)
    end
    return theta
end

# Correlation: lkjcholesky latent through the CholeskyCorrTransform (tanh rows),
# unit-scaled into an mvnormaldense scale_tril. Kept at d = 2 so the fast suite
# stays cheap; the packed L[1,1] == 1 diagonal is a point-mass coordinate that
# sbc reports as p = NaN (issue #226), so only the free correlation entry and
# its derived partner are calibrated.
const sbc_corr_zeros2 = [0.0, 0.0]
const sbc_corr_ones2 = [1.0, 1.0]
@tea static function sbc_corr_model(zeros2_arg, ones2_arg, n)
    Omega ~ lkjcholesky(2, 2.0)
    Ltril = scale_cholesky(ones2_arg, Omega)
    for i = 1:n
        {:y => i} ~ mvnormaldense(zeros2_arg, Ltril)
    end
    return Omega
end

@testset "sbc_calibration" begin
    @testset "sbc_uniformity_checker" begin
        # near-uniform ranks over 0:24 -> comfortable p-value
        uniform_ranks = repeat(0:24, 4)
        @test UncertainTea._sbc_uniformity_pvalue(uniform_ranks, 24, 5) > 0.5
        # all mass on one rank -> vanishing p-value
        @test UncertainTea._sbc_uniformity_pvalue(fill(2, 100), 24, 5) < 1e-12
        # uneven binning must not false-alarm: 6 possible ranks, 20 requested
        # bins (capped to 6), 4 requested bins (6 ranks -> 2+1+1+2 per bin)
        balanced = repeat(0:5, 20)
        @test UncertainTea._sbc_uniformity_pvalue(balanced, 5, 20) > 0.5
        @test UncertainTea._sbc_uniformity_pvalue(balanced, 5, 4) > 0.5
    end

    @testset "sbc_conjugate_passes" begin
        result = sbc(
            sbc_conjugate_model;
            num_simulations=60,
            num_posterior_draws=24,
            num_warmup=40,
            rng=MersenneTwister(42),
        )
        @test size(result.ranks) == (1, 60)
        @test all(0 .<= result.ranks .<= 24)
        @test result.num_posterior_draws == 24
        @test length(result.parameter_names) == 1
        @test occursin("mu", result.parameter_names[1])
        @test result.pvalues[1] > 0.01
        @test !has_warnings(result)
        shown = repr(MIME"text/plain"(), result)
        @test occursin("SBCResult", shown)
        @test occursin("no warnings", shown)
    end

    @testset "sbc_transformed_parameter" begin
        # 128 simulations: the rank-uniformity chi-square p-value is high
        # variance at 50 simulations, so the metric-aware U-turn / invalid-
        # subtree fixes (which shifted the seeded posterior draws) tipped one
        # marginal p-value below the 0.01 false-alarm guard on the 1.10 CI
        # entry. More simulations stabilize the histogram; both parameters
        # clear 0.01 with margin on 1.10 and latest.
        result = sbc(
            sbc_scale_model;
            num_simulations=128,
            num_posterior_draws=24,
            num_warmup=80,
            thin=2,
            rng=MersenneTwister(7),
        )
        @test size(result.ranks) == (2, 128)
        @test all(0 .<= result.ranks .<= 24)
        @test all(result.pvalues .> 0.01)
        @test !has_warnings(result)
    end

    @testset "sbc_detects_broken_sampler" begin
        # no adaptation and an absurd step size freeze the chain, so the true
        # parameter's rank concentrates at the extremes
        result = sbc(
            sbc_conjugate_model;
            num_simulations=60,
            num_posterior_draws=24,
            num_warmup=0,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            step_size=25.0,
            rng=MersenneTwister(42),
        )
        @test result.pvalues[1] < 1e-12
        @test has_warnings(result)
        @test occursin("deviates from uniform", result.warnings[1])
    end

    @testset "sbc_explicit_observations_and_errors" begin
        explicit = sbc(
            sbc_conjugate_model;
            num_simulations=12,
            num_posterior_draws=12,
            num_warmup=30,
            observation_addresses=[(:y,)],
            rng=MersenneTwister(3),
        )
        @test size(explicit.ranks) == (1, 12)

        @test_throws ArgumentError sbc(
            sbc_conjugate_model;
            num_simulations=0,
            num_posterior_draws=8,
            num_warmup=10,
        )
        @test_throws ArgumentError sbc(
            sbc_conjugate_model;
            num_simulations=4,
            num_posterior_draws=8,
            num_warmup=10,
            thin=0,
        )
    end

    @testset "sbc_bounded_transform" begin
        # Logit transform: the beta latent's rank must stay uniform.
        result = sbc(
            sbc_bounded_model,
            (4,);
            num_simulations=80,
            num_posterior_draws=24,
            num_warmup=60,
            rng=MersenneTwister(11),
        )
        @test size(result.ranks) == (1, 80)
        @test all(0 .<= result.ranks .<= 24)
        @test occursin("p", result.parameter_names[1])
        @test result.pvalues[1] > 0.01
        @test !has_warnings(result)
    end

    @testset "sbc_simplex_transform" begin
        # Stick-breaking transform: every dirichlet component's rank uniform.
        result = sbc(
            sbc_simplex_model,
            (6,);
            num_simulations=96,
            num_posterior_draws=24,
            num_warmup=80,
            rng=MersenneTwister(12),
        )
        @test size(result.ranks) == (3, 96)
        @test all(0 .<= result.ranks .<= 24)
        @test all(result.pvalues .> 0.01)
        @test !has_warnings(result)
    end

    @testset "sbc_correlation_transform" begin
        # CholeskyCorrTransform: the free correlation entry (packed index 2) and
        # its derived diagonal (index 3) must calibrate; the unit diagonal
        # (index 1, L[1,1] == 1) is a point-mass coordinate reported as NaN.
        result = sbc(
            sbc_corr_model,
            (sbc_corr_zeros2, sbc_corr_ones2, 8);
            num_simulations=96,
            num_posterior_draws=24,
            num_warmup=80,
            rng=MersenneTwister(13),
        )
        @test size(result.ranks) == (3, 96)
        @test isnan(result.pvalues[1]) # structural unit diagonal, not calibratable
        @test result.pvalues[2] > 0.01 # the free correlation
        @test result.pvalues[3] > 0.01
        @test !has_warnings(result)
        shown = repr(MIME"text/plain"(), result)
        @test occursin("n/a (constant prior)", shown)
    end

    @testset "sbc_constant_prior_not_flagged" begin
        # A point-mass prior coordinate must never false-alarm even when its rank
        # is degenerately pinned (issue #226): the unit diagonal above pins every
        # rank to 0, which a naive uniformity test would flag as p ~ 0.
        result = sbc(
            sbc_corr_model,
            (sbc_corr_zeros2, sbc_corr_ones2, 8);
            num_simulations=48,
            num_posterior_draws=24,
            num_warmup=60,
            rng=MersenneTwister(21),
        )
        @test all(result.ranks[1, :] .== 0) # the constant coordinate is pinned
        @test isnan(result.pvalues[1])
        @test !any(occursin("Omega[1]", w) for w in result.warnings)
    end
end

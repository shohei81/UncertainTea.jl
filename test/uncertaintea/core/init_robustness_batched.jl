# Initialization robustness for batched NUTS (issue #162): a non-finite
# starting point is re-drawn instead of throwing outright (only for re-drawable
# inits -- prior/uniform/Pathfinder), and `init=:uniform` offers the Stan/NumPyro
# Uniform(-2, 2) unconstrained protocol.

# Conjugate gaussian, always-finite everywhere.
@tea static function initr_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end
initr_gauss_cm = choicemap((:y, 0.3))

# A wide prior on an unconstrained latent whose likelihood scale overflows:
# sigma = exp(logscale) -> Inf once logscale exceeds ~709.78, which drives the
# initial logjoint/gradient non-finite for the far-out prior draws. At 32 chains
# under MersenneTwister(14) exactly two chains (6, 10) start non-finite, so the
# retry loop is exercised deterministically.
@tea static function initr_overflow()
    logscale ~ normal(0.0, 400.0)
    {:y} ~ normal(0.0, exp(logscale))
    return logscale
end
initr_overflow_cm = choicemap((:y, 0.0))

@testset "init_robustness_batched" begin
    @testset "helper_nonfinite_columns" begin
        logjoint = [0.0, Inf, -1.0, NaN, 2.0]
        gradient = [1.0 2.0 3.0 4.0 -Inf]  # 1 x 5; column 5 gradient non-finite
        @test UncertainTea._nonfinite_init_columns(logjoint, gradient) == [2, 4, 5]
        @test UncertainTea._nonfinite_init_columns([0.0, 1.0], [1.0 2.0]) == Int[]
    end

    @testset "helper_redrawable" begin
        @test UncertainTea._init_is_redrawable(nothing)
        @test !UncertainTea._init_is_redrawable(zeros(1, 4))
        @test !UncertainTea._init_is_redrawable([0.1, 0.2])
    end

    @testset "uniform_init_bounds" begin
        # Every unconstrained coordinate must land in [-2, 2) (open above).
        positions = UncertainTea._initial_batched_hmc_positions(
            initr_gauss, (), initr_gauss_cm, nothing, MersenneTwister(3), 1, 1, 64; init=:uniform,
        )
        @test all(-2.0 .<= positions .< 2.0)
        # Not all identical -- the draw is actually random per chain/coordinate.
        @test length(unique(positions)) > 1
    end

    @testset "prior_default_unchanged" begin
        # The default path (:prior) must consume the RNG and produce positions
        # identically to the pre-#162 signature (no init kwarg).
        a = UncertainTea._initial_batched_hmc_positions(
            initr_gauss, (), initr_gauss_cm, nothing, MersenneTwister(7), 1, 1, 8,
        )
        b = UncertainTea._initial_batched_hmc_positions(
            initr_gauss, (), initr_gauss_cm, nothing, MersenneTwister(7), 1, 1, 8; init=:prior,
        )
        @test a == b
    end

    @testset "argument_validation" begin
        @test_throws ArgumentError batched_nuts(
            initr_gauss, (), initr_gauss_cm; num_chains=2, num_samples=2, num_warmup=1, init=:bogus,
        )
        @test_throws ArgumentError batched_nuts(
            initr_gauss, (), initr_gauss_cm; num_chains=2, num_samples=2, num_warmup=1, init_max_retries=-1,
        )
    end

    @testset "fixed_nonfinite_fails_fast" begin
        # A fixed initial_params array is not re-drawable, so a non-finite value
        # throws immediately rather than looping to init_max_retries.
        @test_throws ArgumentError batched_nuts(
            initr_gauss, (), initr_gauss_cm;
            num_chains=2, num_samples=2, num_warmup=1,
            initial_params=fill(Inf, 1, 2),
        )
    end

    @testset "retry_recovers_nonfinite_prior" begin
        # retries disabled: the two far-out chains make init throw.
        @test_throws ArgumentError batched_nuts(
            initr_overflow, (), initr_overflow_cm;
            num_chains=32, num_samples=3, num_warmup=3,
            init_max_retries=0, rng=MersenneTwister(14),
        )
        # default retries: the non-finite chains are re-drawn and sampling runs.
        chains = batched_nuts(
            initr_overflow, (), initr_overflow_cm;
            num_chains=32, num_samples=3, num_warmup=3,
            rng=MersenneTwister(14),
        )
        @test chains isa UncertainTea.HMCChains
    end

    @testset "uniform_init_samples_finite" begin
        chains = batched_nuts(
            initr_gauss, (), initr_gauss_cm;
            num_chains=4, num_samples=20, num_warmup=20,
            init=:uniform, rng=MersenneTwister(11),
        )
        @test chains isa UncertainTea.HMCChains
    end
end

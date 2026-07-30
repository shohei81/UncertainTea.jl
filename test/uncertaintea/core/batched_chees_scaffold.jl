# ChEES-HMC scaffold (issue #161, increment 1). `batched_chees` is Halton-jittered
# fixed-length HMC with a shared dual-averaging step size + pooled diagonal mass;
# the cross-chain ChEES trajectory-length adaptation is deferred to increment 2
# (see docs/chees-hmc.md). These tests pin the scaffold: it samples the conjugate
# gauss posterior, is deterministic under a seed, validates its arguments, and the
# Halton generator emits the exact van der Corput sequence.

@testset "batched_chees scaffold" begin
    @tea static function chees_conjugate()
        mu ~ normal(0.0, 1.0)
        {:y} ~ normal(mu, 1.0)
        return mu
    end
    chees_constraints = choicemap((:y, 0.3))
    # Posterior for mu | y=0.3 is N(0.15, 0.5): mean 0.15, variance 0.5.

    @testset "halton base-2 van der Corput" begin
        expected = [1 / 2, 1 / 4, 3 / 4, 1 / 8, 5 / 8, 3 / 8, 7 / 8, 1 / 16, 9 / 16, 5 / 16]
        for (index, value) in enumerate(expected)
            @test UncertainTea._halton_base2(index) == value
        end
        @test_throws ArgumentError UncertainTea._halton_base2(0)
    end

    @testset "gauss posterior recovery" begin
        chees = batched_chees(
            chees_conjugate,
            (),
            chees_constraints;
            num_chains=8,
            num_samples=2000,
            num_warmup=500,
            rng=MersenneTwister(161),
        )
        summary = summarize(chees)
        @test summary[1].mean ≈ 0.15 atol = 0.05
        @test summary[1].sd ≈ sqrt(0.5) atol = 0.08
        @test summary[1].rhat < 1.1
    end

    @testset "determinism under a seed" begin
        run_chees() = batched_chees(
            chees_conjugate,
            (),
            chees_constraints;
            num_chains=4,
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(2718),
        )
        first_run = posterior_array(run_chees(); space=:unconstrained)
        second_run = posterior_array(run_chees(); space=:unconstrained)
        @test first_run == second_run
    end

    @testset "jitter_amount=0 fixed-length smoke" begin
        # With no jitter every iteration integrates exactly num_leapfrog_steps and
        # the sampler still recovers the posterior.
        chees0 = batched_chees(
            chees_conjugate,
            (),
            chees_constraints;
            num_chains=8,
            num_samples=1500,
            num_warmup=400,
            num_leapfrog_steps=12,
            jitter_amount=0.0,
            rng=MersenneTwister(99),
        )
        @test all(steps -> all(==(12), steps), (c.integration_steps for c in chees0.chains))
        summary0 = summarize(chees0)
        @test summary0[1].mean ≈ 0.15 atol = 0.06
        @test summary0[1].sd ≈ sqrt(0.5) atol = 0.1
    end

    @testset "jitter varies trajectory length" begin
        chees_jit = batched_chees(
            chees_conjugate,
            (),
            chees_constraints;
            num_chains=2,
            num_samples=64,
            num_warmup=0,
            num_leapfrog_steps=10,
            jitter_amount=1.0,
            rng=MersenneTwister(7),
        )
        steps = chees_jit.chains[1].integration_steps
        @test all(>=(10), steps)
        @test any(>(10), steps)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError batched_chees(
            chees_conjugate, (), chees_constraints;
            num_chains=2, num_samples=10, jitter_amount=1.5,
        )
        @test_throws ArgumentError batched_chees(
            chees_conjugate, (), chees_constraints;
            num_chains=2, num_samples=10, jitter_amount=-0.1,
        )
        @test_throws ArgumentError batched_chees(
            chees_conjugate, (), chees_constraints;
            num_chains=2, num_samples=10, init=:bogus,
        )
        # Device backend is increment 4 and not yet supported.
        @test_throws ArgumentError batched_chees(
            chees_conjugate, (), chees_constraints;
            num_chains=2, num_samples=10, backend=:some_backend,
        )
    end
end

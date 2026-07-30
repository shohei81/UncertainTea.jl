# ChEES-HMC scaffold (issue #161). `batched_chees` is Halton-jittered HMC with a
# shared dual-averaging step size + pooled diagonal mass; the cross-chain ChEES
# trajectory-length adaptation (increment 2) is exercised separately in
# batched_chees_adaptation.jl. These tests pin the scaffold surface: it samples the
# conjugate gauss posterior, is deterministic under a seed, validates its arguments,
# and the Halton generator emits the exact van der Corput sequence.
#
# NOTE (increment 2): the jittered step count is now
# `L_iter = max(1, ceil(jitter_val * T / ε))` with
# `jitter_val = halton(i)*jitter_amount + (1 - jitter_amount)` and `T` the adapted
# trajectory TIME, matching BlackJAX. This REPLACED the increment-1 formula
# `floor(num_leapfrog_steps * (1 + jitter_amount * h))`; the jitter tests below are
# written against the new (ceil / fraction-of-full) semantics.

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
        # With no jitter jitter_val == 1, so L_iter = ceil(T/ε). Freezing both T
        # (adapt_trajectory_length=false) and ε (adapt_step_size=false) keeps
        # T/ε == num_leapfrog_steps exactly, so every iteration integrates exactly
        # num_leapfrog_steps and the sampler still recovers the posterior.
        chees0 = batched_chees(
            chees_conjugate,
            (),
            chees_constraints;
            num_chains=8,
            num_samples=1500,
            num_warmup=400,
            num_leapfrog_steps=12,
            jitter_amount=0.0,
            adapt_trajectory_length=false,
            adapt_step_size=false,
            rng=MersenneTwister(99),
        )
        @test all(steps -> all(==(12), steps), (c.integration_steps for c in chees0.chains))
        summary0 = summarize(chees0)
        @test summary0[1].mean ≈ 0.15 atol = 0.06
        @test summary0[1].sd ≈ sqrt(0.5) atol = 0.1
    end

    @testset "jitter varies trajectory length" begin
        # New semantics: jitter takes a fraction in [1-jitter_amount, 1) of the
        # full trajectory, so with jitter_amount=1 and no warmup (T frozen at
        # num_leapfrog_steps*ε), L_iter = max(1, ceil(jitter_val * num_leapfrog))
        # ranges over 1..num_leapfrog and varies iteration to iteration.
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
        @test all(>=(1), steps)
        @test all(<=(10), steps)
        @test any(<(10), steps)
        @test length(unique(steps)) > 1
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

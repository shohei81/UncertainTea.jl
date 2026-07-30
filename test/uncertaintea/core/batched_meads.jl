# MEADS -- Maximum-Eigenvalue Adaptation of Damping and Step-size (issue #233).
# These tests pin `batched_meads`: the largest-eigenvalue trace-ratio estimator
# (paper Algorithm 2), the persistent-slice drift (Algorithm 1), argument
# validation, and the statistical guarantees -- posterior recovery on a conjugate
# Gaussian and an ill-conditioned diagonal Gaussian (mean/sd, rank-normalized split
# R-hat, low divergence), continuous self-tuning of step/damping/mass (no warmup
# phase), determinism under a seed, and SBC rank-uniformity. The adaptation matches
# Hoffman & Sountsov, "Tuning-Free Generalized Hamiltonian Monte Carlo" (AISTATS
# 2022); see docs/chees-hmc.md.
#
# CI hygiene: NO `using Statistics`; the sampler's own summarize/ess/rhat provide
# the moments and diagnostics. Chains/draws are kept modest.

@testset "batched_meads" begin
    @tea static function meads_conjugate()
        mu ~ normal(0.0, 1.0)
        {:y} ~ normal(mu, 1.0)
        return mu
    end
    conj_constraints = choicemap((:y, 0.3))
    # Posterior for mu | y=0.3 is N(0.15, 0.5): mean 0.15, sd sqrt(0.5).

    # Diagonal Gaussian whose marginal scales span 100x (0.1 .. 10). No data, so the
    # posterior equals the prior; a good sampler must mix all scales at once.
    @tea static function meads_illconditioned()
        a ~ normal(0.0, 1.0)
        b ~ normal(0.0, 10.0)
        c ~ normal(0.0, 0.1)
        return (a, b, c)
    end

    @testset "largest-eigenvalue trace-ratio estimator" begin
        # Fewer than two members -> no information -> 0.0.
        @test UncertainTea._meads_max_eig(reshape([1.0, 2.0], 2, 1)) == 0.0
        @test UncertainTea._meads_max_eig(Matrix{Float64}(undef, 3, 0)) == 0.0
        # Rank-1 ensemble (all columns equal to v): the only nonzero eigenvalue of
        # the outer-product second moment is ||v||^2, which the ratio recovers
        # exactly. v = [3, 4] -> ||v||^2 = 25.
        v = [3.0, 4.0]
        columns = hcat(v, v, v, v)
        @test isapprox(UncertainTea._meads_max_eig(columns), 25.0; atol=1e-10)
        # A single dominant axis with tiny orthogonal jitter: ratio ~ the big axis'
        # squared scale. Columns along e1 with magnitudes {2, -2, 2, -2}: S is rank 1
        # again with ||col||^2 = 4, so the estimate is 4.
        e1 = [1.0, 0.0, 0.0]
        big = hcat(2 .* e1, -2 .* e1, 2 .* e1, -2 .* e1)
        @test isapprox(UncertainTea._meads_max_eig(big), 4.0; atol=1e-10)
    end

    @testset "persistent-slice drift stays in [-1, 1)" begin
        @test UncertainTea._meads_slice_drift(0.9, 0.2) ≈ -0.9 atol = 1e-12
        @test UncertainTea._meads_slice_drift(-0.9, 0.0) ≈ -0.9 atol = 1e-12
        for u in (-0.99, -0.5, 0.0, 0.4, 0.999), d in (0.0, 0.1, 0.5, 0.9)
            drifted = UncertainTea._meads_slice_drift(u, d)
            @test -1.0 <= drifted < 1.0
        end
    end

    @testset "argument validation" begin
        @test_throws ArgumentError batched_meads(
            meads_conjugate, (), conj_constraints; num_chains=4, num_samples=10, num_folds=1)
        # num_chains must be >= 2 * num_folds.
        @test_throws ArgumentError batched_meads(
            meads_conjugate, (), conj_constraints; num_chains=6, num_samples=10, num_folds=4)
        @test_throws ArgumentError batched_meads(
            meads_conjugate, (), conj_constraints; num_chains=8, num_samples=0)
        @test_throws ArgumentError batched_meads(
            meads_conjugate, (), conj_constraints; num_chains=8, num_samples=10, init=:bogus)
    end

    @testset "conjugate recovery + continuous self-tuning" begin
        trace = Any[]
        chain = batched_meads(
            meads_conjugate,
            (),
            conj_constraints;
            num_chains=16,
            num_samples=1500,
            num_warmup=400,
            rng=MersenneTwister(233),
            _adapt_trace=trace,
        )
        s = summarize(chain)
        @test s[1].mean ≈ 0.15 atol = 0.05
        @test s[1].sd ≈ sqrt(0.5) atol = 0.08
        @test s[1].rhat < 1.05
        @test minimum(ess(chain)) > 400.0
        # No warmup dual-averaging phase: adaptation runs EVERY iteration (warmup +
        # sampling), so the trace has one entry per total iteration and the adapted
        # step/damping/mass are finite and sensible throughout.
        @test length(trace) == 1900
        @test all(t -> isfinite(t.step) && 0.0 < t.step <= 1.0, trace)
        @test all(t -> isfinite(t.damping) && 0.0 <= t.damping <= 1.0, trace)
        @test all(t -> isfinite(t.max_std) && t.max_std > 0.0, trace)
        # The damping floor 1/(t*eps) decays with t, so late-iteration damping is no
        # larger than early damping (adaptation settles, not blows up).
        @test trace[end].damping <= trace[1].damping + 0.2
    end

    @testset "ill-conditioned recovery (100x scale span)" begin
        trace = Any[]
        chain = batched_meads(
            meads_illconditioned,
            (),
            choicemap();
            num_chains=32,
            num_samples=2000,
            num_warmup=600,
            rng=MersenneTwister(2024),
            _adapt_trace=trace,
        )
        s = summarize(chain; space=:unconstrained)
        for (index, target_sd) in enumerate((1.0, 10.0, 0.1))
            @test abs(s[index].mean) < 0.15 * target_sd
            @test s[index].sd ≈ target_sd rtol = 0.12
            @test s[index].rhat < 1.05
        end
        @test minimum(ess(chain; space=:unconstrained)) > 800.0
        total_divergent = sum(sum(c.divergent) for c in chain.chains)
        @test total_divergent == 0
        # The diagonal preconditioner tracks the widest marginal (sd 10).
        @test trace[end].max_std > 5.0
    end

    @testset "deterministic under a fixed seed" begin
        run() = batched_meads(
            meads_illconditioned,
            (),
            choicemap();
            num_chains=16,
            num_samples=150,
            num_warmup=50,
            rng=MersenneTwister(7),
        )
        a = run()
        b = run()
        @test posterior_array(a; space=:unconstrained) == posterior_array(b; space=:unconstrained)
    end

    @testset "SBC calibration with sampler=:meads" begin
        @tea static function meads_sbc_model()
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu, 1.0)
            return mu
        end
        result = sbc(
            meads_sbc_model;
            num_simulations=150,
            num_posterior_draws=63,
            num_warmup=200,
            thin=2,
            sampler=:meads,
            num_chains=8,
            rng=MersenneTwister(20260731),
        )
        @test !has_warnings(result)
        @test all(p -> p > 1e-3, result.pvalues)
    end
end

# ChEES-HMC trajectory-length adaptation (issue #161, increment 2). These tests
# pin the cross-chain ChEES adaptation added to `batched_chees`: the whitened
# per-chain gradient, the scalar Adam-on-log-T ascent, the harmonic-mean step-size
# target, the jitter/step-count reconciliation, and the statistical guarantees
# (warmup converges T, posterior recovery on conjugate + ill-conditioned targets,
# ESS not degraded vs fixed-L, SBC calibration). The estimator matches
# `blackjax.adaptation.chees_adaptation`; see docs/chees-hmc.md.

@testset "batched_chees adaptation" begin
    @tea static function chees_adapt_conjugate()
        mu ~ normal(0.0, 1.0)
        {:y} ~ normal(mu, 1.0)
        return mu
    end
    conj_constraints = choicemap((:y, 0.3))
    # Posterior for mu | y=0.3 is N(0.15, 0.5): mean 0.15, sd sqrt(0.5).

    # Diagonal Gaussian whose marginal scales span 100x (0.1 .. 10). No data, so the
    # posterior equals the prior; a good sampler must mix all scales at once.
    @tea static function chees_illconditioned()
        a ~ normal(0.0, 1.0)
        b ~ normal(0.0, 10.0)
        c ~ normal(0.0, 0.1)
        return (a, b, c)
    end

    @testset "jitter value + step-count reconciliation" begin
        # jitter_val = halton(i)*amount + (1 - amount) in [1-amount, 1).
        @test UncertainTea._chees_jitter_value(0.0, 1) == 1.0
        @test UncertainTea._chees_jitter_value(0.0, 7) == 1.0
        @test UncertainTea._chees_jitter_value(1.0, 1) == UncertainTea._halton_base2(1)
        for i = 1:20
            v = UncertainTea._chees_jitter_value(0.5, i)
            @test 0.5 <= v <= 1.0
        end
        # jitter_amount = 0 => jitter_val = 1 => L = ceil(T/ε) = num_leapfrog at
        # the initial T = num_leapfrog * ε, EXACTLY (float round-off snapped).
        for L in (1, 3, 10, 12, 37, 100), eps in (0.1, 0.037, 1.6, 0.9)
            T0 = L * eps
            @test UncertainTea._chees_jittered_leapfrog_steps(T0, eps, 0.0, 5) == L
        end
        # ceil semantics for a fractional jitter, clamped to >= 1.
        @test UncertainTea._chees_leapfrog_steps_from_jitter(0.5, 10 * 0.1, 0.1) == 5
        @test UncertainTea._chees_leapfrog_steps_from_jitter(0.31, 10 * 0.1, 0.1) == 4  # ceil(3.1)
        @test UncertainTea._chees_leapfrog_steps_from_jitter(1e-6, 10 * 0.1, 0.1) == 1  # clamped
    end

    @testset "scalar Adam ascent" begin
        # With b1 = 0 (BlackJAX ChEES config) the first-moment estimate is the raw
        # gradient, so a constant positive gradient yields a positive (ascending)
        # update whose magnitude -> learning_rate as the second moment settles.
        adam = UncertainTea._ChEESAdamState(0.5, 0.0, 0.95, 1e-8)
        u1 = UncertainTea._chees_adam_ascend!(adam, 2.0)
        @test u1 > 0
        local u
        for _ = 1:200
            u = UncertainTea._chees_adam_ascend!(adam, 2.0)
        end
        @test isapprox(u, 0.5; atol=1e-3)   # mhat/sqrt(vhat) -> 1, times lr
        # Negative gradient descends log T.
        adam2 = UncertainTea._ChEESAdamState(0.5, 0.0, 0.95, 1e-8)
        @test UncertainTea._chees_adam_ascend!(adam2, -3.0) < 0
    end

    @testset "harmonic-mean acceptance statistic" begin
        # Harmonic mean over non-divergent chains; divergent chains masked out.
        hm = UncertainTea._chees_harmonic_mean_accept([0.5, 1.0, 0.25], Bool[false, false, false])
        @test isapprox(hm, 3 / (2 + 1 + 4); atol=1e-12)
        hm2 = UncertainTea._chees_harmonic_mean_accept([0.5, 0.0], Bool[false, true])
        @test hm2 == 0.5   # the p=0 divergent chain is excluded
        @test UncertainTea._chees_harmonic_mean_accept([0.5], Bool[true]) == 0.0
    end

    @testset "warmup converges T (conjugate)" begin
        trace = Float64[]
        chees = batched_chees(
            chees_adapt_conjugate,
            (),
            conj_constraints;
            num_chains=16,
            num_samples=2000,
            num_warmup=600,
            rng=MersenneTwister(161),
            _trajectory_trace=trace,
        )
        # T moved from its warm-start and settled to a finite, sensible value.
        @test length(trace) == 600
        @test all(isfinite, trace)
        T_final = trace[end]
        @test 0.5 < T_final < 200.0
        # The last quarter of warmup is stable (no runaway / oscillation blow-up).
        tail = trace[(end-149):end]
        @test maximum(tail) / minimum(tail) < 3.0
        summary = summarize(chees)
        @test summary[1].mean ≈ 0.15 atol = 0.05
        @test summary[1].sd ≈ sqrt(0.5) atol = 0.08
        @test summary[1].rhat < 1.05
    end

    @testset "ill-conditioned recovery + T convergence" begin
        trace = Float64[]
        chees = batched_chees(
            chees_illconditioned,
            (),
            choicemap();
            num_chains=32,
            num_samples=2500,
            num_warmup=800,
            rng=MersenneTwister(2024),
            _trajectory_trace=trace,
        )
        @test all(isfinite, trace)
        @test 0.1 < trace[end] < 1000.0
        summary = summarize(chees; space=:unconstrained)
        for (index, target_sd) in enumerate((1.0, 10.0, 0.1))
            @test abs(summary[index].mean) < 0.15 * target_sd
            @test summary[index].sd ≈ target_sd rtol = 0.1
            @test summary[index].rhat < 1.05
        end
        @test minimum(ess(chees; space=:unconstrained)) > 1000.0
    end

    @testset "adaptation does not degrade ESS vs fixed-L" begin
        # Start both at a deliberately over-long fixed length (L=40): fixed-L wastes
        # gradients integrating too far, while adaptation pulls T back to an
        # efficient value. Fair comparison uses ESS PER GRADIENT (matched budget).
        function run_case(adapt)
            r = batched_chees(
                chees_illconditioned,
                (),
                choicemap();
                num_chains=32,
                num_samples=2500,
                num_warmup=800,
                num_leapfrog_steps=40,
                adapt_trajectory_length=adapt,
                rng=MersenneTwister(99),
            )
            min_ess = minimum(ess(r; space=:unconstrained))
            total_grad = sum(sum(c.integration_steps) for c in r.chains)
            return min_ess, total_grad
        end
        fixed_ess, fixed_grad = run_case(false)
        adapt_ess, adapt_grad = run_case(true)
        fixed_eff = fixed_ess / fixed_grad
        adapt_eff = adapt_ess / adapt_grad
        @info "ChEES ESS comparison (ill-conditioned, L0=40)" fixed_min_ess = fixed_ess adapt_min_ess =
            adapt_ess fixed_ess_per_grad = fixed_eff adapt_ess_per_grad = adapt_eff efficiency_ratio =
            adapt_eff / fixed_eff
        # Adaptation must not materially degrade efficiency; here it improves it.
        @test adapt_eff >= 0.9 * fixed_eff
    end

    @testset "adaptation off freezes T and stays deterministic" begin
        run_off() = begin
            trace = Float64[]
            chain = batched_chees(
                chees_adapt_conjugate,
                (),
                conj_constraints;
                num_chains=4,
                num_samples=100,
                num_warmup=150,
                num_leapfrog_steps=10,
                adapt_trajectory_length=false,
                rng=MersenneTwister(7),
                _trajectory_trace=trace,
            )
            return chain, trace
        end
        chain1, trace1 = run_off()
        chain2, trace2 = run_off()
        # T never changes when adaptation is off.
        @test length(unique(round.(trace1; digits=10))) == 1
        # Deterministic under a fixed seed.
        @test posterior_array(chain1; space=:unconstrained) == posterior_array(chain2; space=:unconstrained)
    end

    @testset "SBC calibration with sampler=:chees" begin
        @tea static function chees_sbc_model()
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu, 1.0)
            return mu
        end
        result = sbc(
            chees_sbc_model;
            num_simulations=200,
            num_samples=63,
            num_warmup=200,
            thin=2,
            sampler=:chees,
            num_chains=8,
            rng=MersenneTwister(20260730),
        )
        @test !has_warnings(result)
        @test all(p -> p > 1e-3, result.pvalues)
    end
end

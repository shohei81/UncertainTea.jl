# Issue #148: `_logfactorial_like` computes log-factorials of data counts via
# Float64 `loggamma` instead of summing `log(k)` in the caller's (possibly
# dual) arithmetic. These tests pin equality with the old O(n) summation, the
# exact-cancellation behavior of the binomial coefficient at k=0 / k=n, and
# that gradients through latent rate/p are unchanged (the count term is a data
# constant with zero derivative).
@testset "dist_count_logfactorial" begin
    lf_FD = UncertainTea.ForwardDiff

    # the old O(count) summation, kept here as the reference
    lf_reference = function (value, n::Integer)
        total = zero(value)
        unit = one(value)
        for k = 2:n
            total += log(unit * k)
        end
        return total
    end

    lf_counts = (0, 1, 2, 10, 250, 10_000)

    for n in lf_counts
        lf_new = UncertainTea._logfactorial_like(0.7, n)
        lf_old = lf_reference(0.7, n)
        if n < 2
            @test lf_new == 0.0
            @test lf_old == 0.0
        else
            @test lf_new ≈ lf_old rtol = 1e-12
        end
    end

    # element type follows the caller's arithmetic (Float32 stays Float32,
    # duals stay duals with zero count-derivative)
    @test UncertainTea._logfactorial_like(0.5f0, 7) isa Float32
    @test UncertainTea.logpdf(poisson(2.0f0), 5) isa Float32
    lf_dual = UncertainTea._logfactorial_like(lf_FD.Dual(0.5, 1.0), 12)
    @test lf_FD.value(lf_dual) ≈ lf_reference(0.5, 12) rtol = 1e-12
    @test lf_FD.partials(lf_dual, 1) == 0.0

    # public logpdfs and the backend scalar scorers agree with the values
    # rebuilt from the reference summation
    lf_lambda = 3.25
    lf_trials = 10_000
    lf_p = 0.37
    lf_successes = 2.5
    for k in lf_counts
        lf_poisson_expected = k * log(lf_lambda) - lf_lambda - lf_reference(lf_lambda, k)
        @test UncertainTea.logpdf(poisson(lf_lambda), k) ≈ lf_poisson_expected rtol = 1e-12
        @test UncertainTea._backend_poisson_logpdf(lf_lambda, k) ≈ lf_poisson_expected rtol = 1e-12

        lf_binomial_expected =
            lf_reference(lf_p, lf_trials) - lf_reference(lf_p, k) - lf_reference(lf_p, lf_trials - k) +
            (k == 0 ? zero(lf_p) : k * log(lf_p)) +
            (k == lf_trials ? zero(lf_p) : (lf_trials - k) * log1p(-lf_p))
        @test UncertainTea.logpdf(UncertainTea.binomial(lf_trials, lf_p), k) ≈ lf_binomial_expected rtol = 1e-12
        @test UncertainTea._backend_binomial_logpdf(lf_trials, lf_p, k) ≈ lf_binomial_expected rtol = 1e-12

        lf_negbin_expected =
            UncertainTea.loggamma(k + lf_successes) - UncertainTea.loggamma(lf_successes) -
            lf_reference(lf_p, k) + lf_successes * log(lf_p) + k * log1p(-lf_p)
        @test UncertainTea.logpdf(negativebinomial(lf_successes, lf_p), k) ≈ lf_negbin_expected rtol = 1e-12
        @test UncertainTea._backend_negativebinomial_logpdf(lf_successes, lf_p, k) ≈ lf_negbin_expected rtol = 1e-12
    end

    # binomial coefficient at extreme counts: log n! - log k! - log (n-k)! must
    # cancel exactly at k=0 and k=n even for huge n, and the full logpdf stays
    # the pure tail term
    lf_huge = 10^6
    @test UncertainTea._logbinomial_like(0.3, lf_huge, 0) == 0.0
    @test UncertainTea._logbinomial_like(0.3, lf_huge, lf_huge) == 0.0
    @test UncertainTea.logpdf(UncertainTea.binomial(lf_huge, 0.3), 0) ≈ lf_huge * log1p(-0.3) rtol = 1e-12
    @test UncertainTea.logpdf(UncertainTea.binomial(lf_huge, 0.3), lf_huge) ≈ lf_huge * log(0.3) rtol = 1e-12
    @test isfinite(UncertainTea._logbinomial_like(0.3, lf_huge, lf_huge ÷ 2))

    # gradients through a latent rate: ForwardDiff through the full CPU logjoint
    # must match ForwardDiff through the old-summation reference density
    @tea static function lf_poisson_model()
        rate ~ exponential(1.0f0)
        {:y} ~ poisson(rate)
        return rate
    end

    lf_poisson_constraints = choicemap((:y, 250))
    lf_poisson_reference =
        rate -> -rate + 250 * log(rate) - rate - lf_reference(rate, 250)
    for rate in (0.5, 2.5, 300.0)
        lf_grad_new = lf_FD.derivative(r -> logjoint(lf_poisson_model, [r], (), lf_poisson_constraints), rate)
        lf_grad_old = lf_FD.derivative(lf_poisson_reference, rate)
        @test lf_grad_new ≈ lf_grad_old rtol = 1e-12
        @test lf_grad_new ≈ 250 / rate - 2 rtol = 1e-12
    end

    # gradients through a latent binomial success probability
    @tea static function lf_binomial_model()
        p ~ beta(2.0f0, 2.0f0)
        {:y} ~ binomial(1000, p)
        return p
    end

    lf_binomial_constraints = choicemap((:y, 500))
    lf_binomial_reference =
        p ->
            log(6.0) + log(p) + log1p(-p) + lf_reference(p, 1000) - lf_reference(p, 500) -
            lf_reference(p, 500) + 500 * log(p) + 500 * log1p(-p)
    for p in (0.05, 0.4, 0.95)
        lf_grad_new = lf_FD.derivative(q -> logjoint(lf_binomial_model, [q], (), lf_binomial_constraints), p)
        lf_grad_old = lf_FD.derivative(lf_binomial_reference, p)
        @test lf_grad_new ≈ lf_grad_old rtol = 1e-12
    end
end

# --- issue #345: extreme counts go through the stirlerr/saddle-point core -----
#
# Past ~1e8 counts the naive spellings difference ~n*log(n)-sized loggamma and
# k*log(p) terms; the rounding swamped the O(1) result and even flipped its
# sign (binomial(1e16, 0.5) at n/2 scored +35 against the true -18.65,
# poisson(1e16) at k = lambda scored exactly 0.0). The kernels now switch to
# R-dbinom-style stirlerr/bd0 saddle-point forms above
# `_COUNT_SADDLE_THRESHOLD`; everything below keeps the historical exact path.
# All references here are 256-bit BigFloat evaluations of the naive formula.
@testset "dist_count_extreme_345" begin
    setprecision(BigFloat, 256) do
        cx_pois_ref =
            (lambda, k) -> Float64(big(k) * log(big(lambda)) - big(lambda) - UncertainTea.loggamma(big(k) + 1))
        cx_binom_ref =
            (n, p, k) -> Float64(
                UncertainTea.loggamma(big(n) + 1) - UncertainTea.loggamma(big(k) + 1) -
                UncertainTea.loggamma(big(n) - big(k) + 1) +
                big(k) * log(big(p)) + (big(n) - big(k)) * log1p(-big(p)),
            )
        cx_nb_ref =
            (r, p, k) -> Float64(
                UncertainTea.loggamma(big(k) + big(r)) - UncertainTea.loggamma(big(r)) -
                UncertainTea.loggamma(big(k) + 1) + big(r) * log(big(p)) + big(k) * log1p(-big(p)),
            )

        # binomial(10^12..10^18, 0.5) at k = n/2 and in the tails: a log
        # PROBABILITY, so <= 0, and tight against BigFloat
        for cx_e in (12, 14, 16, 18)
            cx_n = Int(10)^cx_e
            cx_half = cx_n ÷ 2
            cx_sigma = round(Int, sqrt(cx_n / 4))
            for cx_k in (cx_half, cx_half + 3 * cx_sigma, cx_half - 7 * cx_sigma, 1, cx_n - 1)
                cx_lp = UncertainTea.logpdf(UncertainTea.binomial(cx_n, 0.5), cx_k)
                @test cx_lp <= 0
                @test cx_lp ≈ cx_binom_ref(cx_n, 0.5, cx_k) rtol = 1e-8
                @test UncertainTea._backend_binomial_logpdf(cx_n, 0.5, cx_k) == cx_lp
            end
            # the exact k = 0 / k = n tail identities survive at extreme n
            @test UncertainTea.logpdf(UncertainTea.binomial(cx_n, 0.5), 0) ≈ cx_n * log1p(-0.5) rtol = 1e-12
            @test UncertainTea.logpdf(UncertainTea.binomial(cx_n, 0.5), cx_n) ≈ cx_n * log(0.5) rtol = 1e-12
        end

        # an asymmetric p keeps the same accuracy
        @test UncertainTea.logpdf(UncertainTea.binomial(Int(10)^14, 0.3), 3 * Int(10)^13) ≈
              cx_binom_ref(Int(10)^14, 0.3, 3 * Int(10)^13) rtol = 1e-8

        # poisson at k = lambda and a 5-sigma tail
        for cx_lambda in (1.0e12, 1.0e16)
            cx_k = Int(cx_lambda)
            cx_lp = UncertainTea.logpdf(poisson(cx_lambda), cx_k)
            @test cx_lp <= 0
            @test cx_lp ≈ cx_pois_ref(cx_lambda, cx_k) rtol = 1e-8
            cx_tail = cx_k + 5 * round(Int, sqrt(cx_lambda))
            @test UncertainTea.logpdf(poisson(cx_lambda), cx_tail) ≈ cx_pois_ref(cx_lambda, cx_tail) rtol =
                1e-8
        end

        # negativebinomial with the mass centered at 1e12 / 1e16 (p = r/(r+mean)),
        # at k = mean and off-center
        for cx_mean in (1.0e12, 1.0e16), cx_r in (2.5, 50.0)
            cx_p = cx_r / (cx_r + cx_mean)
            for cx_k in (round(Int, cx_mean), round(Int, 1.7 * cx_mean), round(Int, 0.2 * cx_mean))
                cx_lp = UncertainTea.logpdf(negativebinomial(cx_r, cx_p), cx_k)
                @test cx_lp <= 0
                @test cx_lp ≈ cx_nb_ref(cx_r, cx_p, cx_k) rtol = 1e-8
            end
        end

        # counts beyond typemax(Int): integer-valued floats score on the float
        # path instead of throwing a raw InexactError
        cx_huge = 1.0e19
        @test UncertainTea._poisson_count(cx_huge) === 1.0e19
        cx_huge_lp = UncertainTea.logpdf(poisson(cx_huge), cx_huge)
        @test isfinite(cx_huge_lp) && cx_huge_lp <= 0
        @test cx_huge_lp ≈ cx_pois_ref(cx_huge, BigInt(10)^19) rtol = 1e-8
        # 1.5e19 is ALSO integer-valued at Float64 (exactly 15 * 10^18): it is
        # a far-tail mass point -- finite and hugely negative, not -Inf
        cx_far = UncertainTea.logpdf(poisson(cx_huge), 1.5e19)
        @test isfinite(cx_far) && cx_far <= 0
        @test cx_far ≈ cx_pois_ref(cx_huge, BigInt(15) * BigInt(10)^18) rtol = 1e-8
        # non-integer floats keep scoring -Inf
        @test UncertainTea.logpdf(poisson(cx_huge), 2.5) == -Inf
        @test UncertainTea.logpdf(poisson(cx_huge), 1.0e10 + 0.5) == -Inf

        # the log-binomial-coefficient helper itself is accurate on the
        # entropy/stirlerr form for huge n
        @test UncertainTea._logbinomial_like(0.5, Int(10)^12, Int(10)^12 ÷ 2) ≈ Float64(
            UncertainTea.loggamma(big(10)^12 + 1) - 2 * UncertainTea.loggamma(big(5) * big(10)^11 + 1),
        ) rtol = 1e-10
    end

    # below the threshold the historical exact path is untouched bit-for-bit
    @test UncertainTea._backend_poisson_logpdf(3.25, 10_000) ==
          10_000 * log(3.25) - 3.25 - UncertainTea.loggamma(10_001.0)
    @test UncertainTea._logbinomial_like(0.3, 10^6, 10^6 ÷ 2) ==
          UncertainTea.loggamma(10^6 + 1.0) - 2 * UncertainTea.loggamma(10^6 ÷ 2 + 1.0)
end

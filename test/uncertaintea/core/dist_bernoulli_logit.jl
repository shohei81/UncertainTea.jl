# Issue #149: bernoullilogit(eta) scores the logit-parameterized Bernoulli in
# the numerically stable log-scale form `x*eta - log1p(exp(eta))`, so the
# logpdf and its gradient stay finite where `bernoulli(1/(1+exp(-eta)))`
# saturates to -Inf/NaN (|eta| >~ 37 in Float64).

@testset "dist_bernoulli_logit" begin
    logistic(eta) = 1 / (1 + exp(-eta))

    # support handling mirrors bernoulli: Bool or a numeric 0/1, else -Inf
    bl_lp(eta, x) = UncertainTea.logpdf(bernoullilogit(eta), x)
    @test bl_lp(0.7, true) ≈ log(logistic(0.7)) atol = 1e-12
    @test bl_lp(0.7, false) ≈ log(1 - logistic(0.7)) atol = 1e-12
    @test bl_lp(0.7, 1) ≈ log(logistic(0.7)) atol = 1e-12
    @test bl_lp(0.7, 0) ≈ log(1 - logistic(0.7)) atol = 1e-12
    @test bl_lp(0.7, 1.0) ≈ log(logistic(0.7)) atol = 1e-12
    @test bl_lp(0.7, 0.0) ≈ log(1 - logistic(0.7)) atol = 1e-12
    @test bl_lp(0.7, 2) == -Inf
    @test bl_lp(0.7, -1) == -Inf
    @test bl_lp(0.7, 0.5) == -Inf
    @test bl_lp(0.7, NaN) == -Inf
    @test bl_lp(0.7, :yes) == -Inf

    # parity with bernoulli(logistic(eta)) at moderate eta, for both outcomes
    for eta in (-2.0, 0.0, 1.5)
        p = logistic(eta)
        @test bl_lp(eta, true) ≈ UncertainTea.logpdf(bernoulli(p), true) atol = 1e-10
        @test bl_lp(eta, false) ≈ UncertainTea.logpdf(bernoulli(p), false) atol = 1e-10
    end

    # STABILITY: at saturating eta the logpdf and its gradient stay finite,
    # where the sigmoid spelling collapses to -Inf / NaN
    for eta in (-90.0, -40.0, 40.0, 90.0)
        for x in (true, false)
            lp = bl_lp(eta, x)
            @test isfinite(lp)
            g = UncertainTea.ForwardDiff.derivative(e -> UncertainTea.logpdf(bernoullilogit(e), x), eta)
            @test isfinite(g)
            # d/d_eta [x*eta - log1p(exp(eta))] = x - logistic(eta)
            expected = (x ? 1.0 : 0.0) - logistic(eta)
            @test g ≈ expected atol = 1e-9
        end
    end
    # the naive sigmoid form really does collapse where p saturates to 1.0:
    # scoring the `false` outcome gives log1p(-1) = -Inf and a NaN gradient,
    # which is exactly what bernoullilogit avoids.
    for eta in (40.0, 90.0)
        @test logistic(eta) == 1.0
        @test UncertainTea.logpdf(bernoulli(logistic(eta)), false) == -Inf
        @test isfinite(bl_lp(eta, false))
    end

    # model-level: an observation site scores like the interpreter reference
    @tea static function bl_model(eta)
        {:y} ~ bernoullilogit(eta)
    end
    @test assess(bl_model, (1.2,), choicemap((:y, true))) ≈ bl_lp(1.2, true) atol = 1e-10
    @test assess(bl_model, (1.2,), choicemap((:y, 2))) == -Inf
    # bernoullilogit vs bernoulli(logistic) agree at moderate eta (ties #149/#150)
    @tea static function b_model(p)
        {:y} ~ bernoulli(p)
    end
    @test assess(bl_model, (0.3,), choicemap((:y, true))) ≈
          assess(b_model, (logistic(0.3),), choicemap((:y, true))) atol = 1e-10
end

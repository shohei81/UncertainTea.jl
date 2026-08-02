# Zero-inflated count distributions (issue #292): structural zeros mixed with a
# Poisson / NegativeBinomial count. CPU-reference only. Statistics is
# unavailable in the harness, so use local helpers.

zi_mean(x) = sum(x) / length(x)

@tea static function zi_model(n)
    logit_p ~ normal(0.0, 1.5)
    log_lambda ~ normal(0.0, 1.0)
    p = 1 / (1 + exp(-logit_p))
    lambda = exp(log_lambda)
    for i = 1:n
        {:y => i} ~ zeroinflatedpoisson(p, lambda)
    end
    return p
end

@testset "dist_zero_inflated" begin
    @testset "ZIP: normalized, exact zero mass, degenerate limits" begin
        d = zeroinflatedpoisson(0.3, 2.0)
        @test sum(exp(UncertainTea.logpdf(d, k)) for k = 0:80) ≈ 1.0 rtol = 1e-12
        @test exp(UncertainTea.logpdf(d, 0)) ≈ 0.3 + 0.7 * exp(-2.0) rtol = 1e-12
        @test exp(UncertainTea.logpdf(d, 3)) ≈ 0.7 * exp(UncertainTea.logpdf(poisson(2.0), 3)) rtol = 1e-12
        # p = 0 degenerates to the plain Poisson exactly
        @test UncertainTea.logpdf(zeroinflatedpoisson(0.0, 2.0), 3) ==
              UncertainTea.logpdf(poisson(2.0), 3)
        # p = 1 puts all mass at zero
        @test UncertainTea.logpdf(zeroinflatedpoisson(1.0, 2.0), 0) == 0.0
        @test UncertainTea.logpdf(zeroinflatedpoisson(1.0, 2.0), 1) == -Inf
        # off-support
        @test UncertainTea.logpdf(d, -1) == -Inf
        @test UncertainTea.logpdf(d, 2.5) == -Inf
    end

    @testset "ZINB: normalized and consistent with the count component" begin
        dz = zeroinflatednegativebinomial(0.25, 3.0, 0.4)
        @test sum(exp(UncertainTea.logpdf(dz, k)) for k = 0:400) ≈ 1.0 rtol = 1e-10
        @test exp(UncertainTea.logpdf(dz, 5)) ≈
              0.75 * exp(UncertainTea.logpdf(negativebinomial(3.0, 0.4), 5)) rtol = 1e-12
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError zeroinflatedpoisson(-0.1, 2.0)
        @test_throws ArgumentError zeroinflatedpoisson(1.1, 2.0)
        @test_throws ArgumentError zeroinflatedpoisson(0.3, -1.0)
        @test_throws ArgumentError zeroinflatednegativebinomial(0.3, 0.0, 0.4)
        @test_throws ArgumentError zeroinflatednegativebinomial(0.3, 2.0, 0.0)
    end

    @testset "rand inflates zeros" begin
        rng = MersenneTwister(292)
        hi = [rand(rng, zeroinflatedpoisson(0.6, 5.0)) for _ = 1:400]
        lo = [rand(rng, zeroinflatedpoisson(0.05, 5.0)) for _ = 1:400]
        @test all(>=(0), hi)
        @test count(iszero, hi) > count(iszero, lo)
    end

    @testset "backend/device honestly unsupported (CPU-reference only)" begin
        @test !UncertainTea.backend_report(zi_model).supported
        @test !device_lowering_report(zi_model)[1]
    end

    @testset "NUTS recovers the inflation probability and rate" begin
        rng = MersenneTwister(7)
        n = 150
        p_true, lambda_true = 0.35, 3.0
        ys = Float64[rand(rng, zeroinflatedpoisson(p_true, lambda_true)) for _ = 1:n]
        cm = choicemap([(:y => i, ys[i]) for i = 1:n])
        chain = first(nuts(zi_model, (n,), cm; num_samples=400, num_warmup=400, rng=MersenneTwister(11)))
        draws = chain.constrained_samples
        @test all(isfinite, draws)
        p_post = zi_mean(1 ./ (1 .+ exp.(-draws[1, :])))
        lambda_post = zi_mean(exp.(draws[2, :]))
        @test abs(p_post - p_true) < 0.15
        @test abs(lambda_post - lambda_true) < 0.6
    end
end

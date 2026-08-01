# von Mises distribution for circular data (issue #293). CPU-reference only.
# Statistics is unavailable in the harness, so use local helpers.

using SpecialFunctions: besselix

vm_mean(x) = sum(x) / length(x)

@tea static function vm_model(n)
    mu ~ normal(0.0, 2.0)
    log_kappa ~ normal(0.5, 1.0)
    for i = 1:n
        {:y => i} ~ vonmises(mu, exp(log_kappa))
    end
    return mu
end

@testset "dist_von_mises" begin
    @testset "log I0 matches SpecialFunctions across regimes" begin
        for x in (0.05, 1.0, 15.0, 50.0, 299.0, 301.0, 700.0)
            ref = log(besselix(0, x)) + x
            @test UncertainTea._log_besseli0(x) ≈ ref rtol = 1e-12
        end
    end

    @testset "density is normalized, periodic, and unimodal at mu" begin
        d = vonmises(0.5, 2.0)
        xs = range(-pi, pi; length=20001)
        @test sum(exp(UncertainTea.logpdf(d, x)) for x in xs) * step(xs) ≈ 1.0 rtol = 1e-4
        @test UncertainTea.logpdf(d, 0.3) ≈ UncertainTea.logpdf(d, 0.3 + 2pi) rtol = 1e-12
        @test UncertainTea.logpdf(d, 0.5) > UncertainTea.logpdf(d, 0.5 + 1.0)
        @test UncertainTea.logpdf(d, 0.5) > UncertainTea.logpdf(d, 0.5 - 1.0)
        @test_throws ArgumentError vonmises(0.0, -1.0)
    end

    @testset "kappa limits" begin
        # kappa -> 0: approaches the circular uniform 1/(2pi)
        near_uniform = vonmises(0.0, 1e-8)
        @test exp(UncertainTea.logpdf(near_uniform, 1.234)) ≈ 1 / (2pi) rtol = 1e-6
        # large kappa: log-density differences match normal(mu, 1/sqrt(kappa))
        kappa = 400.0
        sharp = vonmises(0.0, kappa)
        delta = UncertainTea.logpdf(sharp, 0.05) - UncertainTea.logpdf(sharp, 0.0)
        @test delta ≈ -kappa * (1 - cos(0.05)) rtol = 1e-12
        @test delta ≈ -0.05^2 * kappa / 2 rtol = 1e-2
    end

    @testset "rand wraps to [-pi, pi) and concentrates around mu" begin
        rng = MersenneTwister(293)
        draws = [rand(rng, vonmises(1.0, 8.0)) for _ = 1:400]
        @test all(x -> -pi <= x < pi, draws)
        # circular mean of concentrated draws sits near mu
        cmean = atan(vm_mean(sin.(draws)), vm_mean(cos.(draws)))
        @test abs(cmean - 1.0) < 0.15
    end

    @testset "backend/device honestly unsupported (CPU-reference only)" begin
        @test !UncertainTea.backend_report(vm_model).supported
        @test !device_lowering_report(vm_model)[1]
    end

    @testset "NUTS recovers the mean direction and concentration" begin
        rng = MersenneTwister(7)
        n = 80
        mu_true, kappa_true = 0.8, 4.0
        ys = Float64[rand(rng, vonmises(mu_true, kappa_true)) for _ = 1:n]
        cm = choicemap([(:y => i, ys[i]) for i = 1:n])
        chain = nuts(vm_model, (n,), cm; num_samples=400, num_warmup=400, rng=MersenneTwister(11))
        draws = chain.constrained_samples
        @test all(isfinite, draws)
        @test abs(vm_mean(draws[1, :]) - mu_true) < 0.3
        @test abs(vm_mean(exp.(draws[2, :])) - kappa_true) < 1.8
    end
end

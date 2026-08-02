# Gaussian process regression family (issue #262): a zero-mean GP marginal
# likelihood with an RBF kernel, scored through the dense multivariate-normal
# marginal. CPU-reference only (dense Cholesky, honestly unsupported on the
# backend/device). Statistics is not available in the harness, so use local
# helpers.

using LinearAlgebra

gp_mean(x) = sum(x) / length(x)

@tea static function gp_hyper_model(X, n)
    logl ~ normal(0.0, 1.0)
    logv ~ normal(0.0, 1.0)
    logn ~ normal(-1.0, 1.0)
    {:y} ~ gaussianprocess(X, exp(logl), exp(logv), exp(logn))
    return logl
end

@testset "dist_gaussian_process" begin
    gp_rng = MersenneTwister(262)
    gp_n = 30
    gp_X = reshape(sort(rand(gp_rng, gp_n) .* 5), 1, gp_n)   # 1 x N
    gp_D2 = [(gp_X[1, i] - gp_X[1, j])^2 for i = 1:gp_n, j = 1:gp_n]
    gp_true_l, gp_true_v, gp_true_noise = 1.0, 1.0, 0.2
    gp_Ktrue = gp_true_v^2 .* exp.(-0.5 .* gp_D2 ./ gp_true_l^2) + gp_true_noise^2 .* Matrix(I, gp_n, gp_n)
    gp_y = cholesky(Symmetric(gp_Ktrue)).L * randn(gp_rng, gp_n)

    @testset "logpdf matches the mvnormal marginal" begin
        gp = gaussianprocess(gp_X, gp_true_l, gp_true_v, gp_true_noise)
        # hand-computed N(0, K) log-density (with the same 1e-8 jitter)
        gp_Kref = gp_true_v^2 .* exp.(-0.5 .* gp_D2 ./ gp_true_l^2) +
                  (gp_true_noise^2 + 1e-8) .* Matrix(I, gp_n, gp_n)
        gp_ref = -0.5 * gp_y' * (gp_Kref \ gp_y) - 0.5 * logdet(gp_Kref) - 0.5 * gp_n * log(2pi)
        @test UncertainTea.logpdf(gp, gp_y) ≈ gp_ref rtol = 1e-10

        # the 1-D vector-input convenience matches the explicit 1 x N matrix
        gp_vec = gaussianprocess(vec(gp_X), gp_true_l, gp_true_v, gp_true_noise)
        @test UncertainTea.logpdf(gp_vec, gp_y) ≈ UncertainTea.logpdf(gp, gp_y) rtol = 1e-12

        # an observation of the wrong length is rejected
        @test_throws ArgumentError UncertainTea.logpdf(gp, gp_y[1:(gp_n-1)])
    end

    @testset "ARD (per-dimension lengthscale)" begin
        ard_rng = MersenneTwister(2731)
        ard_d, ard_n = 4, 25
        ard_X = randn(ard_rng, ard_d, ard_n)
        ard_y = randn(ard_rng, ard_n)

        # an ARD vector of equal lengthscales reproduces the isotropic scalar kernel
        ard_iso = gaussianprocess(ard_X, 0.8, 1.2, 0.3)
        ard_equal = gaussianprocess(ard_X, fill(0.8, ard_d), 1.2, 0.3)
        @test UncertainTea.logpdf(ard_equal, ard_y) ≈ UncertainTea.logpdf(ard_iso, ard_y) rtol = 1e-12

        # a genuine ARD kernel matches the hand-computed N(0, K) density
        ard_l = [0.5, 1.0, 2.0, 4.0]
        ard_v, ard_noise = 1.3, 0.25
        ard_K = [
            ard_v^2 * exp(-0.5 * sum((ard_X[:, i] .- ard_X[:, j]) .^ 2 ./ ard_l .^ 2)) +
            (ard_noise^2 + 1e-8) * (i == j) for i = 1:ard_n, j = 1:ard_n
        ]
        ard_ref = -0.5 * ard_y' * (ard_K \ ard_y) - 0.5 * logdet(ard_K) - 0.5 * ard_n * log(2pi)
        ard_gp = gaussianprocess(ard_X, ard_l, ard_v, ard_noise)
        @test UncertainTea.logpdf(ard_gp, ard_y) ≈ ard_ref rtol = 1e-10

        # an ARD lengthscale of the wrong dimension is rejected
        @test_throws ArgumentError gaussianprocess(ard_X, [1.0, 2.0], ard_v, ard_noise)

        # rand still draws a finite length-N sample under an ARD kernel
        ard_sample = rand(MersenneTwister(1), ard_gp)
        @test length(ard_sample) == ard_n
        @test all(isfinite, ard_sample)
    end

    @testset "rand draws a finite length-N prior sample" begin
        gp = gaussianprocess(gp_X, 1.0, 1.0, 0.3)
        sample = rand(MersenneTwister(1), gp)
        @test length(sample) == gp_n
        @test all(isfinite, sample)
    end

    @testset "backend/device honestly unsupported (CPU-reference only)" begin
        @test !UncertainTea.backend_report(gp_hyper_model).supported
        @test !device_lowering_report(gp_hyper_model)[1]
    end

    @testset "NUTS recovers the hyperparameters" begin
        gp_cm = choicemap((:y, gp_y))
        gp_chain = first(nuts(
            gp_hyper_model, (gp_X, gp_n), gp_cm;
            num_samples=400, num_warmup=400, rng=MersenneTwister(7),
        ))
        gp_draws = gp_chain.constrained_samples
        @test all(isfinite, gp_draws)
        # noise is the best-identified hyperparameter at this N; lengthscale and
        # signal variance are only weakly identified, so gate them loosely.
        @test abs(exp(gp_mean(gp_draws[3, :])) - gp_true_noise) < 0.1   # noise ~ 0.2
        @test 0.3 < exp(gp_mean(gp_draws[1, :])) < 3.0                  # lengthscale in a sane band
        @test 0.3 < exp(gp_mean(gp_draws[2, :])) < 3.0                  # signal sd in a sane band
    end
end

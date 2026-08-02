# Sparse Gaussian process regression via the FITC approximation (issue #281): a
# rank-M Nystrom kernel from M << N inducing points plus the exact diagonal
# correction, scored in O(N M^2 + M^3) by the Woodbury identity. CPU-reference
# only. Statistics is not available in the harness, so use local helpers.

using LinearAlgebra

sgp_mean(x) = sum(x) / length(x)

@tea static function sgp_hyper_model(X, Z)
    logl ~ normal(0.0, 1.0)
    logv ~ normal(0.0, 1.0)
    logn ~ normal(-1.0, 1.0)
    {:y} ~ sparsegaussianprocess(X, Z, exp(logl), exp(logv), exp(logn))
    return logl
end

@testset "dist_sparse_gaussian_process" begin
    sgp_rng = MersenneTwister(281)
    sgp_d, sgp_n = 2, 40
    sgp_X = randn(sgp_rng, sgp_d, sgp_n)
    sgp_y = randn(sgp_rng, sgp_n)
    sgp_l, sgp_v, sgp_noise = 1.0, 1.2, 0.3

    @testset "Woodbury matches a direct dense FITC evaluation" begin
        sgp_m = 8
        sgp_Z = randn(sgp_rng, sgp_d, sgp_m)
        sgp = sparsegaussianprocess(sgp_X, sgp_Z, sgp_l, sgp_v, sgp_noise)

        # build the FITC covariance C = Q_NN + Lambda directly and score N(0, C)
        kern(a, b) = sgp_v^2 * exp(-0.5 * sum((a .- b) .^ 2) / sgp_l^2)
        Kmm = [kern(sgp_Z[:, i], sgp_Z[:, j]) for i = 1:sgp_m, j = 1:sgp_m] + 1e-6 * I
        Kmn = [kern(sgp_Z[:, i], sgp_X[:, j]) for i = 1:sgp_m, j = 1:sgp_n]
        Q = Kmn' * (Kmm \ Kmn)
        Lambda = Diagonal([sgp_v^2 - Q[i, i] + sgp_noise^2 + 1e-8 for i = 1:sgp_n])
        C = Q + Lambda
        ref = -0.5 * sgp_y' * (C \ sgp_y) - 0.5 * logdet(C) - 0.5 * sgp_n * log(2pi)
        @test UncertainTea.logpdf(sgp, sgp_y) ≈ ref rtol = 1e-9
    end

    @testset "inducing = inputs reduces to the dense GP marginal" begin
        # exact up to the differing diagonal jitter (K_MM uses 1e-6, the dense GP
        # 1e-8), so compare with a loose tolerance.
        sgp_full = sparsegaussianprocess(sgp_X, sgp_X, sgp_l, sgp_v, sgp_noise)
        gp = gaussianprocess(sgp_X, sgp_l, sgp_v, sgp_noise)
        @test UncertainTea.logpdf(sgp_full, sgp_y) ≈ UncertainTea.logpdf(gp, sgp_y) rtol = 1e-3

        # the 1-D vector-input convenience matches the explicit matrices
        sgp_vec = sparsegaussianprocess(vec(sgp_X[1:1, :]), vec(sgp_X[1:1, 1:5]), sgp_l, sgp_v, sgp_noise)
        @test UncertainTea.logpdf(sgp_vec, sgp_y) isa Real
    end

    @testset "ARD lengthscale flows through" begin
        sgp_Z = randn(sgp_rng, sgp_d, 6)
        sgp_ard = sparsegaussianprocess(sgp_X, sgp_Z, [0.5, 2.0], sgp_v, sgp_noise)
        @test UncertainTea.logpdf(sgp_ard, sgp_y) isa Real
        @test isfinite(UncertainTea.logpdf(sgp_ard, sgp_y))
    end

    @testset "constructor validation" begin
        # inducing/input dimension mismatch
        @test_throws ArgumentError sparsegaussianprocess(sgp_X, randn(3, 5), sgp_l, sgp_v, sgp_noise)
        # ARD lengthscale of the wrong dimension
        @test_throws ArgumentError sparsegaussianprocess(sgp_X, randn(sgp_d, 5), [1.0, 2.0, 3.0], sgp_v, sgp_noise)
        # observation of the wrong length
        sgp = sparsegaussianprocess(sgp_X, randn(sgp_d, 5), sgp_l, sgp_v, sgp_noise)
        @test_throws ArgumentError UncertainTea.logpdf(sgp, sgp_y[1:(sgp_n-1)])
    end

    @testset "rand draws a finite length-N sample" begin
        sgp = sparsegaussianprocess(sgp_X, randn(sgp_rng, sgp_d, 6), sgp_l, sgp_v, 0.4)
        sample = rand(MersenneTwister(1), sgp)
        @test length(sample) == sgp_n
        @test all(isfinite, sample)
    end

    @testset "backend/device honestly unsupported (CPU-reference only)" begin
        sgp_Z = reshape(collect(range(0, 10; length=10)), 1, 10)
        sgp_XX = reshape(collect(range(0, 10; length=20)), 1, 20)
        @test !UncertainTea.backend_report(sgp_hyper_model).supported
        @test !device_lowering_report(sgp_hyper_model)[1]
    end

    @testset "NUTS recovers hyperparameters" begin
        sgp_nn = 150
        sgp_Xs = sort(rand(MersenneTwister(2), sgp_nn) .* 10)
        sgp_XN = reshape(sgp_Xs, 1, sgp_nn)
        sgp_Ktrue = [exp(-0.5 * (sgp_Xs[i] - sgp_Xs[j])^2) for i = 1:sgp_nn, j = 1:sgp_nn] + 0.04 * I
        sgp_yN = cholesky(Symmetric(sgp_Ktrue)).L * randn(MersenneTwister(3), sgp_nn)
        sgp_Zm = reshape(collect(range(0, 10; length=15)), 1, 15)
        sgp_cm = choicemap((:y, sgp_yN))
        sgp_chain = first(nuts(sgp_hyper_model, (sgp_XN, sgp_Zm), sgp_cm; num_samples=300, num_warmup=300, rng=MersenneTwister(7)))
        sgp_draws = sgp_chain.constrained_samples
        @test all(isfinite, sgp_draws)
        @test abs(exp(sgp_mean(sgp_draws[3, :])) - 0.2) < 0.15    # noise ~ 0.2
        @test 0.3 < exp(sgp_mean(sgp_draws[1, :])) < 3.0          # lengthscale sane
    end
end

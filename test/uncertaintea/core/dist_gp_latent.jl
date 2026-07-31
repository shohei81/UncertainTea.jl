# Direct latent-function GP inference via `gp_cholesky` (issue #279): where
# `gaussianprocess` scores the analytic zero-mean marginal `N(0, K)` (Gaussian
# likelihood only), `gp_cholesky` returns the kernel Cholesky factor so the latent
# function values `f ~ N(0, K)` are sampled directly through `mvnormaldense` and
# can drive any likelihood. CPU-reference. Statistics is unavailable in the
# harness, so use local helpers.

using LinearAlgebra

gpl_mean(x) = sum(x) / length(x)
gpl_cor(a, b) = begin
    ma = gpl_mean(a)
    mb = gpl_mean(b)
    da = a .- ma
    db = b .- mb
    sum(da .* db) / sqrt(sum(abs2, da) * sum(abs2, db))
end

# GP regression written with a LATENT function f + Gaussian likelihood.
@tea static function gpl_regression(X)
    f ~ mvnormaldense((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), gp_cholesky(X, 1.0, 1.0, 1.0e-6))
    for i = 1:10
        {:y => i} ~ normal(f[i], 0.3)
    end
    return f[1]
end

# GP classification: latent f + Bernoulli/logit likelihood (a non-Gaussian
# likelihood the analytic marginal cannot handle).
@tea static function gpl_classification(X)
    logl ~ normal(0.0, 0.5)
    L = gp_cholesky(X, exp(logl), 2.0, 1.0e-6)
    f ~ mvnormaldense((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), L)
    for i = 1:10
        {:y => i} ~ bernoullilogit(f[i])
    end
    return logl
end

@testset "dist_gp_latent" begin
    @testset "gp_cholesky reconstructs the RBF kernel" begin
        gpl_rng = MersenneTwister(279)
        gpl_X = reshape(sort(rand(gpl_rng, 8) .* 4), 1, 8)
        L = gp_cholesky(gpl_X, 0.8, 1.3, 0.1)
        @test istril(L)
        @test all(diag(L) .> 0)
        gpl_Kref =
            1.3^2 .* exp.(-0.5 .* [(gpl_X[1, i] - gpl_X[1, j])^2 for i = 1:8, j = 1:8] ./ 0.8^2) +
            (0.1^2 + 1e-8) .* Matrix(I, 8, 8)
        @test L * L' ≈ gpl_Kref rtol = 1e-10

        # ARD lengthscale flows through the helper too
        gpl_X2 = randn(MersenneTwister(1), 3, 6)
        L2 = gp_cholesky(gpl_X2, [0.5, 1.0, 2.0], 1.0, 0.05)
        @test size(L2) == (6, 6)
        @test istril(L2)
    end

    @testset "latent-f GP regression recovers the analytic GP posterior mean" begin
        gpl_n = 10
        gpl_l, gpl_v, gpl_jit, gpl_sig = 1.0, 1.0, 1e-6, 0.3
        gpl_Xs = collect(range(0, 5; length=gpl_n))
        gpl_X = reshape(gpl_Xs, 1, gpl_n)
        gpl_K = [
            gpl_v^2 * exp(-0.5 * (gpl_Xs[i] - gpl_Xs[j])^2 / gpl_l^2) + (gpl_jit^2 + 1e-8) * (i == j)
            for i = 1:gpl_n, j = 1:gpl_n
        ]
        gpl_rng = MersenneTwister(3)
        gpl_y = cholesky(Symmetric(gpl_K)).L * randn(gpl_rng, gpl_n) .+ gpl_sig .* randn(gpl_rng, gpl_n)
        gpl_cm = choicemap([(:y => i, gpl_y[i]) for i = 1:gpl_n])

        # closed-form GP regression posterior mean of f: K (K + sigma^2 I)^-1 y
        gpl_analytic = gpl_K * ((gpl_K + gpl_sig^2 * I) \ gpl_y)

        gpl_chain = nuts(gpl_regression, (gpl_X,), gpl_cm; num_samples=800, num_warmup=800, rng=MersenneTwister(7))
        gpl_draws = gpl_chain.constrained_samples          # rows 1..10 are f[1..10]
        @test all(isfinite, gpl_draws)
        gpl_fpost = [gpl_mean(gpl_draws[i, :]) for i = 1:gpl_n]
        @test gpl_cor(gpl_fpost, gpl_analytic) > 0.99
        @test maximum(abs.(gpl_fpost .- gpl_analytic)) < 0.15
    end

    @testset "latent-f GP classification runs and separates the classes" begin
        gpl_n = 10
        gpl_Xs = collect(range(-3, 3; length=gpl_n))
        gpl_X = reshape(gpl_Xs, 1, gpl_n)
        # a steep latent -> near-deterministic labels (clear class structure)
        gpl_ftrue = 3.0 .* gpl_Xs
        gpl_rng = MersenneTwister(11)
        gpl_ys = Float64[rand(gpl_rng) < 1 / (1 + exp(-ft)) ? 1.0 : 0.0 for ft in gpl_ftrue]
        gpl_cm = choicemap([(:y => i, gpl_ys[i]) for i = 1:gpl_n])

        gpl_chain = nuts(gpl_classification, (gpl_X,), gpl_cm; num_samples=500, num_warmup=500, rng=MersenneTwister(7))
        gpl_draws = gpl_chain.constrained_samples          # row 1 = logl, rows 2..11 = f
        @test all(isfinite, gpl_draws)
        gpl_fpost = [gpl_mean(gpl_draws[1+i, :]) for i = 1:gpl_n]
        # the latent function must be larger where the label is 1
        @test gpl_mean(gpl_fpost[gpl_ys .== 1]) > gpl_mean(gpl_fpost[gpl_ys .== 0])
    end
end

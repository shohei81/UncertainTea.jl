# Direct latent-function GP inference via `gp_cholesky` (issue #279): where
# `gaussianprocess` scores the analytic zero-mean marginal `N(0, K)` (Gaussian
# likelihood only), `gp_cholesky` returns the kernel Cholesky factor so the latent
# function values `f ~ N(0, K)` are sampled directly through `mvnormaldense` and
# can drive any likelihood. As of issue #289 the zero mean no longer needs a
# literal-length tuple: `f ~ mvnormaldense(zeros(n), L)` with a runtime `n`
# resolves the latent's dimension from the model arguments, so ONE model runs
# at any data size (tested below). CPU-reference. Statistics is unavailable in
# the harness, so use local helpers.

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

# GP classification with a RUNTIME-length latent f (issue #289): `n` is a model
# argument, so the same model runs at any data size — no literal-length mean.
@tea static function gpl_classification_rt(X, n)
    logl ~ normal(0.0, 0.5)
    L = gp_cholesky(X, exp(logl), 2.0, 1.0e-6)
    f ~ mvnormaldense(zeros(n), L)
    for i = 1:n
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

        gpl_chain = first(nuts(gpl_regression, (gpl_X,), gpl_cm; num_samples=800, num_warmup=800, rng=MersenneTwister(7)))
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

        gpl_chain = first(nuts(gpl_classification, (gpl_X,), gpl_cm; num_samples=500, num_warmup=500, rng=MersenneTwister(7)))
        gpl_draws = gpl_chain.constrained_samples          # row 1 = logl, rows 2..11 = f
        @test all(isfinite, gpl_draws)
        gpl_fpost = [gpl_mean(gpl_draws[1+i, :]) for i = 1:gpl_n]
        # the latent function must be larger where the label is 1
        @test gpl_mean(gpl_fpost[gpl_ys .== 1]) > gpl_mean(gpl_fpost[gpl_ys .== 0])
    end

    @testset "runtime-length latent f (issue #289): twin equality + two sizes in one session" begin
        # twin equality: the runtime-length spelling scores bitwise identically
        # to the literal-length spelling at n = 10
        gpl_n = 10
        gpl_Xs = collect(range(-3, 3; length=gpl_n))
        gpl_X = reshape(gpl_Xs, 1, gpl_n)
        gpl_rng = MersenneTwister(17)
        gpl_ys = Float64[x > 0 ? 1.0 : 0.0 for x in gpl_Xs]
        gpl_cm = choicemap([(:y => i, gpl_ys[i]) for i = 1:gpl_n])
        gpl_params = vcat([0.1], 0.3 .* randn(gpl_rng, gpl_n))
        @test logjoint_unconstrained(gpl_classification_rt, gpl_params, (gpl_X, gpl_n), gpl_cm) ==
              logjoint_unconstrained(gpl_classification, gpl_params, (gpl_X,), gpl_cm)
        @test logjoint_gradient_unconstrained(gpl_classification_rt, gpl_params, (gpl_X, gpl_n), gpl_cm) ==
              logjoint_gradient_unconstrained(gpl_classification, gpl_params, (gpl_X,), gpl_cm)

        # nuts smoke at TWO data sizes in one session: the plan re-specializes
        # per (signature, dims) and both chains sample the right-sized latent
        gpl_m = 6
        gpl_Xs2 = collect(range(-3, 3; length=gpl_m))
        gpl_X2 = reshape(gpl_Xs2, 1, gpl_m)
        gpl_ys2 = Float64[x > 0 ? 1.0 : 0.0 for x in gpl_Xs2]
        gpl_cm2 = choicemap([(:y => i, gpl_ys2[i]) for i = 1:gpl_m])
        gpl_chain10 = first(nuts(
            gpl_classification_rt, (gpl_X, gpl_n), gpl_cm;
            num_samples=300, num_warmup=300, rng=MersenneTwister(7),
        ))
        gpl_chain6 = first(nuts(
            gpl_classification_rt, (gpl_X2, gpl_m), gpl_cm2;
            num_samples=300, num_warmup=300, rng=MersenneTwister(7),
        ))
        @test size(gpl_chain10.constrained_samples, 1) == 1 + gpl_n
        @test size(gpl_chain6.constrained_samples, 1) == 1 + gpl_m
        @test all(isfinite, gpl_chain10.constrained_samples)
        @test all(isfinite, gpl_chain6.constrained_samples)
        # class separation holds at both sizes (rows 2..n+1 are f)
        for (chain, ys, count) in ((gpl_chain10, gpl_ys, gpl_n), (gpl_chain6, gpl_ys2, gpl_m))
            fpost = [gpl_mean(chain.constrained_samples[1+i, :]) for i = 1:count]
            @test gpl_mean(fpost[ys .== 1]) > gpl_mean(fpost[ys .== 0])
        end
    end
end

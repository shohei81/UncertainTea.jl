# GP kernel specifications (issue #290): Matérn 3/2 & 5/2, periodic, and
# sum/product composition alongside the existing RBF, usable in
# `gaussianprocess(X, kernel, noise)`, `sparsegaussianprocess(X, Z, kernel,
# noise)`, and `gp_cholesky(X, kernel, noise)`. The positional
# `gaussianprocess(X, l, v, noise)` form stays the RBF shorthand, byte-for-byte.

using LinearAlgebra

gpk_mean(x) = sum(x) / length(x)

@testset "dist_gp_kernels" begin
    gpk_rng = MersenneTwister(290)
    gpk_n = 20
    gpk_X = reshape(sort(rand(gpk_rng, gpk_n) .* 5), 1, gpk_n)
    gpk_y = randn(gpk_rng, gpk_n)

    @testset "positional RBF shorthand equals the kernel-spec RBF exactly" begin
        @test UncertainTea.logpdf(gaussianprocess(gpk_X, 1.0, 1.2, 0.3), gpk_y) ==
              UncertainTea.logpdf(gaussianprocess(gpk_X, rbf_kernel(1.0, 1.2), 0.3), gpk_y)
        @test UncertainTea.logpdf(sparsegaussianprocess(gpk_X, gpk_X[:, 1:6], 1.0, 1.2, 0.3), gpk_y) ==
              UncertainTea.logpdf(sparsegaussianprocess(gpk_X, gpk_X[:, 1:6], rbf_kernel(1.0, 1.2), 0.3), gpk_y)
    end

    @testset "matern32 / matern52 / periodic match hand-computed marginals" begin
        l, v, nz = 0.8, 1.1, 0.3
        m32(r) = v^2 * (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
        m52(r) = v^2 * (1 + sqrt(5) * r + 5 * r^2 / 3) * exp(-sqrt(5) * r)
        per(r) = v^2 * exp(-2 * sin(pi * r / 2.0)^2 / l^2)
        for (kernel, kfun, scaled) in (
            (matern32_kernel(l, v), m32, true),
            (matern52_kernel(l, v), m52, true),
            (periodic_kernel(l, v, 2.0), per, false),
        )
            K = [
                kfun(abs(gpk_X[1, i] - gpk_X[1, j]) / (scaled ? l : 1.0)) + (nz^2 + 1e-8) * (i == j)
                for i = 1:gpk_n, j = 1:gpk_n
            ]
            ref = -0.5 * dot(gpk_y, K \ gpk_y) - 0.5 * logdet(K) - 0.5 * gpk_n * log(2pi)
            @test UncertainTea.logpdf(gaussianprocess(gpk_X, kernel, nz), gpk_y) ≈ ref rtol = 1e-9
        end
    end

    @testset "sum / product composition" begin
        ka = matern32_kernel(0.8, 1.0)
        kb = rbf_kernel(2.0, 0.7)
        lp_a = UncertainTea.logpdf(gaussianprocess(gpk_X, ka, 0.3), gpk_y)
        @test isfinite(UncertainTea.logpdf(gaussianprocess(gpk_X, kernel_sum(ka, kb), 0.3), gpk_y))
        # locally periodic: the workhorse composition
        lp_lp = UncertainTea.logpdf(
            gaussianprocess(gpk_X, kernel_product(periodic_kernel(1.0, 1.0, 2.0), rbf_kernel(3.0, 1.0)), 0.3),
            gpk_y,
        )
        @test isfinite(lp_lp)
        @test lp_lp != lp_a
    end

    @testset "FITC with a non-RBF kernel reduces to the dense GP at Z = X" begin
        k = matern52_kernel(1.0, 1.0)
        dense = UncertainTea.logpdf(gaussianprocess(gpk_X, k, 0.3), gpk_y)
        fitc = UncertainTea.logpdf(sparsegaussianprocess(gpk_X, gpk_X, k, 0.3), gpk_y)
        @test dense ≈ fitc rtol = 1e-3    # differing diagonal jitter only
    end

    @testset "gp_cholesky with a kernel spec reconstructs the covariance" begin
        k = matern32_kernel(0.9, 1.3)
        L = gp_cholesky(gpk_X, k, 0.1)
        @test istril(L)
        Kref = UncertainTea._gp_kernel_covariance(k, gpk_X, 0.1)
        @test L * L' ≈ Kref rtol = 1e-10
    end

    @testset "ARD lengthscale validation flows through kernels" begin
        Xd = randn(MersenneTwister(1), 2, 6)
        @test isfinite(
            UncertainTea.logpdf(gaussianprocess(Xd, matern32_kernel([0.5, 2.0], 1.0), 0.2), randn(MersenneTwister(2), 6)),
        )
        @test_throws ArgumentError gaussianprocess(Xd, matern32_kernel([0.5, 2.0, 1.0], 1.0), 0.2)
    end

    @testset "NUTS with a latent Matern lengthscale (diagonal r == 0 guard)" begin
        # the Matern diagonal is r == 0 where sqrt has an infinite derivative;
        # the element guard keeps the dual gradient finite
        gpk_m32 = @tea static function gpk_gp_m32(X)
            logl ~ normal(0.0, 1.0)
            logn ~ normal(-1.0, 1.0)
            {:y} ~ gaussianprocess(X, matern32_kernel(exp(logl), 1.0), exp(logn))
            return logl
        end
        chain = nuts(gpk_m32, (gpk_X,), choicemap((:y, gpk_y)); num_samples=100, num_warmup=100, rng=MersenneTwister(7))
        @test all(isfinite, chain.constrained_samples)
    end
end

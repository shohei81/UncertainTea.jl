# Elliptical slice sampling (issue #294): gradient-free, tuning-free updates
# for a latent vector with a multivariate-Gaussian prior — the natural sampler
# for the GP latent-function models (#280). Statistics is unavailable in the
# harness, so use local helpers.

using LinearAlgebra

ess_mean(x) = sum(x) / length(x)

@testset "elliptical_slice" begin
    @testset "flat likelihood samples the prior" begin
        rng = MersenneTwister(294)
        n = 4
        A = randn(rng, n, n)
        L = cholesky(Symmetric(A * A' + n * I)).L
        result = elliptical_slice(f -> 0.0, L; num_samples=4000, num_warmup=200, rng=rng)
        @test size(result.samples) == (n, 4000)
        Sigma = L * L'
        sample_cov = result.samples * result.samples' ./ 4000
        @test maximum(abs.(vec(sample_cov .- Sigma))) < 0.6 * maximum(abs.(Sigma))
        @test maximum(abs, [ess_mean(result.samples[i, :]) for i = 1:n]) < 0.5
        @test result.average_likelihood_evaluations >= 1
    end

    @testset "Gaussian likelihood recovers the conjugate posterior" begin
        # prior N(0, Sigma), likelihood y | f ~ N(f, sigma^2 I):
        # posterior N(Sigma (Sigma + sigma^2 I)^-1 y, (Sigma^-1 + I/sigma^2)^-1)
        rng = MersenneTwister(7)
        n = 5
        A = randn(rng, n, n)
        Sigma = Symmetric(A * A' + n * I)
        L = cholesky(Sigma).L
        sigma = 0.5
        y = randn(rng, n)
        loglik(f) = -sum(abs2, y .- f) / (2 * sigma^2)
        result = elliptical_slice(loglik, L; num_samples=6000, num_warmup=500, rng=MersenneTwister(11))
        post_mean_ref = Sigma * ((Sigma + sigma^2 * I) \ y)
        post_mean = [ess_mean(result.samples[i, :]) for i = 1:n]
        @test maximum(abs.(post_mean .- post_mean_ref)) < 0.15
    end

    @testset "GP classification latents move under a logit likelihood" begin
        rng = MersenneTwister(280)
        m = 10
        Xs = collect(range(-3.0, 3.0; length=m))
        L = gp_cholesky(reshape(Xs, 1, m), rbf_kernel(1.5, 1.5), 1e-4)
        labels = Float64.(Xs .> 0)
        loglik(f) = sum(UncertainTea.logpdf(bernoullilogit(f[i]), labels[i]) for i = 1:m)
        result = elliptical_slice(loglik, L; num_samples=1500, num_warmup=300, rng=rng)
        fpost = [ess_mean(result.samples[i, :]) for i = 1:m]
        # the latent function separates the classes
        @test ess_mean(fpost[labels .== 1]) > ess_mean(fpost[labels .== 0])
        @test all(isfinite, result.samples)
    end

    @testset "validation" begin
        L = Matrix{Float64}(I, 3, 3)
        @test_throws ArgumentError elliptical_slice(f -> 0.0, L; num_samples=0)
        @test_throws ArgumentError elliptical_slice(f -> 0.0, randn(3, 2); num_samples=5)
        @test_throws ArgumentError elliptical_slice(f -> 0.0, L; num_samples=5, initial=[1.0, 2.0])
        @test_throws ArgumentError elliptical_slice(f -> -Inf, L; num_samples=5)
    end
end

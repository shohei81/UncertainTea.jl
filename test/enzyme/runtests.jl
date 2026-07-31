# Tests for the Enzyme reverse-mode extension `reverse_mode_gradient` (issue #268).
#
# NOT part of the package test target and NOT run in CI (Enzyme is heavy and
# version-sensitive). See test/enzyme/Project.toml for setup instructions.
#
# Loading Enzyme activates UncertainTeaEnzymeExt, which supplies the
# `reverse_mode_gradient` method. Every case checks reverse-mode against the
# authoritative ForwardDiff gradient.

using Random, Test, LinearAlgebra
using UncertainTea
const UT = UncertainTea
using ForwardDiff
using Enzyme   # activates UncertainTeaEnzymeExt

@testset "UncertainTeaEnzymeExt reverse_mode_gradient" begin
    @testset "GP hyperparameter gradient matches ForwardDiff" begin
        rng = MersenneTwister(262)
        n = 30
        X = reshape(sort(rand(rng, n) .* 5), 1, n)
        Ktrue = exp.(-0.5 .* [(X[1, i] - X[1, j])^2 for i = 1:n, j = 1:n]) + 0.04 .* Matrix(I, n, n)
        y = cholesky(Symmetric(Ktrue)).L * randn(rng, n)
        gp_nlml(h) = UT.logpdf(gaussianprocess(X, exp(h[1]), exp(h[2]), exp(h[3])), y)

        h0 = [0.0, 0.0, -1.0]
        fd = ForwardDiff.gradient(gp_nlml, h0)
        rev = reverse_mode_gradient(gp_nlml, h0)
        @test rev ≈ fd rtol = 1e-8
        @test length(rev) == 3
    end

    @testset "pure high-P coupled logjoint matches ForwardDiff" begin
        # normal(0,1) prior + per-element nonlinear neighbour coupling; nothing
        # sufficient-statistics-fuses, the class where reverse-mode scales O(P)
        # against ForwardDiff's O(P^2) (see bench/reverse_mode/).
        function coupled_logjoint(x, y)
            T = eltype(x)
            c = T(0.9189385332046727)
            lp = zero(T)
            @inbounds for i in eachindex(x)
                lp += -x[i]^2 / 2 - c
            end
            @inbounds for i = 1:(length(x)-1)
                z = (y[i] - (tanh(x[i]) + T(0.5) * x[i+1])) / T(0.3)
                lp += -z^2 / 2 - log(T(0.3)) - c
            end
            return lp
        end

        rng = MersenneTwister(268)
        P = 100
        x = randn(rng, P)
        y = [tanh(x[i]) + 0.5 * x[i+1] + 0.3 * randn(rng) for i = 1:(P-1)]
        obj(u) = coupled_logjoint(u, y)
        fd = ForwardDiff.gradient(obj, x)
        rev = reverse_mode_gradient(obj, x)
        @test rev ≈ fd rtol = 1e-8
        @test length(rev) == P
    end

    @testset "matches a closed-form gradient" begin
        # f(x) = -||x - a||^2 / 2  =>  grad = a - x
        a = [1.0, -2.0, 0.5, 3.0]
        f(x) = -sum((x .- a) .^ 2) / 2
        x0 = [0.2, 0.1, -0.3, 1.5]
        @test reverse_mode_gradient(f, x0) ≈ (a .- x0) rtol = 1e-10
    end
end

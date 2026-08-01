# Single-source scalar kernels (issue #285): every hand-derived partials kernel
# in src/distributions/scalar_kernels.jl is pinned against ForwardDiff applied to
# the corresponding logpdf kernel, at multiple in-support points. This is THE
# check that the one declaration each family now has is internally consistent —
# the batched analytic gradients and the CPU/backend logpdfs both consume it.

using ForwardDiff

# (label, n_params, logpdf(v) with v = [params..., x], partials(v) reordered to
# ForwardDiff's (dparams..., dx) channel order)
scalar_kernel_cases = [
    (
        "normal", 2,
        v -> UncertainTea._backend_normal_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._normal_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[0.3, 1.2, -0.4], [-1.0, 0.5, 2.0]],
    ),
    (
        "lognormal", 2,
        v -> UncertainTea._backend_lognormal_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._lognormal_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[0.2, 0.8, 1.5], [-0.5, 1.3, 0.4]],
    ),
    (
        "exponential", 1,
        v -> UncertainTea._backend_exponential_logpdf(v[1], v[2]),
        v -> begin
            (dx, dp...) = UncertainTea._exponential_logpdf_partials(v[1], v[2])
            (dp..., dx)
        end,
        [[1.5, 0.7], [0.4, 2.5]],
    ),
    (
        "gamma", 2,
        v -> UncertainTea._backend_gamma_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._gamma_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[2.0, 1.5, 0.9], [0.7, 0.3, 2.4]],
    ),
    (
        "inversegamma", 2,
        v -> UncertainTea._backend_inversegamma_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._inversegamma_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[2.5, 1.2, 0.8], [1.1, 0.6, 1.9]],
    ),
    (
        "weibull", 2,
        v -> UncertainTea._backend_weibull_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._weibull_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[1.5, 2.0, 1.1], [0.8, 0.5, 0.3]],
    ),
    (
        "beta", 2,
        v -> UncertainTea._backend_beta_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._beta_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[2.0, 3.0, 0.3], [0.7, 1.4, 0.85]],
    ),
    (
        "studentt", 3,
        v -> UncertainTea._backend_studentt_logpdf(v[1], v[2], v[3], v[4]),
        v -> begin
            (dx, dp...) = UncertainTea._studentt_logpdf_partials(v[1], v[2], v[3], v[4])
            (dp..., dx)
        end,
        [[4.0, 0.2, 1.1, -0.7], [2.5, -1.0, 0.6, 1.8]],
    ),
    (
        "laplace", 2,
        v -> UncertainTea._backend_laplace_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._laplace_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[0.5, 1.0, 1.7], [-0.3, 0.4, -1.2]],
    ),
    (
        "cauchy", 2,
        v -> UncertainTea._backend_cauchy_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._cauchy_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[0.1, 0.9, 1.3], [-2.0, 2.2, -0.5]],
    ),
    (
        "halfnormal", 1,
        v -> UncertainTea._backend_halfnormal_logpdf(v[1], v[2]),
        v -> begin
            (dx, dp...) = UncertainTea._halfnormal_logpdf_partials(v[1], v[2])
            (dp..., dx)
        end,
        [[1.2, 0.6], [0.5, 1.4]],
    ),
    (
        "halfcauchy", 1,
        v -> UncertainTea._backend_halfcauchy_logpdf(v[1], v[2]),
        v -> begin
            (dx, dp...) = UncertainTea._halfcauchy_logpdf_partials(v[1], v[2])
            (dp..., dx)
        end,
        [[0.8, 0.9], [2.0, 0.3]],
    ),
    (
        "logistic", 2,
        v -> UncertainTea._backend_logistic_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._logistic_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[0.4, 1.1, -0.9], [-1.5, 0.7, 0.2]],
    ),
    (
        "gumbel", 2,
        v -> UncertainTea._backend_gumbel_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._gumbel_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[0.3, 1.3, 0.8], [-0.6, 0.5, -1.1]],
    ),
    (
        "pareto", 2,
        v -> UncertainTea._backend_pareto_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._pareto_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[0.5, 2.0, 1.4], [1.0, 3.5, 2.2]],
    ),
    (
        "frechet", 2,
        v -> UncertainTea._backend_frechet_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._frechet_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[1.8, 1.2, 0.9], [0.9, 0.7, 2.1]],
    ),
    (
        "rayleigh", 1,
        v -> UncertainTea._backend_rayleigh_logpdf(v[1], v[2]),
        v -> begin
            (dx, dp...) = UncertainTea._rayleigh_logpdf_partials(v[1], v[2])
            (dp..., dx)
        end,
        [[1.1, 0.8], [0.6, 1.7]],
    ),
    (
        "inversegaussian", 2,
        v -> UncertainTea._backend_inversegaussian_logpdf(v[1], v[2], v[3]),
        v -> begin
            (dx, dp...) = UncertainTea._inversegaussian_logpdf_partials(v[1], v[2], v[3])
            (dp..., dx)
        end,
        [[1.3, 2.1, 0.9], [0.7, 0.5, 1.6]],
    ),
]

@testset "scalar_kernel_partials" begin
    @testset "partials match ForwardDiff on the logpdf kernel" begin
        for (label, _, kernel, partials, points) in scalar_kernel_cases
            for v in points
                fd = ForwardDiff.gradient(kernel, v)
                analytic = collect(partials(v))
                @test isapprox(analytic, fd; rtol=1e-8, atol=1e-10)
            end
        end
    end

    @testset "uniform: bound partials match, value channel is zero" begin
        for v in ([0.0, 2.0, 0.7], [-1.0, 1.5, 0.2])
            fd = ForwardDiff.gradient(u -> UncertainTea._backend_uniform_logpdf(u[1], u[2], u[3]), v)
            dlower, dupper = UncertainTea._uniform_logpdf_partials(v[1], v[2], v[3])
            @test isapprox([dlower, dupper], fd[1:2]; rtol=1e-10)
            @test fd[3] == 0.0
        end
    end

    @testset "CPU logpdf delegation matches the kernels" begin
        # a spot check that the delegated CPU-reference logpdfs return the kernel
        # values exactly (same code path now, but this pins the wiring)
        @test UncertainTea.logpdf(gamma(2.0, 1.5), 0.9) ==
              UncertainTea._backend_gamma_logpdf(2.0, 1.5, 0.9)
        @test UncertainTea.logpdf(weibull(1.0, 2.0), 0.0) ==
              UncertainTea._backend_weibull_logpdf(1.0, 2.0, 0.0)   # boundary case preserved
        @test UncertainTea.logpdf(beta(2.0, 3.0), 1.5) == -Inf       # off support
        @test UncertainTea.logpdf(studentt(4.0f0, 0.0f0, 1.0f0), 0.5f0) isa Float32  # eltype preserved
    end
end

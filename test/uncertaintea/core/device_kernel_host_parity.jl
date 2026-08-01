# Device-vs-host kernel parity (issue #285, stage 3). The device logpdf kernels
# in src/device/math.jl are INTENTIONALLY separate implementations from the
# single-source host kernels in src/distributions/scalar_kernels.jl: they are
# branchless (ifelse, exception-free), carry Float32-exact constants, skip
# parameter validation (plans validate upstream), and avoid SpecialFunctions
# (pure-Julia Lanczos loggamma). Merging them would need a kernel DSL that costs
# more than the duplication it removes. Instead, this test pins every
# device/host kernel PAIR per family at Float64 over in-support and off-support
# points, so the two implementations can never silently diverge -- the
# correctness value of unification without the redesign.
#
# Tolerance: formulas are identical up to the special-function implementation
# (Lanczos vs SpecialFunctions loggamma agree to ~1e-13 relative at Float64).
#
# Off-support probes are limited to families whose branchless device body is
# TOTAL: an ifelse evaluates both arms, so a log-based body throws DomainError
# on the host for a negative x (on the GPU it yields a NaN that the ifelse
# selects away). The model-level device parity suite covers those paths.

const UTD = UncertainTea

# (label, host(v), device(v), in-support points v = [params..., x], off-support x)
device_host_parity_cases = [
    ("normal",
        v -> UTD._backend_normal_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_normal_logpdf(v[1], v[2], v[3]),
        [[0.3, 1.2, -0.4], [-1.0, 0.5, 2.0]], nothing),
    ("lognormal",
        v -> UTD._backend_lognormal_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_lognormal_logpdf(v[1], v[2], v[3]),
        [[0.2, 0.8, 1.5], [-0.5, 1.3, 0.4]], nothing),
    ("exponential",
        v -> UTD._backend_exponential_logpdf(v[1], v[2]),
        v -> UTD._device_exponential_logpdf(v[1], v[2]),
        [[1.5, 0.7], [0.4, 2.5]], [1.5, -0.3]),
    ("gamma",
        v -> UTD._backend_gamma_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_gamma_logpdf(v[1], v[2], v[3]),
        [[2.0, 1.5, 0.9], [0.7, 0.3, 2.4]], nothing),
    ("inversegamma",
        v -> UTD._backend_inversegamma_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_inversegamma_logpdf(v[1], v[2], v[3]),
        [[2.5, 1.2, 0.8], [1.1, 0.6, 1.9]], nothing),
    ("weibull",
        v -> UTD._backend_weibull_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_weibull_logpdf(v[1], v[2], v[3]),
        [[1.5, 2.0, 1.1], [0.8, 0.5, 0.3]], nothing),
    ("beta",
        v -> UTD._backend_beta_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_beta_logpdf(v[1], v[2], v[3]),
        [[2.0, 3.0, 0.3], [0.7, 1.4, 0.85]], nothing),
    ("studentt",
        v -> UTD._backend_studentt_logpdf(v[1], v[2], v[3], v[4]),
        v -> UTD._device_studentt_logpdf(v[1], v[2], v[3], v[4]),
        [[4.0, 0.2, 1.1, -0.7], [2.5, -1.0, 0.6, 1.8], [30.0, 0.0, 1.0, 0.5]], nothing),
    ("laplace",
        v -> UTD._backend_laplace_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_laplace_logpdf(v[1], v[2], v[3]),
        [[0.5, 1.0, 1.7], [-0.3, 0.4, -1.2]], nothing),
    ("cauchy",
        v -> UTD._backend_cauchy_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_cauchy_logpdf(v[1], v[2], v[3]),
        [[0.1, 0.9, 1.3], [-2.0, 2.2, -0.5]], nothing),
    ("halfnormal",
        v -> UTD._backend_halfnormal_logpdf(v[1], v[2]),
        v -> UTD._device_halfnormal_logpdf(v[1], v[2]),
        [[1.2, 0.6], [0.5, 1.4]], [1.2, -0.4]),
    ("halfcauchy",
        v -> UTD._backend_halfcauchy_logpdf(v[1], v[2]),
        v -> UTD._device_halfcauchy_logpdf(v[1], v[2]),
        [[0.8, 0.9], [2.0, 0.3]], [0.8, -0.4]),
    ("logistic",
        v -> UTD._backend_logistic_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_logistic_logpdf(v[1], v[2], v[3]),
        [[0.4, 1.1, -0.9], [-1.5, 0.7, 0.2]], nothing),
    ("gumbel",
        v -> UTD._backend_gumbel_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_gumbel_logpdf(v[1], v[2], v[3]),
        [[0.3, 1.3, 0.8], [-0.6, 0.5, -1.1]], nothing),
    ("frechet",
        v -> UTD._backend_frechet_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_frechet_logpdf(v[1], v[2], v[3]),
        [[1.8, 1.2, 0.9], [0.9, 0.7, 2.1]], nothing),
    ("rayleigh",
        v -> UTD._backend_rayleigh_logpdf(v[1], v[2]),
        v -> UTD._device_rayleigh_logpdf(v[1], v[2]),
        [[1.1, 0.8], [0.6, 1.7]], nothing),
    ("inversegaussian",
        v -> UTD._backend_inversegaussian_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_inversegaussian_logpdf(v[1], v[2], v[3]),
        [[1.3, 2.1, 0.9], [0.7, 0.5, 1.6]], nothing),
    ("bernoulli",
        v -> UTD._backend_bernoulli_logpdf(v[1], v[2]),
        v -> UTD._device_bernoulli_logpdf(v[1], v[2]),
        [[0.35, 1.0], [0.35, 0.0], [0.9, 1.0]], [0.35, 0.5]),
    ("bernoullilogit",
        v -> UTD._backend_bernoullilogit_logpdf(v[1], v[2]),
        v -> UTD._device_bernoullilogit_logpdf(v[1], v[2]),
        [[0.7, 1.0], [0.7, 0.0], [-2.4, 1.0]], [0.7, 0.5]),
    ("poisson",
        v -> UTD._backend_poisson_logpdf(v[1], v[2]),
        v -> UTD._device_poisson_logpdf(v[1], v[2]),
        [[2.4, 0.0], [2.4, 3.0], [0.6, 7.0]], [2.4, 2.5]),
    ("geometric",
        v -> UTD._backend_geometric_logpdf(v[1], v[2]),
        v -> UTD._device_geometric_logpdf(v[1], v[2]),
        [[0.3, 0.0], [0.3, 4.0], [0.9, 2.0]], [0.3, 1.5]),
    ("negativebinomial",
        v -> UTD._backend_negativebinomial_logpdf(v[1], v[2], v[3]),
        v -> UTD._device_negativebinomial_logpdf(v[1], v[2], v[3]),
        [[3.0, 0.45, 0.0], [3.0, 0.45, 5.0], [1.5, 0.7, 2.0]], [3.0, 0.45, 2.3]),
]

@testset "device_kernel_host_parity" begin
    @testset "per-family device kernels match the single-source host kernels" begin
        for (label, host, device, points, off) in device_host_parity_cases
            for v in points
                h = host(v)
                d = device(v)
                @test isfinite(h)
                @test isapprox(d, h; rtol=1e-10, atol=1e-10)
            end
            if off !== nothing
                @test host(off) == -Inf
                @test device(off) == -Inf
            end
        end
    end

    @testset "binomial (pre-classified device signature)" begin
        for (n, k, p) in ((10, 0, 0.4), (10, 4, 0.4), (10, 10, 0.4), (7, 3, 0.85))
            h = UTD._backend_binomial_logpdf(n, p, float(k))
            d = UTD._device_binomial_logpdf(n, k, p)
            @test isapprox(d, h; rtol=1e-10, atol=1e-10)
        end
    end

    @testset "categorical (tuple probabilities)" begin
        probs = (0.2, 0.5, 0.3)
        for x in (1.0, 2.0, 3.0)
            h = UTD._backend_categorical_logpdf(probs, x)
            d = UTD._device_categorical_logpdf(probs, x)
            @test isapprox(d, h; rtol=1e-12)
        end
        @test UTD._backend_categorical_logpdf(probs, 4.0) == -Inf
        @test UTD._device_categorical_logpdf(probs, 4.0) == -Inf
    end
end

# Stein Variational Gradient Descent (issue #236).
#
# SVGD is a deterministic PARTICLE method, not MCMC: it is validated by
# posterior-MOMENT recovery and particle SPREAD (variance + correlation), not by
# Rhat/ESS. The device leg runs on KernelAbstractions.CPU() (the reference
# backend); it keeps the RNG host-side, so it matches the host path within
# tolerance (here it is even bitwise identical, since the CPU()-backend Float64
# gradient is deterministic and the same seed gives the same particle init).
#
# NO `using Statistics`: local mean/variance/covariance helpers below.

using KernelAbstractions: CPU

svgd_mean(x) = sum(x) / length(x)
function svgd_var(x)
    m = svgd_mean(x)
    accumulator = 0.0
    for value in x
        accumulator += (value - m)^2
    end
    return accumulator / (length(x) - 1)
end
function svgd_cor(xs, ys)
    mx = svgd_mean(xs)
    my = svgd_mean(ys)
    covariance = 0.0
    var_x = 0.0
    var_y = 0.0
    for index in eachindex(xs, ys)
        dx = xs[index] - mx
        dy = ys[index] - my
        covariance += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    end
    return covariance / sqrt(var_x * var_y)
end

# Conjugate gaussian: mu ~ N(0,1), y ~ N(mu,1) observed at y = 0.3.
# Posterior precision = 2 -> variance 0.5, mean = 0.3/2 = 0.15.
@tea static function svgd_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# Correlated 2-D prior: z ~ N(0, Sigma) with Sigma = L L', L = [1 0; 0.8 0.6],
# so Sigma = [[1, 0.8], [0.8, 1]] -- unit marginal variances, correlation 0.8.
# The posterior IS this prior (no observation), a known correlated Gaussian: a
# genuine spread check (both variances AND the correlation), not just the mean.
const svgd_L = [1.0 0.0; 0.8 0.6]
@tea static function svgd_correlated_2d(L)
    z ~ mvnormaldense([0.0, 0.0], L)
    return z
end

@testset "svgd_conjugate_moments" begin
    result = batched_svgd(
        svgd_conjugate_gauss,
        (),
        choicemap((:y, 0.3));
        num_particles=80,
        num_iterations=300,
        learning_rate=0.1,
        rng=MersenneTwister(236),
    )

    @test result isa SVGDResult
    @test result.gradient_backend == :backend_native
    @test result.num_particles == 80
    @test result.num_iterations == 300
    @test size(result.unconstrained_particles) == (1, 80)
    @test size(result.constrained_particles) == (1, 80)
    @test all(isfinite, result.unconstrained_particles)
    @test length(result.kernel_scale_history) == 300
    @test all(>(0.0), result.kernel_scale_history)
    # the update direction decays as the particles settle
    @test result.direction_norm_history[end] < result.direction_norm_history[1]

    particles = vec(result.unconstrained_particles)
    @test isapprox(svgd_mean(particles), 0.15; atol=0.05)
    @test isapprox(svgd_var(particles), 0.5; atol=0.15)

    # constrained == unconstrained for an unbounded scalar latent
    @test result.constrained_particles ≈ result.unconstrained_particles atol = 1e-8
    @test isapprox(particle_mean(result; space=:unconstrained)[1], 0.15; atol=0.05)
    @test isapprox(particle_covariance(result; space=:unconstrained)[1, 1], 0.5; atol=0.15)
end

@testset "svgd_determinism" begin
    kwargs = (num_particles=40, num_iterations=120, learning_rate=0.1)
    first_result = batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); kwargs..., rng=MersenneTwister(99),
    )
    second_result = batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); kwargs..., rng=MersenneTwister(99),
    )
    @test first_result.unconstrained_particles == second_result.unconstrained_particles
    @test first_result.kernel_scale_history == second_result.kernel_scale_history
end

@testset "svgd_correlated_spread" begin
    result = batched_svgd(
        svgd_correlated_2d,
        (svgd_L,),
        choicemap();
        num_particles=150,
        num_iterations=500,
        learning_rate=0.1,
        rng=MersenneTwister(7),
    )

    @test size(result.unconstrained_particles) == (2, 150)
    first_dimension = result.unconstrained_particles[1, :]
    second_dimension = result.unconstrained_particles[2, :]

    # means near zero
    @test isapprox(svgd_mean(first_dimension), 0.0; atol=0.15)
    @test isapprox(svgd_mean(second_dimension), 0.0; atol=0.15)
    # both marginal variances recovered (a collapse would drive these to ~0)
    @test isapprox(svgd_var(first_dimension), 1.0; atol=0.25)
    @test isapprox(svgd_var(second_dimension), 1.0; atol=0.25)
    # the correlation sign AND magnitude (target 0.8)
    correlation = svgd_cor(first_dimension, second_dimension)
    @test correlation > 0.5
    @test isapprox(correlation, 0.8; atol=0.15)

    covariance = particle_covariance(result; space=:unconstrained)
    @test isapprox(covariance[1, 2], 0.8; atol=0.25)
end

@testset "svgd_device_cpu_matches_host" begin
    backend = CPU()
    kwargs = (
        num_particles=80,
        num_iterations=300,
        learning_rate=0.1,
    )
    host = batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); kwargs..., rng=MersenneTwister(236),
    )
    device = batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); kwargs..., backend=backend, rng=MersenneTwister(236),
    )
    @test device.gradient_backend == :device
    host_particles = vec(host.unconstrained_particles)
    device_particles = vec(device.unconstrained_particles)
    @test isapprox(svgd_mean(device_particles), svgd_mean(host_particles); atol=0.05)
    @test isapprox(svgd_var(device_particles), svgd_var(host_particles); atol=0.05)
    @test isapprox(svgd_mean(device_particles), 0.15; atol=0.05)
end

@testset "svgd_argument_validation" begin
    @test_throws ArgumentError batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); num_particles=0, num_iterations=10,
    )
    @test_throws ArgumentError batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); num_particles=8, num_iterations=0,
    )
    @test_throws ArgumentError batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); num_particles=8, num_iterations=10, learning_rate=-1.0,
    )
    @test_throws ArgumentError batched_svgd(
        svgd_conjugate_gauss, (), choicemap((:y, 0.3)); num_particles=8, num_iterations=10, bandwidth=-2.0,
    )
end

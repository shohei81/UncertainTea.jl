# PR 46: device-resident batched HMC and ADVI inner loops.
#
# All tests run on KernelAbstractions.CPU() (the authoritative reference backend);
# the GPU smoke test under test/gpu/ mirrors a subset on a MetalBackend at Float32.
# The device path keeps the RNG host-side, so its results are STATISTICALLY
# equivalent to -- not bitwise identical to -- the CPU `batched_hmc`/`batched_advi`
# path; the tests below compare posterior/variational analytics within tolerance.

using KernelAbstractions: CPU

# Local mean/std helpers (Statistics is not imported by the test harness).
devh_mean(x) = sum(x) / length(x)
function devh_std(x)
    m = devh_mean(x)
    accumulator = 0.0
    for value in x
        accumulator += (value - m)^2
    end
    return sqrt(accumulator / (length(x) - 1))
end

# Conjugate gaussian: mu ~ N(0,1), y ~ N(mu,1) observed at y = 0.3.
# Posterior precision = 2 -> variance 0.5 (sd sqrt(0.5)), mean = 0.3/2 = 0.15.
@tea static function devh_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# backend-supported but device-UNSUPPORTED (issue #67 made marginalize=:enumerate
# device-lowerable; #134 did the same for broadcast normal): an lkjcholesky latent
# above the device dimension cap stays a clean non-lowerable example.
@tea static function devh_unsupported_model()
    Omega ~ lkjcholesky(9, 2.0)
    {:y} ~ normal(0.0, 1.0)
    return Omega
end

# issue #70: a non-differentiable point (sqrt(abs(x)) has an infinite/NaN
# derivative at x = 0) makes every particle gradient non-finite while the
# objective value stays finite.
@tea static function devh_badgrad_model()
    x ~ normal(0.0, 1.0)
    s = sqrt(abs(x))
    {:y} ~ normal(s, 1.0)
end

@testset "devh_hmc_conjugate" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        # 4 chains x 600/300 rather than 2 x 300/150: the posterior-std smoke
        # check below must hold for every supported randn stream (the ziggurat
        # changed in Julia 1.12), and shorter seeded runs leave the std
        # estimate of the autocorrelated chains too noisy -- the pre-1.12
        # stream under-adapts the step size at 150 warmup (acceptance pins at
        # 1.0, std comes out at half its true value) and single short chains
        # can under-disperse on any stream. Exact device-vs-host parity is
        # covered separately by devh_hmc_vs_cpu.
        num_chains=4,
        num_samples=600,
        num_warmup=300,
        backend=backend,
        precision=Float64,
        rng=MersenneTwister(46),
    )

    samples = posterior_array(chains)
    @test all(isfinite, samples)
    @test isapprox(devh_mean(samples), 0.15; atol=0.1)
    @test isapprox(devh_std(samples), sqrt(0.5); atol=0.15)
    @test all(<(1.2), rhat(chains))
    accept = acceptancerate(chains)
    @test 0.5 < accept < 1.0
end

@testset "devh_hmc_vs_cpu" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    device_chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=300,
        num_warmup=150,
        backend=backend,
        precision=Float64,
        rng=MersenneTwister(52),
    )
    cpu_chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=300,
        num_warmup=150,
        # compare like with like: the device leg runs shared adaptation
        # (issue #137 made per-chain the host default)
        per_chain_adaptation=false,
        rng=MersenneTwister(52),
    )
    @test isapprox(devh_mean(posterior_array(device_chains)), devh_mean(posterior_array(cpu_chains)); atol=0.1)
end

@testset "devh_hmc_float32" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    chains = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=300,
        num_warmup=150,
        backend=backend,
        precision=Float32,
        rng=MersenneTwister(46),
    )
    samples = posterior_array(chains)
    # HMCChain stores Float64; the device ran in Float32 and downloads convert up.
    @test eltype(samples) === Float64
    @test all(isfinite, samples)
    @test isapprox(devh_mean(samples), 0.15; atol=0.1)
end

@testset "devh_advi_conjugate" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    result = batched_advi(
        devh_conjugate_gauss,
        (),
        constraints;
        num_steps=300,
        backend=backend,
        rng=MersenneTwister(46),
    )
    mean_estimate = variational_mean(result)
    @test length(mean_estimate) == 1
    @test isapprox(mean_estimate[1], 0.15; atol=0.1)
    @test all(isfinite, result.elbo_history)
    quartile = length(result.elbo_history) ÷ 4
    first_quartile = devh_mean(result.elbo_history[1:quartile])
    last_quartile = devh_mean(result.elbo_history[(end-quartile+1):end])
    @test last_quartile > first_quartile
end

@testset "devh_device_errors" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    # Per-chain adaptation is now SUPPORTED on the device backend (issue #137): it
    # routes to the pooled-mass / per-chain-step device HMC path and converges.
    per_chain_hmc = batched_hmc(
        devh_conjugate_gauss,
        (),
        constraints;
        num_chains=4,
        num_samples=300,
        num_warmup=200,
        num_leapfrog_steps=8,
        per_chain_adaptation=true,
        backend=backend,
        rng=MersenneTwister(514),
    )
    per_chain_hmc_draws = posterior_array(per_chain_hmc)
    @test all(isfinite, per_chain_hmc_draws)
    @test isapprox(devh_mean(vec(per_chain_hmc_draws)), 0.15; atol=0.1)
    # An unsupported (over-cap lkjcholesky latent) model raises the lowering ArgumentError.
    marg_error = try
        batched_hmc(devh_unsupported_model, (), choicemap((:y, 0.4)); num_chains=2, num_samples=10, backend=backend)
        nothing
    catch err
        err
    end
    @test marg_error isa ArgumentError
    @test occursin("device_lowering_report", sprint(showerror, marg_error))

    advi_error = try
        batched_advi(devh_unsupported_model, (), choicemap((:y, 0.4)); num_steps=10, backend=backend)
        nothing
    catch err
        err
    end
    @test advi_error isa ArgumentError
    @test occursin("device_lowering_report", sprint(showerror, advi_error))
end

# issue #70: the device ADVI step must guard the DOWNLOADED gradients (not only
# the objective values). With initial_log_scale extremely negative the particles
# collapse onto x = 0, where sqrt(abs(x)) has a non-finite derivative: the
# objective value is finite but every gradient is NaN. The device path must throw
# the same ArgumentError as the CPU path instead of returning a NaN state.
@testset "devh_advi_nonfinite_gradient_guard" begin
    backend = CPU()
    kwargs = (
        num_steps=1,
        num_particles=2,
        initial_params=[0.0],
        initial_log_scale=-1000.0,
        rng=MersenneTwister(1),
    )
    # CPU reference throws; the device path must match.
    @test_throws ArgumentError batched_advi(devh_badgrad_model, (), choicemap((:y, 0.0)); kwargs...)
    device_error = try
        batched_advi(devh_badgrad_model, (), choicemap((:y, 0.0)); backend=backend, kwargs...)
        nothing
    catch err
        err
    end
    @test device_error isa ArgumentError
    @test occursin("non-finite", sprint(showerror, device_error))
end

# issue #108: same-seed CPU and device mean-field ADVI must see an identical RNG
# stream. The CPU path shapes its gradient cache without a real draw, so no RNG is
# burned before the loop -- matching the device path, which has no pre-loop draw.
@testset "devh_advi_same_seed_cpu_device" begin
    backend = CPU()
    constraints = choicemap((:y, 0.3))
    seed = 20260721
    common = (num_steps=25, num_particles=8, initial_params=[0.0], initial_log_scale=0.0)
    cpu = batched_advi(devh_conjugate_gauss, (), constraints; common..., rng=MersenneTwister(seed))
    dev = batched_advi(devh_conjugate_gauss, (), constraints; common..., backend=backend, rng=MersenneTwister(seed))
    @test maximum(abs.(cpu.elbo_history .- dev.elbo_history)) < 1e-13
    @test maximum(abs.(cpu.location .- dev.location)) < 1e-13
    @test maximum(abs.(cpu.log_scale .- dev.log_scale)) < 1e-13
end

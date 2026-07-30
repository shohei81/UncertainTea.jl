# Device-resident ChEES-HMC (issue #161, increment 4). These tests pin the device
# analogue of the CPU `batched_chees`: the jittered device trajectory (one sync per
# iteration), the warmup-only cross-chain ChEES trajectory-length adaptation
# (proposal/momentum download on the host, reusing the CPU `_chees_trajectory_update!`
# verbatim), and the statistical guarantees (posterior recovery + `T` convergence).
#
# The primary leg runs on KernelAbstractions.CPU() (the authoritative reference
# backend) at Float64: the device path keeps the RNG host-side, so its results are
# STATISTICALLY -- and here numerically, being Float64 on the same host RNG shape --
# comparable to the CPU `batched_chees`. The Metal (Float32) leg under test/gpu/
# mirrors the posterior-recovery smoke test at a looser tolerance.

using KernelAbstractions: CPU

# Local mean/std helpers (Statistics is not imported by the test harness).
devc_mean(x) = sum(x) / length(x)
function devc_std(x)
    m = devc_mean(x)
    accumulator = 0.0
    for value in x
        accumulator += (value - m)^2
    end
    return sqrt(accumulator / (length(x) - 1))
end

# Conjugate gaussian: mu ~ N(0,1), y ~ N(mu,1) observed at y = 0.3.
# Posterior for mu | y=0.3 is N(0.15, 0.5): mean 0.15, sd sqrt(0.5).
@tea static function devc_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# Diagonal Gaussian whose marginal scales span 100x (0.1 .. 10). No data, so the
# posterior equals the prior; a good sampler must mix all scales at once.
@tea static function devc_illconditioned()
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 10.0)
    c ~ normal(0.0, 0.1)
    return (a, b, c)
end

@testset "device ChEES-HMC (issue #161 increment 4)" begin
    conj_constraints = choicemap((:y, 0.3))

    @testset "conjugate gauss recovery + T convergence (CPU backend)" begin
        backend = CPU()
        trace = Float64[]
        chees = batched_chees(
            devc_conjugate_gauss,
            (),
            conj_constraints;
            num_chains=16,
            num_samples=2000,
            num_warmup=600,
            backend=backend,
            precision=Float64,
            rng=MersenneTwister(161),
            _trajectory_trace=trace,
        )

        samples = posterior_array(chees)
        @test all(isfinite, samples)
        @test isapprox(devc_mean(samples), 0.15; atol=0.05)
        @test isapprox(devc_std(samples), sqrt(0.5); atol=0.08)

        summary = summarize(chees)
        @test summary[1].mean ≈ 0.15 atol = 0.05
        @test summary[1].sd ≈ sqrt(0.5) atol = 0.08
        @test summary[1].rhat < 1.05

        # T moved from its warm-start and settled to a finite, sensible value (the CPU
        # path settles near T ≈ 15 here) with a stable tail (no runaway/oscillation).
        @test length(trace) == 600
        @test all(isfinite, trace)
        @test 0.5 < trace[end] < 200.0
        tail = trace[(end-149):end]
        @test maximum(tail) / minimum(tail) < 3.0
    end

    @testset "device CPU backend ≈ host batched_chees" begin
        # Same seed + same shape on both paths. On CPU()/Float64 with a host-side RNG,
        # the device ChEES and host ChEES agree on posterior moments and converge T to
        # a similar value (statistical, not bitwise, equivalence -- the leapfrog runs
        # in device kernels with a different arithmetic reduction order).
        host_trace = Float64[]
        device_trace = Float64[]
        common = (
            num_chains=16,
            num_samples=1500,
            num_warmup=600,
        )
        host = batched_chees(
            devc_conjugate_gauss, (), conj_constraints;
            common...,
            rng=MersenneTwister(20260730),
            _trajectory_trace=host_trace,
        )
        device = batched_chees(
            devc_conjugate_gauss, (), conj_constraints;
            common...,
            backend=CPU(),
            precision=Float64,
            rng=MersenneTwister(20260730),
            _trajectory_trace=device_trace,
        )

        host_summary = summarize(host)
        device_summary = summarize(device)
        @test device_summary[1].mean ≈ host_summary[1].mean atol = 0.03
        @test device_summary[1].sd ≈ host_summary[1].sd atol = 0.05
        @test device_summary[1].rhat < 1.05
        # Adapted T lands in the same neighborhood on both paths.
        @test device_trace[end] ≈ host_trace[end] rtol = 0.5
        @test 0.5 < device_trace[end] < 200.0
    end

    @testset "ill-conditioned recovery + T convergence (CPU backend)" begin
        backend = CPU()
        trace = Float64[]
        chees = batched_chees(
            devc_illconditioned,
            (),
            choicemap();
            num_chains=32,
            num_samples=2500,
            num_warmup=800,
            backend=backend,
            precision=Float64,
            rng=MersenneTwister(2024),
            _trajectory_trace=trace,
        )
        @test all(isfinite, trace)
        @test 0.1 < trace[end] < 1000.0
        summary = summarize(chees; space=:unconstrained)
        for (index, target_sd) in enumerate((1.0, 10.0, 0.1))
            @test abs(summary[index].mean) < 0.2 * target_sd
            @test summary[index].sd ≈ target_sd rtol = 0.12
            @test summary[index].rhat < 1.05
        end
    end

    @testset "adaptation off freezes T and downloads nothing for adaptation" begin
        # With adapt_trajectory_length=false the device sampling+warmup loop never
        # downloads the proposal/momentum for a ChEES update, and T never moves. This
        # is the structural "sampling is a pure device loop" guarantee: the only extra
        # warmup transfer is the ChEES download, which this path elides entirely.
        trace = Float64[]
        chain = batched_chees(
            devc_conjugate_gauss,
            (),
            conj_constraints;
            num_chains=4,
            num_samples=100,
            num_warmup=150,
            num_leapfrog_steps=10,
            adapt_trajectory_length=false,
            backend=CPU(),
            precision=Float64,
            rng=MersenneTwister(7),
            _trajectory_trace=trace,
        )
        @test length(unique(round.(trace; digits=10))) == 1
        @test all(isfinite, posterior_array(chain))
    end

    @testset "backend argument validation" begin
        @test_throws ArgumentError batched_chees(
            devc_conjugate_gauss, (), conj_constraints;
            num_chains=4, num_samples=10, num_warmup=10, backend=:not_a_backend,
        )
    end
end

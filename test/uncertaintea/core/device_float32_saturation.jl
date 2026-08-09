# Issue #367 item 3: device Float32 saturation audit, pinned on the
# KernelAbstractions CPU() reference backend (the CI-testable stand-in for
# Metal; test/gpu mirrors parity on real hardware).
#
# The device kernels are branchless and run at the workspace precision, so the
# saturation boundaries sit elsewhere in Float32 than in the host's Float64:
#   * exp(theta) is subnormal for theta in ~(-103.97, -87.34) and exactly 0.0
#     below, so the positive-support log-latent boundary is ~ -87, not ~ -708;
#   * sigmoid(theta) rounds to exactly 1.0f0 just below theta ~ 17, so the
#     logit-family cliff of issue #343 is ~ 17, not ~ 36.7.
#
# Audit outcome (this file pins it):
#   * The Float32 exp-subnormal band WAS a real hazard: values built from the
#     few surviving mantissa bits were silently wrong by up to O(30) nats
#     (lognormal at theta = -103) with finite gradients -- nothing rejected.
#     Fixed minimally by mirroring the host issue-#345/#367 boundary in the
#     device kernels (`_device_positive_normal`, a branchless `>= floatmin`
#     support test in the value precision) for the same seven families:
#     lognormal, gamma, inversegamma, weibull, frechet, rayleigh,
#     inversegaussian. The band now scores -Inf, so device integrators reject
#     on the VALUE.
#   * Past-boundary points (exp underflow to 0.0, sigmoid rounding to 1.0)
#     already scored -Inf; the accompanying gradient is the finite
#     Jacobian-only derivative because the branchless neginf selection drops
#     the density's derivative channel. That is NOT the silent hazard of
#     issue #343 -- the -Inf value rejects -- and it deliberately mirrors the
#     host single ForwardDiff path at exp underflow (issue #354's documented
#     residual), so the #354 NaN-poison semantics are not replicated in the
#     fragile branchless kernels (device numerics pinned by #218/#298).
#   * The logit pre-cliff band (theta ~ 13..17 at Float32) keeps a
#     ~eps32 * exp(theta) error envelope on value and gradient while both stay
#     finite -- the Float32 analogue of the host theta = 30..36.7 band, with
#     the same root cause (the kernel sees only the rounded sigmoid value) and
#     the same fused-density fix dependency (issue #367 item 1).

@tea static function dfs_gamma()
    x ~ gamma(2.0, 1.0)
end
@tea static function dfs_lognormal()
    x ~ lognormal(0.0, 1.0)
end
@tea static function dfs_weibull()
    x ~ weibull(2.0, 1.5)
end
@tea static function dfs_inversegamma()
    x ~ inversegamma(3.0, 2.0)
end
@tea static function dfs_frechet()
    x ~ frechet(2.0, 1.5)
end
@tea static function dfs_rayleigh()
    x ~ rayleigh(1.5)
end
@tea static function dfs_inversegaussian()
    x ~ inversegaussian(1.0, 2.0)
end
@tea static function dfs_exponential()
    x ~ exponential(1.5)
end
@tea static function dfs_beta_bern()
    p ~ beta(2.0, 2.0)
    {:y} ~ bernoulli(p)
end

dfs_guarded = (
    dfs_gamma,
    dfs_lognormal,
    dfs_weibull,
    dfs_inversegamma,
    dfs_frechet,
    dfs_rayleigh,
    dfs_inversegaussian,
)

dfs_point(m, theta, args, cm; precision=Float32) = device_batched_logjoint_gradient(
    m, reshape([precision(theta)], 1, 1), args, cm; precision=precision)

@testset "device_f32_exp_subnormal_boundary" begin
    # In the Float32 subnormal band and past underflow the guarded families
    # score -Inf32: rejection is value-driven. The gradient is the finite
    # Jacobian-only derivative (see header), never NaN.
    for dfs_model in dfs_guarded, dfs_theta in (-90.0, -100.0, -120.0)
        v, g = dfs_point(dfs_model, dfs_theta, (), choicemap())
        @test v[1] == -Inf32
        @test !isnan(g[1, 1])
    end

    # Just outside the band exp32(theta) is a normal Float32: the guarded
    # families still score finite values that track the Float64 host truth.
    for dfs_model in (dfs_gamma, dfs_lognormal, dfs_weibull, dfs_rayleigh)
        v, g = dfs_point(dfs_model, -80.0, (), choicemap())
        vref = batched_logjoint_unconstrained(dfs_model, reshape([-80.0], 1, 1), ())
        gref = batched_logjoint_gradient_unconstrained(dfs_model, reshape([-80.0], 1, 1), ())
        @test isfinite(v[1])
        @test Float64(v[1]) ≈ vref[1] rtol = 1e-4
        @test Float64(g[1, 1]) ≈ gref[1, 1] rtol = 1e-4
    end

    # Audited no-guard family: exponential's composite is exact through the
    # band and past underflow (no log(x)/1/x value term).
    for dfs_theta in (-90.0, -120.0)
        v, g = dfs_point(dfs_exponential, dfs_theta, (), choicemap())
        @test isfinite(v[1])
        @test g[1, 1] ≈ 1.0f0 atol = 1e-6
    end

    # The boundary follows the VALUE precision: at Float64 the same thetas are
    # far from the Float64 boundary and stay finite, while the Float64
    # subnormal band (host issue #367 item 2) rejects on device too.
    for dfs_model in (dfs_gamma, dfs_lognormal)
        v64, g64 = dfs_point(dfs_model, -100.0, (), choicemap(); precision=Float64)
        gref = batched_logjoint_gradient_unconstrained(dfs_model, reshape([-100.0], 1, 1), ())
        @test isfinite(v64[1])
        @test g64[1, 1] ≈ gref[1, 1] rtol = 1e-9
        vband, gband = dfs_point(dfs_model, -720.0, (), choicemap(); precision=Float64)
        @test vband[1] == -Inf
        @test !isnan(gband[1, 1])
    end
end

@testset "device_f32_logit_cliff" begin
    dfs_cm = choicemap(:y => true)

    # Past the Float32 sigmoid cliff the value saturates to -Inf32 (rejection
    # is value-driven); the gradient stays the finite Jacobian-only -1.0f0
    # (see header: the host #354 poison is deliberately not mirrored).
    for dfs_theta in (18.0, 20.0, 30.0)
        v, g = dfs_point(dfs_beta_bern, dfs_theta, (), dfs_cm)
        @test v[1] == -Inf32
        @test !isnan(g[1, 1])
    end

    # Pre-cliff Float32 band: finite value and gradient inside the
    # ~eps32 * exp(theta) envelope of the Float64 host truth (factor 4
    # margin) -- the Float32 analogue of the host theta = 30..36.7 residual.
    for dfs_theta in (13.0, 15.0, 16.0)
        v, g = dfs_point(dfs_beta_bern, dfs_theta, (), dfs_cm)
        vref = batched_logjoint_unconstrained(dfs_beta_bern, reshape([dfs_theta], 1, 1), (), dfs_cm)
        gref = batched_logjoint_gradient_unconstrained(dfs_beta_bern, reshape([dfs_theta], 1, 1), (), dfs_cm)
        dfs_tol = 4 * eps(Float32) * exp(dfs_theta) + 1e-5
        @test isfinite(v[1])
        @test Float64(v[1]) ≈ vref[1] atol = dfs_tol
        @test isfinite(g[1, 1])
        @test Float64(g[1, 1]) ≈ gref[1, 1] atol = dfs_tol
    end

    # Normal range stays tight against the host reference.
    for dfs_theta in (-1.5, 0.3, 2.0)
        v, g = dfs_point(dfs_beta_bern, dfs_theta, (), dfs_cm)
        vref = batched_logjoint_unconstrained(dfs_beta_bern, reshape([dfs_theta], 1, 1), (), dfs_cm)
        gref = batched_logjoint_gradient_unconstrained(dfs_beta_bern, reshape([dfs_theta], 1, 1), (), dfs_cm)
        @test Float64(v[1]) ≈ vref[1] atol = 1e-5
        @test Float64(g[1, 1]) ≈ gref[1, 1] atol = 1e-5
    end
end

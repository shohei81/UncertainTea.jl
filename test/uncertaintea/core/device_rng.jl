# issue #154 (increment 1): on-device counter-based RNG (Philox4x32-10).
#
# This is the self-contained prerequisite for the persistent per-chain NUTS tree
# kernel: because a data-dependent tree cannot use host-pre-drawn randomness, the
# kernel must draw its own. These tests pin down three things:
#   1. Bit-exact correctness of `philox4x32` against the published Random123 /
#      Salmon et al. (2011) known-answer test (KAT) vectors -- the algorithm is a
#      standard, so we validate against the standard rather than ourselves. The
#      hardcoded vectors below were additionally cross-checked, during
#      development, against Random123.jl v1.7.1 over 200k random (key, counter)
#      pairs with zero mismatches (Random123 is NOT a package dependency, so the
#      vectors are frozen here rather than recomputed at test time).
#   2. Cross-backend determinism: the SAME coordinate produces bit-identical
#      draws from a KernelAbstractions CPU() kernel and from a direct host call
#      (the Metal leg is a smoke test under test/gpu/, since the core suite has no
#      Metal dependency). Counter-based RNG is a pure function of its inputs, so
#      "determinism" here means the device path introduces no divergence. NOTE
#      the honest boundary confirmed by the Metal smoke test: `philox4x32` and
#      `device_rand_uniform` are bit-identical ACROSS backends too (pure integer
#      mixing + exact divide), but `device_rand_normal` uses Box-Muller's log/cos
#      whose Metal libm differs in the last bits -- so it is bit-identical only
#      WITHIN a backend (the === checks below all run on CPU()) and merely
#      statistically equivalent across CPU vs Metal (issue #154 risk (a)).
#   3. Distributional sanity: `device_rand_uniform` is ~Uniform(0,1) and
#      `device_rand_normal` is ~N(0,1) over a large coordinate sweep.
#
# Run standalone:
#   julia --project=. -e 'using Test,Random,UncertainTea; \
#     include("test/uncertaintea/fixtures.jl"); \
#     include("test/uncertaintea/core/device_rng.jl")'

using KernelAbstractions: @kernel, @index, CPU, allocate, synchronize
const drng_KA = UncertainTea.KernelAbstractions

# --- KA kernels exercising the device RNG through the real @kernel path --------
# One kernel per primitive, dispatching on `eltype(out)` so a single kernel body
# serves both Float32 (Metal) and Float64 (CPU()) without a type-parameterized
# @kernel signature.

@kernel function _drng_philox_block_kernel!(out)
    # out is 4 x N; column i draws the Philox block for a distinct counter.
    i = @index(Global)
    blk = UncertainTea.philox4x32(
        (UInt32(0x243f6a88), UInt32(i)),
        (UInt32(0x13198a2e), UInt32(i) * UInt32(2654435761), UInt32(7), UInt32(0)),
    )
    @inbounds out[1, i] = blk[1]
    @inbounds out[2, i] = blk[2]
    @inbounds out[3, i] = blk[3]
    @inbounds out[4, i] = blk[4]
end

@kernel function _drng_uniform_kernel!(out)
    i = @index(Global)
    T = eltype(out)
    @inbounds out[i] =
        UncertainTea.device_rand_uniform(T, UInt32(0), UInt32(0), UInt32(0), UInt32(i))
end

@kernel function _drng_normal_kernel!(out)
    i = @index(Global)
    T = eltype(out)
    @inbounds out[i] =
        UncertainTea.device_rand_normal(T, UInt32(0), UInt32(0), UInt32(1), UInt32(i))
end

# Host mirrors (the same pure functions, called directly) for the determinism
# comparison and the KAT/distribution checks.
_drng_host_uniform(::Type{T}, i) where {T} =
    UncertainTea.device_rand_uniform(T, UInt32(0), UInt32(0), UInt32(0), UInt32(i))
_drng_host_normal(::Type{T}, i) where {T} =
    UncertainTea.device_rand_normal(T, UInt32(0), UInt32(0), UInt32(1), UInt32(i))

@testset "device_rng_philox_kat" begin
    # Canonical Philox4x32-10 known-answer vectors (key, counter) -> block.
    # These are the Random123 kat_vectors entries for philox4x32 R=10.
    kat = [
        ((0x00000000, 0x00000000),
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
        ((0xffffffff, 0xffffffff),
            (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
            (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
        ((0xa4093822, 0x299f31d0),
            (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
            (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
    ]
    for (key, ctr, expected) in kat
        got = UncertainTea.philox4x32(UInt32.(key), UInt32.(ctr))
        @test got === UInt32.(expected)
    end

    # Bijection sanity: distinct counters give distinct blocks (no collisions
    # over a sweep), and the block is genuinely a full NTuple{4,UInt32}.
    blocks = Set{NTuple{4,UInt32}}()
    for i = 1:5000
        push!(blocks, UncertainTea.philox4x32((UInt32(0), UInt32(0)), (UInt32(i), UInt32(0), UInt32(0), UInt32(0))))
    end
    @test length(blocks) == 5000
end

@testset "device_rng_cross_backend_determinism" begin
    # The CPU() KernelAbstractions kernel must reproduce the host-side pure calls
    # BIT-FOR-BIT (===), for the raw block and for both derived draws.
    backend = drng_KA.CPU()
    N = 4096

    # raw Philox block through the kernel vs host
    blk_dev = drng_KA.allocate(backend, UInt32, 4, N)
    _drng_philox_block_kernel!(backend)(blk_dev; ndrange=N)
    drng_KA.synchronize(backend)
    for i = 1:N
        host = UncertainTea.philox4x32(
            (UInt32(0x243f6a88), UInt32(i)),
            (UInt32(0x13198a2e), UInt32(i) * UInt32(2654435761), UInt32(7), UInt32(0)),
        )
        @test (blk_dev[1, i], blk_dev[2, i], blk_dev[3, i], blk_dev[4, i]) === host
    end

    for T in (Float32, Float64)
        u_dev = drng_KA.allocate(backend, T, N)
        _drng_uniform_kernel!(backend)(u_dev; ndrange=N)
        drng_KA.synchronize(backend)
        @test all(i -> u_dev[i] === _drng_host_uniform(T, i), 1:N)

        n_dev = drng_KA.allocate(backend, T, N)
        _drng_normal_kernel!(backend)(n_dev; ndrange=N)
        drng_KA.synchronize(backend)
        @test all(i -> n_dev[i] === _drng_host_normal(T, i), 1:N)
    end
end

@testset "device_rng_uniform_range_and_distribution" begin
    for T in (Float32, Float64)
        N = 1_000_000
        us = [_drng_host_uniform(T, i) for i = 1:N]
        # strictly in [0, 1) -- never 1.0 (the normal transform relies on this).
        @test all(u -> zero(T) <= u < one(T), us)
        @test eltype(us) === T
        # first two moments of Uniform(0,1): mean 1/2, variance 1/12.
        m = sum(Float64.(us)) / N
        v = sum(x -> (Float64(x) - m)^2, us) / N
        @test isapprox(m, 0.5; atol=2e-3)
        @test isapprox(v, 1 / 12; atol=2e-3)
        # coarse 10-bin histogram: every decile ~0.1 of the mass.
        counts = zeros(Int, 10)
        for u in us
            b = clamp(Int(floor(Float64(u) * 10)) + 1, 1, 10)
            counts[b] += 1
        end
        for c in counts
            @test isapprox(c / N, 0.1; atol=3e-3)
        end
    end
end

@testset "device_rng_normal_distribution" begin
    for T in (Float32, Float64)
        N = 1_000_000
        zs = [_drng_host_normal(T, i) for i = 1:N]
        @test all(isfinite, zs)
        @test eltype(zs) === T
        m = sum(Float64.(zs)) / N
        v = sum(x -> (Float64(x) - m)^2, zs) / N
        @test isapprox(m, 0.0; atol=5e-3)   # mean 0
        @test isapprox(v, 1.0; atol=1e-2)   # variance 1
        # tail coverage: ~68.27% within 1 sigma, ~95.45% within 2 sigma.
        within1 = count(z -> abs(Float64(z)) <= 1.0, zs) / N
        within2 = count(z -> abs(Float64(z)) <= 2.0, zs) / N
        @test isapprox(within1, 0.6827; atol=5e-3)
        @test isapprox(within2, 0.9545; atol=5e-3)
    end
end

@testset "device_rng_stream_independence" begin
    # Different stream_ids at the same (chain, iteration, draw) must not correlate:
    # a persistent kernel uses distinct streams for momentum vs accept draws.
    N = 200_000
    a = [UncertainTea.device_rand_uniform(Float64, UInt32(3), UInt32(9), UInt32(0), UInt32(i)) for i = 1:N]
    b = [UncertainTea.device_rand_uniform(Float64, UInt32(3), UInt32(9), UInt32(1), UInt32(i)) for i = 1:N]
    @test a != b
    ma = sum(a) / N
    mb = sum(b) / N
    corr = (sum(a .* b) / N - ma * mb) / (sqrt(sum(x -> (x - ma)^2, a) / N) * sqrt(sum(x -> (x - mb)^2, b) / N))
    @test abs(corr) < 1e-2   # essentially uncorrelated streams
end

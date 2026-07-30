# Manual Metal GPU smoke test for the device-resident batched logjoint.
#
# NOT part of the package test target and NOT run in CI. Requires a functional Metal
# GPU (Apple Silicon). See test/gpu/Project.toml for setup instructions.
#
# It runs the same parity checks as test/uncertaintea/core/device_lowering_parity.jl, but on a
# Metal.MetalBackend at Float32, and compares against the authoritative CPU
# `batched_logjoint_unconstrained` reference computed in Float64.

using Random
using Test
using UncertainTea
using Metal

@tea static function gpu_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

@tea static function gpu_bernoulli_model(n)
    p ~ beta(2.0, 3.0)
    for i = 1:n
        {:z => i} ~ bernoulli(p)
    end
    return p
end

function gpu_check_float32(dev32, ref)
    ok = true
    for (d, r) in zip(dev32, ref)
        ok &= isapprox(Float64(d), r; rtol=1e-3, atol=1e-3 * max(1.0, abs(r)))
    end
    return ok
end

@testset "device Metal GPU smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()

    # Float64 must be rejected on Metal via the precision guard.
    @test_throws ArgumentError DeviceBatchedWorkspace(gpu_gauss_model, 2; backend=backend, precision=Float64)

    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    ref = batched_logjoint_unconstrained(gpu_gauss_model, params, (5,), cm)
    dev = device_batched_logjoint(gpu_gauss_model, Float32.(params), (5,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)

    zs = [1.0, 0.0, 1.0, 1.0]
    cmb = choicemap((:z => i, zs[i]) for i = 1:4)
    pb = reshape([0.3, -0.8, 1.5], 1, 3)
    refb = batched_logjoint_unconstrained(gpu_bernoulli_model, pb, (4,), cmb)
    devb = device_batched_logjoint(gpu_bernoulli_model, Float32.(pb), (4,), cmb; backend=backend, precision=Float32)
    @test gpu_check_float32(devb, refb)
end

@testset "device Metal GPU gradient smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU gradient smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()

    # gaussian + gamma(log) latent with loop observations.
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    gref = batched_logjoint_gradient_unconstrained(gpu_gauss_model, params, (5,), cm)
    vref = batched_logjoint_unconstrained(gpu_gauss_model, params, (5,), cm)
    v, g = device_batched_logjoint_gradient(gpu_gauss_model, Float32.(params), (5,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
    @test gpu_check_float32(Float64.(v), vref)

    # bernoulli + beta(logit) latent.
    zs = [1.0, 0.0, 1.0, 1.0]
    cmb = choicemap((:z => i, zs[i]) for i = 1:4)
    pb = reshape([0.3, -0.8, 1.5], 1, 3)
    grefb = batched_logjoint_gradient_unconstrained(gpu_bernoulli_model, pb, (4,), cmb)
    _, gb = device_batched_logjoint_gradient(gpu_bernoulli_model, Float32.(pb), (4,), cmb; backend=backend, precision=Float32)
    @test gpu_check_float32(vec(Float64.(gb)), vec(grefb))
end

# --- issue #12 group 1 families: logjoint + gradient parity smoke ------------

@tea static function gpu_heavytail_model(n)
    loc ~ studentt(4.0, 0.0, 1.0)
    s ~ inversegamma(3.0, 2.0)
    w ~ weibull(2.0, 1.5)
    for i = 1:n
        {:y => i} ~ studentt(5.0, loc, s + w)
    end
    return loc
end

@tea static function gpu_count_model(n)
    p ~ beta(2.0, 2.0)
    {:k} ~ binomial(20, p)
    {:g} ~ geometric(p)
    {:nb} ~ negativebinomial(4.0, p)
    for i = 1:n
        {:c => i} ~ categorical(p / 2.0, p / 2.0, 1.0 - p)
    end
    return p
end

@testset "device Metal GPU group-1 family smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU group-1 smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()

    # continuous: studentt (identity) + inversegamma/weibull (log) latents.
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2; -0.4 0.2 0.6]
    ref = batched_logjoint_unconstrained(gpu_heavytail_model, params, (5,), cm)
    dev = device_batched_logjoint(gpu_heavytail_model, Float32.(params), (5,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)
    gref = batched_logjoint_gradient_unconstrained(gpu_heavytail_model, params, (5,), cm)
    _, g = device_batched_logjoint_gradient(
        gpu_heavytail_model,
        Float32.(params),
        (5,),
        cm;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))

    # discrete: binomial / geometric / negativebinomial / categorical observations.
    cs = [1.0, 3.0, 2.0, 3.0]
    cm2 = choicemap((:k, 7.0), (:g, 3.0), (:nb, 5.0), ((:c => i, cs[i]) for i = 1:4)...)
    params2 = reshape([0.3, -0.8, 1.5], 1, 3)
    ref2 = batched_logjoint_unconstrained(gpu_count_model, params2, (4,), cm2)
    dev2 = device_batched_logjoint(gpu_count_model, Float32.(params2), (4,), cm2; backend=backend, precision=Float32)
    @test gpu_check_float32(dev2, ref2)
    g2ref = batched_logjoint_gradient_unconstrained(gpu_count_model, params2, (4,), cm2)
    _, g2 = device_batched_logjoint_gradient(
        gpu_count_model,
        Float32.(params2),
        (4,),
        cm2;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g2)), vec(g2ref))
end

# --- issue #12 group 2: truncated families smoke ------------------------------

@tea static function gpu_truncated_model(n)
    m ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 2.0)
    {:h} ~ truncatednormal(m, 1.0, 0.0, Inf)
    for i = 1:n
        {:y => i} ~ truncatednormal(m, s, -1.0, 2.0)
        {:t => i} ~ truncatedstudentt(5.0, m, s, -2.0, 2.0)
    end
    return m
end

@testset "device Metal GPU truncated family smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU truncated smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    ys = [0.4, -0.7, 1.1]
    ts = [1.5, -0.2, 0.8]
    cm = choicemap((:h, 0.6), ((:y => i, ys[i]) for i = 1:3)..., ((:t => i, ts[i]) for i = 1:3)...)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2]
    ref = batched_logjoint_unconstrained(gpu_truncated_model, params, (3,), cm)
    dev = device_batched_logjoint(gpu_truncated_model, Float32.(params), (3,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)
    gref = batched_logjoint_gradient_unconstrained(gpu_truncated_model, params, (3,), cm)
    _, g = device_batched_logjoint_gradient(
        gpu_truncated_model,
        Float32.(params),
        (3,),
        cm;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
end

# --- issue #12 group 3 phase 1: diagonal mvnormal smoke -----------------------

@tea static function gpu_mvnormal_model(n)
    state ~ mvnormal([0.0, 1.0], [1.5, 0.8])
    s ~ gamma(2.0, 2.0)
    {:w} ~ mvnormal([0.3, -0.2], [1.0, 2.0])
    for i = 1:n
        {:v => i} ~ mvnormal([0.5, -0.5], [1.0, s])
    end
    return s
end

@testset "device Metal GPU mvnormal smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU mvnormal smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    vs = [[0.2, -0.1], [1.0, 0.4], [-0.3, 0.9]]
    cm = choicemap((:w, [0.1, -0.4]), ((:v => i, vs[i]) for i = 1:3)...)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2; -0.4 0.2 0.6]
    ref = batched_logjoint_unconstrained(gpu_mvnormal_model, params, (3,), cm)
    dev = device_batched_logjoint(gpu_mvnormal_model, Float32.(params), (3,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)
    gref = batched_logjoint_gradient_unconstrained(gpu_mvnormal_model, params, (3,), cm)
    _, g = device_batched_logjoint_gradient(
        gpu_mvnormal_model,
        Float32.(params),
        (3,),
        cm;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
end

# --- issue #12 group 3 phase 2: dirichlet smoke --------------------------------

@tea static function gpu_dirichlet_model(n)
    theta ~ dirichlet([2.0, 3.0, 4.0])
    s ~ gamma(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ normal(0.5, s)
    end
    return s
end

@testset "device Metal GPU dirichlet smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU dirichlet smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    ys = [0.4, -0.7, 1.1]
    cm = choicemap((:y => i, ys[i]) for i = 1:3)
    params = [0.3 -0.5; -0.2 0.4; 0.1 0.6]
    ref = batched_logjoint_unconstrained(gpu_dirichlet_model, params, (3,), cm)
    dev = device_batched_logjoint(gpu_dirichlet_model, Float32.(params), (3,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)
    gref = batched_logjoint_gradient_unconstrained(gpu_dirichlet_model, params, (3,), cm)
    _, g = device_batched_logjoint_gradient(
        gpu_dirichlet_model,
        Float32.(params),
        (3,),
        cm;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
end

# --- issue #12 group 3 phase 3: mvnormaldense smoke ----------------------------

@tea static function gpu_mvdense_model(L)
    state ~ mvnormaldense([0.0, 1.0], L)
    m ~ normal(0.0, 1.0)
    {:y} ~ mvnormaldense([m, m], L)
    return m
end

@testset "device Metal GPU mvnormaldense smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU mvnormaldense smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    L = [1.2 0.0; 0.4 0.9]
    cm = choicemap((:y, [0.4, -0.2]))
    params = [0.5 -0.3; 0.1 0.7; -0.4 0.2]
    ref = batched_logjoint_unconstrained(gpu_mvdense_model, params, (L,), cm)
    dev = device_batched_logjoint(gpu_mvdense_model, Float32.(params), (L,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)
    gref = batched_logjoint_gradient_unconstrained(gpu_mvdense_model, params, (L,), cm)
    _, g = device_batched_logjoint_gradient(
        gpu_mvdense_model,
        Float32.(params),
        (L,),
        cm;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
end

# --- issue #50: mixture-of-normals smoke ---------------------------------------

@tea static function gpu_mixture_model(n)
    x ~ mixture((0.5, 0.5), normal(-2.0, 0.5), normal(2.0, 0.5))
    s ~ gamma(2.0, 2.0)
    {:z} ~ normal(x * 2.0, 0.3)
    for i = 1:n
        {:y => i} ~ mixture((0.3, 0.7), normal(x, s), normal(2.0, 0.5))
    end
    return x
end

@testset "device Metal GPU mixture smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU mixture smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    ys = [1.4, -1.8, 2.2]
    cm = choicemap((:z, 3.5), ((:y => i, ys[i]) for i = 1:3)...)
    params = [1.8 -2.1; 0.1 0.6]
    ref = batched_logjoint_unconstrained(gpu_mixture_model, params, (3,), cm)
    dev = device_batched_logjoint(gpu_mixture_model, Float32.(params), (3,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)
    gref = batched_logjoint_gradient_unconstrained(gpu_mixture_model, params, (3,), cm)
    _, g = device_batched_logjoint_gradient(
        gpu_mixture_model,
        Float32.(params),
        (3,),
        cm;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
end

# --- issue #57: lkjcholesky smoke ------------------------------------------------

@tea static function gpu_lkj_model(n)
    Omega ~ lkjcholesky(3, 2.0)
    s ~ gamma(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ normal(0.5, s)
    end
    return s
end

@testset "device Metal GPU lkjcholesky smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU lkjcholesky smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    ys = [0.4, -0.7, 1.1]
    cm = choicemap((:y => i, ys[i]) for i = 1:3)
    params = [0.3 -0.5; -0.2 0.4; 0.7 -0.1; 0.1 0.6]
    ref = batched_logjoint_unconstrained(gpu_lkj_model, params, (3,), cm)
    dev = device_batched_logjoint(gpu_lkj_model, Float32.(params), (3,), cm; backend=backend, precision=Float32)
    @test gpu_check_float32(dev, ref)
    gref = batched_logjoint_gradient_unconstrained(gpu_lkj_model, params, (3,), cm)
    _, g = device_batched_logjoint_gradient(
        gpu_lkj_model,
        Float32.(params),
        (3,),
        cm;
        backend=backend,
        precision=Float32,
    )
    @test gpu_check_float32(vec(Float64.(g)), vec(gref))
end

# --- device-resident batched HMC / ADVI smoke (PR 46) ------------------------
# Mirrors test/uncertaintea/core/device_hmc_advi.jl on a Metal.MetalBackend at Float32.
# RNG stays host-side, so results are statistically (not bitwise) equivalent to the
# CPU path; we only assert finite results and posterior/variational-mean sanity.

@tea static function gpu_conjugate_gauss()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

@testset "device Metal GPU batched HMC/ADVI smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU HMC/ADVI smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    constraints = choicemap((:y, 0.3))

    chains = batched_hmc(
        gpu_conjugate_gauss,
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
    @test all(isfinite, samples)
    @test isapprox(sum(samples) / length(samples), 0.15; atol=0.15)
    @test all(<(1.3), rhat(chains))

    result = batched_advi(
        gpu_conjugate_gauss,
        (),
        constraints;
        num_steps=300,
        backend=backend,
        precision=Float32,
        rng=MersenneTwister(46),
    )
    @test all(isfinite, result.elbo_history)
    @test isapprox(variational_mean(result)[1], 0.15; atol=0.15)
end

# --- device-resident masked batched NUTS smoke (issue #2) --------------------
# Mirrors test/uncertaintea/core/device_masked_nuts.jl on a Metal.MetalBackend at Float32.
# The masked doubling trajectory runs device-resident: the per-leaf P x C arrays and
# tree ops (leapfrog, checkpoints, U-turn) live on the device. The default Tier-2 path
# (issue #152) pre-draws each round's RNG and runs the per-leaf accept/select on the
# device too, reading the subtree state back once per round; results are statistically
# (not bitwise) equivalent to the CPU masked path.

@testset "device Metal GPU batched NUTS smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU NUTS smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    constraints = choicemap((:y, 0.3))

    chains = batched_nuts(
        gpu_conjugate_gauss,
        (),
        constraints;
        num_chains=2,
        num_samples=300,
        num_warmup=150,
        tree_strategy=:masked,
        backend=backend,
        precision=Float32,
        rng=MersenneTwister(46),
    )
    samples = posterior_array(chains)
    @test all(isfinite, samples)
    @test isapprox(sum(samples) / length(samples), 0.15; atol=0.15)
    @test all(<(1.3), rhat(chains))
end

# --- device-resident ChEES-HMC smoke (issue #161 increment 4) ----------------
# Mirrors test/uncertaintea/core/device_chees.jl on a Metal.MetalBackend at Float32.
# The Halton-jittered leapfrog trajectory runs device-resident (one sync per
# iteration); the cross-chain ChEES trajectory-length adaptation runs on the host
# during warmup from a per-iteration proposal/momentum download. RNG stays host-side,
# so results are statistically (not bitwise) equivalent to the CPU path; we assert
# finite results, posterior-mean sanity, and that the adapted T converges finite.

@testset "device Metal GPU batched ChEES smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU ChEES smoke test."
        @test true
        return
    end

    backend = Metal.MetalBackend()
    constraints = choicemap((:y, 0.3))

    trace = Float64[]
    chains = batched_chees(
        gpu_conjugate_gauss,
        (),
        constraints;
        num_chains=32,
        num_samples=2000,
        num_warmup=600,
        backend=backend,
        precision=Float32,
        rng=MersenneTwister(161),
        _trajectory_trace=trace,
    )
    samples = posterior_array(chains)
    @test all(isfinite, samples)
    @test isapprox(sum(samples) / length(samples), 0.15; atol=0.15)
    @test all(<(1.3), rhat(chains))
    # Adapted trajectory time converged to a finite, sensible value.
    @test length(trace) == 600
    @test all(isfinite, trace)
    @test 0.5 < trace[end] < 200.0
end

@tea static function gpu_noncentered_funnel()
    v ~ normal(0.0, 3.0)
    x ~ normal(0.0, exp(v / 2); reparam=:noncentered)
    {:y} ~ normal(x, 10.0)
    return v
end

@testset "device Metal GPU noncentered normal parity" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU smoke test."
        @test true
        return
    end
    constraints = choicemap((:y, 0.5))
    points = Float64[0.3 -0.5; 0.9 0.2]
    supported, _ = device_lowering_report(gpu_noncentered_funnel)
    @test supported
    values, gradients = device_batched_logjoint_gradient(
        gpu_noncentered_funnel,
        Float32.(points),
        (),
        constraints;
        backend=Metal.MetalBackend(),
        precision=Float32,
    )
    reference_values =
        [logjoint_unconstrained(gpu_noncentered_funnel, points[:, i], (), constraints) for i = 1:2]
    reference_gradients = hcat(
        [logjoint_gradient_unconstrained(gpu_noncentered_funnel, points[:, i], (), constraints) for i = 1:2]...,
    )
    # Metal runs Float32
    @test Float64.(collect(values)) ≈ reference_values atol = 1e-4
    @test Float64.(collect(gradients)) ≈ reference_gradients atol = 1e-4
end

# --- issue #154 (increment 1): on-device Philox4x32-10 RNG Metal smoke ---------
# Mirrors test/uncertaintea/core/device_rng.jl's cross-backend determinism leg on
# a real Metal.MetalBackend. The honest boundary, MEASURED on Apple M4:
#   * The integer Philox block and the `device_rand_uniform` draw ARE
#     bit-identical (===) between Metal and the host, because they are pure
#     integer mixing plus an exact integer->float divide.
#   * `device_rand_normal` is NOT bit-identical across backends: Box-Muller uses
#     `log`/`cos`, whose Metal libm differs from the host libm in the last bits
#     (measured max abs diff ~5e-7 in Float32, ~1854/4096 draws differ). This is
#     exactly issue #154's risk (a) -- the Gaussian transform is only
#     STATISTICALLY equivalent across backends, to be gated by the #121 harness,
#     not bitwise. We therefore assert the normal matches the host to a few-ULP
#     tolerance (not ===) and is finite.
using KernelAbstractions

@kernel function _gpu_drng_block_kernel!(out)
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

@kernel function _gpu_drng_uniform_kernel!(out)
    i = @index(Global)
    @inbounds out[i] =
        UncertainTea.device_rand_uniform(Float32, UInt32(0), UInt32(0), UInt32(0), UInt32(i))
end

@kernel function _gpu_drng_normal_kernel!(out)
    i = @index(Global)
    @inbounds out[i] =
        UncertainTea.device_rand_normal(Float32, UInt32(0), UInt32(0), UInt32(1), UInt32(i))
end

@testset "device Metal GPU Philox RNG smoke" begin
    if !Metal.functional()
        @info "Metal GPU not functional; skipping GPU smoke test."
        @test true
        return
    end
    backend = Metal.MetalBackend()
    N = 4096

    # raw block: Metal vs host, bit-for-bit
    blk_dev = KernelAbstractions.allocate(backend, UInt32, 4, N)
    _gpu_drng_block_kernel!(backend)(blk_dev; ndrange=N)
    KernelAbstractions.synchronize(backend)
    blk_host = Array(blk_dev)
    for i = 1:N
        ref = UncertainTea.philox4x32(
            (UInt32(0x243f6a88), UInt32(i)),
            (UInt32(0x13198a2e), UInt32(i) * UInt32(2654435761), UInt32(7), UInt32(0)),
        )
        @test (blk_host[1, i], blk_host[2, i], blk_host[3, i], blk_host[4, i]) === ref
    end

    # Float32 uniform: Metal vs host bit-identical, and in [0, 1)
    u_dev = KernelAbstractions.allocate(backend, Float32, N)
    _gpu_drng_uniform_kernel!(backend)(u_dev; ndrange=N)
    KernelAbstractions.synchronize(backend)
    u_host = Array(u_dev)
    @test all(
        i -> u_host[i] ===
             UncertainTea.device_rand_uniform(Float32, UInt32(0), UInt32(0), UInt32(0), UInt32(i)),
        1:N,
    )
    @test all(u -> 0.0f0 <= u < 1.0f0, u_host)

    # Float32 normal: only STATISTICALLY equal across backends (Box-Muller's
    # log/cos libm differs on Metal), so compare to the host to a few-ULP
    # tolerance rather than ===, and check finiteness + distributional sanity.
    n_dev = KernelAbstractions.allocate(backend, Float32, N)
    _gpu_drng_normal_kernel!(backend)(n_dev; ndrange=N)
    KernelAbstractions.synchronize(backend)
    n_host = Array(n_dev)
    n_ref = [
        UncertainTea.device_rand_normal(Float32, UInt32(0), UInt32(0), UInt32(1), UInt32(i))
        for i = 1:N
    ]
    @test all(isfinite, n_host)
    @test maximum(abs.(n_host .- n_ref)) < 1.0f-5   # few-ULP transcendental drift
    @test abs(sum(n_host) / N) < 0.1                # mean ~ 0
    @test abs(sum(x -> x^2, n_host) / N - 1) < 0.1  # variance ~ 1
end

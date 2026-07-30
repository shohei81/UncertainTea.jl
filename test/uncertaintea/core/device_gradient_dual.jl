# PR 6.2: device-resident batched gradient of the UNCONSTRAINED logjoint.
#
# One fused KernelAbstractions kernel over a 2D ndrange (parameter_index, batch_index)
# forward-differentiates the same device plan the logjoint kernel walks, using the
# scalar `DeviceDual` number seeded on the thread's unconstrained parameter. All tests
# run on KernelAbstractions.CPU(); the GPU smoke test under test/gpu/ mirrors the
# parity checks on a Metal backend at Float32.
#
# The device gradient takes UNCONSTRAINED parameters and differentiates through the
# in-kernel transform (including its log-abs-det), so the authoritative CPU
# counterparts are `batched_logjoint_gradient_unconstrained` (gradients) and
# `batched_logjoint_unconstrained` (values).

const devg_FD = UncertainTea.ForwardDiff
const DeviceDual = UncertainTea.DeviceDual

# --- shared device-lowerable models (mirroring device_lowering_parity.jl) -----------------------
@tea static function devg_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

@tea static function devg_lognormal_gamma_model(n)
    a ~ lognormal(0.0, 1.0)
    b ~ gamma(3.0, 2.0)
    scale = exp(a)
    for i = 1:n
        {:y => i} ~ normal(b, scale)
    end
    return a
end

@tea static function devg_bernoulli_model(n)
    p ~ beta(2.0, 3.0)
    for i = 1:n
        {:z => i} ~ bernoulli(p)
    end
    return p
end

# backend-supported but device-UNSUPPORTED negative example. marginalize=:enumerate
# became device-lowerable in issue #67 (and broadcast-normal in #134), so an
# lkjcholesky latent above the device dimension cap is a genuinely-still-unsupported
# case: the backend scores it, the device rejects on the compile-time cap.
@tea static function devg_unsupported_model()
    Omega ~ lkjcholesky(9, 2.0)
    return Omega
end

# issue #12 group 1 mirrors of device_lowering_parity.jl.
@tea static function devg_heavytail_model(n)
    loc ~ studentt(4.0, 0.0, 1.0)
    s ~ inversegamma(3.0, 2.0)
    w ~ weibull(2.0, 1.5)
    for i = 1:n
        {:y => i} ~ studentt(5.0, loc, s + w)
    end
    return loc
end

@tea static function devg_count_model(n)
    p ~ beta(2.0, 2.0)
    {:k} ~ binomial(20, p)
    {:g} ~ geometric(p)
    {:nb} ~ negativebinomial(4.0, p)
    for i = 1:n
        {:c => i} ~ categorical(p / 2.0, p / 2.0, 1.0 - p)
    end
    return p
end

@tea static function devg_named_trials_model(n)
    p ~ beta(2.0, 2.0)
    trials = n + 1
    {:k} ~ binomial(trials, p)
    return p
end

# poisson observations with a gamma(log) latent rate: regression for round() on
# DeviceDual (the discrete-support check inside the dual gradient walk).
@tea static function devg_poisson_model(n)
    lam ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ poisson(lam)
    end
    return lam
end

@testset "devg_dual_algebra" begin
    # For each supported DeviceDual operation, the .deriv channel must match
    # ForwardDiff.derivative of the equivalent scalar function on a grid of points.
    grid = collect(-2.5:0.37:2.5)
    posgrid = collect(0.1:0.23:3.0)

    dseed(x) = DeviceDual(x, 1.0)   # seed derivative 1

    # unary op set: (device function on dual, scalar function, domain)
    unary = [
        (x -> exp(dseed(x)), exp, grid),
        (x -> log(dseed(x)), log, posgrid),
        (x -> log1p(dseed(x)), log1p, posgrid),
        (x -> sqrt(dseed(x)), sqrt, posgrid),
        (x -> sin(dseed(x)), sin, grid),
        (x -> cos(dseed(x)), cos, grid),
        (x -> abs(dseed(x)), abs, filter(!iszero, grid)),
        (x -> -dseed(x), x -> -x, grid),
        (x -> dseed(x)^3, x -> x^3, grid),                 # integer power
        (x -> dseed(x)^2, x -> x^2, grid),
        (x -> dseed(x)^0.5, x -> x^0.5, posgrid),          # real power
        (x -> dseed(x)^2.5, x -> x^2.5, posgrid),
        (x -> inv(dseed(x)), inv, filter(!iszero, grid)),
    ]
    for (dfun, sfun, domain) in unary
        for x in domain
            @test dfun(x).deriv ≈ devg_FD.derivative(sfun, x) atol = 1e-12
            @test dfun(x).value ≈ sfun(x) atol = 1e-12
        end
    end

    # binary ops seeding one or the other argument.
    for x in grid, y in grid
        @test (dseed(x) + DeviceDual(y, 0.0)).deriv ≈ devg_FD.derivative(t -> t + y, x) atol = 1e-12
        @test (dseed(x) - DeviceDual(y, 0.0)).deriv ≈ devg_FD.derivative(t -> t - y, x) atol = 1e-12
        @test (dseed(x) * DeviceDual(y, 0.0)).deriv ≈ devg_FD.derivative(t -> t * y, x) atol = 1e-12
        # * seeding both (product rule): d/dx (x*x) at x
        @test (dseed(x) * dseed(x)).deriv ≈ devg_FD.derivative(t -> t * t, x) atol = 1e-12
    end
    for x in grid, y in filter(!iszero, grid)
        @test (dseed(x) / DeviceDual(y, 0.0)).deriv ≈ devg_FD.derivative(t -> t / y, x) atol = 1e-12
    end

    # min / max / clamp at non-tie points.
    for x in grid
        @test min(dseed(x), DeviceDual(0.5, 0.0)).deriv ≈ devg_FD.derivative(t -> min(t, 0.5), x) atol = 1e-12
        @test max(dseed(x), DeviceDual(0.5, 0.0)).deriv ≈ devg_FD.derivative(t -> max(t, 0.5), x) atol = 1e-12
        @test clamp(dseed(x), DeviceDual(-1.0, 0.0), DeviceDual(1.0, 0.0)).deriv ≈
              devg_FD.derivative(t -> clamp(t, -1.0, 1.0), x) atol = 1e-12
    end

    # promotion with a plain real scalar and isbits-ness.
    @test (dseed(1.3) + 2.0).deriv ≈ 1.0 atol = 1e-12
    @test (2.0 * dseed(1.3)).deriv ≈ 2.0 atol = 1e-12
    @test isbits(DeviceDual(1.0, 2.0))
    @test DeviceDual(1.0f0, 2.0f0) isa DeviceDual{Float32}
end

@testset "devg_gradient_parity_f64" begin
    # --- model 1: gaussian with loop observations, gamma(log) latent ---
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    v, g = device_batched_logjoint_gradient(devg_gauss_model, params, (5,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_gauss_model, params, (5,), cm)
    vref = batched_logjoint_unconstrained(devg_gauss_model, params, (5,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12

    # --- model 2: lognormal + gamma latents, exp-derived arg ---
    ys2 = [0.4, 1.1, -0.7]
    cm2 = choicemap((:y => i, ys2[i]) for i = 1:3)
    params2 = [0.5 -0.3 0.9; 0.1 0.7 -0.2]
    v2, g2 = device_batched_logjoint_gradient(devg_lognormal_gamma_model, params2, (3,), cm2)
    g2ref = batched_logjoint_gradient_unconstrained(devg_lognormal_gamma_model, params2, (3,), cm2)
    v2ref = batched_logjoint_unconstrained(devg_lognormal_gamma_model, params2, (3,), cm2)
    @test g2 ≈ g2ref rtol = 1e-10
    @test v2 ≈ v2ref rtol = 1e-12

    # --- model 3: bernoulli observations, beta(logit) latent ---
    zs = [1.0, 0.0, 1.0, 1.0]
    cm3 = choicemap((:z => i, zs[i]) for i = 1:4)
    params3 = reshape([0.3, -0.8, 1.5], 1, 3)
    v3, g3 = device_batched_logjoint_gradient(devg_bernoulli_model, params3, (4,), cm3)
    g3ref = batched_logjoint_gradient_unconstrained(devg_bernoulli_model, params3, (4,), cm3)
    v3ref = batched_logjoint_unconstrained(devg_bernoulli_model, params3, (4,), cm3)
    @test g3 ≈ g3ref rtol = 1e-10
    @test v3 ≈ v3ref rtol = 1e-12
end

@testset "devg_gradient_parity_f32" begin
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    gref = batched_logjoint_gradient_unconstrained(devg_gauss_model, params, (5,), cm)
    _, g32 = device_batched_logjoint_gradient(devg_gauss_model, Float32.(params), (5,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3

    ys2 = [0.4, 1.1, -0.7]
    cm2 = choicemap((:y => i, ys2[i]) for i = 1:3)
    params2 = [0.5 -0.3 0.9; 0.1 0.7 -0.2]
    g2ref = batched_logjoint_gradient_unconstrained(devg_lognormal_gamma_model, params2, (3,), cm2)
    _, g2_32 = device_batched_logjoint_gradient(devg_lognormal_gamma_model, Float32.(params2), (3,), cm2; precision=Float32)
    @test Float64.(g2_32) ≈ g2ref rtol = 1e-3 atol = 1e-3

    zs = [1.0, 0.0, 1.0, 1.0]
    cm3 = choicemap((:z => i, zs[i]) for i = 1:4)
    params3 = reshape([0.3, -0.8, 1.5], 1, 3)
    g3ref = batched_logjoint_gradient_unconstrained(devg_bernoulli_model, params3, (4,), cm3)
    _, g3_32 = device_batched_logjoint_gradient(devg_bernoulli_model, Float32.(params3), (4,), cm3; precision=Float32)
    @test Float64.(g3_32) ≈ g3ref rtol = 1e-3 atol = 1e-3
end

@testset "devg_gradient_parity_group1" begin
    # --- continuous: studentt / inversegamma / weibull ---
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2; -0.4 0.2 0.6]
    v, g = device_batched_logjoint_gradient(devg_heavytail_model, params, (5,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_heavytail_model, params, (5,), cm)
    vref = batched_logjoint_unconstrained(devg_heavytail_model, params, (5,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12
    _, g32 = device_batched_logjoint_gradient(devg_heavytail_model, Float32.(params), (5,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3

    # --- discrete: binomial / geometric / negativebinomial / categorical ---
    cs = [1.0, 3.0, 2.0, 3.0]
    cm2 = choicemap((:k, 7.0), (:g, 3.0), (:nb, 5.0), ((:c => i, cs[i]) for i = 1:4)...)
    params2 = reshape([0.3, -0.8, 1.5], 1, 3)
    v2, g2 = device_batched_logjoint_gradient(devg_count_model, params2, (4,), cm2)
    g2ref = batched_logjoint_gradient_unconstrained(devg_count_model, params2, (4,), cm2)
    v2ref = batched_logjoint_unconstrained(devg_count_model, params2, (4,), cm2)
    @test g2 ≈ g2ref rtol = 1e-10
    @test v2 ≈ v2ref rtol = 1e-12
    _, g2_32 = device_batched_logjoint_gradient(devg_count_model, Float32.(params2), (4,), cm2; precision=Float32)
    @test Float64.(g2_32) ≈ g2ref rtol = 1e-3 atol = 1e-3

    # --- binomial trials via a named (index-slot) deterministic binding ---
    cmk = choicemap((:k, 7.0))
    paramsk = reshape([0.3, -0.8, 1.5], 1, 3)
    vk, gk = device_batched_logjoint_gradient(devg_named_trials_model, paramsk, (10,), cmk)
    gkref = batched_logjoint_gradient_unconstrained(devg_named_trials_model, paramsk, (10,), cmk)
    vkref = batched_logjoint_unconstrained(devg_named_trials_model, paramsk, (10,), cmk)
    @test gk ≈ gkref rtol = 1e-10
    @test vk ≈ vkref rtol = 1e-12

    # --- poisson round-on-dual regression ---
    cmp = choicemap((:y => 1, 3.0), (:y => 2, 1.0))
    paramsp = reshape([0.2, -0.5], 1, 2)
    vp, gp = device_batched_logjoint_gradient(devg_poisson_model, paramsp, (2,), cmp)
    gpref = batched_logjoint_gradient_unconstrained(devg_poisson_model, paramsp, (2,), cmp)
    vpref = batched_logjoint_unconstrained(devg_poisson_model, paramsp, (2,), cmp)
    @test gp ≈ gpref rtol = 1e-10
    @test vp ≈ vpref rtol = 1e-12
end

@tea static function devg_truncated_model(n)
    m ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 2.0)
    {:h} ~ truncatednormal(m, 1.0, 0.0, Inf)
    for i = 1:n
        {:y => i} ~ truncatednormal(m, s, -1.0, 2.0)
        {:t => i} ~ truncatedstudentt(5.0, m, s, -2.0, 2.0)
    end
    return m
end

@testset "devg_gradient_parity_truncated" begin
    # gradients flow through the analytic-derivative duals of erf/erfc and the
    # t-CDF (d/dz = pdf; the literal-nu channel is genuinely zero), including a
    # one-sided Inf bound whose derivative must pin to zero rather than NaN.
    ys = [0.4, -0.7, 1.1]
    ts = [1.5, -0.2, 0.8]
    cm = choicemap((:h, 0.6), ((:y => i, ys[i]) for i = 1:3)..., ((:t => i, ts[i]) for i = 1:3)...)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2]
    v, g = device_batched_logjoint_gradient(devg_truncated_model, params, (3,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_truncated_model, params, (3,), cm)
    vref = batched_logjoint_unconstrained(devg_truncated_model, params, (3,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12
    _, g32 = device_batched_logjoint_gradient(devg_truncated_model, Float32.(params), (3,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3
end

@tea static function devg_mvnormal_model(n)
    state ~ mvnormal([0.0, 1.0], [1.5, 0.8])
    s ~ gamma(2.0, 2.0)
    {:w} ~ mvnormal([0.3, -0.2], [1.0, 2.0])
    for i = 1:n
        {:v => i} ~ mvnormal([0.5, -0.5], [1.0, s])
    end
    return s
end

@testset "devg_gradient_parity_mvnormal" begin
    # per-component dual seeding across the latent vector rows, dual sigma
    # flowing into an in-loop vector observation, and the strided cursor
    vs = [[0.2, -0.1], [1.0, 0.4], [-0.3, 0.9]]
    cm = choicemap((:w, [0.1, -0.4]), ((:v => i, vs[i]) for i = 1:3)...)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2; -0.4 0.2 0.6]
    v, g = device_batched_logjoint_gradient(devg_mvnormal_model, params, (3,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_mvnormal_model, params, (3,), cm)
    vref = batched_logjoint_unconstrained(devg_mvnormal_model, params, (3,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12
    _, g32 = device_batched_logjoint_gradient(devg_mvnormal_model, Float32.(params), (3,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3
end

@tea static function devg_dirichlet_group3_model(n)
    theta ~ dirichlet([2.0, 3.0, 4.0])
    s ~ gamma(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ normal(0.5, s)
    end
    return s
end

@tea static function devg_dirichlet_obs_model()
    a ~ gamma(2.0, 1.0)
    {:w} ~ dirichlet([a, 2.0, 3.0])
    return a
end

@testset "devg_gradient_parity_dirichlet" begin
    # duals flow through the register softmax (simplex Jacobian + log-abs-det
    # derivative) and through a latent-flowing concentration on the observed side
    ys = [0.4, -0.7, 1.1]
    cm = choicemap((:y => i, ys[i]) for i = 1:3)
    params = [0.3 -0.5; -0.2 0.4; 0.1 0.6]
    v, g = device_batched_logjoint_gradient(devg_dirichlet_group3_model, params, (3,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_dirichlet_group3_model, params, (3,), cm)
    vref = batched_logjoint_unconstrained(devg_dirichlet_group3_model, params, (3,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12
    _, g32 = device_batched_logjoint_gradient(devg_dirichlet_group3_model, Float32.(params), (3,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3

    cm_obs = choicemap((:w, [0.2, 0.3, 0.5]))
    params_obs = reshape([0.1, -0.4], 1, 2)
    v2, g2 = device_batched_logjoint_gradient(devg_dirichlet_obs_model, params_obs, (), cm_obs)
    g2ref = batched_logjoint_gradient_unconstrained(devg_dirichlet_obs_model, params_obs, (), cm_obs)
    @test g2 ≈ g2ref rtol = 1e-10
end

# issue #57: lkjcholesky mirrors of device_lowering_parity.jl.
@tea static function devg_lkj_group_model(n)
    Omega ~ lkjcholesky(3, 2.0)
    s ~ gamma(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ normal(0.5, s)
    end
    return s
end

@tea static function devg_lkj_obs_model()
    e ~ gamma(2.0, 1.0)
    {:L} ~ lkjcholesky(3, e)
    return e
end

@testset "devg_gradient_parity_lkjcholesky" begin
    # duals flow through the register tanh/stick constrain (z-space analytic
    # gradient + log-abs-det derivative) and through a latent-flowing eta (the
    # loggamma normalizer chain) on the observed side
    ys = [0.4, -0.7, 1.1]
    cm = choicemap((:y => i, ys[i]) for i = 1:3)
    params = [0.3 -0.5; -0.2 0.4; 0.7 -0.1; 0.1 0.6]
    v, g = device_batched_logjoint_gradient(devg_lkj_group_model, params, (3,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_lkj_group_model, params, (3,), cm)
    vref = batched_logjoint_unconstrained(devg_lkj_group_model, params, (3,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12
    _, g32 = device_batched_logjoint_gradient(devg_lkj_group_model, Float32.(params), (3,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3

    cm_obs = choicemap((:L, [1.0, 0.6, 0.3, 0.8, -0.4, sqrt(0.75)]))
    params_obs = reshape([0.1, -0.4], 1, 2)
    v2, g2 = device_batched_logjoint_gradient(devg_lkj_obs_model, params_obs, (), cm_obs)
    g2ref = batched_logjoint_gradient_unconstrained(devg_lkj_obs_model, params_obs, (), cm_obs)
    @test g2 ≈ g2ref rtol = 1e-10

    # Float32 tanh saturation (|z| ~ 12 rounds a partial correlation to +-1)
    # degrades the value to -Inf but must NOT emit NaN gradients: the
    # eagerly-evaluated log1p(-1) / sqrt(0) duals carry 0/0 derivatives that
    # the constrain's saturation guards discard (codex review of issue #57)
    params_sat = Float32.([12.0 12.0; 12.0 0.4; 12.0 -0.6; 0.1 0.1])
    v_sat, g_sat = device_batched_logjoint_gradient(devg_lkj_group_model, params_sat, (3,), cm; precision=Float32)
    @test all(==(-Inf32), v_sat)
    @test !any(isnan, g_sat)
end

@tea static function devg_mvdense_model(L)
    state ~ mvnormaldense([0.0, 1.0], L)
    m ~ normal(0.0, 1.0)
    {:y} ~ mvnormaldense([m, m], L)
    return m
end

@testset "devg_gradient_parity_mvnormaldense" begin
    # duals flow through the unrolled forward substitution; the packed factor
    # rows are constant duals (zero derivative), matching the CPU contract
    L = [1.2 0.0; 0.4 0.9]
    cm = choicemap((:y, [0.4, -0.2]))
    params = [0.5 -0.3; 0.1 0.7; -0.4 0.2]
    v, g = device_batched_logjoint_gradient(devg_mvdense_model, params, (L,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_mvdense_model, params, (L,), cm)
    vref = batched_logjoint_unconstrained(devg_mvdense_model, params, (L,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12
    _, g32 = device_batched_logjoint_gradient(devg_mvdense_model, Float32.(params), (L,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3
end

@tea static function devg_mixture_device_model(n)
    x ~ mixture((0.5, 0.5), normal(-2.0, 0.5), normal(2.0, 0.5))
    s ~ gamma(2.0, 2.0)
    {:z} ~ normal(x * 2.0, 0.3)
    for i = 1:n
        {:y => i} ~ mixture((0.3, 0.7), normal(x, s), normal(2.0, 0.5))
    end
    return x
end

@testset "devg_gradient_parity_mixture" begin
    # duals flow through the max-shifted log-sum-exp (responsibility weighting)
    # and through the materialized scalar binding into a downstream observation
    ys = [1.4, -1.8, 2.2]
    cm = choicemap((:z, 3.5), ((:y => i, ys[i]) for i = 1:3)...)
    params = [1.8 -2.1; 0.1 0.6]
    v, g = device_batched_logjoint_gradient(devg_mixture_device_model, params, (3,), cm)
    gref = batched_logjoint_gradient_unconstrained(devg_mixture_device_model, params, (3,), cm)
    vref = batched_logjoint_unconstrained(devg_mixture_device_model, params, (3,), cm)
    @test g ≈ gref rtol = 1e-10
    @test v ≈ vref rtol = 1e-12
    _, g32 = device_batched_logjoint_gradient(devg_mixture_device_model, Float32.(params), (3,), cm; precision=Float32)
    @test Float64.(g32) ≈ gref rtol = 1e-3 atol = 1e-3
end

@testset "devg_unsupported_throws" begin
    err = try
        device_batched_logjoint_gradient(devg_unsupported_model, reshape([0.1], 1, 1), ())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("device_lowering_report", err.msg)
end

@testset "devg_workspace_reuse" begin
    ys = [0.4, -0.7, 1.1, 0.2]
    cm = choicemap((:y => i, ys[i]) for i = 1:4)
    ws = DeviceBatchedWorkspace(devg_gauss_model, 3; args=(4,), constraints=cm)

    params_a = [0.5 -0.3 1.2; 0.1 0.7 -0.2]
    params_b = params_a .+ 0.75
    v_a, g_a = device_batched_logjoint_gradient!(ws, params_a)
    v_b, g_b = device_batched_logjoint_gradient!(ws, params_b)
    v_a2, g_a2 = device_batched_logjoint_gradient!(ws, params_a)

    g_ref_a = batched_logjoint_gradient_unconstrained(devg_gauss_model, params_a, (4,), cm)
    g_ref_b = batched_logjoint_gradient_unconstrained(devg_gauss_model, params_b, (4,), cm)
    @test g_a ≈ g_ref_a rtol = 1e-10
    @test g_b ≈ g_ref_b rtol = 1e-10
    # second/third calls do not leave stale state.
    @test g_a2 == g_a
    @test v_a2 == v_a
end

# issue #38 regression: model arguments stage into the device slot storage
@tea static function devg_argument_model(offset)
    tau ~ lognormal(0.0, 0.5)
    theta ~ normal(offset, tau)
    {:y} ~ normal(theta, 0.5)
    return theta
end

# rebinding an argument symbol would let kernels overwrite the staged slot
@tea static function devg_rebinding_model(x)
    x = x + 1.0
    theta ~ normal(x, 1.0)
    {:y} ~ normal(theta, 0.5)
    return theta
end

@testset "devg_argument_staging" begin
    rebind_supported, rebind_issues = device_lowering_report(devg_rebinding_model)
    @test !rebind_supported
    @test any(occursin("rebinding", issue) for issue in rebind_issues)

    constraints = choicemap((:y, 0.4))
    points = [0.2 -0.3; 0.7 0.1]
    values, gradients = device_batched_logjoint_gradient(
        devg_argument_model,
        points,
        (2.5,),
        constraints;
    )
    @test collect(values) ≈
          [logjoint_unconstrained(devg_argument_model, points[:, i], (2.5,), constraints) for i = 1:2] atol = 1e-10
    @test collect(gradients) ≈ hcat(
        [logjoint_gradient_unconstrained(devg_argument_model, points[:, i], (2.5,), constraints) for i = 1:2]...,
    ) atol = 1e-10

    # per-chain argument tuples stage per column
    batch_args = [(1.0,), (3.0,)]
    batch_values, batch_gradients = device_batched_logjoint_gradient(
        devg_argument_model,
        points,
        batch_args,
        constraints;
    )
    @test collect(batch_values) ≈
          [logjoint_unconstrained(devg_argument_model, points[:, i], batch_args[i], constraints) for i = 1:2] atol =
        1e-10
    @test collect(batch_gradients) ≈ hcat(
        [
            logjoint_gradient_unconstrained(devg_argument_model, points[:, i], batch_args[i], constraints) for
            i = 1:2
        ]...,
    ) atol = 1e-10
end

# issue #104: near tanh saturation (row square-sums approaching 1) the device
# lkjcholesky gradient used to drift from the analytic gradient because the dual
# chain went through the subtractive `1 - sum_sqs`. The constrain now tracks the
# leftover variance MULTIPLICATIVELY (remaining *= 1 - tanh(z)^2), which makes the
# forward-mode derivative well-conditioned. The truth here is a 256-bit BigFloat
# central finite difference of the CPU logjoint (independent of either gradient
# formula); the device now matches it to FP noise on in-support states.
@tea static function devg_lkj8_model()
    Omega ~ lkjcholesky(8, 2.0)
    return Omega
end

@testset "devg_lkjcholesky_saturation_gradient" begin
    nz = (8 * 7) ÷ 2
    base = Float64[0.3, -0.5, 0.2, 0.7, -0.1, 0.4, -0.3, 0.6, 0.9, -1.1,
        1.3, -1.4, 1.15, 1.35, -1.25, 1.45, -1.15, 1.55, 1.05, -1.35,
        1.25, -1.45, 1.12, 1.32, -1.22, 1.42, -1.12, 1.52]

    bigfloat_fd(params, i; h=1e-30) = setprecision(BigFloat, 256) do
        p = BigFloat.(params)
        pp = copy(p)
        pp[i, 1] += h
        pm = copy(p)
        pm[i, 1] -= h
        vp = batched_logjoint_unconstrained(devg_lkj8_model, pp, (), choicemap())[1]
        vm = batched_logjoint_unconstrained(devg_lkj8_model, pm, (), choicemap())[1]
        return Float64((vp - vm) / (2 * h))
    end

    # s = 2.0 drives some z up to ~3.0 (row square-sums near 1) while staying in
    # support; this is where the old dual chain drifted (~1e-4 at |z|~2-3).
    for s in (1.0, 2.0)
        params = reshape(base .* s, nz, 1)
        v, g = device_batched_logjoint_gradient(devg_lkj8_model, params, (), choicemap())
        @test isfinite(v[1]) # in-support column
        worst_row = argmax(abs.(vec(g) .- vec(bigfloat_fd.(Ref(params), 1:nz))))
        fd = bigfloat_fd(params, worst_row)
        # device gradient is within FP noise of the exact (BigFloat FD) gradient
        @test isapprox(g[worst_row, 1], fd; rtol=1e-8, atol=1e-8)
    end

    # Benign region: device still agrees with the CPU analytic gradient tightly.
    params = reshape(base .* 0.5, nz, 1)
    _, g = device_batched_logjoint_gradient(devg_lkj8_model, params, (), choicemap())
    gref = batched_logjoint_gradient_unconstrained(devg_lkj8_model, params, (), choicemap())
    @test g ≈ gref rtol = 1e-9
end

# --- issue #153: observation-parallel tiled gradient ----------------------------
# For a large observation loop the device lowers the loop to a tiled partial-sum
# kernel + per-chain reduction (occupancy over the serial re-scan). The tile
# reduction reassociates the observation sum, so the result is only STATISTICALLY
# equal to the untiled serial scan; the authoritative CPU reference is the same
# `batched_logjoint(_gradient)_unconstrained` used everywhere else, and Float64
# agreement to rtol ~1e-6 confirms the reassociation is the only change. Below
# `DEVICE_GRADIENT_TILE_MIN_OBS` the serial scan is kept, so small models stay on
# the bitwise-identical path (covered by every other parity test in this file).

@tea static function devg_tiled_glm(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ mvnormal((0.0, 0.0, 0.0, 0.0), (2.5, 2.5, 2.5, 2.5))
    for i = 1:n
        {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
    end
    return alpha
end

@testset "devg_tiled_gradient_parity" begin
    tile_min = UncertainTea.DEVICE_GRADIENT_TILE_MIN_OBS

    # gauss: large loop engages tiling, matches the serial reference to Float64 rtol.
    let n = 1000, B = 4, rng = MersenneTwister(153)
        ys = randn(rng, n)
        cm = choicemap((:y => i, ys[i]) for i = 1:n)
        params = randn(rng, 2, B)
        ws = DeviceBatchedWorkspace(devg_gauss_model, B; precision=Float64, args=(n,), constraints=cm)
        @test !isnothing(ws.tiled_gradient)
        @test ws.tiled_gradient.n_obs == n
        # shared constraints/args -> single staged observation column
        @test size(ws.observed_device, 2) == 1
        v, g = device_batched_logjoint_gradient!(ws, params)
        @test v ≈ batched_logjoint_unconstrained(devg_gauss_model, params, (n,), cm) rtol = 1e-6
        @test g ≈ batched_logjoint_gradient_unconstrained(devg_gauss_model, params, (n,), cm) rtol = 1e-6
    end

    # logistic GLM (fused linear predictor): tiled body seeds the per-coefficient
    # gradient correctly; no count-family step, so the integer mirror is dropped.
    let n = 800, D = 4, B = 4, rng = MersenneTwister(154)
        X = randn(rng, D, n)
        ys = Float64.(rand(rng, Bool, n))
        cm = choicemap((:y => i, ys[i]) for i = 1:n)
        params = 0.3 .* randn(rng, 1 + D, B)
        ws = DeviceBatchedWorkspace(devg_tiled_glm, B; precision=Float64, args=(X, n), constraints=cm)
        @test !isnothing(ws.tiled_gradient)
        @test size(ws.observed_device, 2) == 1
        @test size(ws.observed_int_device) == (1, 1)   # no count family -> dropped
        v, g = device_batched_logjoint_gradient!(ws, params)
        @test v ≈ batched_logjoint_unconstrained(devg_tiled_glm, params, (X, n), cm) rtol = 1e-6
        @test g ≈ batched_logjoint_gradient_unconstrained(devg_tiled_glm, params, (X, n), cm) rtol = 1e-6
    end

    # just below the threshold keeps the serial scan (no tiled descriptor)
    let n = tile_min - 1, B = 3, rng = MersenneTwister(155)
        ys = randn(rng, n)
        cm = choicemap((:y => i, ys[i]) for i = 1:n)
        params = randn(rng, 2, B)
        ws = DeviceBatchedWorkspace(devg_gauss_model, B; precision=Float64, args=(n,), constraints=cm)
        @test isnothing(ws.tiled_gradient)
        v, g = device_batched_logjoint_gradient!(ws, params)
        @test g ≈ batched_logjoint_gradient_unconstrained(devg_gauss_model, params, (n,), cm) rtol = 1e-9
    end

    # per-chain (batched-argument) observations keep the C-column staging AND still
    # tile: the accessor broadcasts a shared column but indexes per-chain otherwise.
    let n = 400, B = 3, rng = MersenneTwister(156)
        ys = randn(rng, n)
        cm = choicemap((:y => i, ys[i]) for i = 1:n)
        batched_args = [(n,) for _ = 1:B]
        params = randn(rng, 2, B)
        ws = DeviceBatchedWorkspace(devg_gauss_model, B; precision=Float64, args=batched_args, constraints=cm)
        @test size(ws.observed_device, 2) == B    # genuinely per-chain columns
        @test !isnothing(ws.tiled_gradient)
        v, g = device_batched_logjoint_gradient!(ws, params)
        @test g ≈ batched_logjoint_gradient_unconstrained(devg_gauss_model, params, (n,), cm) rtol = 1e-6
    end
end

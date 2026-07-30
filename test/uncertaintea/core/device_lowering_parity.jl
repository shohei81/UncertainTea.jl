# PR 42: first real device-execution phase -- device lowering + a device-resident
# batched (unconstrained) logjoint fused into a single KernelAbstractions kernel.
# All tests here run on KernelAbstractions.CPU() (the default backend); the GPU
# smoke test under test/gpu/ mirrors these on a MetalBackend and is intentionally
# not wired into CI.
#
# The device path takes UNCONSTRAINED parameters and folds the transform log-abs-det
# in-kernel, so the authoritative CPU counterpart is `batched_logjoint_unconstrained`.

# --- shared device-lowerable models -----------------------------------------
# gaussian mean/scale with a log-transformed (gamma) latent and loop observations.
@tea static function dev_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# lognormal + gamma latents with an exp-derived scale argument.
@tea static function dev_lognormal_gamma_model(n)
    a ~ lognormal(0.0, 1.0)
    b ~ gamma(3.0, 2.0)
    scale = exp(a)
    for i = 1:n
        {:y => i} ~ normal(b, scale)
    end
    return a
end

# bernoulli observations governed by a logit-transformed (beta) latent probability.
@tea static function dev_bernoulli_model(n)
    p ~ beta(2.0, 3.0)
    for i = 1:n
        {:z => i} ~ bernoulli(p)
    end
    return p
end

# backend-supported but device-UNSUPPORTED negative example. marginalize=:enumerate
# became device-lowerable in issue #67, and broadcast-normal became device-lowerable
# in #134, so both are no longer valid "unsupported" cases; an lkjcholesky latent
# above the device dimension cap is a genuinely-still-unsupported one (the backend
# scores it, the device rejects on the compile-time cap).
@tea static function dev_unsupported_model()
    Omega ~ lkjcholesky(9, 2.0)
    return Omega
end

# issue #12 group 1 (continuous): studentt (identity latent), inversegamma and
# weibull (log latents), studentt loop observations with a derived scale.
@tea static function dev_heavytail_model(n)
    loc ~ studentt(4.0, 0.0, 1.0)
    s ~ inversegamma(3.0, 2.0)
    w ~ weibull(2.0, 1.5)
    for i = 1:n
        {:y => i} ~ studentt(5.0, loc, s + w)
    end
    return loc
end

# issue #12 group 1 (discrete): binomial/geometric/negativebinomial observations and
# categorical loop observations, all driven by one beta (logit) latent probability.
@tea static function dev_count_model(n)
    p ~ beta(2.0, 2.0)
    {:k} ~ binomial(20, p)
    {:g} ~ geometric(p)
    {:nb} ~ negativebinomial(4.0, p)
    for i = 1:n
        {:c => i} ~ categorical(p / 2.0, p / 2.0, 1.0 - p)
    end
    return p
end

# helper: assert Float32 device output matches a Float64 reference with the
# rtol 1e-4 / atol 1e-4*max(1,|ref|) contract, elementwise.
function dev_check_float32(dev32, ref)
    ok = true
    for (d, r) in zip(dev32, ref)
        ok &= isapprox(Float64(d), r; rtol=1e-4, atol=1e-4 * max(1.0, abs(r)))
    end
    return ok
end

@testset "dev_lowering_report" begin
    supported, issues = device_lowering_report(dev_gauss_model)
    @test supported
    @test isempty(issues)

    supported_ln, _ = device_lowering_report(dev_lognormal_gamma_model)
    @test supported_ln
    supported_bern, _ = device_lowering_report(dev_bernoulli_model)
    @test supported_bern

    marg_supported, marg_issues = device_lowering_report(dev_unsupported_model)
    @test !marg_supported
    @test !isempty(marg_issues)
    @test any(occursin("caps lkjcholesky", issue) for issue in marg_issues)
end

@testset "dev_plan_isbits" begin
    ys = [0.4, -0.7, 1.1, 0.2]
    cm = choicemap((:y => i, ys[i]) for i = 1:4)
    ws = DeviceBatchedWorkspace(dev_gauss_model, 3; args=(4,), constraints=cm)
    @test isbits(ws.plan)
    @test ws.plan isa DeviceExecutionPlan{Float64}

    ws32 = DeviceBatchedWorkspace(dev_gauss_model, 3; args=(4,), constraints=cm, precision=Float32)
    @test isbits(ws32.plan)
end

@testset "dev_device_loggamma_grid" begin
    max_rel64 = 0.0
    max_rel32 = 0.0
    for x = 0.05:0.05:20.0
        ref = UncertainTea.loggamma(x)
        d64 = UncertainTea._device_loggamma(x)
        max_rel64 = max(max_rel64, abs(d64 - ref) / max(1.0, abs(ref)))
        d32 = UncertainTea._device_loggamma(Float32(x))
        max_rel32 = max(max_rel32, abs(Float64(d32) - ref) / max(1.0, abs(ref)))
    end
    @test max_rel64 < 1e-10
    # Float32 accuracy is bounded by ~1 ulp of Float32 (~1.2e-7) accumulated over
    # the Lanczos sum; the achievable worst-case relative error is just above 1e-6.
    @test max_rel32 < 2e-6
end

@testset "dev_numerical_parity" begin
    # --- model 1: gaussian with loop observations, log latent ---
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2 0.05; 0.1 0.7 -0.2 0.4]
    dev = device_batched_logjoint(dev_gauss_model, params, (5,), cm)
    ref = batched_logjoint_unconstrained(dev_gauss_model, params, (5,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_gauss_model, Float32.(params), (5,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    # --- model 2: lognormal + gamma latents, exp-derived arg ---
    ys2 = [0.4, 1.1, -0.7]
    cm2 = choicemap((:y => i, ys2[i]) for i = 1:3)
    params2 = [0.5 -0.3 0.9; 0.1 0.7 -0.2]
    dev2 = device_batched_logjoint(dev_lognormal_gamma_model, params2, (3,), cm2)
    ref2 = batched_logjoint_unconstrained(dev_lognormal_gamma_model, params2, (3,), cm2)
    @test dev2 ≈ ref2 rtol = 1e-12
    dev2_32 = device_batched_logjoint(dev_lognormal_gamma_model, Float32.(params2), (3,), cm2; precision=Float32)
    @test dev_check_float32(dev2_32, ref2)

    # --- model 3: bernoulli observations, logit latent ---
    zs = [1.0, 0.0, 1.0, 1.0]
    cm3 = choicemap((:z => i, zs[i]) for i = 1:4)
    params3 = reshape([0.3, -0.8, 1.5], 1, 3)
    dev3 = device_batched_logjoint(dev_bernoulli_model, params3, (4,), cm3)
    ref3 = batched_logjoint_unconstrained(dev_bernoulli_model, params3, (4,), cm3)
    @test dev3 ≈ ref3 rtol = 1e-12
    dev3_32 = device_batched_logjoint(dev_bernoulli_model, Float32.(params3), (4,), cm3; precision=Float32)
    @test dev_check_float32(dev3_32, ref3)
end

@testset "dev_numerical_parity_group1" begin
    supported_ht, ht_issues = device_lowering_report(dev_heavytail_model)
    @test supported_ht
    @test isempty(ht_issues)
    supported_ct, ct_issues = device_lowering_report(dev_count_model)
    @test supported_ct
    @test isempty(ct_issues)

    # --- continuous: studentt / inversegamma / weibull ---
    ys = [0.4, -0.7, 1.1, 0.2, 0.9]
    cm = choicemap((:y => i, ys[i]) for i = 1:5)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2; -0.4 0.2 0.6]
    dev = device_batched_logjoint(dev_heavytail_model, params, (5,), cm)
    ref = batched_logjoint_unconstrained(dev_heavytail_model, params, (5,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_heavytail_model, Float32.(params), (5,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    # --- discrete: binomial / geometric / negativebinomial / categorical ---
    cs = [1.0, 3.0, 2.0, 3.0]
    cm2 = choicemap((:k, 7.0), (:g, 3.0), (:nb, 5.0), ((:c => i, cs[i]) for i = 1:4)...)
    params2 = reshape([0.3, -0.8, 1.5], 1, 3)
    dev2 = device_batched_logjoint(dev_count_model, params2, (4,), cm2)
    ref2 = batched_logjoint_unconstrained(dev_count_model, params2, (4,), cm2)
    @test dev2 ≈ ref2 rtol = 1e-12
    dev2_32 = device_batched_logjoint(dev_count_model, Float32.(params2), (4,), cm2; precision=Float32)
    @test dev_check_float32(dev2_32, ref2)

    # out-of-support discrete observations must land on -Inf, matching the CPU path.
    cm_oob = choicemap((:k, 25.0), (:g, 3.0), (:nb, 5.0), ((:c => i, cs[i]) for i = 1:4)...)
    dev_oob = device_batched_logjoint(dev_count_model, params2, (4,), cm_oob)
    @test all(==(-Inf), dev_oob)
    cm_cat_oob = choicemap((:k, 7.0), (:g, 3.0), (:nb, 5.0), (:c => 1, 4.0), ((:c => i, cs[i]) for i = 2:4)...)
    dev_cat_oob = device_batched_logjoint(dev_count_model, params2, (4,), cm_cat_oob)
    @test all(==(-Inf), dev_cat_oob)

    # near-integer (non-integral) observations are out of support on the CPU path
    # and must stay out of support on the device (exact-integer check).
    cm_frac = choicemap((:k, 7.0000001), (:g, 3.0), (:nb, 5.0), ((:c => i, cs[i]) for i = 1:4)...)
    @test all(==(-Inf), device_batched_logjoint(dev_count_model, params2, (4,), cm_frac))
    @test all(==(-Inf), batched_logjoint_unconstrained(dev_count_model, params2, (4,), cm_frac))
    cm_cat_frac = choicemap((:k, 7.0), (:g, 3.0), (:nb, 5.0), (:c => 1, 2.0000001), ((:c => i, cs[i]) for i = 2:4)...)
    @test all(==(-Inf), device_batched_logjoint(dev_count_model, params2, (4,), cm_cat_frac))
    @test all(==(-Inf), batched_logjoint_unconstrained(dev_count_model, params2, (4,), cm_cat_frac))
end

# binomial trials via a named deterministic binding: the symbol is classified as an
# index slot, which the kernel must materialize on device (codex review of issue #12
# group 1; previously read uninitialized scratch).
@tea static function dev_named_trials_model(n)
    p ~ beta(2.0, 2.0)
    trials = n + 1
    {:k} ~ binomial(trials, p)
    return p
end

@testset "dev_named_trials_index_slot" begin
    supported, issues = device_lowering_report(dev_named_trials_model)
    @test supported
    @test isempty(issues)

    cm = choicemap((:k, 7.0))
    params = reshape([0.3, -0.8, 1.5], 1, 3)
    dev = device_batched_logjoint(dev_named_trials_model, params, (10,), cm)
    ref = batched_logjoint_unconstrained(dev_named_trials_model, params, (10,), cm)
    @test dev ≈ ref rtol = 1e-12
end

# issue #12 group 2: truncated families (observed-only on the backend path).
# truncatednormal with a one-sided (Inf) bound and a two-sided loop observation;
# truncatedstudentt with the lowering-required literal nu.
@tea static function dev_truncated_model(n)
    m ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 2.0)
    {:h} ~ truncatednormal(m, 1.0, 0.0, Inf)
    for i = 1:n
        {:y => i} ~ truncatednormal(m, s, -1.0, 2.0)
        {:t => i} ~ truncatedstudentt(5.0, m, s, -2.0, 2.0)
    end
    return m
end

@tea static function dev_far_tail_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatednormal(m, 1.0, 37.49, Inf)
    return m
end

@testset "dev_device_erf_tcdf_grid" begin
    # Cody erf/erfc vs SpecialFunctions (the CPU reference)
    max_abs = 0.0
    for x = -6.0:0.01:6.0
        max_abs = max(
            max_abs,
            abs(UncertainTea._device_erf(x) - UncertainTea.erf(x)),
            abs(UncertainTea._device_erfc(x) - UncertainTea.erfc(x)),
        )
    end
    @test max_abs < 1e-14
    max_rel_tail = 0.0
    for x = 4.0:0.05:26.0
        reference = UncertainTea.erfc(x)
        max_rel_tail = max(max_rel_tail, abs(UncertainTea._device_erfc(x) - reference) / reference)
    end
    @test max_rel_tail < 1e-13

    # continued-fraction t-CDF vs the beta_inc-based CPU `_std_t_cdf`
    max_t = 0.0
    for nu in (1.5, 3.0, 5.0, 12.0), z = -8.0:0.05:8.0
        max_t = max(max_t, abs(UncertainTea._device_std_t_cdf(z, nu) - UncertainTea._std_t_cdf(z, nu)))
    end
    @test max_t < 1e-13

    # subnormal erfc tail: representable values must survive down to the true
    # Float64 underflow point (~27.2), not collapse at Cody's XBIG (codex review)
    max_rel_subnormal = 0.0
    for x = 26.0:0.01:27.4
        reference = UncertainTea.erfc(x)
        device = UncertainTea._device_erfc(x)
        if reference > 0
            max_rel_subnormal = max(max_rel_subnormal, abs(device - reference) / reference)
        else
            @test device == 0.0
        end
    end
    @test max_rel_subnormal < 1e-13

    # Float32 sanity on the same surfaces
    @test abs(Float64(UncertainTea._device_erf(0.8f0)) - UncertainTea.erf(0.8)) < 1e-6
    @test abs(Float64(UncertainTea._device_std_t_cdf(1.3f0, 5.0f0)) - UncertainTea._std_t_cdf(1.3, 5.0)) < 1e-5
end

@testset "dev_numerical_parity_truncated" begin
    ys = [0.4, -0.7, 1.1]
    ts = [1.5, -0.2, 0.8]
    cm = choicemap((:h, 0.6), ((:y => i, ys[i]) for i = 1:3)..., ((:t => i, ts[i]) for i = 1:3)...)

    # PR-4 (issue #95): device lowering is signature-aware. The static `:h`
    # truncated choice is an OBSERVATION only when constrained; left unconstrained
    # it is a truncated latent, which the device path does not lower. Report for
    # the conditioning this test scores (`:h`, `:y`, `:t` observed).
    supported, issues = device_lowering_report(dev_truncated_model; constraints=cm)
    @test supported
    @test isempty(issues)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2]
    dev = device_batched_logjoint(dev_truncated_model, params, (3,), cm)
    ref = batched_logjoint_unconstrained(dev_truncated_model, params, (3,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_truncated_model, Float32.(params), (3,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    # out-of-support observations land on -Inf on both paths
    cm_oob = choicemap((:h, -0.5), ((:y => i, ys[i]) for i = 1:3)..., ((:t => i, ts[i]) for i = 1:3)...)
    @test all(==(-Inf), device_batched_logjoint(dev_truncated_model, params, (3,), cm_oob))
    @test all(==(-Inf), batched_logjoint_unconstrained(dev_truncated_model, params, (3,), cm_oob))

    # far-tail one-sided bound: the erfc subnormal tail keeps the device finite
    # and matching the CPU reference (codex review regression)
    tail_params = reshape([0.2, -0.4], 1, 2)
    dev_tail = device_batched_logjoint(dev_far_tail_model, tail_params, (), choicemap((:y, 37.6)))
    ref_tail = batched_logjoint_unconstrained(dev_far_tail_model, tail_params, (), choicemap((:y, 37.6)))
    @test all(isfinite, dev_tail)
    @test dev_tail ≈ ref_tail rtol = 1e-10

    # degenerate lower == upper bounds: the CPU reference throws (studentt) or
    # hits log(0); the exception-free device path must degrade to -Inf, never +Inf
    @test UncertainTea._device_truncatedstudentt_logpdf(5.0, 0.0, 1.0, 1.0, 1.0, 1.0) == -Inf
    @test UncertainTea._device_truncatednormal_logpdf(0.0, 1.0, 1.0, 1.0, 1.0) == -Inf
end

# narrow valid intervals at Float32: the CDF difference quantizes (codex review);
# the accurate-complement beta_inc and the midpoint fallback keep the device
# finite and close to the Float64 CPU reference.
@tea static function dev_narrow_tt_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatedstudentt(5.0, m, 1.0, -1.0e-4, 1.0e-4)
    return m
end

@tea static function dev_narrow_tn_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatednormal(m, 1.0, 0.5, 0.5001)
    return m
end

# one-sided far-tail truncations at Float32 (codex review round 3): the normal
# side loses erfc to underflow around z ~ 13 (Float32) and must switch to the
# asymptotic log(erfc); the Student-t side computes the tail probability
# directly via the symmetry S(z) = cdf(-z) instead of cancelling against 1.
@tea static function dev_onesided_tail_tn_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatednormal(m, 1.0, 15.0, Inf)
    return m
end

@tea static function dev_onesided_tail_tt_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatedstudentt(5.0, m, 1.0, 100.0, Inf)
    return m
end

@testset "dev_truncated_onesided_tail_f32" begin
    params32 = reshape(Float32[0.01, -0.02], 1, 2)

    dev_tn = device_batched_logjoint(dev_onesided_tail_tn_model, params32, (), choicemap((:y, 15.2)); precision=Float32)
    ref_tn = batched_logjoint_unconstrained(dev_onesided_tail_tn_model, Float64.(params32), (), choicemap((:y, 15.2)))
    @test all(isfinite, dev_tn)
    @test dev_check_float32(dev_tn, ref_tn)
    _, g_tn = device_batched_logjoint_gradient(
        dev_onesided_tail_tn_model,
        params32,
        (),
        choicemap((:y, 15.2));
        precision=Float32,
    )
    g_tn_ref = batched_logjoint_gradient_unconstrained(dev_onesided_tail_tn_model, Float64.(params32), (), choicemap((:y, 15.2)))
    @test all(isfinite, g_tn)
    @test dev_check_float32(vec(g_tn), vec(g_tn_ref))

    dev_tt = device_batched_logjoint(dev_onesided_tail_tt_model, params32, (), choicemap((:y, 105.0)); precision=Float32)
    ref_tt = batched_logjoint_unconstrained(dev_onesided_tail_tt_model, Float64.(params32), (), choicemap((:y, 105.0)))
    @test all(isfinite, dev_tt)
    @test dev_check_float32(dev_tt, ref_tt)
    _, g_tt = device_batched_logjoint_gradient(
        dev_onesided_tail_tt_model,
        params32,
        (),
        choicemap((:y, 105.0));
        precision=Float32,
    )
    g_tt_ref =
        batched_logjoint_gradient_unconstrained(dev_onesided_tail_tt_model, Float64.(params32), (), choicemap((:y, 105.0)))
    @test all(isfinite, g_tt)
    @test dev_check_float32(vec(g_tt), vec(g_tt_ref))
end

# two-sided far-tail interval at Float32 (codex review round 4): both plain erfc
# values underflow, so the same-sign branch must go through the erfcx identity.
@tea static function dev_twosided_tail_tn_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatednormal(m, 1.0, 15.0, 15.1)
    return m
end

# large literal nu at Float32 (codex review round 5): the loggamma half-ratio
# keeps the Student-t density and CDF from losing the O(1) gamma-ratio to
# rounding of two ~nu log nu sized loggammas.
@tea static function dev_large_nu_tt_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatedstudentt(1.0e5, m, 1.0, -1.0, 2.0)
    return m
end

@testset "dev_truncated_extreme_f32" begin
    params32 = reshape(Float32[0.01, -0.02], 1, 2)

    dev_tn = device_batched_logjoint(dev_twosided_tail_tn_model, params32, (), choicemap((:y, 15.02)); precision=Float32)
    ref_tn = batched_logjoint_unconstrained(dev_twosided_tail_tn_model, Float64.(params32), (), choicemap((:y, 15.02)))
    @test all(isfinite, dev_tn)
    @test dev_check_float32(dev_tn, ref_tn)
    _, g_tn = device_batched_logjoint_gradient(
        dev_twosided_tail_tn_model,
        params32,
        (),
        choicemap((:y, 15.02));
        precision=Float32,
    )
    @test all(isfinite, g_tn)
    @test dev_check_float32(
        vec(g_tn),
        vec(batched_logjoint_gradient_unconstrained(dev_twosided_tail_tn_model, Float64.(params32), (), choicemap((:y, 15.02)))),
    )

    dev_nu = device_batched_logjoint(dev_large_nu_tt_model, params32, (), choicemap((:y, 0.5)); precision=Float32)
    ref_nu = batched_logjoint_unconstrained(dev_large_nu_tt_model, Float64.(params32), (), choicemap((:y, 0.5)))
    @test dev_check_float32(dev_nu, ref_nu)
    @test abs(Float64(UncertainTea._device_std_t_cdf(1.0f0, 1.0f5)) - UncertainTea._std_t_cdf(1.0, 1.0e5)) < 1e-6
end

# one-sided light-tail truncatedstudentt (codex review round 6): the plain tail
# CDF underflows before its log, so the one-sided normalizer computes in log
# space. Since issue #43 the CPU reference computes its normalizer the same way
# and stays finite here, so both precisions compare against it directly.
@tea static function dev_lighttail_tt_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatedstudentt(1.0e5, m, 1.0, 15.0, Inf)
    return m
end

# finite interval wholly in a deep light tail (codex review round 7): both plain
# CDFs underflow, so the finite-bounds normalizer must difference LOG CDFs via
# expm1 (the CPU reference does the same since issue #43).
@tea static function dev_deeptail_interval_tt_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatedstudentt(1.0e5, m, 1.0, 15.0, 16.0)
    return m
end

@testset "dev_truncated_deeptail_interval" begin
    params32 = reshape(Float32[0.01, -0.02], 1, 2)
    cm = choicemap((:y, 15.2))
    dev32 = device_batched_logjoint(dev_deeptail_interval_tt_model, params32, (), cm; precision=Float32)
    ref = batched_logjoint_unconstrained(dev_deeptail_interval_tt_model, Float64.(params32), (), cm)
    dev64 = device_batched_logjoint(dev_deeptail_interval_tt_model, Float64.(params32), (), cm)
    @test all(isfinite, ref) # issue #43: the CPU reference stays finite now
    @test dev64 ≈ ref rtol = 1e-10
    @test dev_check_float32(dev32, ref)
    # the mass above the upper bound is a ~e^-15.5 (~1.9e-7) relative correction,
    # which the expm1 log-difference resolves; the one-sided normalizer agrees to
    # exactly that order
    ref_onesided = batched_logjoint_unconstrained(dev_lighttail_tt_model, Float64.(params32), (), cm)
    @test ref ≈ ref_onesided rtol = 1e-6
    @test !isapprox(ref, ref_onesided; rtol=1e-9) # the correction is real, not noise
end

@testset "dev_truncated_lighttail_logspace" begin
    @test isapprox(
        UncertainTea._device_std_t_log_cdf(-15.0, 1.0e5),
        UncertainTea._std_t_log_cdf(-15.0, 1.0e5);
        rtol=1e-12,
    )

    params32 = reshape(Float32[0.01, -0.02], 1, 2)
    cm = choicemap((:y, 15.2))
    dev32 = device_batched_logjoint(dev_lighttail_tt_model, params32, (), cm; precision=Float32)
    ref = batched_logjoint_unconstrained(dev_lighttail_tt_model, Float64.(params32), (), cm)
    dev64 = device_batched_logjoint(dev_lighttail_tt_model, Float64.(params32), (), cm)
    @test all(isfinite, ref) # issue #43: the CPU reference stays finite now
    @test dev64 ≈ ref rtol = 1e-10
    @test dev_check_float32(dev32, ref)
    _, g32 = device_batched_logjoint_gradient(dev_lighttail_tt_model, params32, (), cm; precision=Float32)
    gref = batched_logjoint_gradient_unconstrained(dev_lighttail_tt_model, Float64.(params32), (), cm)
    _, g64 = device_batched_logjoint_gradient(dev_lighttail_tt_model, Float64.(params32), (), cm)
    @test all(isfinite, gref)
    # the deep-tail pdf/cdf ratio amplifies the implementation gap between the
    # device special functions (Lanczos loggamma, continued-fraction beta) and
    # the CPU SpecialFunctions to a few 1e-9 relative
    @test g64 ≈ gref rtol = 1e-8
    # a deep-tail pdf/cdf gradient ratio carries a slightly looser Float32 error
    @test all(isapprox(Float64(a), b; rtol=2e-3, atol=2e-3) for (a, b) in zip(vec(g32), vec(gref)))
end

# ultra-narrow same-sign interval (codex review round 8): `s` in the erfcx
# difference sinks under the Float32 rounding noise floor, so the sqrt(eps)
# threshold must hand over to the log-space midpoint form.
@tea static function dev_ultranarrow_tn_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ truncatednormal(m, 1.0, 0.5, 0.500001)
    return m
end

@testset "dev_truncated_narrow_interval_f32" begin
    params32 = reshape(Float32[0.01, -0.02], 1, 2)
    dev_tt = device_batched_logjoint(dev_narrow_tt_model, params32, (), choicemap((:y, 0.0)); precision=Float32)
    ref_tt = batched_logjoint_unconstrained(dev_narrow_tt_model, Float64.(params32), (), choicemap((:y, 0.0)))
    @test all(isfinite, dev_tt)
    @test dev_check_float32(dev_tt, ref_tt)

    dev_tn = device_batched_logjoint(dev_narrow_tn_model, params32, (), choicemap((:y, 0.50005)); precision=Float32)
    ref_tn = batched_logjoint_unconstrained(dev_narrow_tn_model, Float64.(params32), (), choicemap((:y, 0.50005)))
    @test all(isfinite, dev_tn)
    @test dev_check_float32(dev_tn, ref_tn)

    cm_ultra = choicemap((:y, 0.5000005))
    dev_ultra = device_batched_logjoint(dev_ultranarrow_tn_model, params32, (), cm_ultra; precision=Float32)
    ref_ultra = batched_logjoint_unconstrained(dev_ultranarrow_tn_model, Float64.(params32), (), cm_ultra)
    @test all(isfinite, dev_ultra)
    @test all(isapprox(Float64(d), r; rtol=2e-3, atol=2e-3 * max(1.0, abs(r))) for (d, r) in zip(dev_ultra, ref_ultra))
    _, g_ultra = device_batched_logjoint_gradient(dev_ultranarrow_tn_model, params32, (), cm_ultra; precision=Float32)
    g_ultra_ref = batched_logjoint_gradient_unconstrained(dev_ultranarrow_tn_model, Float64.(params32), (), cm_ultra)
    @test all(isfinite, g_ultra)
    @test all(
        isapprox(Float64(d), r; rtol=2e-3, atol=2e-3 * max(1.0, abs(r))) for (d, r) in zip(vec(g_ultra), vec(g_ultra_ref))
    )
end

# issue #12 group 3 phase 1: diagonal mvnormal. Latent (VectorIdentity, two
# unconstrained rows) mixed with a scalar latent, plus a top-level and an
# in-loop vector observation (the staged cursor advances by the dimension).
@tea static function dev_mvnormal_model(n)
    state ~ mvnormal([0.0, 1.0], [1.5, 0.8])
    s ~ gamma(2.0, 2.0)
    {:w} ~ mvnormal([0.3, -0.2], [1.0, 2.0])
    for i = 1:n
        {:v => i} ~ mvnormal([0.5, -0.5], [1.0, s])
    end
    return s
end

# reading a component of a vector binding is not representable (today the
# backend already rejects the `ref` expression; if it ever lowers, the device
# read/write audit rejects the unmaterialized binding read instead).
@tea static function dev_mvnormal_binding_read_model()
    state ~ mvnormal([0.0, 1.0], [1.5, 0.8])
    {:y} ~ normal(state[1], 1.0)
    return state
end

# above the compile-time dimension cap: honest rejection.
@tea static function dev_mvnormal_bigdim_model()
    big ~ mvnormal(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    return big
end

@testset "dev_numerical_parity_mvnormal" begin
    supported, issues = device_lowering_report(dev_mvnormal_model)
    @test supported
    @test isempty(issues)

    vs = [[0.2, -0.1], [1.0, 0.4], [-0.3, 0.9]]
    cm = choicemap((:w, [0.1, -0.4]), ((:v => i, vs[i]) for i = 1:3)...)
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2; -0.4 0.2 0.6] # state (2 rows) + s (1 row)
    dev = device_batched_logjoint(dev_mvnormal_model, params, (3,), cm)
    ref = batched_logjoint_unconstrained(dev_mvnormal_model, params, (3,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_mvnormal_model, Float32.(params), (3,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    read_supported, read_issues = device_lowering_report(dev_mvnormal_binding_read_model)
    @test !read_supported
    @test !isempty(read_issues)

    big_supported, big_issues = device_lowering_report(dev_mvnormal_bigdim_model)
    @test !big_supported
    @test any(occursin("caps vector dimensions", issue) for issue in big_issues)
end

# issue #12 group 3 phase 2: dirichlet — the first dimension-changing stride
# (a K-simplex latent reads K-1 unconstrained rows; K staged rows observed).
# The scalar latent AFTER the simplex exercises the issue #36 placement freedom.
@tea static function dev_dirichlet_group3_model(n)
    theta ~ dirichlet([2.0, 3.0, 4.0])
    s ~ gamma(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ normal(0.5, s)
    end
    return s
end

# observed dirichlet with a latent-flowing concentration.
@tea static function dev_dirichlet_obs_model()
    a ~ gamma(2.0, 1.0)
    {:w} ~ dirichlet([a, 2.0, 3.0])
    return a
end

@testset "dev_numerical_parity_dirichlet" begin
    supported, issues = device_lowering_report(dev_dirichlet_group3_model)
    @test supported
    @test isempty(issues)

    ys = [0.4, -0.7, 1.1]
    cm = choicemap((:y => i, ys[i]) for i = 1:3)
    params = [0.3 -0.5; -0.2 0.4; 0.1 0.6] # theta (2 unconstrained rows) + s
    dev = device_batched_logjoint(dev_dirichlet_group3_model, params, (3,), cm)
    ref = batched_logjoint_unconstrained(dev_dirichlet_group3_model, params, (3,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_dirichlet_group3_model, Float32.(params), (3,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    cm_obs = choicemap((:w, [0.2, 0.3, 0.5]))
    params_obs = reshape([0.1, -0.4], 1, 2)
    dev_obs = device_batched_logjoint(dev_dirichlet_obs_model, params_obs, (), cm_obs)
    ref_obs = batched_logjoint_unconstrained(dev_dirichlet_obs_model, params_obs, (), cm_obs)
    @test dev_obs ≈ ref_obs rtol = 1e-12

    # an observation off the simplex lands on -Inf on both paths
    cm_off = choicemap((:w, [0.2, 0.3, 0.9]))
    @test all(==(-Inf), device_batched_logjoint(dev_dirichlet_obs_model, params_obs, (), cm_off))
    @test all(==(-Inf), batched_logjoint_unconstrained(dev_dirichlet_obs_model, params_obs, (), cm_off))

    # a NEGATIVE component must reject through the support check, not blow up in
    # the eagerly-evaluated log (codex review: log(negative) throws a DomainError
    # on the CPU KernelAbstractions backend where the reference returns -Inf)
    cm_neg = choicemap((:w, [-0.2, 0.7, 0.5]))
    @test all(==(-Inf), device_batched_logjoint(dev_dirichlet_obs_model, params_obs, (), cm_neg))
    @test all(==(-Inf), batched_logjoint_unconstrained(dev_dirichlet_obs_model, params_obs, (), cm_neg))
end

# issue #57: lkjcholesky. A latent constrains d(d-1)/2 unconstrained rows
# through the register-resident tanh/stick construction; the scalar latent
# AFTER the packed factor exercises the issue #36 placement freedom.
@tea static function dev_lkj_group_model(n)
    Omega ~ lkjcholesky(3, 2.0)
    s ~ gamma(2.0, 2.0)
    for i = 1:n
        {:y => i} ~ normal(0.5, s)
    end
    return s
end

# observed packed factor with a latent-flowing concentration.
@tea static function dev_lkj_obs_model()
    e ~ gamma(2.0, 1.0)
    {:L} ~ lkjcholesky(3, e)
    return e
end

# over the quadratic-unroll dimension cap
@tea static function dev_lkj_big_model()
    Omega ~ lkjcholesky(9, 2.0)
    return Omega
end

# packed column-major lower triangle of a valid d=3 correlation cholesky
# factor: rows (1), (0.6, 0.8), (0.3, -0.4, sqrt(0.75)) all unit-norm.
dev_lkj_packed() = [1.0, 0.6, 0.3, 0.8, -0.4, sqrt(0.75)]

@testset "dev_numerical_parity_lkjcholesky" begin
    supported, issues = device_lowering_report(dev_lkj_group_model)
    @test supported
    @test isempty(issues)

    ys = [0.4, -0.7, 1.1]
    cm = choicemap((:y => i, ys[i]) for i = 1:3)
    params = [0.3 -0.5; -0.2 0.4; 0.7 -0.1; 0.1 0.6] # Omega (3 unconstrained rows) + s
    dev = device_batched_logjoint(dev_lkj_group_model, params, (3,), cm)
    ref = batched_logjoint_unconstrained(dev_lkj_group_model, params, (3,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_lkj_group_model, Float32.(params), (3,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    cm_obs = choicemap((:L, dev_lkj_packed()))
    params_obs = reshape([0.1, -0.4], 1, 2)
    dev_obs = device_batched_logjoint(dev_lkj_obs_model, params_obs, (), cm_obs)
    ref_obs = batched_logjoint_unconstrained(dev_lkj_obs_model, params_obs, (), cm_obs)
    @test dev_obs ≈ ref_obs rtol = 1e-12

    # a non-positive diagonal rejects through the support check, not a throw in
    # the eagerly-evaluated log (the dirichlet log(abs) discipline)
    bad_diag = dev_lkj_packed()
    bad_diag[4] = -0.8 # L[2,2]
    cm_bad = choicemap((:L, bad_diag))
    @test all(==(-Inf), device_batched_logjoint(dev_lkj_obs_model, params_obs, (), cm_bad))
    @test all(==(-Inf), batched_logjoint_unconstrained(dev_lkj_obs_model, params_obs, (), cm_bad))

    # a row with norm > 1 lands on -Inf on both paths
    long_row = dev_lkj_packed()
    long_row[3] = 0.9 # L[3,1]: row 3 square sum now exceeds one
    cm_long = choicemap((:L, long_row))
    @test all(==(-Inf), device_batched_logjoint(dev_lkj_obs_model, params_obs, (), cm_long))
    @test all(==(-Inf), batched_logjoint_unconstrained(dev_lkj_obs_model, params_obs, (), cm_long))

    # the dimension cap rejects with a message naming it
    big_supported, big_issues = device_lowering_report(dev_lkj_big_model)
    @test !big_supported
    @test any(occursin("caps lkjcholesky", issue) for issue in big_issues)
end

# issue #12 group 3 phase 3: mvnormaldense. The constant scale_tril factor
# rides the observation buffer as d(d+1)/2 packed rows ahead of any value rows.
@tea static function dev_mvdense_model(L)
    state ~ mvnormaldense([0.0, 1.0], L)
    m ~ normal(0.0, 1.0)
    {:y} ~ mvnormaldense([m, m], L)
    return m
end

@testset "dev_numerical_parity_mvnormaldense" begin
    supported, issues = device_lowering_report(dev_mvdense_model)
    @test supported
    @test isempty(issues)

    L = [1.2 0.0; 0.4 0.9]
    cm = choicemap((:y, [0.4, -0.2]))
    params = [0.5 -0.3; 0.1 0.7; -0.4 0.2] # state (2 rows) + m
    dev = device_batched_logjoint(dev_mvdense_model, params, (L,), cm)
    ref = batched_logjoint_unconstrained(dev_mvdense_model, params, (L,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_mvdense_model, Float32.(params), (L,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    # per-batch factors stage per column through the shared observation buffer
    batch_args = [(L,), ([1.5 0.0; -0.2 0.7],)]
    params2 = params[:, 1:2]
    dev_batch = device_batched_logjoint(dev_mvdense_model, params2, batch_args, cm)
    ref_batch = batched_logjoint_unconstrained(dev_mvdense_model, params2, batch_args, cm)
    @test dev_batch ≈ ref_batch rtol = 1e-12
end

# issue #50: mixture of normals. The VALUE is scalar (Identity latent), so the
# binding is materialized and may feed downstream expressions; the observed form
# carries a latent-flowing component mean through the log-sum-exp.
@tea static function dev_mixture_device_model(n)
    x ~ mixture((0.5, 0.5), normal(-2.0, 0.5), normal(2.0, 0.5))
    s ~ gamma(2.0, 2.0)
    {:z} ~ normal(x * 2.0, 0.3)
    for i = 1:n
        {:y => i} ~ mixture((0.3, 0.7), normal(x, s), normal(2.0, 0.5))
    end
    return x
end

# latent-simplex mixture weights stay a backend fallback (generic-slot weights),
# so the device path must keep reporting them unsupported.
@tea static function dev_mixture_latent_weights_model()
    w ~ dirichlet([2.0, 2.0])
    x ~ mixture(w, normal(-1.0, 1.0), normal(1.0, 1.0))
    return x
end

@testset "dev_numerical_parity_mixture" begin
    supported, issues = device_lowering_report(dev_mixture_device_model)
    @test supported
    @test isempty(issues)

    ys = [1.4, -1.8, 2.2]
    cm = choicemap((:z, 3.5), ((:y => i, ys[i]) for i = 1:3)...)
    params = [1.8 -2.1; 0.1 0.6] # x + s
    dev = device_batched_logjoint(dev_mixture_device_model, params, (3,), cm)
    ref = batched_logjoint_unconstrained(dev_mixture_device_model, params, (3,), cm)
    @test dev ≈ ref rtol = 1e-12
    dev32 = device_batched_logjoint(dev_mixture_device_model, Float32.(params), (3,), cm; precision=Float32)
    @test dev_check_float32(dev32, ref)

    weights_supported, _ = device_lowering_report(dev_mixture_latent_weights_model)
    @test !weights_supported
end

# an argument rebound into a host-only loop bound: the emitted index deterministic
# is pruned (no kernel expression reads it), so the rebinding audit must not fire.
@tea static function dev_rebound_loop_bound_model(n)
    n = n + 1
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0)
    end
    return mu
end

# a loop-body binding read after the loop: a zero-trip loop would leave the slot
# unwritten on device, so the audit must reject the shape.
@tea static function dev_postloop_read_model(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        c = i * 1.0
    end
    {:y} ~ normal(c, 1.0)
    return mu
end

# a random choice binding feeding a loop bound: host staging can never resolve
# the range, so lowering must reject it instead of leaking a staging error.
@tea static function dev_choice_loop_bound_model(N)
    p ~ beta(2.0, 2.0)
    n = {:n} ~ binomial(N, p)
    for i = 1:n
        {:y => i} ~ normal(0.0, 1.0)
    end
    return p
end

@testset "dev_slot_read_write_audit" begin
    sup_rebind, rebind_issues = device_lowering_report(dev_rebound_loop_bound_model)
    @test sup_rebind
    @test isempty(rebind_issues)
    cm = choicemap((:y => 1, 0.4), (:y => 2, -0.7), (:y => 3, 1.1))
    params = reshape([0.3, -0.8], 1, 2)
    dev = device_batched_logjoint(dev_rebound_loop_bound_model, params, (2,), cm)
    ref = batched_logjoint_unconstrained(dev_rebound_loop_bound_model, params, (2,), cm)
    @test dev ≈ ref rtol = 1e-12

    sup_postloop, postloop_issues = device_lowering_report(dev_postloop_read_model)
    @test !sup_postloop
    @test any(occursin("not materialized on the device", issue) for issue in postloop_issues)

    sup_bound, bound_issues = device_lowering_report(dev_choice_loop_bound_model)
    @test !sup_bound
    @test any(occursin("random choice binding", issue) for issue in bound_issues)
end

@testset "dev_workspace_reuse" begin
    ys = [0.4, -0.7, 1.1, 0.2]
    cm = choicemap((:y => i, ys[i]) for i = 1:4)
    ws = DeviceBatchedWorkspace(dev_gauss_model, 3; args=(4,), constraints=cm)

    params_a = [0.5 -0.3 1.2; 0.1 0.7 -0.2]
    params_b = params_a .+ 0.75
    result_a = device_batched_logjoint!(ws, params_a)
    result_b = device_batched_logjoint!(ws, params_b)
    # a third call with params_a again must reproduce result_a exactly (no stale state).
    result_a2 = device_batched_logjoint!(ws, params_a)

    ref_a = batched_logjoint_unconstrained(dev_gauss_model, params_a, (4,), cm)
    ref_b = batched_logjoint_unconstrained(dev_gauss_model, params_b, (4,), cm)
    @test result_a ≈ ref_a rtol = 1e-12
    @test result_b ≈ ref_b rtol = 1e-12
    @test result_a2 == result_a
end

@testset "dev_batched_constraints" begin
    ys = [0.4, -0.7, 1.1, 0.2]
    params = [0.5 -0.3 1.2; 0.1 0.7 -0.2]

    # identical-structure constraint vectors (per-column observation values).
    cm_vec = [choicemap((:y => i, ys[i] + 0.1 * b) for i = 1:4) for b = 1:3]
    dev = device_batched_logjoint(dev_gauss_model, params, (4,), cm_vec)
    ref = batched_logjoint_unconstrained(dev_gauss_model, params, (4,), cm_vec)
    @test dev ≈ ref rtol = 1e-12

    # mismatched structure (a column missing an address) must throw during staging.
    cm_bad = [
        choicemap((:y => i, ys[i]) for i = 1:4),
        choicemap((:y => i, ys[i]) for i = 1:2),
        choicemap((:y => i, ys[i]) for i = 1:4),
    ]
    @test_throws Exception device_batched_logjoint(dev_gauss_model, params, (4,), cm_bad)
end

@testset "dev_unsupported_fallback" begin
    # A device-unsupported model must raise a clear error pointing at the report.
    err = try
        device_batched_logjoint(dev_unsupported_model, reshape([0.1], 1, 1), ())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("device_lowering_report", err.msg)
end

# issue #71: integer distribution inputs (binomial trials/count) are staged into a
# separate integer observation buffer, so Float32 (2^24 mantissa) can no longer
# collapse `n` and `k` -- the exact `n - k` distinction survives into the log
# binomial coefficient, keeping the gradient exact at any compute precision. The
# coefficient's absolute value is computed in the compute type T (Float64 on the
# CPU, Float32 on Metal per issue #218); in Float32 it loses a constant offset for
# huge n, which the value assertions below account for.
@tea static function dev_large_binomial(n)
    p ~ beta(2.0, 2.0)
    {:k} ~ binomial(n, p)
end

@testset "dev_binomial_integer_exactness" begin
    n = 2^24 + 1
    k = n - 1
    cm = choicemap((:k, k))
    @test Float32(n) == Float32(k) # both collapse to 16777216f0 in Float32

    params32 = reshape(Float32[16.0], 1, 1)
    v32, g32 = device_batched_logjoint_gradient(dev_large_binomial, params32, (n,), cm; precision=Float32)
    gref32 = batched_logjoint_gradient_unconstrained(dev_large_binomial, params32, (n,), cm)
    # The (n - k) = 1 distinction is preserved, so the gradient matches the CPU
    # Float32 reference (which keeps n, k as exact Ints) instead of collapsing.
    @test Float64(g32[1]) ≈ Float64(gref32[1]) rtol = 1e-5
    @test abs(Float64(g32[1])) > 0.9 # was ~5e-2 (corrupted) before the fix

    # Float64: the device computes the log binomial coefficient from the exact
    # integers via a difference of two large loggammas, so for huge n it carries
    # ~1e-9 relative cancellation error vs the CPU exact factorial sum. This is
    # pre-existing device behavior (the old binomial used the same F64 loggamma
    # differences) -- unchanged by the fix -- so the huge-n value agrees loosely.
    params64 = reshape(Float64[16.0], 1, 1)
    v64 = device_batched_logjoint(dev_large_binomial, params64, (n,), cm; precision=Float64)
    vref64 = batched_logjoint_unconstrained(dev_large_binomial, params64, (n,), cm)
    @test v64 ≈ vref64 rtol = 1e-6
    # Float32: the log binomial coefficient must be computed in the compute type T
    # (Float32) rather than always in Float64, because Metal has no Float64 and the
    # F64 helpers fail to compile there (issue #218). For huge n ~ 2^24 the constant
    # log C(n, n-1) = log(n) ~ 16.6 is then unresolvable: loggamma(n+1) ~ 2.5e8 has a
    # ~16-ULP Float32 gap that swallows the ~16.6 difference. This is a fundamental
    # Float32 limit (identical on Metal) and affects only the constant offset -- the
    # gradient wrt p is exact (the coefficient is constant in p, asserted above), so
    # inference is unaffected. The Float32 logjoint therefore drops exactly log(n):
    lost = log(Float64(n)) # = log C(n, n - 1), the coefficient Float32 cannot resolve
    @test isapprox(Float64(v32[1]) + lost, v64[1]; rtol=1e-3)

    # Small-count binomial is unchanged: Float64 matches the CPU tightly and
    # Float32 matches within precision. Use a moderate p (logit 0.5 -> ~0.62); the
    # huge-n case above pins p ~ 1, which is a degenerate Float32 regime for the
    # log1p(-p) term unrelated to the integer-exactness fix.
    small = choicemap((:k, 7))
    small_params64 = reshape(Float64[0.5], 1, 1)
    small_params32 = reshape(Float32[0.5], 1, 1)
    v64_small = device_batched_logjoint(dev_large_binomial, small_params64, (20,), small; precision=Float64)
    ref64_small = batched_logjoint_unconstrained(dev_large_binomial, small_params64, (20,), small)
    @test v64_small ≈ ref64_small rtol = 1e-12
    dev_small = device_batched_logjoint(dev_large_binomial, small_params32, (20,), small; precision=Float32)
    @test Float64.(dev_small) ≈ Float64.(ref64_small) rtol = 1e-4
end

# issue #83: device_lowering_report must enforce the same loop-range constraints
# staging enforces (unit step, Int32-representable bounds/trip count), rather than
# deferring them to a staging error after the report claims support.
@tea static function dev_nonunit_step_loop(n)
    m ~ normal(0.0, 1.0)
    for i = 1:2:n
        {:y => i} ~ normal(m, 1.0)
    end
    return m
end

@tea static function dev_big_bound_loop()
    m ~ normal(0.0, 1.0)
    for i = 2147483648:2147483648
        {:y => i} ~ normal(m, 1.0)
    end
    return m
end

@tea static function dev_big_trip_loop()
    m ~ normal(0.0, 1.0)
    for i = 1:3000000000
        {:y => i} ~ normal(m, 1.0)
    end
    return m
end

@testset "dev_loop_iterable_report_checks" begin
    sup_step, iss_step = device_lowering_report(dev_nonunit_step_loop)
    @test !sup_step
    @test any(occursin("unit-step", issue) for issue in iss_step)

    sup_bound, iss_bound = device_lowering_report(dev_big_bound_loop)
    @test !sup_bound
    @test any(occursin("Int32", issue) for issue in iss_bound)

    sup_trip, iss_trip = device_lowering_report(dev_big_trip_loop)
    @test !sup_trip
    @test any(occursin("trip count", issue) for issue in iss_trip)
end

# issue #84: the device categorical scorer must validate the probability vector
# (each entry in [0, 1] and sum to 1), mapping invalid vectors to a non-finite
# result per the exception-free contract, matching the CPU (which throws).
@tea static function dev_invalid_categorical()
    p ~ beta(2.0, 2.0)
    {:y} ~ categorical(p, p)
end

@tea static function dev_norm_categorical()
    q ~ beta(2.0, 2.0)
    {:y} ~ categorical(q, 1.0 - q)
end

@testset "dev_categorical_probability_validation" begin
    # p = 0.3 -> probabilities (0.3, 0.3) sum to 0.6 (not normalized).
    params = reshape([log(0.3 / 0.7)], 1, 1)
    cm = choicemap((:y, 1))
    sup, _ = device_lowering_report(dev_invalid_categorical)
    @test sup # lowering is fine; validation happens at scoring time
    val = device_batched_logjoint(dev_invalid_categorical, params, (), cm)
    @test all(==(-Inf), val) # was a finite ~-2.53 before the fix; CPU throws
    # the gradient-kernel value channel agrees (the -Inf constant carries a zero
    # derivative, so only the reported logjoint VALUE flags the rejection)
    gval, _ = device_batched_logjoint_gradient(dev_invalid_categorical, params, (), cm)
    @test all(==(-Inf), gval)

    # A properly normalized categorical still scores finitely and matches the CPU.
    params_ok = reshape([0.2], 1, 1)
    cm_ok = choicemap((:y, 2))
    dev_ok = device_batched_logjoint(dev_norm_categorical, params_ok, (), cm_ok)
    ref_ok = batched_logjoint_unconstrained(dev_norm_categorical, params_ok, (), cm_ok)
    @test dev_ok ≈ ref_ok rtol = 1e-12
end

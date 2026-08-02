# PR 41: truncated normal and truncated Student-t distributions.
# Contract: `truncatednormal(mu, sigma, lower, upper)` and
# `truncatedstudentt(nu, mu, sigma, lower, upper)` renormalize the base density
# over [lower, upper] (infinite bounds allowed). They work as observations with
# any bound expression; as latents (parameter slots) they require literal
# (static) bounds so a fixed unconstraining transform can be chosen. These
# families are CPU-reference only and are honestly reported unsupported by the
# GPU backend, but models still run through the compiled logjoint and the
# batched ForwardDiff fallback.
#
# SpecialFunctions/ForwardDiff are dependencies of UncertainTea (not of the
# test sandbox), so reference them through the package namespace.
trunc_erf = UncertainTea.erf
trunc_beta_inc = UncertainTea.beta_inc
const trunc_ForwardDiff = UncertainTea.ForwardDiff
trunc_mean(xs) = sum(xs) / length(xs)

# --- logpdf against hand-computed references -----------------------------
@testset "trunc_logpdf_reference" begin
    # Standard-normal CDF via erf for the reference normalization.
    trunc_Phi(z) = (1 + trunc_erf(z / sqrt(2))) / 2

    tn = truncatednormal(0.0, 1.0, -1.0, 2.0)
    trunc_x = 0.5
    trunc_base_n = -0.5 * log(2pi) - trunc_x^2 / 2
    trunc_ref_n = trunc_base_n - log(trunc_Phi(2.0) - trunc_Phi(-1.0))
    @test UncertainTea.logpdf(tn, trunc_x) ≈ trunc_ref_n atol = 1e-12

    # Outside the support is -Inf on both sides.
    @test UncertainTea.logpdf(tn, -1.5) == -Inf
    @test UncertainTea.logpdf(tn, 2.5) == -Inf

    # Half-open (0, Inf): folded/renormalized standard half-normal.
    tn_half = truncatednormal(0.0, 1.0, 0.0, Inf)
    trunc_xh = 0.7
    trunc_ref_half = (-0.5 * log(2pi) - trunc_xh^2 / 2) - log(1 - trunc_Phi(0.0))
    @test UncertainTea.logpdf(tn_half, trunc_xh) ≈ trunc_ref_half atol = 1e-12
    @test UncertainTea.logpdf(tn_half, -0.1) == -Inf

    # Truncated Student-t via the regularized incomplete beta function.
    trunc_t_cdf(z, nu) =
        z > 0 ? 1 - trunc_beta_inc(nu / 2, 0.5, nu / (nu + z^2))[1] / 2 :
        trunc_beta_inc(nu / 2, 0.5, nu / (nu + z^2))[1] / 2
    tt = truncatedstudentt(5.0, 0.0, 1.0, -2.0, 2.0)
    trunc_z = 0.3
    trunc_base_t = UncertainTea.logpdf(studentt(5.0, 0.0, 1.0), trunc_z)
    trunc_ref_t = trunc_base_t - log(trunc_t_cdf(2.0, 5.0) - trunc_t_cdf(-2.0, 5.0))
    @test UncertainTea.logpdf(tt, trunc_z) ≈ trunc_ref_t atol = 1e-12
    @test UncertainTea.logpdf(tt, -3.0) == -Inf
    @test UncertainTea.logpdf(tt, 2.1) == -Inf
end

# --- density normalizes to one over a finite support --------------------
@testset "trunc_normalization" begin
    function trunc_trapezoid(dist, lower, upper)
        n = 20001
        xs = range(lower, upper; length=n)
        h = (upper - lower) / (n - 1)
        vals = [exp(UncertainTea.logpdf(dist, x)) for x in xs]
        return h * (sum(vals) - (vals[1] + vals[end]) / 2)
    end

    @test trunc_trapezoid(truncatednormal(0.3, 1.2, -1.0, 2.5), -1.0, 2.5) ≈ 1.0 atol = 1e-4
    @test trunc_trapezoid(truncatedstudentt(5.0, 0.0, 1.0, -2.0, 2.0), -2.0, 2.0) ≈ 1.0 atol = 1e-4
end

# --- sampling ------------------------------------------------------------
@testset "trunc_rand" begin
    trunc_rng = MersenneTwister(20240701)
    trunc_hn = truncatednormal(0.0, 1.0, 0.0, Inf)
    trunc_draws = [rand(trunc_rng, trunc_hn) for _ = 1:20000]
    @test all(>=(0.0), trunc_draws)
    # Half-normal(0,1) mean is sqrt(2/pi).
    @test trunc_mean(trunc_draws) ≈ sqrt(2 / pi) atol = 0.02

    trunc_rng_t = MersenneTwister(987654321)
    trunc_tt = truncatedstudentt(4.0, 0.0, 1.0, -1.5, 2.5)
    trunc_t_draws = [rand(trunc_rng_t, trunc_tt) for _ = 1:10000]
    @test all(x -> -1.5 <= x <= 2.5, trunc_t_draws)
end

# --- unconstraining transforms ------------------------------------------
@testset "trunc_transforms" begin
    trunc_cases = (
        (BoundedTransform(-1.0, 2.0), [-3.0, -0.5, 0.0, 1.2, 3.5]),
        (LowerBoundedTransform(0.5), [-2.0, -0.3, 0.0, 1.7, 4.0]),
        (UpperBoundedTransform(4.0), [-2.0, -0.3, 0.0, 1.7, 4.0]),
    )
    for (trunc_t, trunc_points) in trunc_cases
        for trunc_u in trunc_points
            trunc_c = UncertainTea.to_constrained(trunc_t, trunc_u)
            @test UncertainTea.to_unconstrained(trunc_t, trunc_c) ≈ trunc_u atol = 1e-10
            trunc_fd = log(abs(trunc_ForwardDiff.derivative(
                x -> UncertainTea.to_constrained(trunc_t, x),
                trunc_u,
            )))
            @test UncertainTea.logabsdetjac(trunc_t, trunc_u) ≈ trunc_fd atol = 1e-8
        end
    end

    # BoundedTransform maps into the open interval; single-sided ones respect
    # the finite bound.
    @test -1.0 < UncertainTea.to_constrained(BoundedTransform(-1.0, 2.0), 5.0) < 2.0
    @test UncertainTea.to_constrained(LowerBoundedTransform(0.5), -8.0) > 0.5
    @test UncertainTea.to_constrained(UpperBoundedTransform(4.0), -8.0) < 4.0
end

# --- latent end-to-end: half-normal scale prior --------------------------
@testset "trunc_latent_end_to_end" begin
    @tea static function trunc_scale_model(y)
        sigma ~ truncatednormal(0.0f0, 5.0f0, 0.0f0, Inf)
        {:obs} ~ normal(0.0, sigma)
    end

    # The latent gets a single lower-bounded parameter slot.
    trunc_layout = parameterlayout(trunc_scale_model)
    @test parametercount(trunc_layout) == 1
    @test parametervaluecount(trunc_layout) == 1
    @test trunc_layout.slots[1].transform isa LowerBoundedTransform

    # generate/logjoint agreement over the full joint. The trace fixes both the
    # latent and the observation so its log_weight is the full joint; logjoint
    # scores `:sigma` as a latent (from the parameter vector) and conditions on
    # the observation `:obs` only -- constraining `:sigma` too would classify it
    # as an observation (docs/constraint-driven-conditioning.md, issue #95).
    trunc_full = choicemap((:sigma, 0.7f0), (:obs, 1.3f0))
    trunc_trace, _ = generate(trunc_scale_model, (nothing,), trunc_full; rng=MersenneTwister(3))
    trunc_pv = parameter_vector(trunc_trace)
    @test trunc_trace.log_weight ≈
          logjoint(trunc_scale_model, trunc_pv, (nothing,), choicemap((:obs, 1.3f0))) atol = 1e-6

    # NUTS runs and produces finite, strictly positive constrained samples.
    trunc_obs = choicemap((:obs, 1.3f0))
    trunc_chain = first(nuts(
        trunc_scale_model,
        (nothing,),
        trunc_obs;
        num_samples=100,
        num_warmup=100,
        rng=MersenneTwister(5),
    ))
    @test size(trunc_chain.constrained_samples, 2) == 100
    @test all(isfinite, trunc_chain.constrained_samples)
    @test all(>(0.0), trunc_chain.constrained_samples)

    # Batched logjoint (ForwardDiff fallback) matches the per-column value.
    trunc_u = transform_to_unconstrained(trunc_scale_model, [0.7])
    trunc_params = hcat(trunc_u, trunc_u .+ 0.4, trunc_u .- 0.6)
    trunc_batched = batched_logjoint_unconstrained(trunc_scale_model, trunc_params, (nothing,), trunc_obs)
    trunc_percol = [
        logjoint_unconstrained(trunc_scale_model, trunc_params[:, i], (nothing,), trunc_obs)
        for i = 1:size(trunc_params, 2)
    ]
    @test trunc_batched ≈ trunc_percol atol = 1e-6
end

# --- dynamic-bound latents are rejected ----------------------------------
@testset "trunc_dynamic_bound_rejection" begin
    # The rejection is raised while resolving the latent's parameter transform.
    @test_throws ArgumentError UncertainTea._supported_distribution_family(
        :(truncatednormal(0.0f0, 1.0f0, a, 1.0f0)),
    )
    @test_throws ArgumentError UncertainTea._supported_distribution_family(
        :(truncatedstudentt(5.0f0, 0.0f0, 1.0f0, a, b)),
    )

    # ... and it surfaces at macro-expansion of a model that declares such a latent.
    trunc_bad_model = :(@tea static function trunc_bad(a)
        x ~ truncatednormal(0.0f0, 1.0f0, a, 1.0f0)
        {:o} ~ normal(x, 1.0)
    end)
    @test_throws ArgumentError macroexpand(@__MODULE__, trunc_bad_model)

    # Static bounds remain accepted.
    @test UncertainTea._supported_distribution_family(
        :(truncatednormal(0.0f0, 1.0f0, -2.0, 3.0)),
    ) === :truncatednormal
end

# --- backend honestly reports the families as unsupported ----------------
@testset "trunc_backend_report" begin
    @tea static function trunc_report_model(y)
        sigma ~ truncatednormal(0.0f0, 5.0f0, 0.0f0, Inf)
        {:obs} ~ normal(0.0, sigma)
    end
    trunc_report = backend_report(trunc_report_model)
    @test trunc_report.supported == false
    @test any(issue -> occursin("truncatednormal", issue), trunc_report.issues)

    @tea static function trunc_report_model_t(y)
        z ~ truncatedstudentt(5.0f0, 0.0f0, 1.0f0, -2.0, 2.0)
        {:obs} ~ normal(z, 1.0)
    end
    trunc_report_t = backend_report(trunc_report_model_t)
    @test trunc_report_t.supported == false
    @test any(issue -> occursin("truncatedstudentt", issue), trunc_report_t.issues)
end

# --- issue #43: light-tail truncated Student-t normalizers stay finite ----
# The plain normalizer computed 1 - cdf(za) and cancelled to zero at Float64
# for light tails (large nu, far cutoff), scoring +Inf; the log-space
# normalizer (_t_log_normalizer, the CPU port of the device implementation)
# keeps every path finite and mutually consistent.
@testset "trunc_t_lighttail_normalizer" begin
    # one-sided: log Z = log S(15) for an effectively-normal tail
    @test isfinite(UncertainTea._t_log_normalizer(1.0e5, 15.0, Inf))
    @test UncertainTea._t_log_normalizer(1.0e5, 15.0, Inf) ≈
          UncertainTea._std_t_log_cdf(-15.0, 1.0e5) rtol = 1e-12
    # healthy domain agrees with the plain difference
    @test UncertainTea._t_log_normalizer(5.0, -2.0, 2.0) ≈
          log(UncertainTea._std_t_cdf(2.0, 5.0) - UncertainTea._std_t_cdf(-2.0, 5.0)) rtol = 1e-12

    lighttail = truncatedstudentt(1.0e5, 0.0, 1.0, 15.0, Inf)
    lp = UncertainTea.logpdf(lighttail, 15.2)
    @test isfinite(lp)
    # nu = 1e5 tracks the truncated normal to O(z^4/nu) ~ 5e-3 this deep in the
    # tail (the genuine tail correction, not a numerical error)
    @test lp ≈ UncertainTea.logpdf(truncatednormal(0.0, 1.0, 15.0, Inf), 15.2) atol = 1e-2

    # deep-tail finite interval: both plain CDFs underflow; the expm1 difference
    # resolves the ~e^-15.5 mass above the upper bound
    interval = truncatedstudentt(1.0e5, 0.0, 1.0, 15.0, 16.0)
    lp_interval = UncertainTea.logpdf(interval, 15.2)
    @test isfinite(lp_interval)
    @test lp_interval ≈ lp rtol = 1e-6
    @test !isapprox(lp_interval, lp; rtol=1e-9)

    # Float32 (codex review): the value-space incomplete beta underflows at
    # Float32 long before its log does (the log-cdf widens to Float64
    # internally), and the gradient's pdf/Z hazard ratios must form in log
    # space -- exp(-log Z) alone overflows Float32 into Inf * 0 = NaN
    lp32 = UncertainTea.logpdf(truncatedstudentt(1.0f5, 0.0f0, 1.0f0, 15.0f0, Inf32), 15.2f0)
    @test isfinite(lp32)
    # with the base density's normalizing constant Float64-widened too
    # (issue #53), the Float32 result sits at representation-rounding distance
    @test Float64(lp32) ≈ lp atol = 1e-4

    @tea static function trunc_t_f32_tail_model(offset)
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ truncatedstudentt(1.0f5, mu, 1.0f0, 14.0f0, Inf)
        return mu
    end
    cm32 = choicemap((:y, 14.2f0))
    params32 = reshape(Float32[0.01, -0.02], 1, 2)
    values32 = batched_logjoint_unconstrained(trunc_t_f32_tail_model, params32, (0.0f0,), cm32)
    gradients32 = batched_logjoint_gradient_unconstrained(trunc_t_f32_tail_model, params32, (0.0f0,), cm32)
    @test all(isfinite, values32)
    @test all(isfinite, gradients32)
    gradients64 =
        batched_logjoint_gradient_unconstrained(trunc_t_f32_tail_model, Float64.(params32), (0.0,), cm32)
    # the -k + pdf/Z cancellation amplifies rounding; ~1e-3 relative is the
    # achievable Float32 quality here (was NaN, then ~400% before the
    # Float64-widened log-pdf)
    @test all(isapprox(Float64(a), b; rtol=5e-3, atol=5e-3) for (a, b) in zip(vec(gradients32), vec(gradients64)))
end

# --- issues #96/#97: far-tail truncation windows score honestly -----------
# The value-space normalizers underflowed once the tail probability left the
# float range: truncatednormal returned +Inf (log of a zero erfc difference)
# and truncatedstudentt fell through a NaN guard into the midpoint form on a
# width-1 window (~0.0). Both now mirror the device log-space implementations,
# which double as the parity reference here.
@testset "trunc_far_tail_logpdf" begin
    # truncatednormal [40, 41] at 40.5: 1024-bit BigFloat reference (issue #96)
    tn_ref = -16.435496519450883
    @test UncertainTea.logpdf(truncatednormal(0.0, 1.0, 40.0, 41.0), 40.5) ≈ tn_ref rtol = 1e-12
    @test UncertainTea._backend_truncatednormal_logpdf(0.0, 1.0, 40.0, 41.0, 40.5) ≈ tn_ref rtol = 1e-12
    # CPU == device parity in the far tail (was an infinite disagreement)
    @test UncertainTea.logpdf(truncatednormal(0.0, 1.0, 40.0, 41.0), 40.5) ≈
          UncertainTea._device_truncatednormal_logpdf(0.0, 1.0, 40.0, 41.0, 40.5) rtol = 1e-12

    # one-sided far tail (erfc underflows at za/sqrt(2) > ~27.2)
    tn_one_sided = UncertainTea.logpdf(truncatednormal(0.0, 1.0, 39.0, Inf), 39.5)
    @test isfinite(tn_one_sided)
    @test tn_one_sided ≈ UncertainTea._device_truncatednormal_logpdf(0.0, 1.0, 39.0, Inf, 39.5) rtol = 1e-9

    # Float32 far tail (erfc32 underflows from z ~ 13); reference is the
    # Float64 evaluation of the same window
    tn32 = UncertainTea.logpdf(truncatednormal(0.0f0, 1.0f0, 15.0f0, 16.0f0), 15.5f0)
    @test isfinite(tn32)
    @test Float64(tn32) ≈ UncertainTea.logpdf(truncatednormal(0.0, 1.0, 15.0, 16.0), 15.5) atol = 1e-4

    # ulp-narrow interval: the value-space erfc difference cancelled to a
    # wrong finite value (36.0110); the log-space midpoint form is exact
    @test UncertainTea.logpdf(truncatednormal(0.0, 1.0, 1.0, nextfloat(1.0)), 1.0) ≈
          UncertainTea._device_truncatednormal_logpdf(0.0, 1.0, 1.0, nextfloat(1.0), 1.0) rtol = 1e-12

    # moderate tails keep their values (device already agreed there)
    for (a, b, x) in ((-1.0, 2.0, 0.5), (2.0, 3.0, 2.5), (-8.0, -7.0, -7.5), (0.0, Inf, 0.7))
        @test UncertainTea.logpdf(truncatednormal(0.0, 1.0, a, b), x) ≈
              UncertainTea._device_truncatednormal_logpdf(0.0, 1.0, a, b, x) rtol = 1e-12
    end

    # truncatedstudentt nu = 1e5, [40, 41] at 40.5: 256-bit BigFloat quadrature
    # reference (issue #97); was silently ~0.0
    tt_ref = -16.13071622789926
    @test UncertainTea.logpdf(truncatedstudentt(1.0e5, 0.0, 1.0, 40.0, 41.0), 40.5) ≈ tt_ref rtol = 1e-9
    @test UncertainTea._backend_truncatedstudentt_logpdf(1.0e5, 0.0, 1.0, 40.0, 41.0, 40.5) ≈ tt_ref rtol =
        1e-9
    @test UncertainTea.logpdf(truncatedstudentt(1.0e5, 0.0, 1.0, 40.0, 41.0), 40.5) ≈
          UncertainTea._device_truncatedstudentt_logpdf(1.0e5, 0.0, 1.0, 40.0, 41.0, 40.5) rtol = 1e-9

    # the log-CDF itself stays finite past the value-space underflow (z ~ 39)
    @test UncertainTea._std_t_log_cdf(-40.0, 1.0e5) ≈ -798.2679656412453 rtol = 1e-11
    # one-sided far-tail window
    tt_one_sided = UncertainTea.logpdf(truncatedstudentt(1.0e5, 0.0, 1.0, 40.0, Inf), 40.5)
    @test isfinite(tt_one_sided)
    @test tt_one_sided ≈ UncertainTea._device_truncatedstudentt_logpdf(1.0e5, 0.0, 1.0, 40.0, Inf, 40.5) rtol =
        1e-9

    # Float32 far tail (widened internally, narrowed on return)
    tt32 = UncertainTea.logpdf(truncatedstudentt(1.0f5, 0.0f0, 1.0f0, 40.0f0, 41.0f0), 40.5f0)
    @test isfinite(tt32)
    @test Float64(tt32) ≈ tt_ref atol = 1e-4

    # moderate/heavy tails keep their value-space accuracy (regression guard
    # for the log-space incomplete beta against SpecialFunctions.beta_inc)
    for (z, nu) in ((-2.0, 5.0), (2.0, 5.0), (-0.3, 4.0), (-8.0, 30.0), (0.5, 1.0))
        x = nu / (nu + z^2)
        reg = trunc_beta_inc(nu / 2, 0.5, x)[1]
        ref = z < 0 ? log(reg) - log(2.0) : log1p(-reg / 2)
        @test UncertainTea._std_t_log_cdf(z, nu) ≈ ref rtol = 1e-13 atol = 1e-15
    end

    # gradients through the far-tail normalizers stay finite and correct
    trunc_fd_dmu = trunc_ForwardDiff.derivative(
        m -> UncertainTea.logpdf(truncatednormal(m, 1.0, 40.0, 41.0), 40.5),
        0.0,
    )
    @test isfinite(trunc_fd_dmu)
    trunc_fd_step = 1.0e-6
    trunc_fd_central =
        (
            UncertainTea.logpdf(truncatednormal(trunc_fd_step, 1.0, 40.0, 41.0), 40.5) -
            UncertainTea.logpdf(truncatednormal(-trunc_fd_step, 1.0, 40.0, 41.0), 40.5)
        ) / (2 * trunc_fd_step)
    @test trunc_fd_dmu ≈ trunc_fd_central rtol = 1e-5
    trunc_fd_dmu_t = trunc_ForwardDiff.derivative(
        m -> UncertainTea.logpdf(truncatedstudentt(1.0e5, m, 1.0, 40.0, 41.0), 40.5),
        0.0,
    )
    @test isfinite(trunc_fd_dmu_t)
end

# --- issue #72: sampler stays in-support in far tails ----------------------
# The inverse-CDF sampler proceeded even when both CDFs rounded to the same
# float, escaping the support ([10, 11] gave ~8.2, [-11, -10] gave -Inf). The
# survival-space inversion plus tail rejection sampling keeps every draw
# inside [lower, upper] with the correct truncated law.
@testset "trunc_far_tail_rand" begin
    # exact truncated moments over [a, b] via BigFloat survival functions;
    # left-tail windows go through the mirror symmetry so the BigFloat
    # difference itself never cancels (erfc(-z) parks the tail next to 2)
    function trunc_exact_moments(a, b)
        if b <= 0
            m_mirror, v_mirror = trunc_exact_moments(-b, -a)
            return -m_mirror, v_mirror
        end
        S(z) = isinf(z) ? big(0) : UncertainTea.erfc(big(z) / sqrt(big(2))) / 2
        phi(z) = isinf(z) ? big(0) : exp(-big(z)^2 / 2) / sqrt(2 * big(pi))
        Z = S(a) - S(b)
        m = (phi(a) - phi(b)) / Z
        v = 1 + (big(isinf(a) ? 0 : a) * phi(a) - big(isinf(b) ? 0 : b) * phi(b)) / Z - m^2
        return Float64(m), Float64(v)
    end

    trunc_tail_rng = MersenneTwister(20260719)
    for (lower, upper) in ((10.0, 11.0), (-11.0, -10.0), (10.0, Inf), (40.0, 41.0), (-41.0, -40.0))
        dist = truncatednormal(0.0, 1.0, lower, upper)
        draws = [rand(trunc_tail_rng, dist) for _ = 1:8000]
        @test all(x -> lower <= x <= upper, draws)
        m_exact, v_exact = trunc_exact_moments(lower, upper)
        m_emp = trunc_mean(draws)
        v_emp = trunc_mean((draws .- m_emp) .^ 2)
        # 5-sigma bands on the empirical mean and a loose one on the variance
        @test abs(m_emp - m_exact) <= 5 * sqrt(v_exact / length(draws))
        @test v_emp ≈ v_exact rtol = 0.15
    end

    # Float32 right tail (issue repro gave ~5.29f0, far outside [15, 16])
    trunc_tail32 = truncatednormal(0.0f0, 1.0f0, 15.0f0, 16.0f0)
    draws32 = [rand(trunc_tail_rng, trunc_tail32) for _ = 1:8000]
    @test eltype(draws32) === Float32
    @test all(x -> 15.0f0 <= x <= 16.0f0, draws32)
    m32_exact, v32_exact = trunc_exact_moments(15.0, 16.0)
    @test abs(trunc_mean(Float64.(draws32)) - m32_exact) <= 5 * sqrt(v32_exact / length(draws32))

    # ulp-narrow window: degenerate but still in-support
    trunc_ulp = truncatednormal(0.0, 1.0, 1.0, nextfloat(1.0))
    @test all(x -> 1.0 <= x <= nextfloat(1.0), [rand(trunc_tail_rng, trunc_ulp) for _ = 1:50])

    # healthy straddling window: the classical inverse-CDF path still matches
    # its exact moments
    trunc_mid = truncatednormal(0.0, 1.0, -1.0, 2.0)
    mid_draws = [rand(trunc_tail_rng, trunc_mid) for _ = 1:8000]
    @test all(x -> -1.0 <= x <= 2.0, mid_draws)
    m_mid, v_mid = trunc_exact_moments(-1.0, 2.0)
    @test abs(trunc_mean(mid_draws) - m_mid) <= 5 * sqrt(v_mid / length(mid_draws))
end

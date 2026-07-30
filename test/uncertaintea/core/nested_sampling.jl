# Nested sampling (issue #234): a CPU-only, gradient-free log-evidence estimator.
#
# The evidence is the oracle. On conjugate models with a KNOWN marginal
# likelihood, the estimated log Z must match the analytic value within a few
# times nested sampling's own information-based uncertainty (sqrt(H/N)) -- this
# is the make-or-break correctness check, because the whole point of nested
# sampling is the evidence. We also check posterior-moment recovery from the
# importance-weighted dead points, bimodal mode coverage, determinism, and that
# the two constrained-replacement strategies (:rwmh, :rejection) agree with the
# analytic evidence.
#
# Sizes are kept modest for CI (nested sampling runs one model evaluation per
# random-walk step). No `using Statistics`: local weighted-moment helpers below.

@testset "nested_sampling" begin
    # --- local helpers (no Statistics / SpecialFunctions dependency) ---
    ns_logfact(m::Int) = m <= 1 ? 0.0 : sum(log, 2:m)
    # log B(p, q) for POSITIVE INTEGER p, q via factorials (exact in Float64 here).
    ns_logbeta_int(p::Int, q::Int) = ns_logfact(p - 1) + ns_logfact(q - 1) - ns_logfact(p + q - 1)
    ns_logaddexp(a, b) = (m=max(a, b); m + log(exp(a - m) + exp(b - m)))
    ns_normlogpdf(x, mu, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - mu) / s)^2
    function ns_wmean(values, weights)
        total = 0.0
        for j in eachindex(values, weights)
            total += weights[j] * values[j]
        end
        return total
    end
    function ns_wvar(values, weights, mean_value)
        total = 0.0
        for j in eachindex(values, weights)
            total += weights[j] * (values[j] - mean_value)^2
        end
        return total
    end

    # ------------------------------------------------------------------
    # TEST 1: Beta-Bernoulli -- 1-D bounded latent (logit transform).
    # theta ~ Beta(2,3); k successes in n trials.
    # Z = B(2+k, 3+n-k) / B(2,3); posterior is Beta(2+k, 3+n-k).
    # ------------------------------------------------------------------
    @tea static function ns_beta_bernoulli(n)
        theta ~ beta(2.0, 3.0)
        for i = 1:n
            {:y => i} ~ bernoulli(theta)
        end
        return theta
    end

    bb_n, bb_k = 12, 8
    bb_ys = vcat(ones(Int, bb_k), zeros(Int, bb_n - bb_k))
    bb_cm = choicemap((:y => i, bb_ys[i]) for i = 1:bb_n)
    bb_logZ = ns_logbeta_int(2 + bb_k, 3 + bb_n - bb_k) - ns_logbeta_int(2, 3)
    bb_post_mean = (2.0 + bb_k) / (2.0 + 3.0 + bb_n)

    bb = nested_sampling(
        ns_beta_bernoulli, (bb_n,), bb_cm;
        num_live_points=80, num_walk=25, dlogz=0.05, rng=MersenneTwister(11),
    )

    # log Z vs analytic within a few sigma (the key correctness check).
    @test log_evidence_error(bb) > 0.0
    @test log_evidence_error(bb) < 1.0
    @test abs(log_evidence(bb) - bb_logZ) < 4 * log_evidence_error(bb)
    @test information(bb) > 0.0

    # posterior mean of theta from importance-weighted dead points.
    bb_thetas = bb.constrained_samples[1, :]
    bb_mean = ns_wmean(bb_thetas, bb.normalized_weights)
    @test isapprox(bb_mean, bb_post_mean; atol=0.03)

    # weights are a normalized distribution; ess and sample count are sane.
    @test sum(bb.normalized_weights) ≈ 1.0
    @test ess(bb) > 1.0
    @test numsamples(bb) == size(bb.constrained_samples, 2)
    @test bb.num_iterations >= 1

    # ------------------------------------------------------------------
    # TEST 2: Gaussian-Gaussian -- 1-D unbounded latent (identity transform).
    # mu ~ N(0, 2^2); y_i ~ N(mu, 1). Analytic marginal likelihood and Gaussian
    # posterior below.
    # ------------------------------------------------------------------
    @tea static function ns_gauss_gauss(n)
        mu ~ normal(0.0, 2.0)
        for i = 1:n
            {:y => i} ~ normal(mu, 1.0)
        end
        return mu
    end

    gg_ys = [0.8, 1.2, 0.5, 1.5, 0.9]
    gg_n = length(gg_ys)
    gg_cm = choicemap((:y => i, gg_ys[i]) for i = 1:gg_n)
    gg_m0, gg_s0, gg_sig = 0.0, 2.0, 1.0
    gg_ybar = sum(gg_ys) / gg_n
    gg_ss = sum((y - gg_ybar)^2 for y in gg_ys)
    gg_logZ =
        -gg_n / 2 * log(2π * gg_sig^2) - gg_ss / (2 * gg_sig^2) +
        0.5 * log(2π * gg_sig^2 / gg_n) +
        ns_normlogpdf(gg_ybar, gg_m0, sqrt(gg_s0^2 + gg_sig^2 / gg_n))
    gg_prec = 1 / gg_s0^2 + gg_n / gg_sig^2
    gg_post_mean = (gg_m0 / gg_s0^2 + sum(gg_ys) / gg_sig^2) / gg_prec
    gg_post_var = 1 / gg_prec

    gg = nested_sampling(
        ns_gauss_gauss, (gg_n,), gg_cm;
        num_live_points=80, num_walk=25, dlogz=0.05, rng=MersenneTwister(12),
    )
    @test abs(log_evidence(gg) - gg_logZ) < 4 * log_evidence_error(gg)
    gg_mus = gg.constrained_samples[1, :]
    gg_mean = ns_wmean(gg_mus, gg.normalized_weights)
    gg_var = ns_wvar(gg_mus, gg.normalized_weights, gg_mean)
    @test isapprox(gg_mean, gg_post_mean; atol=0.1)
    @test isapprox(gg_var, gg_post_var; rtol=0.4)

    # ------------------------------------------------------------------
    # TEST 3: Bimodal posterior -- mixture PRIOR with a well-separated pair of
    # modes and a broad observation, so both modes survive. Live points populate
    # both modes, so nested sampling recovers both (a genuinely multimodal case)
    # AND the analytic mixture-of-Gaussians evidence.
    # ------------------------------------------------------------------
    @tea static function ns_bimodal()
        x ~ mixture((0.5, 0.5), normal(-4.0, 0.5), normal(4.0, 0.5))
        d ~ normal(x, 2.0)
        return x
    end
    bi_cm = choicemap((:d, 0.0))
    bi_sd = sqrt(0.5^2 + 2.0^2)
    bi_logZ = ns_logaddexp(
        log(0.5) + ns_normlogpdf(0.0, -4.0, bi_sd),
        log(0.5) + ns_normlogpdf(0.0, 4.0, bi_sd),
    )

    bi = nested_sampling(
        ns_bimodal, (), bi_cm;
        num_live_points=120, num_walk=30, dlogz=0.05, rng=MersenneTwister(13),
    )
    @test abs(log_evidence(bi) - bi_logZ) < 4 * log_evidence_error(bi)

    # both modes carry substantial posterior mass (multimodal recovery).
    bi_xs = bi.constrained_samples[1, :]
    bi_neg = sum(bi.normalized_weights[j] for j in eachindex(bi_xs) if bi_xs[j] < 0.0)
    bi_pos = sum(bi.normalized_weights[j] for j in eachindex(bi_xs) if bi_xs[j] >= 0.0)
    @test bi_neg > 0.2
    @test bi_pos > 0.2

    # ------------------------------------------------------------------
    # TEST 4: :rejection replacement (exact i.i.d. prior|constraint draws) agrees
    # with the analytic evidence -- the gold-standard cross-check for :rwmh.
    # ------------------------------------------------------------------
    bb_rej = nested_sampling(
        ns_beta_bernoulli, (bb_n,), bb_cm;
        num_live_points=50, replacement=:rejection, dlogz=0.1,
        max_rejection_attempts=200_000, rng=MersenneTwister(21),
    )
    @test bb_rej.replacement === :rejection
    @test abs(log_evidence(bb_rej) - bb_logZ) < 4 * log_evidence_error(bb_rej)

    # ------------------------------------------------------------------
    # TEST 5: determinism under a fixed seed.
    # ------------------------------------------------------------------
    d1 = nested_sampling(ns_beta_bernoulli, (bb_n,), bb_cm; num_live_points=40, rng=MersenneTwister(7))
    d2 = nested_sampling(ns_beta_bernoulli, (bb_n,), bb_cm; num_live_points=40, rng=MersenneTwister(7))
    @test log_evidence(d1) == log_evidence(d2)
    @test log_evidence_error(d1) == log_evidence_error(d2)
    @test d1.constrained_samples == d2.constrained_samples
    @test d1.normalized_weights == d2.normalized_weights

    # ------------------------------------------------------------------
    # TEST 6: predict from nested-sampling draws (smoke: resample by weights).
    # ------------------------------------------------------------------
    bb_pred = predict(ns_beta_bernoulli, (bb_n,), bb; num_draws=20, rng=MersenneTwister(5))
    @test length(bb_pred) == 20

    # ------------------------------------------------------------------
    # TEST 7: argument validation.
    # ------------------------------------------------------------------
    @test_throws ArgumentError nested_sampling(ns_beta_bernoulli, (bb_n,), bb_cm; num_live_points=1)
    @test_throws ArgumentError nested_sampling(ns_beta_bernoulli, (bb_n,), bb_cm; replacement=:bogus)
    @test_throws ArgumentError nested_sampling(ns_beta_bernoulli, (bb_n,), bb_cm; num_walk=0)

    # show renders the headline fields.
    @test occursin("NestedSamplingResult", sprint(show, bb))
    @test occursin("log_evidence", sprint(show, bb))
end

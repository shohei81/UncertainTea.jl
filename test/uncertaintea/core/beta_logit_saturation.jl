# Issue #343: a beta latent behind LogitTransform through the sigmoid cliff.
#
# Once sigmoid(theta) rounds to exactly 1.0 (theta >~ 36.74 in Float64), the
# value path scores -Inf while the gradient paths used to drop the beta
# partials and return the FINITE transform-Jacobian derivative (exactly -1.0)
# -- a silently wrong gradient that leapfrog gradient guards never rejected.
# The fix makes the gradient non-finite whenever a latent-flowing value lands
# off the support: the batched analytic path poisons the skipped partials with
# NaN (`_poison_offsupport_value_gradient!`), and the single ForwardDiff path
# gets NaN partials from the kernels' off-support return
# (`_offsupport_neginf`).
#
# Below the cliff (theta ~ 30..36.7) the density term (beta - 1) * log1p(-v)
# still loses accuracy because 1 - v carries the absolute rounding error of
# v = sigmoid(theta), i.e. a relative error up to ~ eps * exp(theta). That
# band is a documented residual of the transform-agnostic kernel design (the
# exact fix needs the fused unconstrained-space density, the issue-#105
# treatment extended to the density term); the assertions below pin the error
# inside the eps * exp(theta) envelope. The batched analytic gradient is
# exempt from the band: its 1/(1-v) value partial and the v*(1-v) chain-rule
# factor are formed from the SAME rounded 1 - v and cancel exactly, so it
# tracks the true composite to ~1e-12 right up to the cliff.

# Exact composite for p ~ beta(2,2); y ~ bernoulli(p), y = 1, in unconstrained
# theta: L = log(6) + 3 log sigma(theta) + 2 log(1 - sigma(theta)) (density +
# bernoulli + logit Jacobian), dL/dtheta = 3 - 5 sigma(theta).
function bls_reference(theta::Float64)
    bls_t = BigFloat(theta)
    bls_s = inv(1 + exp(-bls_t))
    bls_value = log(BigFloat(6)) + 3 * log(bls_s) + 2 * log1p(-bls_s)
    bls_gradient = 3 - 5 * bls_s
    return Float64(bls_value), Float64(bls_gradient)
end

@testset "beta_logit_saturation" begin
    @tea static function bls_model()
        p ~ beta(2.0, 2.0)
        {:y} ~ bernoulli(p)
    end
    bls_cm = choicemap(:y => true)

    # Past the cliff: the logjoint saturates to -Inf (true value is finite,
    # -theta * 2 - log(6) + O(exp(-theta)) -- the value-path residual) and the
    # unconstrained gradient must be NON-FINITE on the single AND batched
    # paths so integrator guards reject; no finite-wrong -1.0 survives.
    for bls_theta in (37.0, 38.0, 40.0)
        @test logjoint_unconstrained(bls_model, [bls_theta], (), bls_cm) == -Inf
        bls_g = logjoint_gradient_unconstrained(bls_model, [bls_theta], (), bls_cm)
        @test !isfinite(bls_g[1])
        bls_bv = batched_logjoint_unconstrained(bls_model, reshape([bls_theta], 1, 1), (), bls_cm)
        @test bls_bv[1] == -Inf
        bls_bg = batched_logjoint_gradient_unconstrained(bls_model, reshape([bls_theta], 1, 1), (), bls_cm)
        @test !isfinite(bls_bg[1, 1])
    end

    # A saturated column must not leak NaN into its batch neighbors, and the
    # negative side has no cliff (sigmoid(-38) is tiny but nonzero).
    bls_batch = reshape([1.25, 38.0, -38.0], 1, 3)
    bls_batch_values = batched_logjoint_unconstrained(bls_model, bls_batch, (), bls_cm)
    bls_batch_gradient = batched_logjoint_gradient_unconstrained(bls_model, bls_batch, (), bls_cm)
    @test isfinite(bls_batch_values[1])
    @test bls_batch_values[2] == -Inf
    @test isfinite(bls_batch_values[3])
    @test isfinite(bls_batch_gradient[1, 1])
    @test !isfinite(bls_batch_gradient[1, 2])
    @test bls_batch_gradient[1, 3] ≈ bls_reference(-38.0)[2] atol = 1e-9

    # Pre-cliff error band: the 1 - sigmoid(theta) cancellation bounds both
    # value paths and the ForwardDiff gradient by ~ eps * exp(theta) (factor 4
    # margin); the batched analytic gradient stays exact (see header).
    for bls_theta = 30.0:1.0:36.0
        bls_expected_value, bls_expected_gradient = bls_reference(bls_theta)
        bls_tol = 4 * eps() * exp(bls_theta) + 1e-9
        @test logjoint_unconstrained(bls_model, [bls_theta], (), bls_cm) ≈ bls_expected_value atol = bls_tol
        bls_g = logjoint_gradient_unconstrained(bls_model, [bls_theta], (), bls_cm)
        @test bls_g[1] ≈ bls_expected_gradient atol = bls_tol
        bls_bv = batched_logjoint_unconstrained(bls_model, reshape([bls_theta], 1, 1), (), bls_cm)
        @test bls_bv[1] ≈ bls_expected_value atol = bls_tol
        bls_bg = batched_logjoint_gradient_unconstrained(bls_model, reshape([bls_theta], 1, 1), (), bls_cm)
        @test bls_bg[1, 1] ≈ bls_expected_gradient atol = 1e-9
    end

    # Sanity: the normal range is untouched -- the single (ForwardDiff) and
    # batched analytic gradients both match the exact composite tightly.
    for bls_theta = -4.5:1.5:4.5
        bls_expected_value, bls_expected_gradient = bls_reference(bls_theta)
        @test logjoint_unconstrained(bls_model, [bls_theta], (), bls_cm) ≈ bls_expected_value atol = 1e-12
        bls_g = logjoint_gradient_unconstrained(bls_model, [bls_theta], (), bls_cm)
        @test bls_g[1] ≈ bls_expected_gradient atol = 1e-12
        bls_bg = batched_logjoint_gradient_unconstrained(bls_model, reshape([bls_theta], 1, 1), (), bls_cm)
        @test bls_bg[1, 1] ≈ bls_expected_gradient atol = 1e-12
        @test bls_bg[1, 1] ≈ bls_g[1] atol = 1e-12
    end

    # OBSERVED off-support values keep the previous skip semantics on both
    # paths: the value carries no derivative seed, so the -Inf evaluation
    # keeps a finite gradient for unrelated latents (here d/dmu of the normal
    # prior) instead of being poisoned.
    @tea static function bls_observed_model()
        mu ~ normal(0.0, 1.0)
        {:x} ~ beta(2.0, 2.0)
    end
    bls_obs_cm = choicemap(:x => 1.5)
    @test logjoint_unconstrained(bls_observed_model, [0.7], (), bls_obs_cm) == -Inf
    bls_obs_g = logjoint_gradient_unconstrained(bls_observed_model, [0.7], (), bls_obs_cm)
    @test bls_obs_g[1] ≈ -0.7 atol = 1e-12
    bls_obs_bg =
        batched_logjoint_gradient_unconstrained(bls_observed_model, reshape([0.7], 1, 1), (), bls_obs_cm)
    @test bls_obs_bg[1, 1] ≈ -0.7 atol = 1e-12

    # Positive-support analogue of the same skip-partials pattern: a gamma
    # latent whose exp(theta) underflows to exactly 0.0 scores -Inf, and the
    # batched analytic gradient is now poisoned instead of returning the
    # finite Jacobian-only +1.0. (The single ForwardDiff path cannot be
    # poisoned there -- the value's partials underflow WITH the value, unlike
    # the logit case -- so its rejection rests on the -Inf value guard.)
    @tea static function bls_gamma_model()
        x ~ gamma(2.0, 1.0)
    end
    @test logjoint_unconstrained(bls_gamma_model, [-800.0], ()) == -Inf
    bls_gamma_bv = batched_logjoint_unconstrained(bls_gamma_model, reshape([-800.0], 1, 1), ())
    @test bls_gamma_bv[1] == -Inf
    bls_gamma_bg = batched_logjoint_gradient_unconstrained(bls_gamma_model, reshape([-800.0], 1, 1), ())
    @test !isfinite(bls_gamma_bg[1, 1])
end

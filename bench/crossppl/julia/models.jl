# Canonical benchmark models, shared shapes with the Stan/NumPyro definitions.
# Priors follow the literature-standard forms so every framework states the
# identical joint density (see bench/crossppl/README.md).

# Eight schools (Rubin 1981): mu ~ N(0,5), tau ~ HalfCauchy(5),
# theta_j ~ N(mu, tau), y_j ~ N(theta_j, sigma_j).  J = 8 is literal because
# `iid` requires a literal count.
@tea static function bench_eight_schools_centered(sigma)
    mu ~ normal(0.0, 5.0)
    tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
    theta ~ iid(normal(mu, tau), 8)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

@tea static function bench_eight_schools_noncentered(sigma)
    mu ~ normal(0.0, 5.0)
    tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

# Logistic regression: alpha ~ N(0,2.5), beta_d ~ N(0,2.5),
# y_i ~ BernoulliLogit(alpha + x_i'beta).  D = 8 is literal; the observation
# loop is loop-addressed because only `normal.` broadcasts.  The observation is
# the logit-scale `bernoullilogit` family (issue #149) so the linear predictor
# `alpha + sum(beta .* X[:, i])` backend-lowers to the fused analytic path
# (issue #150), matching Stan's `bernoulli_logit` and NumPyro's
# `Bernoulli(logits=...)` sides — same joint density, stable parameterization.
# The i.i.d. N(0, 2.5) coefficient prior is written as a diagonal `mvnormal` (an
# identical joint density) so the whole plan, prior included, rides the backend.
@tea static function bench_logistic(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ mvnormal(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5),
    )
    for i = 1:n
        {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
    end
    return alpha
end

# Large logistic regression (D=16, N=8000): the same GLM shape as
# `bench_logistic`, scaled up so the per-observation D-dim dot product over N
# observations makes the gradient genuinely heavy.  This is the model that
# exercises the device analytic path (issue #135) and the many-chains scaling
# sweep — the small logistic/gauss models are dominated by device dispatch
# overhead, so device work cannot be measured on them.  Identical priors to
# `bench_logistic` (alpha ~ N(0,2.5), beta_d ~ N(0,2.5) as a diagonal
# `mvnormal`), so both the host analytic path (issue #150) and the device
# analytic path (issue #135) apply.  D = 16 is literal (mvnormal tuple length)
# and is the device coefficient-dimension cap (DEVICE_MAX_VECTOR_DIMENSION);
# N = 8000 keeps D*N at the intended heavy per-gradient budget while the model
# still lowers to the device.  The observation loop is loop-addressed because
# only `normal.` broadcasts.
@tea static function bench_logistic_large(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ mvnormal(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5),
    )
    for i = 1:n
        {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
    end
    return alpha
end

# Poisson GLM in the BROADCAST observation form (issue #317): the whole count
# vector is one addressed choice, `{:y} ~ poisson.(exp.(a .+ b .* x))` — the
# same vectorized formulation NumPyro states naturally and Stan writes as
# `y ~ poisson_log(a + b * x)`. Rides the backend-native dense broadcast step
# with the analytic batched gradient (issue #287); the first cross-PPL case on
# a post-#300 fast path.
@tea static function bench_poisson_glm(x, n)
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ poisson.(exp.(a .+ b .* x))
    return a
end

# J=32 hierarchical Gaussian (eight-schools shape, CENTERED, log-normal tau) —
# the P >= 24 (34-parameter) model for the reverse-mode (Enzyme) leg
# (issue #317). Centered + lognormal deliberately: the reverse tier engages for
# plain iid latents, while the noncentered `reparam` machinery and the
# truncated-t tau both fail Enzyme's type analysis today (the adtype guard
# would silently fall back to forward). The eight_schools twins keep measuring
# the canonical funnel; this model measures the gradient tiers at real P.
@tea static function bench_schools_large(sigma)
    mu ~ normal(0.0, 5.0)
    log_tau ~ normal(0.0, 1.0)
    theta ~ iid(normal(mu, exp(log_tau)), 32)
    for i = 1:32
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

# Gaussian mean/scale estimation with a loop-addressed observation vector —
# the device-supported form (same shape as test/gpu gpu_gauss_model) used for
# the chain-count scaling sweep.  The logistic model above now rides the HOST
# batched analytic path via the fused GLM linear predictor (issue #150); the
# device (KernelAbstractions) leg for that same shape is the follow-on #135.
@tea static function bench_gauss(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# Discrete-latent finite mixture (issue #224): a 2-component Gaussian mixture with
# unknown, ORDERED component means and a shared scale, marginalized over the
# per-observation component indicator via the single-site `mixture` machinery
# (issue #13's acceptance oracle -- the same marginal density Stan hand-writes
# with `log_mix`/`log_sum_exp`). The ordering `mu2 = mu1 + exp(log_gap)` makes the
# model identifiable so the cross-PPL gate compares well-defined posteriors
# instead of label-switched ones. CPU-only (enumerate/mixture is device-
# unsupported until #67).
@tea static function bench_mixture(n)
    mu1 ~ normal(0.0, 3.0)
    log_gap ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ mixture((0.4, 0.6), normal(mu1, s), normal(mu1 + exp(log_gap), s))
    end
    return mu1
end

# lkjcholesky correlation model (issue #224): a d=2 multivariate normal with an
# LKJ Cholesky correlation prior and per-dimension log-normal scales. Exercises
# the CholeskyCorrTransform + packed logpdf (#49/#57) against Stan's
# `lkj_corr_cholesky` + `multi_normal_cholesky` and NumPyro's `LKJCholesky`. The
# latent `Omega` is the packed lower-triangular correlation factor (column-major:
# [L11, L21, L22]); the free correlation is `Omega[2]`. d is kept at 2 so the
# device dense/cholesky caps are not a constraint. CPU-reference-only (the LKJ
# family is device-unsupported).
@tea static function bench_lkj(zeros2, n)
    Omega ~ lkjcholesky(2, 2.0)
    tau ~ iid(lognormal(0.0, 0.5), 2)
    Ltril = scale_cholesky(tau, Omega)
    for i = 1:n
        {:y => i} ~ mvnormaldense(zeros2, Ltril)
    end
    return Omega
end

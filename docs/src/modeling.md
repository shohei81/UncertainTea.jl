# Modeling

This page covers the modeling features beyond the basics in
[Getting Started](getting-started.md): vectorized observations for GLMs,
the Gaussian-process suite, structured and specialized distributions, and how
gradients are chosen for a compiled model.

## Vectorized observations (broadcast GLMs)

A whole observation vector can be a single addressed choice, written with
Julia's dot syntax. The canonical linear-regression spelling:

```julia
@tea static function linreg(x, n)
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ normal.(a .+ b .* x, 0.5)
    return a
end

constraints = choicemap((:y, ys))   # ys::Vector{Float64}, length n
chain = nuts(linreg, (x, n), constraints; num_samples=500, num_warmup=500)
```

The broadcast form scores element-wise like the equivalent
`for i = 1:n; {:y => i} ~ normal(...); end` loop, but stays a single dense
plan step with an analytic batched gradient — the fast path for GLM-style
models. Beyond `normal`, the scalar families `poisson`, `bernoulli`,
`bernoullilogit`, `exponential`, and `studentt` support the same spelling,
which covers the standard link functions:

```julia
@tea static function poisson_glm(x, n)
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ poisson.(exp.(a .+ b .* x))     # log link
    return a
end

@tea static function logit_glm(x, n)
    b ~ normal(0.0, 1.0)
    {:y} ~ bernoullilogit.(b .* x)         # logit link, no manual clamping
    return b
end
```

Dotted *function* calls (`exp.(...)`, `log1p.(...)`) are supported inside
broadcast arguments and in deterministic expressions, alongside the dotted
operators. Integer and boolean observation vectors (`Vector{Int}`,
`Vector{Bool}` — natural for count and binary data) are accepted directly in
the constraints.

## Gaussian processes

### Marginal GP regression

[`gaussianprocess`](@ref) is a multivariate distribution over `n` outputs whose
covariance is built from a kernel over input locations (a `d × n` matrix).
The Gaussian noise is integrated out analytically, so a GP regression with
latent hyperparameters is just:

```julia
@tea static function gp_reg(X)
    log_l ~ normal(0.0, 1.0)
    log_v ~ normal(0.0, 1.0)
    log_n ~ normal(-1.0, 1.0)
    {:y} ~ gaussianprocess(X, exp(log_l), exp(log_v), exp(log_n))
    return log_l
end
```

The positional form `gaussianprocess(X, lengthscale, variance, noise)` is the
RBF (squared-exponential) shorthand. A vector `lengthscale` gives ARD —
one lengthscale per input dimension.

### Kernel specifications

Other covariance structures are expressed with kernel constructors and
composition:

- [`rbf_kernel`](@ref)`(lengthscale, variance)`
- [`matern32_kernel`](@ref)`(lengthscale, variance)` /
  [`matern52_kernel`](@ref)`(lengthscale, variance)`
- [`periodic_kernel`](@ref)`(lengthscale, variance, period)`
- [`kernel_sum`](@ref)`(k1, k2)` / [`kernel_product`](@ref)`(k1, k2)`

```julia
# locally periodic: the workhorse composition for seasonal-plus-trend data
k = kernel_product(periodic_kernel(1.0, 1.0, 12.0), rbf_kernel(36.0, 1.0))
{:y} ~ gaussianprocess(X, k, noise)
```

`lengthscale` may be a vector (ARD) in any of the stationary kernels.

### Sparse GPs

For large `n`, [`sparsegaussianprocess`](@ref)`(X, Z, kernel, noise)` is the
FITC approximation with inducing inputs `Z` (a `d × m` matrix, `m ≪ n`):
cost drops from `O(n³)` to `O(n m²)`. At `Z = X` it recovers the dense GP.
The same positional RBF shorthand
`sparsegaussianprocess(X, Z, lengthscale, variance, noise)` exists.

### Latent GP functions and non-Gaussian likelihoods

When the GP prior feeds a non-Gaussian likelihood (classification, counts),
the function values themselves are latent. [`gp_cholesky`](@ref)`(X, kernel,
jitter)` returns the lower-triangular Cholesky factor `L` of the kernel
matrix, which plugs into `mvnormaldense` as the scale factor of the latent
function vector:

```julia
@tea static function gp_classifier(X)
    logl ~ normal(0.0, 0.5)
    L = gp_cholesky(X, exp(logl), 2.0, 1.0e-6)
    f ~ mvnormaldense((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), L)
    for i = 1:10
        {:y => i} ~ bernoullilogit(f[i])
    end
    return logl
end
```

Kernel hyperparameters can themselves be latent, as above — the Cholesky
factor is recomputed per evaluation and differentiated through.

For a fixed-hyperparameter GP prior, [`elliptical_slice`](@ref) samples the
latent function gradient-free — see
[Inference Overview](inference.md#other-inference-tools).

## Structured and specialized distributions

- [`hmm`](@ref)`(init, transition, means, sigma)` — a hidden Markov model with
  Gaussian emissions, scored over an observation sequence with the forward
  algorithm (discrete states marginalized out, so gradient-based samplers
  apply to the continuous parameters).
- [`orderedlogistic`](@ref)`(eta, cutpoints)` — ordinal regression with the
  cumulative-logit link (Likert scales, graded outcomes).
- [`zeroinflatedpoisson`](@ref)`(p, lambda)` /
  [`zeroinflatednegativebinomial`](@ref)`(p, successes, probability)` —
  mixtures of a point mass at zero (probability `p`) with a count
  distribution, for over-dispersed zero-heavy counts.
- [`vonmises`](@ref)`(mu, kappa)` — the circular normal on `[-π, π]` for
  directional data (angles, phases, time-of-day).

These join the scalar and multivariate families listed in
[Getting Started](getting-started.md#distributions).

## Gradients and AD selection

Compiled static models get gradients from a tiered strategy, chosen per model
and conditioning signature:

1. **Analytic backend gradients** — models that lower to the vectorized
   backend (most GLM-style models, including the broadcast forms above) use
   hand-derived per-family partials. No AD involved.
2. **Reverse-mode AD (Enzyme)** — models with a generated scorer and many
   parameters use Enzyme reverse-mode, whose cost is independent of parameter
   count. Loading `Enzyme` activates the extension.
3. **Forward-mode AD (ForwardDiff)** — the default fallback; cost grows with
   parameter count.

The batched samplers accept `adtype`:

- `:auto` (default) — analytic when available; otherwise reverse-mode when
  the model supports it **and** has ≥ 24 parameters (below that, forward mode
  wins on constant factors); otherwise forward.
- `:reverse` — prefer reverse-mode whenever the model supports it
  (host-only; cannot be combined with a device `backend`).
- `:forward` — force forward-mode.

```julia
chain = batched_nuts(model, args, constraints; num_chains=8, adtype=:reverse)
```

Analytic backend support always preempts `adtype` — it is faster than either
AD mode. Use `UncertainTea.backend_report(model)` to check which tier a model
reaches.

## Prior predictive checks

Before conditioning, check that the model generates data on the right scale
with [`prior_predictive`](@ref) — it samples the full joint from the prior and
keeps the observation addresses (the supplied constraints only fix the
observed/latent split; their values are unused):

```julia
draws = prior_predictive(model, args, constraints; num_draws=200)
```

It returns the same `PredictiveDraws` container as `predict`, so the same
plotting/summary code applies to prior and posterior predictive draws.

## Next steps

- [Inference Overview](inference.md) — the sampler map.
- [Logistic Regression example](generated/logistic_regression.md) — a GLM end
  to end.
- [Eight Schools example](generated/eight_schools.md) — hierarchical modeling
  with a non-centered reparameterization.

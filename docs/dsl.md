# Minimal DSL Proposal

Date: 2026-03-10

## Design Goal

The DSL should feel recognizably close to official Gen syntax, while still compiling
to a static GPU-friendly execution plan.

The key principle is:

`Gen-like syntax, static GPU semantics`

This document intentionally corrects the earlier draft after checking the official
Gen documentation.

## What "Closer to Gen" Means

The minimal UncertainTea DSL should borrow these ideas directly:

- `@tea` and `@tea (static)` in the role that `@gen` and `@gen (static)` play in Gen
- tilde syntax for random choices
- explicit choice addresses with `{...}`
- hierarchical addresses with `=>`
- external conditioning through `choicemap` and `generate`

It should not center the language around `@observe`.

## Minimal Surface Syntax

### Scalar Model

```julia
@tea (static) function gaussian_mean()
    mu ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(mu, 1.0f0)
    return mu
end

constraints = choicemap((:y, 0.3f0))
(trace, logw) = generate(gaussian_mean, (), constraints)
```

Semantics:

- `mu ~ normal(...)` creates a random choice, binds it to `mu`, and uses `:mu` as the implicit address
- `{:y} ~ normal(...)` creates an explicitly addressed choice without introducing a named local binding
- `generate` conditions the model with externally supplied constraints

### Explicit Address on the Left-Hand Side

```julia
@tea (static) function latent_scale_model()
    mu ~ normal(0.0f0, 1.0f0)
    log_sigma ~ normal(0.0f0, 1.0f0)
    sigma = exp(log_sigma)
    y = ({:y} ~ normal(mu, sigma))
    return (; mu, sigma, y)
end
```

This style stays close to Gen's explicit addressed choice syntax.

### Constraint-Driven Latent/Observation Classification

A random choice is an **observation** iff its address is present in the
constraints supplied at inference time; otherwise it is a **latent** that
receives a dense parameter slot. **Binding is orthogonal to this
classification** -- `x ~ dist` and `y = ({:a} ~ dist)` only name the value for
downstream use; neither makes a choice a latent or an observation on its own.
This is the same conditioning rule `generate`/`assess` already follow, now
shared by every compiled path (`logjoint`, `logjoint_unconstrained`, gradients,
the batched and device paths, and the diagnostics/predictive APIs). See
[docs/constraint-driven-conditioning.md](constraint-driven-conditioning.md) for
the full design.

Two consequences follow, and both were behavior changes from the earlier
syntactic rule (which classified a choice as a latent iff it was *bound*):

- **Constraining a bound address now conditions on it.** In
  `y = ({:y} ~ normal(mu, sigma))`, constraining `:y` makes it an observation
  everywhere. The compiled scoring paths previously ignored such a constraint
  and scored `:y` as a free latent, silently disagreeing with
  `generate`/`assess`; they now agree.
- **An unbound `{:a} ~ dist` left unconstrained is now a latent.** It gets a
  parameter slot and its prior is scored, instead of raising an error demanding
  a constraint. A program that still constrains such a site is unaffected.

Because the latent set depends on what is constrained, **the parameter-vector
length is a function of the conditioning signature** (the set of constrained
addresses), not of the model alone -- exactly as Stan's parameter block depends
on its data block. Once the conditioning is fixed the layout is fully static and
dense. The raw-parameter-vector APIs therefore validate the vector length
against the signature-specific count and, on mismatch, name the conditioning
(which addresses are observed and which latents remain) rather than reporting a
bare count. `observation_addresses(model, args, constraints)` returns exactly
the constrained-and-present choice addresses under this rule.

### Repeated Choices

```julia
@tea (static) function iid_model(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i in 1:n
        {:y => i} ~ normal(mu, 1.0f0)
    end
    return mu
end

constraints = choicemap((:y => i, ys[i]) for i in eachindex(ys))
(trace, logw) = generate(iid_model, (length(ys),), constraints)
```

This is intentionally closer to Gen than the earlier `@plate`-based draft.

In the static GPU path, the loop extent `n` must be compile-time constant or part of
the shape-specialized execution cache.

### Broadcast (Vectorized) Observations

A dot-call on the right-hand side of `~` scores N observations as ONE dense vector
choice instead of N loop-addressed choices — this is the flagship GPU-lowering form:

```julia
@tea (static) function regression(xs)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(slope .* xs, sigma)
end

constraints = choicemap(:y => ys)          # ONE vector value at address (:y,)
(trace, logw) = generate(regression, (xs,), constraints)
```

Semantics:

- The choice has a single address; its value is a `Vector`.
- Each element scores against the broadcast-elementwise arguments. Arguments may be
  real scalars or vectors of the observation's length (scalar-or-N broadcast only);
  a length mismatch throws at scoring time.
- Only `normal.(...)` is currently supported. Dot-calling any other distribution
  family throws an `ArgumentError` at macro-expansion time.
- **Generate requires a length source**: sampling the choice (i.e. running the model
  with the observation unconstrained) needs at least one vector argument to infer
  the sample length. With all-scalar arguments the observation must be constrained,
  otherwise `generate` throws an informative `ArgumentError`.
- Backend lowering emits a native `BackendBroadcastNormalChoicePlanStep`, so
  `backend_report(model).supported == true` and the batched logjoint / manual
  gradient paths score the vector observation densely per batch column.

### `iid` Latent Vectors

`iid(dist_call, n)` declares a latent vector of `n` independent draws from a scalar
distribution under a single address (and a single `n`-wide parameter slot):

```julia
@tea (static) function factors()
    eps ~ iid(normal(0.0f0, 1.0f0), 12)      # 12-wide slot, VectorIdentityTransform
    scales ~ iid(lognormal(0.0f0, 1.0f0), 3) # 3-wide slot, VectorLogTransform
    return eps
end
```

Rules:

- **`n` must be a literal `Int`** — a non-literal count throws an `ArgumentError`
  at macro-expansion time. This holds for latents and observations alike (kept
  literal-only for simplicity).
- The per-element transform follows the base family: `normal`/`laplace`/`studentt`
  use `VectorIdentityTransform(n)`; `lognormal`/`exponential`/`gamma`/
  `inversegamma`/`weibull` use `VectorLogTransform(n)`; `beta` uses
  `VectorLogitTransform(n)`.
- `iid` may also appear as an observation (constrain a length-`n` vector at the
  address; no parameter slot is created for observations).
- `iid` latents currently run through the compiled/AD fallback rather than the
  backend-native batched path; `backend_report` reports them honestly as
  unsupported.

### Nested Calls

```julia
@tea (static) function step(prev)
    z ~ normal(prev, 1.0f0)
    return z
end

@tea (static) function chain_model(T)
    z = ({:z => 1} ~ step(0.0f0))
    for t in 2:T
        z = ({:z => t} ~ step(z))
    end
    return z
end
```

This preserves the "generative function call at an address" flavor from Gen.

## Conditioning Model

The minimal conditioning model should follow Gen's style:

- models declare random choices
- data is supplied through constraints
- the runtime provides `choicemap`, `generate`, `assess`, and replay

`choicemap` accepts single entries (`:y => 0.3`, `(:y, 0.3)`,
`(:y => i, ys[i])`) as separate arguments or one collection of entries (a
vector, tuple, generator, or Dict of pairs). A 2-tuple whose elements are
both `Pair`s is parsed as a collection of two `address => value` entries —
`choicemap((:a => 1.0, :b => 2.0))` yields the entries `(:a,) => 1.0` and
`(:b,) => 2.0`, consistent with the vector and longer-tuple forms (it is
never one entry whose address is the flattened first pair). Passing such a
tuple as ONE entry among several arguments is ambiguous and throws an
`ArgumentError`.

Possible future sugar:

- `condition(model, constraints)`
- helpers for turning named tuples or arrays into structured choicemaps

But the core semantics should stay external, not inline.

## Static Semantics for `@tea (static)`

The initial GPU-targeted subset should require:

- the set of choices is fixed for a compiled execution plan
- each choice has a fixed shape
- each address is statically enumerable
- loop bounds are static or shape-specialized
- no recursion
- no trans-dimensional structure
- no data-dependent creation of new addresses

Allowed:

- arithmetic and deterministic local computation
- simple control flow that does not change the set of choices
- repeated structure with statically enumerable addresses

Rejected or sent to CPU fallback:

- data-dependent growth of the trace
- runtime-generated address structure
- variable-length latent structure
- dynamic control flow that changes which choices exist

### Branchful Control Flow Is Rejected in Static Models

The linear execution plan has no IR for branches, so `@tea static` REJECTS
branchful control flow at macro expansion time with an `ArgumentError`:

- `if` / `elseif` / `else` blocks and the ternary `?:` operator, anywhere in
  the body (including nested inside loops and `begin` blocks)
- short-circuit `&&` / `||` whose subtree contains a `~` choice or an
  assignment (a value-level `a && b` with neither stays accepted)

Silently accepting them would linearize BOTH branches into one plan and
miscompile `logjoint` (while `generate`/`assess` execute the real branch), so
the two APIs could disagree without any error. Supported alternatives:

- `ifelse(cond, a, b)` for deterministic value selection (both sides are
  evaluated; the set of choices does not change)
- moving the data-dependent branch outside the model and passing its result
  in as a model argument
- static `for` loops for repeated structure

Dynamic-mode models (`@tea function ...` without `static`) may contain
`if`/`else`: `generate` and `assess` execute the model body directly and
score the branch that actually runs. The compiled scoring paths (`logjoint`,
gradients, batched and device evaluation) refuse such models with an
`ArgumentError`, and `backend_report` reports them as unsupported, because
the linearized plan cannot represent the branch.

### Duplicate Choice Addresses Are Rejected

Every random choice needs a unique address per execution. Two choices landing
on the same address would silently overwrite the earlier one (and leave the
recorded log-weight inconsistent with the trace), so:

- fully static address collisions (including collisions produced by inlining
  generative submodel calls) throw an `ArgumentError` at model construction
- loop-generated (template-address) collisions throw an `ArgumentError` when
  the model executes (`generate`/`assess`)

User-side `ChoiceMap` updates keep their update semantics: constructing
`choicemap((:y, 0.1), (:y, 0.2))` still yields one entry with the later
value; only in-model duplicate recording during a single execution errors.

### Default Argument Values

`@tea` model arguments may declare default values, and the defaults work
uniformly across the API: `generate`/`assess` (which execute the model body)
and the compiled scoring entry points (`logjoint`,
`logjoint_unconstrained`, gradients, the batched and device paths,
`pointwise_loglikelihood`, ...) all fill missing TRAILING arguments from the
declared defaults, using Julia's own default-argument semantics (a default
may reference earlier arguments):

```julia
@tea static function scaled(mu = 2.0)
    x ~ normal(mu, 1.0)
end

(trace, _) = generate(scaled)          # samples with mu = 2.0
logjoint(scaled, [trace[:x]])          # scores with mu = 2.0 (same density)
logjoint(scaled, [trace[:x]], (2.0,))  # identical
```

Passing fewer arguments than the non-defaulted prefix requires (or more than
the full signature) throws a `DimensionMismatch`.

## Address Rules

The initial static mode should accept:

- implicit symbol addresses from `x ~ dist(...)`
- explicit addresses like `{:x}`
- hierarchical addresses like `{:layer => 1 => :weight}`
- repeated addresses generated from static loops, such as `{:y => i}`
- tuple addresses such as `{(:y, i)}`, which flatten to the same normalized
  parts as `{:y => i}`

The address language should lower into a normalized `AddressSpec` rather than a
runtime dictionary key scheme.

## Lowering Model

The examples above should lower into something conceptually like:

```julia
ModelSpec(
    choices = [
        ChoiceSpec(:mu, Normal(), shape=(), constraint=Identity()),
        ChoiceSpec(:y, Normal(), shape=(), constraint=Identity())
    ],
    layout = ParameterLayout(offsets=(mu=1,)),
    return_plan = ...
)
```

The backend should then provide:

- `initial_trace(spec, rng, batch_shape)`
- `generate(spec_or_model, args, constraints)`
- `assess(spec_or_model, args, constraints)`
- `logjoint(spec, unconstrained_params, constraints, args)`
- `replay(spec, unconstrained_params, args)`

`assess` requires a complete choicemap covering every choice in the model and
returns the full log density; it throws if any choice is missing.

## Distribution Interface

The initial GPU-targeted distribution set should stay small:

- `normal`
- `lognormal`
- `laplace`
- `exponential`
- `gamma`
- `inversegamma`
- `weibull`
- `beta`
- `dirichlet`
- `bernoulli`
- `binomial`
- `betabinomial`
- `multinomial`
- `discreteuniform`
- `geometric`
- `negativebinomial`
- `poisson`
- `studentt`
- `cauchy`
- `halfnormal`
- `halfcauchy`
- `uniform`
- `logistic`
- `gumbel`
- `frechet`
- `rayleigh`
- `inversegaussian`
- `categorical`
- `truncatednormal`
- `truncatedstudentt`
- `mixture`
- a restricted diagonal `mvnormal`
- `mvnormaldense` (dense covariance via a Cholesky factor)
- `mvstudentt` / `mvstudenttdense` (heavy-tailed multivariate Student-t)
- `lkjcholesky` (LKJ prior over correlation Cholesky factors)
- `wishart` / `inversewishart` (covariance-matrix priors, CPU-reference)
- simple transformed distributions

Requirements:

- backend-compatible `logpdf`
- clear batch semantics
- a well-defined constrained/unconstrained transform when needed
- `binomial` is treated as a built-in distribution inside `@tea`, but direct
  constructor calls outside the DSL should use `UncertainTea.binomial(...)`
  because `Base` already defines a different `binomial`
- `dirichlet` now supports static simplex sizes in both the CPU reference path
  and the current backend-native static subset, with unconstrained HMC/NUTS
  flowing through a simplex transform
- `mvnormal` currently supports static diagonal vector sizes in the CPU
  reference path and unconstrained HMC/NUTS through a vector-valued identity
  transform, and restricted diagonal forms now lower to the backend-native
  subset as the first vector-valued built-in family
- `mvnormaldense(mu, scale_tril)` is the dense-covariance multivariate normal in
  Cholesky parameterization: `scale_tril` is a d×d lower-triangular factor `L`
  with strictly positive diagonal and covariance `L * L'`. Only the lower
  triangle of `scale_tril` is read — any upper-triangular content is ignored —
  so a full runtime matrix (a model argument or deterministic binding) works
  without wrapping. It is backend-native (`BackendMvNormalDenseChoicePlanStep`
  with hand-derived batched analytic gradients, #57) and device-lowered
  (`DeviceMvNormalDenseChoiceStep`, whose forward-substitution unroll caps the
  device dimension at 8) when `scale_tril` is a model argument or a captured
  matrix; an inline literal factor is honestly reported unsupported by
  `backend_report` and takes the ForwardDiff fallback. As a
  **latent** (parameter slot sampled by HMC/NUTS) the mean must have a
  statically known length (vector literal/tuple), mirroring the diagonal
  `mvnormal` rule; with a non-static mean it is observation-only (no slot).
- `mvstudentt(nu, mu, sigma)` and `mvstudenttdense(nu, mu, scale_tril)` are the
  heavy-tailed analogues of `mvnormal` / `mvnormaldense`: a multivariate
  Student-t with `nu` degrees of freedom whose `d` dimensions couple through one
  chi-square mixing variable (a single quadratic form, not a product of
  univariate t's). The diagonal form takes a scale **vector** `sigma`; the dense
  form takes a lower-triangular Cholesky **factor** `scale_tril` (scale matrix
  `L * L'`), reading only the lower triangle. `nu` may flow from a latent (its
  gradient uses the digamma degrees-of-freedom channel). Both are backend-native
  and device-lowered where `mvnormal` / `mvnormaldense` are (the device dense
  path caps `d` at 8, riding the same forward-substitution unroll). As latents
  the mean must have a statically known length, mirroring the Gaussian rule.
- `lkjcholesky(d, eta)` is the LKJ prior over the Cholesky factor of a `d`×`d`
  correlation matrix, scored on the column-major **packed** lower triangle
  (length `d*(d+1)/2`, diagonal included) so the value stays a flat vector. The
  dimension `d` must be a literal integer `>= 2` — for latents and observations
  alike; a non-literal `d` raises an `ArgumentError` at macro-expansion time.
  `eta` may be any expression. Latents flow through `CholeskyCorrTransform`
  (Stan's canonical partial correlation parameterization, `d*(d-1)/2`
  unconstrained coordinates), so HMC/NUTS explores exactly the below-diagonal
  free coordinates. The family is backend-native
  (`BackendLKJCholeskyChoicePlanStep` with hand-derived batched analytic
  gradients, #57) and device-lowered (`DeviceLKJCholeskyChoiceStep`), scoring the
  packed factor through the `CholeskyCorrTransform` on both paths.
  The `scale_cholesky(scales, packed_corr_chol)` helper un-packs the factor and
  scales row `i` by `scales[i]`, producing the dense lower-triangular
  `scale_tril` for `mvnormaldense`; it is a plain function usable as a
  deterministic binding inside `@tea`. A hierarchical covariance prior then
  reads:

  ```julia
  @tea static function hierarchical_cov_model(zero_mean, n)
      Omega ~ lkjcholesky(3, 2.0)              # packed correlation Cholesky
      tau ~ iid(lognormal(0.0f0, 0.3f0), 3)    # per-dimension scales
      Ltril = scale_cholesky(tau, Omega)       # dense scale_tril = diag(tau) * L
      for i in 1:n
          {:y => i} ~ mvnormaldense(zero_mean, Ltril)
      end
      return Omega
  end
  ```
- `gaussianprocess(inputs, lengthscale, variance, noise)` is a zero-mean Gaussian
  process regression likelihood with a squared-exponential (RBF) kernel. `inputs`
  is a `D×N` matrix (one column per point) or a length-`N` vector for 1-D inputs;
  the positive scalars `lengthscale`/`variance`/`noise` are typically `exp` of
  latent log-hyperparameters. As an observation `{:y} ~ gaussianprocess(X, l, v,
  n)` scores the length-`N` output vector under `N(0, K)` with `K[i,j] = v² ·
  exp(-½ Σ_d (x_{d,i} − x_{d,j})² / l_d²) + n² δᵢⱼ`, rebuilding the kernel Cholesky
  each call so the hyperparameter gradient flows through it by ForwardDiff. It is
  CPU-reference only (the dense `O(N³)` Cholesky is not device-lowered) and expects
  centred outputs (the prior mean is zero). `lengthscale` is either a positive
  **scalar** (isotropic) or a length-`D` **vector** for **Automatic Relevance
  Determination** (one lengthscale per input dimension, so an uninformative
  dimension is pruned by a large `l_d`); the `D + 2` hyperparameter gradient of the
  ARD marginal likelihood is a natural `reverse_mode_gradient` target (issue #268)
  when `D` is large. Example:

  ```julia
  @tea static function gp_regression(X, n)
      logl ~ normal(0.0, 1.0)
      logv ~ normal(0.0, 1.0)
      logn ~ normal(-1.0, 1.0)
      {:y} ~ gaussianprocess(X, exp(logl), exp(logv), exp(logn))
      return logl
  end

  # ARD: a per-dimension lengthscale vector (here supplied as a model argument)
  ard_gp(X, lengthscales, n) = gaussianprocess(X, lengthscales, 1.0, 0.2)
  ```

  `gaussianprocess` marginalizes the latent function analytically, so it only fits
  a **Gaussian** likelihood. For **direct latent-function inference** — GP
  classification, count regression, or any non-Gaussian likelihood —
  `gp_cholesky(inputs, lengthscale, variance, noise)` returns the kernel Cholesky
  factor `L` (`K = L L'`) as a deterministic binding (mirroring `scale_cholesky`),
  which feeds the GP prior into `mvnormaldense` so the function values
  `f ~ N(0, K)` are sampled directly. `noise` here is the diagonal jitter/nugget;
  the observation noise lives in the likelihood. Because `f` is a **latent**
  `mvnormaldense`, its zero mean must be a static-length literal/tuple (the same
  rule as any dense-normal latent), which also fixes the number of function values:

  ```julia
  @tea static function gp_classification(X)
      logl ~ normal(0.0, 1.0)
      L = gp_cholesky(X, exp(logl), 1.0, 1e-6)   # kernel Cholesky, deterministic binding
      f ~ mvnormaldense((0.0, 0.0, 0.0, 0.0), L) # latent GP function values (N = 4)
      for i in 1:4
          {:y => i} ~ bernoullilogit(f[i])       # non-Gaussian likelihood
      end
      return logl
  end
  ```
- `sparsegaussianprocess(inputs, inducing, lengthscale, variance, noise)` is the
  **sparse** GP regression likelihood via the FITC approximation, for when the
  dense `O(N³)` Cholesky is too expensive. `inducing` is a `D×M` matrix (or
  length-`M` vector) of `M ≪ N` inducing points; the marginal replaces the full
  kernel `K_NN` with the rank-`M` Nyström approximation `Q_NN = K_NM K_MM⁻¹ K_MN`
  plus the exact diagonal correction `diag(K_NN − Q_NN)`, and scores `N(0, Q_NN +
  diag(K_NN − Q_NN) + noise² I)` in `O(N M² + M³)` via the Woodbury identity.
  `lengthscale` may be scalar or an ARD vector, as with `gaussianprocess`. With
  `inducing = inputs` it reduces to the dense `gaussianprocess`. CPU-reference
  only; expects centred outputs. Measured ~9× faster per `logpdf` than the dense
  GP at `N = 200, M = 15`, with the gap widening as `N` grows.

  ```julia
  @tea static function sparse_gp(X, Z)
      logl ~ normal(0.0, 1.0)
      logn ~ normal(-1.0, 1.0)
      {:y} ~ sparsegaussianprocess(X, Z, exp(logl), 1.0, exp(logn))
      return logl
  end
  ```
- `hmm(init, transition, means, sigma)` is a hidden Markov model with Gaussian
  emissions. `init` is the length-`K` initial-state distribution and `transition`
  the `K×K` row-stochastic transition matrix (fixed dynamics, supplied as model
  arguments); `means` are the `K` per-state emission means and `sigma` the shared
  emission standard deviation, typically latent. As an observation `{:y} ~
  hmm(init, transition, means, sigma)` scores the length-`T` observation sequence
  through the **forward algorithm**, marginalizing the hidden state path in
  `O(T·K²)` via the log-space `α` recursion (the marginal Stan hand-writes with
  `log_sum_exp`). This loop-carried Markov chain is exactly the case that needs the
  forward algorithm rather than per-timestep `marginalize=:enumerate`, whose `Kᵀ`
  path enumeration is intractable. The recursion is differentiable, so HMC/NUTS
  infers the emission parameters; give `means` an identifiable (e.g. ordered)
  parameterization so the marginal posterior is well-defined. It is CPU-reference
  only (the loop-carried recursion is not device-lowered). Latent
  transition/initial simplexes and non-Gaussian emissions are follow-ups. Example:

  ```julia
  @tea static function hmm_regime_model(init, transition, seqlen)
      m1 ~ normal(-1.0, 2.0)
      log_gap ~ normal(0.0, 1.0)                 # ordered means for identifiability
      logs ~ normal(-0.5, 0.5)
      {:y} ~ hmm(init, transition, [m1, m1 + exp(log_gap)], exp(logs))
      return m1
  end
  ```
- `wishart(nu, S)` and `inversewishart(nu, S)` are the classical conjugate
  covariance-matrix priors (`nu` degrees of freedom, `d`×`d` scale matrix `S`,
  requiring `nu > d - 1`). The value is the column-major **packed** lower
  Cholesky factor `L` of the sampled PD matrix `M = L * L'` (length `d*(d+1)/2`),
  the same flat-vector convention as `lkjcholesky`; `logpdf` is the induced
  density over `L` (including the `L -> L*L'` Jacobian), and `rand` uses the
  Bartlett decomposition. Latents unconstrain through the new log-Cholesky
  `CholeskyCovTransform` (`exp` on the Cholesky diagonal, identity below it), so
  HMC/NUTS explores an unconstrained `d*(d+1)/2`-vector. As latents these
  families need a **static square scale-matrix literal** `S` (its side length is
  the value dimension — there is no vector argument to read it from); a
  non-literal `S` makes the choice observation-only. They are **CPU-reference
  only**: `backend_report` honestly reports them unsupported and the device
  rejects them, so batched calls take the ForwardDiff fallback (backend/device
  lowering is a deliberate follow-up). `lkjcholesky` + per-dimension scale priors
  remains the recommended default covariance parameterization; the Wishart pair
  is for interop, conjugacy, and matching published analyses.
- `betabinomial(n, alpha, beta)` is the overdispersed-binomial likelihood (a
  binomial whose success probability is `Beta(alpha, beta)`), with log-pmf
  `logC(n,k) + logB(k+alpha, n-k+beta) - logB(alpha, beta)`. `n` is an integer
  trial count; `alpha`/`beta` may flow from latents (their gradients are
  closed-form digamma differences). As an **observation** it lowers to the
  backend-native batched path with analytic gradients
  (`backend_report(model).supported == true`). Device lowering is honestly
  rejected (it would reuse the Float64 binomial-coefficient helper of #218), so
  it is CPU/backend-only there.
- `multinomial(n, p)` is the compositional-count likelihood over a length-`K`
  simplex `p` (the natural observation partner of the `dirichlet` prior), with
  log-pmf `logGamma(n+1) - sum logGamma(x_i+1) + sum x_i*log(p_i)`. As a vector
  observation it is CPU-reference only (honestly reported unsupported by
  `backend_report`; the batched path uses the ForwardDiff fallback, so
  dirichlet-multinomial models still sample correctly). Backend/device lowering
  of the vector observation, and multinomial **latents**, are out of scope
  (issue #67 / #231).
- `discreteuniform(a, b)` is uniform over the integers `a..b` (inclusive), with
  density `-log(b - a + 1)`; `a`/`b` are integer bounds and the family has no
  continuous parameters (zero gradient). As an **observation** it lowers to the
  backend-native batched path; device lowering is honestly rejected. Enumerated
  discrete-**latent** support (`marginalize=:enumerate` / Gibbs) is not yet
  wired for this family.
- `truncatednormal(mu, sigma, lower, upper)` and
  `truncatedstudentt(nu, mu, sigma, lower, upper)` renormalize the base density
  over `[lower, upper]` (infinite bounds are allowed on either side). As
  **observations** the bounds may be any expression (model arguments,
  deterministic bindings, etc.), and both families lower to the backend-native
  batched path with analytic gradients (`backend_report(model).supported ==
  true`) — with one restriction for `truncatedstudentt`: the normalizer uses the
  regularized incomplete beta, whose `nu`-derivative has no closed form, so the
  backend-native (and ForwardDiff) gradient is only available when `nu` is a
  **constant** (a literal degrees-of-freedom). A latent- or argument-flowing `nu`
  is honestly reported unsupported and runs through the compiled CPU logjoint
  fallback (value only; its gradient is intractable). As **latents** (parameter
  slots sampled by HMC/NUTS) both truncated families draw through a bounded
  parameter transform not implemented in the batched backend, so they fall back
  to the ForwardDiff column path; both bounds must be literal statics — a `Number`
  or `Inf`/`-Inf` — so the unconstraining transform is fixed at model build time:
  both finite uses a scaled-logit `BoundedTransform`, a single finite bound uses
  `LowerBoundedTransform`/`UpperBoundedTransform`, and two infinite bounds degrade
  to `IdentityTransform`. Declaring a truncated latent with a dynamic (non-literal)
  bound raises an `ArgumentError` at macro-expansion time.
- `mixture(weights, components...)` marginalizes a finite mixture with
  `logpdf(mix, x) = logsumexp_k(log(w_k) + logpdf(component_k, x))`. The `weights`
  argument may be a literal tuple/vector or any runtime expression — including a
  latent simplex supplied by a `dirichlet` slot — and is validated (nonnegative,
  summing to 1 within `1e-8`, one per component). Components are inline
  distribution constructor calls with fixed families. Mixtures are CPU-reference
  only (honestly reported unsupported by `backend_report`, but they still run
  through the compiled CPU logjoint and the batched ForwardDiff fallback). As
  **observations** the components may be any families. As **latents** (parameter
  slots sampled by HMC/NUTS) every component must be a real-line location-scale
  family (`normal`, `laplace`, `studentt`) so an `IdentityTransform` is exact;
  declaring a latent mixture with any other component family raises an
  `ArgumentError` at macro-expansion time.
- The scalar prior families `cauchy(mu, sigma)`, `halfnormal(sigma)`,
  `halfcauchy(scale)`, `uniform(a, b)`, `logistic(mu, s)`, and `gumbel(mu, beta)`
  are first-class elementary densities with full CPU-reference, backend-native
  (analytic batched gradient), and device (KernelAbstractions) support — except
  `uniform`, which is CPU + backend only (its bounded transform is not a device
  transform, so a `uniform` choice is honestly reported unsupported by
  `device_report`). `cauchy`/`logistic`/`gumbel` are real-line
  (`IdentityTransform`); `halfnormal`/`halfcauchy` are positive
  (`LogTransform`, like `lognormal`). `uniform` latents unconstrain through a
  scaled-logit `BoundedTransform` and therefore require **literal (static) finite
  bounds** (the same restriction the truncated families use); a dynamic bound is
  allowed only for an observation and otherwise raises an `ArgumentError` at
  macro-expansion time.
- The positive-support / heavy-tail families `pareto(x_m, alpha)`,
  `frechet(shape, scale)`, `rayleigh(scale)`, and `inversegaussian(mu, lambda)`
  (Wald) are first-class elementary densities. `frechet`/`rayleigh`/
  `inversegaussian` are positive (`LogTransform`, like `lognormal`) with full
  CPU-reference, backend-native (analytic batched gradient), and device
  (KernelAbstractions) support. `pareto` has support `x >= x_m`, so a latent
  unconstrains through a `LowerBoundedTransform` and therefore requires a
  **literal (static) positive lower bound** `x_m` (the same static-bounds rule the
  truncated/`uniform` families use); as with `uniform`, that transform is not a
  batched/device transform, so a latent `pareto` honestly falls back to the
  ForwardDiff column path and `device_report` reports it unsupported, while a
  `pareto` observation (with an arbitrary, possibly dynamic `x_m`) is
  backend-native.

## Non-Centered Reparameterization

Hierarchical location-scale latents can opt into the non-centered
parameterization without changing how the model reads:

```julia
@tea (static) function eight_schools()
    mu ~ normal(0.0, 5.0)
    log_tau ~ normal(0.0, 1.5)
    tau = exp(log_tau)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i in 1:8
        {:y => i} ~ normal(theta[i], 10.0)
    end
    return mu
end
```

- The NUTS/HMC parameter slot holds the standardized `z`; traces, choicemaps,
  constrained samples, and summaries keep `theta` at its own address.
- `generate`/`assess` semantics are unchanged (the runtime path stays
  centered); only the sampler geometry changes -- on funnel-shaped posteriors
  this removes the divergences that defeat the centered form.
- Eligible: `normal`, `studentt`, `laplace` (real line), `lognormal`
  (log-space affine), directly or as the base of an `iid` vector latent
  (real-line bases only). The flag must sit on the top-level call of `~`.
- `reparam=:auto` picks `:noncentered` whenever the location or scale is a
  non-literal expression, `:centered` otherwise.
- Scalar noncentered `normal` runs on the backend-native batched and device
  paths; the other variants use the CPU reference path and are reported
  honestly by `backend_report`/`device_lowering_report`.
- See docs/noncentered-reparam.md for the staged design.

## Marginalized Discrete Latents

Finite-support discrete latents can opt into automatic enumeration so
gradient-based samplers work without user-side marginalization:

```julia
@tea (static) function indicator_mixture()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
    return m1
end
```

- `logjoint`/`logjoint_unconstrained` (and everything built on them: HMC/NUTS,
  gradients, the batched per-column path) score the MARGINAL over `z`'s
  support via a logsumexp over the remaining model, so the samplers see only
  the continuous parameters. Providing `z` in the constraints scores the
  plain joint instead (conditioning stays free).
- `generate` still forward-samples `z` into the trace, and `assess` still
  scores the full joint of the choices it is given (it requires `z` like any
  unflagged slotless choice).
- Eligible: `bernoulli` (support `false/true`) and `categorical` with a
  literal probability vector (the support size must be known at macro time;
  the probability values may be expressions, including other latents). The
  flag must sit on the top-level call of `~`, outside loops.
- Each marginalized site multiplies the cost of the model suffix after it by
  its support size (nested sites multiply), so put enumerated latents as late
  in the model as dependencies allow.
- The batched path is backend-native: the flag lowers to a suffix-owning
  backend step (conditioning stays free per column, and nested support
  products beyond 32 are rejected honestly), and batched gradients use the
  analytic responsibility-weighted branch gradients. The device path reports
  the step as unsupported.
- See docs/discrete-enumeration.md for the staged design.

## Discrete Latents via MH-within-Gibbs

Unbounded or large-support discrete latents (poisson, geometric,
negativebinomial, binomial) cannot be enumerated; the `gibbs` sampler
alternates symmetric single-site Metropolis-Hastings updates on them with
NUTS transitions on the continuous block:

```julia
@tea (static) function count_model()
    rate_log ~ normal(1.0, 0.5)
    z ~ poisson(exp(rate_log))
    {:y} ~ normal(1.0 * z, 0.8)
    return z
end

chain = gibbs(count_model, (), choicemap((:y, 6.2)); num_samples=2000, num_warmup=500)
chain.discrete_samples      # sites x samples, addresses in chain.discrete_addresses
discrete_ess(chain)         # split-chain ESS per discrete site
```

- Sites are discovered automatically: every non-observed choice without a
  parameter slot that is not `marginalize=:enumerate`. Bernoulli sites flip,
  literal-probability categoricals propose uniformly over the other
  categories, and the count families take a ±1 integer walk
  (`discrete_tail=p` mixes in a geometric-tailed step for large-count
  posteriors). A model without discrete sites reduces to plain NUTS; one
  without continuous slots to pure single-site MH.
- Models where the SET of latent choices depends on a sampled value
  (latent-bound loops or addresses) are trans-dimensional and rejected at
  construction; marginalize the controlling site instead.
- Difficult initializations retry the prior and accept explicit seeds
  (`initial_discrete`, `initial_params`); NUTS keyword arguments are
  mirrored from [`nuts`](@ref).
- `sbc(model; sampler=:gibbs, observation_addresses=[...], ...)` runs the
  calibration harness with the Gibbs kernel; the observed addresses must be
  named explicitly so the discrete sites stay latent (the default would
  condition every non-slot choice).
- See docs/mh-within-gibbs.md for the design.

## User-Defined Distributions

A distribution defined outside the package participates in `@tea` models on
the CPU reference path once registered with `register_distribution`. The
family stays honestly unsupported in `backend_report`/`device_report` (same
tier as the built-in CPU-only families), so scoring and gradients use the
compiled CPU logjoint and the ForwardDiff column fallback.

The builder must return a subtype of `AbstractTeaDistribution` implementing
`UncertainTea.logpdf(dist, x)` (return `-Inf` outside the support, and keep it
ForwardDiff-Dual-friendly if the family will be a latent) and
`Random.rand(rng::AbstractRNG, dist)`:

```julia
using UncertainTea, Random

struct SkewNormalDist{T<:Real} <: AbstractTeaDistribution
    location::T
    scale::T
    shape::T
    function SkewNormalDist(location::T, scale::T, shape::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("skewnormal requires scale > 0"))
        return new{T}(location, scale, shape)
    end
end

skewnormal(location, scale, shape) = SkewNormalDist(promote(location, scale, shape)...)

function UncertainTea.logpdf(dist::SkewNormalDist, x)
    z = (x - dist.location) / dist.scale
    return log(2) - z^2 / 2 - log(2 * pi) / 2 - log(dist.scale) +
           log(UncertainTea._std_normal_cdf(dist.shape * z))
end

function Random.rand(rng::AbstractRNG, dist::SkewNormalDist)
    delta = dist.shape / sqrt(1 + dist.shape^2)
    u0, v = randn(rng), randn(rng)
    return dist.location + dist.scale * (delta * abs(u0) + sqrt(1 - delta^2) * v)
end

register_distribution(:skewnormal; builder=skewnormal, transform=IdentityTransform())

@tea static function model()
    x ~ skewnormal(0.0, 1.0, 3.0)
    {:y} ~ normal(x, 0.5)
    return x
end
```

`transform` declares the unconstrained parameterization for latent use --
`IdentityTransform()` (real line), `LogTransform()` (positive),
`LogitTransform()` ((0,1)), or `BoundedTransform(lower, upper)`. Omit it for
observation-only families; a latent then gets no parameter slot.

Rules and caveats:

- Register **before** defining models that use the family (registration is
  consulted at model definition).
- Models capture the builder and transform at definition, so re-registering a
  family affects only models defined afterwards.
- Built-in family names and expression primitives (`exp`, `log`, ...) cannot
  be registered.
- Broadcast observations (`family.(...)`) and `iid(family(...), n)` are not
  supported for registered families.

## Inference-Oriented Consequences

### VI / SVI

This syntax lowers cleanly to a fixed latent representation and batch-heavy likelihoods.
The current CPU reference runtime now exposes `batched_advi` on top of the
same static unconstrained parameter layout, and the current backend-native
vector subset (`mvnormal` diagonal, `dirichlet`) flows through that path.

### SMC

Particle axes map naturally onto the same compiled model structure.
The current reference implementation exposes `batched_importance_sampling` and
`batched_sir`, and `batched_smc` now denotes an adaptive tempered multi-stage
SMC bridge from a Gaussian proposal to the target density, with optional
batched rejuvenation moves after resampling stages. The current move kernels
are `:random_walk`, fixed-step tempered `:hmc`, and CPU-reference tempered
`:nuts`.

### HMC

Feasible after the static path exists, but batched `HMC` is a better first target than `NUTS`.

## Recommended First Milestone

The first real subset should include:

1. `@tea (static)`
2. tilde syntax
3. implicit and explicit addresses
4. hierarchical addresses with `=>`
5. `choicemap`
6. `generate`
7. a small distribution set starting with `normal`, `lognormal`,
   `laplace`, `exponential`, `gamma`, `inversegamma`, `weibull`, `beta`,
   `dirichlet`, `bernoulli`, `binomial`, `betabinomial`, `multinomial`,
   `discreteuniform`, `geometric`, `negativebinomial`, `poisson`, `studentt`,
   and `categorical`

That is enough to build a CPU reference backend and a GPU-oriented static lowering path
without committing to full Gen compatibility.

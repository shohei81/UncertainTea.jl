# Changelog

## Unreleased

### Added

- **`to_arviz_dict` coord-dim arrays for vector variables** (#366): the new
  `flatten_vectors::Bool=true` keyword (both methods) keeps the flattened
  per-component `"v[1]"` keys by default; with `flatten_vectors=false` each
  vector parameter — and each indexed observation family in the
  `"log_likelihood"` group — is emitted as a single array with a trailing
  component dimension (`(draw, chain, k)` for `:draw_chain`,
  `(chain, draw, k)` for `:chain_draw`, matching Python `az.from_dict`'s
  `(chain, draw, *shape)`), and the returned dictionary gains top-level
  `"coords"` / `"dims"` entries following the ArviZ `from_dict` convention
  (e.g. `dims["theta"] == ["theta_dim_0"]`,
  `coords["theta_dim_0"] == collect(1:k)`).

## v0.2.0 (2026-08-08)

### Breaking

- **Namespaced public API** (#329): the flat export surface is reorganized
  into a minimal modeling-language top level plus three facade submodules.
  `using UncertainTea` now brings in only the `@tea` DSL, the trace/choicemap
  types, `generate`/`assess`/`logjoint`, and the distribution constructors;
  everything else moved to `UncertainTea.Inference` (samplers, fitters, result
  types/accessors, density and parameter machinery),
  `UncertainTea.Diagnostics` (chain summaries, convergence checks, draw
  export, model comparison, predictive checks), and `UncertainTea.Device`
  (the KernelAbstractions device density APIs). Every name is still defined in
  (and reachable qualified from) the parent module — `UncertainTea.nuts`
  keeps working — only unqualified use after `using UncertainTea` changes.

  Migration: add the submodule `using`s next to `using UncertainTea`.

  | Before (unqualified via `using UncertainTea`) | Now import with |
  | --- | --- |
  | `nuts`, `hmc_chains`, `batched_nuts`, `batched_advi`, `pathfinder`, `gibbs`, `sbc`, `map_estimate`, … and their result types; `logjoint_unconstrained`, `batched_logjoint*`, `parameter_vector`, `transform_to_*`, `reverse_mode_gradient`, … | `using UncertainTea.Inference` |
  | `summarize`, `rhat`, `ess`, `check_diagnostics`, `posterior_array`, `to_arviz_dict`, `to_mcmcchains`, `waic`, `psis_loo`, `loo`, `predict`, `prior_predictive`, … | `using UncertainTea.Diagnostics` |
  | `device_batched_logjoint*`, `device_lowering_report`, `DeviceBatchedWorkspace`, `DeviceExecutionPlan`, `DeviceHMCWorkspace` | `using UncertainTea.Device` |
  | `@tea`, `choicemap`, `generate`, `assess`, `logjoint`, distribution constructors (`normal`, …), `register_distribution`, … | unchanged — `using UncertainTea` |

- **Standardized sampler keyword arguments** (#338): the same concept now has
  one name everywhere. No deprecation shims — old spellings raise a keyword
  argument error.

  | Function | Old kwarg | New kwarg |
  | --- | --- | --- |
  | `pathfinder` | `init` | `initial_params` |
  | `map_estimate` (and via `laplace_approximation` forwarding) | `init` | `initial_params` |
  | `elliptical_slice` | `initial` | `initial_params` |
  | `batched_nuts`, `batched_chees`, `batched_meads`, `batched_svgd` | `init` (`:prior`/`:uniform`) | `init_strategy` |
  | `sbc` (kwarg and `SBCResult` field) | `num_posterior_draws` | `num_samples` |
  | `batched_advi` | `num_steps` | `num_iterations` |
  | `nuts`, `nuts_chains`, `batched_nuts`, `gibbs` | `max_delta_energy` | `divergence_threshold` |

  `initial_params` is now the initial point everywhere, `init_strategy` the
  initialization-strategy Symbol on the batched samplers, `num_iterations` the
  optimizer iteration count (matching `batched_svgd`), and
  `divergence_threshold` the divergence cutoff (matching `hmc`/`batched_hmc`/
  `batched_chees`/`batched_meads`). `num_draws` (predict/prior_predictive/
  pathfinder) and `num_particles` (ADVI MC samples per step; SVGD particles)
  are unchanged. The single-chain `nuts`/`hmc`, `pathfinder`, and
  `map_estimate` docstrings now state that gradients always use ForwardDiff
  (`adtype` selection exists only on the batched samplers).

- **`nuts`/`hmc` return a one-chain `HMCChains`** (#337): the single-chain
  samplers no longer return the bare per-chain `HMCChain`, so their results
  compose with everything that already accepted multi-chain output —
  `summarize`, `rhat`, `ess` (now defined for a single chain via its split
  halves), `posterior_array`, `to_arviz_dict`, `to_mcmcchains`, `predict`, and
  `loo`. The per-chain record is `first(result)` / `result[1]`; `HMCChain`
  itself is unchanged. `MAPResult` gained leading `model`/`args`/`constraints`
  fields (positional constructors change), and the four-argument
  `pointwise_loglikelihood`/`waic`/`psis_loo`/`loo` forms are now typed on the
  posterior-draws interface (plus a raw constrained-draws matrix method), so a
  wrong argument fails with a `MethodError` instead of a field error.

### Added

- **Runtime-length vector latents** (#289):
  `theta ~ mvnormal(zeros(n), ones(n))` with a model argument `n` now works on
  every CPU path (`logjoint`, `logjoint_unconstrained`, gradients, `nuts` and
  the other samplers, `generate`/`assess`) — the latent's dimension is
  resolved from the model arguments at signature-resolution time and cached
  per `(signature, dims)`, so one session can score/sample the same model at
  several `n`. All five argument-sized families are covered: `mvnormal`,
  `mvnormaldense`, `mvstudentt`, `mvstudenttdense` (identity vector
  transform), and `dirichlet` (`SimplexTransform(n)`, parameter dimension
  `n-1`). In particular GP latent-function models no longer need a
  literal-length mean: `f ~ mvnormaldense(zeros(n), gp_cholesky(X, kernel,
  jitter))` runs at any data size. The length must be computable from the
  arguments alone (a latent-dependent length is rejected with an informative
  error); mixed-`n` per-column argument batches are rejected; the
  args-independent layout APIs (`parameterlayout(model)`, 3-argument
  `transform_to_*`) throw an informative error for these models.
  `wishart`/`inversewishart`/`lkjcholesky` stay literal-dimension by design
  (macro-time errors, unchanged). Posture (final): batched scoring
  deliberately falls back to per-column ForwardDiff — no whole-vector backend
  lowering — and the device path stays unsupported;
  `backend_report`/`device_lowering_report` now name the runtime-length
  choice as the one reason instead of the generic per-argument lowering
  errors, and the `iid` runtime-count error points at the runtime-length
  `mvnormal` spelling for i.i.d. normal vectors. Documented in
  docs/src/modeling.md ("Runtime-length vector latents").
- **Distributions.jl adapter** (#340): the new `UncertainTeaDistributionsExt`
  package extension (activated by loading Distributions.jl) makes any
  Distributions.jl univariate distribution usable inside `@tea` models with
  one line — `register_distribution(:skewnormal, Distributions.SkewNormal)`.
  The latent-space transform is read off `Distributions.support` when the
  support is a fixed property of the type (real / positive / unit-interval /
  bounded), with an explicit `support` keyword for parameter-dependent
  supports and observation-only use; discrete families register
  observation-only. The underlying wrapper is exported as
  `wrap_distribution`. Univariate only; runs on the CPU reference path like
  every registered family.
- **Tables.jl interface** (#340): `HMCChains` and `PredictiveDraws` are now
  Tables.jl sources via the `UncertainTeaTablesExt` extension (activated by
  loading Tables.jl, which DataFrames.jl does automatically), so
  `DataFrame(chains)` works: wide layout with `chain`/`draw` columns plus one
  column per parameter (`PredictiveDraws`: `draw` plus one column per
  predictive address).
- **"Working with results" docs** (#340): the inference docs now cover the
  plotting route (`to_mcmcchains` → MCMCChains/StatsPlots recipes), tabular
  access, and the JLD2 persistence pattern (save
  `posterior_array`/`parameter_names`/`to_arviz_dict` output, which reloads
  in a fresh session without the model closures that result structs embed).
- **Posterior-draws interface** (#337): `constrained_draws(result; num_draws,
  rng) -> (draws, names)` — a `num_params x num_draws` constrained-space draw
  matrix plus display names — implemented by every inference result:
  `HMCChains`, `GibbsChain`, `ADVIResult`, `PathfinderResult` (its previously
  missing constrained accessor: the stored unconstrained draws mapped through
  the model transform), `SVGDResult`, `ImportanceSamplingResult` / `SMCResult`
  / `NestedSamplingResult` (weight-resampled), `SIRResult`,
  `EllipticalSliceResult`, `LaplaceResult` (Gaussian draws at the mode), and
  `MAPResult` (the mode as one draw). Exported from
  `UncertainTea.Diagnostics` together with the `PosteriorDrawsResult` union;
  `predict` and `loo`/`psis_loo`/`waic` route through it and therefore accept
  every result type above.
- **`GibbsChain` diagnostics** (#337): `summarize`, `rhat`, `ess`,
  `posterior_array`, and `parameter_names` now work on the continuous block of
  a Gibbs run; `predict` pins each draw's sampled discrete values, and
  `pointwise_loglikelihood`/`loo` condition each draw on that draw's discrete
  values (one column per user observation). The discrete part keeps
  `discrete_ess` and the chain fields.
- **ArviZ export upgrades** (#339): the four-argument
  `to_arviz_dict(model, args, constraints, chains)` emits a
  `"log_likelihood"` group (one matrix per observation address via
  `pointwise_loglikelihood`), shaped like the posterior group. Both methods
  take a `layout` kwarg — `:draw_chain` (default, Julia InferenceObjects) or
  `:chain_draw` (Python `az.from_dict`) — and record the chosen layout plus
  the producing package/version in an `"attrs"` entry, defusing the silent
  2×1000 → 1000-chains round-trip trap. `sample_stats` gains `"step_size"`
  and `"n_steps"`. `predict` now also accepts the four-argument
  `(model, args, constraints, result)` spelling for signature symmetry with
  `loo`/`waic`/`pointwise_loglikelihood`.

### Fixed

- **ForwardDiff 1.x compatibility — boundary validations judge the primal
  value**: ForwardDiff 1.x compares `Dual`s lexicographically at value ties
  (the partials break the tie), so a saturated latent landing EXACTLY on a
  support boundary with live partials (issues #343/#354 — e.g.
  `sigmoid(theta)` rounding to `1.0` with derivative `~8.5e-17`) started
  misfiring boundary-inclusive checks like `0 <= p <= 1` in both directions
  (values that used to pass now threw, and `x > 0` at `Dual(0.0, +1)` wrongly
  admitted the boundary). All Dual-reachable validation, support-membership,
  and boundary-equality comparisons now route through a `_primal` helper so a
  perturbation direction can never flip support membership; semantics under
  ForwardDiff 0.10 are unchanged, and the compat entry widens to `"0.10, 1"`.
  (#326): concrete compiled-plan step types fix the union-selector layouts
  that made Enzyme's type analysis reject `reparam=:noncentered` models, so
  the guarded-automatic (`adtype=:auto`) and explicit `:reverse` paths now
  cover hierarchical non-centered models (eight-schools-style). When
  `adtype=:reverse` is explicitly requested and the model cannot engage
  reverse mode, the sampler now warns once (naming the reason) instead of
  silently falling back to forward mode; `:auto` still falls back silently.
- **`to_mcmcchains` axis order** (#336): the MCMCChains extension permuted
  nothing and passed the `(draws, chains, params)` array straight to
  `MCMCChains.Chains`, which expects `iterations × parameters × chains` — a
  hard error for `num_chains != num_params` and silently swapped labels when
  they matched. The conversion now permutes correctly, appends the per-draw
  sampler statistics (`lp`, `diverging`, `energy`, `tree_depth`,
  `acceptance_rate`) as the `:internals` section, and the docstring documents
  the `summarize`/`ess`/`rhat` name collision with MCMCChains (call the
  UncertainTea versions qualified once both are loaded).
- **`vonmises` sampling no longer hangs at extreme `kappa`** (#342): the
  Best–Fisher rejection sampler looped forever below `kappa ≈ 1.4e-8`
  (cancellation) and above `kappa ≈ 6.7e153` (overflow). Tiny `kappa` now uses
  exact rejection from a circular-uniform proposal, huge `kappa` a wrapped
  normal (exact to double precision there), and both rejection loops bail
  with an error instead of hanging on a numerical pathology.
- **Count families accurate at extreme counts** (#345): `binomial`,
  `poisson`, and `negativebinomial` switch to saddle-point (Loader
  stirlerr/bd0) kernels above counts of `1e8` — the naive spellings returned
  *positive* log-densities at huge counts — accurate to ~2 ulp up to `1e18`
  and accepting integer-valued float counts beyond `typemax(Int)` (to
  `1e19`). `studentt` gets a Stirling-ratio expansion of its log-constant for
  `nu > 1e3` (shared with the truncated-t normalizer), and `gamma` treats
  exp-subnormal boundary values as off-support instead of scoring a silently
  wrong finite value.
- **Saturated-latent boundary values reject instead of returning wrong
  gradients** (#343): when a transformed latent saturates its support
  boundary (sigmoid rounding to exactly 1.0 past `theta ≈ 36.7`, log-transform
  underflow to 0.0), the gradient paths used to drop the density partials and
  return a finite—wrong—transform-Jacobian derivative. Both the batched
  analytic and ForwardDiff paths now return non-finite gradients alongside
  the `-Inf` log-joint, so leapfrog guards reject the point.
- **`orderedlogistic` middle categories no longer cancel** for `eta` far
  below the cutpoints (#344): the log-CDF difference is mirrored to a
  log-CCDF difference on the lower side, fixing `-Inf` values (and `Inf`
  gradients) at `eta <= -37`.
- **NaN observations throw at the data** (#346): a NaN inside a constrained
  observation value used to make `logjoint` and every gradient silently NaN
  (and sampler init failures blamed the parameters). Constraint values are
  now scanned at constraint-resolution/staging time and a NaN throws an
  `ArgumentError` naming the offending address; `Inf` stays allowed (it can
  legitimately score `-Inf`). The scan is stamped per choicemap mutation, so
  the warm path pays one integer comparison.

### Internal

- The signature cache is re-keyed from the conditioning signature alone to
  `(signature, dims)`, where `dims` is a runtime-dimension tuple resolved from
  the model arguments (always `()` today), and the model arguments are now
  threaded through signature resolution (#289, PR-1). Groundwork for
  runtime-length vector latents; no behavior change for existing models,
  except that a LATENT `mvnormal`/`mvstudentt`/`dirichlet`-family choice whose
  length is not statically known now fails at resolution time with an
  informative error naming the address and issue #289, instead of the late
  missing-parameter-slot error. Observed choices with runtime-length arguments
  are unaffected.
- The per-family scoring, gradient, and lowering code paths are generated
  from a single family table with a consistency test (#331), 11 dead internal
  functions are deleted behind a new export smoke test (#334), and
  `evaluator.jl`/`device/plan.jl`/`inference/vi.jl` are split into focused
  files (#332). CI: push-to-main runs are no longer cancelled and main
  requires status checks (#330); a scheduled workflow exercises the Enzyme
  extension (#333).

### Earlier in the 0.2.0 development line

A large feature release. Breaking only in the export surface (see the first
entry); model code written against v0.1.0 runs unchanged.

### Breaking

- **Curated public API** (#284): ~43 compiler-internal types (IR specs,
  execution-plan steps, transforms, mode singletons, lowering-introspection
  helpers) are no longer exported. They remain reachable qualified
  (`UncertainTea.ExecutionPlan`, …) with no semver guarantee. The exported
  surface — the `@tea` DSL, samplers, distributions, diagnostics, result
  types — is the supported public API rendered at `docs/src/api.md`.

### Reverse-mode AD (Enzyme)

- `reverse_mode_gradient(f, x)` and `reverse_mode_gradient(model, params,
  args, constraints)` via the optional Enzyme package extension: `O(1)`-in-P
  gradients for the models the analytic path does not fuse (#269–#271).
- Guarded-automatic gradient-backend selection on the batched samplers:
  `adtype=:auto` (default) engages reverse mode when it is safe and
  beneficial; `:forward`/`:reverse` override. Measured 24.8× faster
  `batched_nuts` on a P=60 non-analytic model (#272, #278).
- GP/HMM hyperparameter models and broadcast GLMs take the reverse-mode path
  via static whole-vector observation staging (#301); Int counts and Bool
  labels stage exactly (#319).

### Modeling

- Vectorized non-Gaussian observations: `{:y} ~ poisson.(exp.(a .+ b .* x))`
  and friends (bernoulli, bernoullilogit, exponential, studentt) lower to a
  backend-native dense step with analytic gradients (#300). Dotted function
  calls (`exp.(...)`) now work in broadcast arguments and compiled
  deterministic expressions.
- Gaussian processes: `gaussianprocess` marginal likelihood with isotropic or
  ARD lengthscales (#266, #274); kernel specifications `rbf_kernel`,
  `matern32_kernel`, `matern52_kernel`, `periodic_kernel`, and
  `kernel_sum`/`kernel_product` composition (#302); `gp_cholesky` for direct
  latent-function inference under non-Gaussian likelihoods (#280);
  `sparsegaussianprocess` FITC inducing-point approximation, `O(NM²+M³)`
  (#282).
- New distributions: `hmm` (Gaussian-emission hidden Markov models via the
  forward algorithm, #267), `orderedlogistic` (#303), `zeroinflatedpoisson` /
  `zeroinflatednegativebinomial` (#304), `vonmises` with an AD-generic
  `log I₀` (#305).
- Automatic marginalization of enumerable discrete latents
  (Rao-Blackwellization, #265).
- Eight new fast-path math primitives: `sin`, `cos`, `tan`, `tanh`, `sinh`,
  `cosh`, `atan`, `expm1` — with analytic chain rules and device dual rules
  (#299).

### Inference & diagnostics

- `elliptical_slice`: gradient-free, tuning-free sampler for latents with a
  multivariate-Gaussian prior — the natural companion to `gp_cholesky` (#306).
- `prior_predictive` for prior workflow checks; `rhat(chains; method=:rank)`
  rank-normalized split-R̂ (Vehtari et al. 2021) (#307).

### Fixes & robustness

- Warn on constraint addresses that match no model choice (silent
  misconditioning, #320).
- Fix `mvnormal` macro-expansion crash with runtime mean/scale arguments
  (#318).
- Broadcast observation staging cached per gradient cache (~5× less warm-call
  allocation on broadcast GLMs, #321).

### Internal

- Single-source scalar distribution kernels: each family declares logpdf +
  analytic partials once; the CPU reference, backend scoring, and batched
  gradients consume the one declaration, and the device kernels are pinned to
  it by a per-family parity test (#296–#298).

## v0.1.0 (2026-08-03)

Initial registered release on the Julia General registry: the static `@tea` DSL with constraint-driven
conditioning, CPU/batched/Metal-device inference (NUTS/HMC/ChEES/MEADS, ADVI,
SVGD, SMC, gibbs, pathfinder, nested sampling), a broad distribution library,
and diagnostics (split-R̂, ESS, MCSE, WAIC, PSIS-LOO, SBC).

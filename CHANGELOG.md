# Changelog

## Unreleased

### Breaking

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

## v0.2.0 (unreleased → pending registration)

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

## v0.1.0

Initial registered release: the static `@tea` DSL with constraint-driven
conditioning, CPU/batched/Metal-device inference (NUTS/HMC/ChEES/MEADS, ADVI,
SVGD, SMC, gibbs, pathfinder, nested sampling), a broad distribution library,
and diagnostics (split-R̂, ESS, MCSE, WAIC, PSIS-LOO, SBC).

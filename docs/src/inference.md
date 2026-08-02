```@meta
CurrentModule = UncertainTea
```

# Inference Overview

UncertainTea ships a family of gradient-based and Monte Carlo samplers, plus
variational and optimization-based approximations. They share one design theme:
keep model structure static so many chains or particles can run together over
dense layouts, first on the CPU reference runtime and then on GPU-friendly
device kernels.

This page is a map of the samplers and the batched/device architecture. The
deeper design rationale lives in the [Design Notes](design-notes.md).

The samplers and their result types live in the `UncertainTea.Inference`
submodule, and the chain summaries / export helpers in
`UncertainTea.Diagnostics` (see the [API Reference](api.md)):

```julia
using UncertainTea               # the modeling language
using UncertainTea.Inference     # nuts_chains, batched_nuts, batched_advi, ...
using UncertainTea.Diagnostics   # summarize, rhat, ess, posterior_array, ...
```

## Single-chain and multi-chain samplers

The reference entry points run standard MCMC on the CPU:

- [`hmc`](@ref) / [`nuts`](@ref) — a single Hamiltonian Monte Carlo or No-U-Turn
  chain with step-size and mass-matrix adaptation.
- [`hmc_chains`](@ref) / [`nuts_chains`](@ref) — several independent chains, each
  its own warmup, combined into an `HMCChains` result.

All of them accept a model, an argument tuple, and a `choicemap` of
observations, and return chains you can pass to [`summarize`](@ref), `rhat`,
`ess`, and the export helpers (`posterior_array`, `to_arviz_dict`,
`to_mcmcchains`).

```julia
chains = nuts_chains(model, args, constraints; num_chains=4, num_samples=500, num_warmup=500)
summary = summarize(chains)
```

## Batched samplers

The batched samplers advance **all chains together** over a dense
parameter-by-chain layout — the layout that lowers to the device. They are the
main path for the GPU-native work.

- [`batched_nuts`](@ref) — many-chain NUTS. `tree_strategy` selects `:hybrid`
  (default), `:masked`, or `:persistent` (a device-only, one-launch-per-iteration
  tree kernel).
- [`batched_hmc`](@ref) — many-chain fixed-length HMC.
- [`batched_chees`](@ref) — ChEES-HMC (Hoffman et al., AISTATS 2021): fixed-length
  jittered HMC whose trajectory length is adapted from **cross-chain**
  statistics. Every lane does identical work each step (no control-flow
  divergence, no per-leaf host sync), so it maps to a handful of kernels with one
  sync per iteration — the GPU-native route. NUTS stays the reference/default.
- `batched_advi`, `batched_importance_sampling`, `batched_sir`, `batched_smc` —
  batched variational inference and particle methods. `batched_advi` selects a
  guide family with `guide`: `:meanfield`, `:fullrank`, and `:lowrank` are
  Gaussian in unconstrained space, while `:flow` stacks affine coupling layers
  (RealNVP-style) on a mean-field base to capture nonlinear correlation and
  skew that the Gaussian guides structurally miss. The objective is chosen with
  `elbo`: the default `:standard` is the reparameterized ELBO, and `:iwae` is
  the importance-weighted bound `L_K = E[log(1/K Σ_k w_k)]` — a strictly tighter
  bound whose importance-sample group size is set by `iwae_samples` (`:iwae` is
  available for `:meanfield`, `:fullrank`, and `:flow`). Both extensions keep
  the model gradient untouched — the particles are still scored positions — so
  the device path is unchanged and only the host-side guide and objective
  bookkeeping differ. `standard_elbo_history` records the plain ELBO alongside
  `elbo_history` so the two bounds can be compared. The flow guide is CPU-only.
- [`batched_svgd`](@ref) — Stein Variational Gradient Descent (Liu & Wang,
  NeurIPS 2016): a **deterministic particle** method between VI and MCMC. `N`
  particles follow a kernelized gradient flow toward the posterior; each
  iteration is one batched unconstrained log-joint gradient over the particle
  matrix (host or `backend`-device) plus an `N × N` RBF-kernel interaction with
  the median-distance bandwidth, stepped by Adam. It returns an `SVGDResult` of
  equal-weight **optimized particles** — not MCMC draws, so there is no
  Rhat/ESS; validate by posterior-moment recovery and particle spread via
  [`particle_mean`](@ref) / [`particle_covariance`](@ref). Caveats (non-goals):
  mode collapse / variance underestimation at small `N` and in high dimensions,
  and bandwidth sensitivity — SVGD is a fast approximate-posterior tool, not a
  NUTS replacement.

Per-chain step-size adaptation is the default across the batched samplers, which
avoids stranding chains whose initial curvature a single shared step size never
fits.

### Gradient backend selection (`adtype`)

The batched samplers accept `adtype=:auto | :forward | :reverse` to choose how
log-joint gradients are computed when no analytic backend gradient applies:
`:auto` (default) picks Enzyme reverse-mode for models with a generated scorer
and ≥ 24 parameters and ForwardDiff otherwise, `:reverse` prefers reverse-mode
whenever the model supports it (host-only), and `:forward` forces ForwardDiff.
See [Modeling — Gradients and AD selection](modeling.md#gradients-and-ad-selection)
for the full tiering.

## The device backend

Passing a `backend` (a `KernelAbstractions.Backend`) to the batched samplers
runs the inner loop **device-resident**. A static, backend-lowerable model is
lowered to an `isbits` `DeviceExecutionPlan` and evaluated inside fused
KernelAbstractions kernels, one thread per batch column. The same code path runs
on `KernelAbstractions.CPU()` (the authoritative reference target) and on any GPU
backend; a package extension declares `Float32` as Metal's precision.

Not every model lowers to the device subset. Use `backend_report` and
`backend_execution_plan` to check support; unsupported models fall back to the
CPU path. The low-level device density APIs (`device_batched_logjoint`,
`device_batched_logjoint_gradient`, and their in-place variants) are exported
from `UncertainTea.Device` for direct use.

Because RNG stays host-side on the device HMC/NUTS paths, device results are
statistically equivalent to — not bitwise identical to — the CPU path.

## Other inference tools

- `pathfinder` — the Pathfinder variational approximation for initialization or
  standalone posterior draws.
- `gibbs` — Metropolis-within-Gibbs for models with discrete structure, with
  `discrete_ess`.
- `map_estimate` / `laplace_approximation` — optimization-based point and
  Gaussian approximations.
- `sbc` — simulation-based calibration, which runs one of the batched samplers
  under the hood and doubles as an end-to-end correctness check.
- `nested_sampling` — static nested sampling (Skilling 2006): a **CPU-only,
  gradient-free** estimator of the log-evidence `log Z` **with an
  information-based uncertainty** (`sqrt(H/N)` via `log_evidence_error`), plus
  importance-weighted posterior draws as a by-product. It is the package's third,
  independent evidence estimator alongside importance sampling and tempered SMC,
  presented as the **multimodal / evidence-with-uncertainty** option rather than
  a faster path for unimodal problems: live points migrate into every surviving
  mode, so well-separated multimodal posteriors are handled naturally. The
  correctness crux is the constrained-prior replacement; the default `:rwmh`
  (a short random-walk Metropolis chain in unconstrained space, seeded at a
  surviving live point, accepting only inside the likelihood constraint) needs no
  gradients and so works on non-differentiable models, while `:rejection` gives
  exact draws for very low-dimensional problems. Continuous latents only (no
  `marginalize=:enumerate`); validated for correctness in low dimension —
  `:rwmh` mixing, hence evidence quality, degrades as dimension grows and needs a
  longer `num_walk`.
- [`elliptical_slice`](@ref) — elliptical slice sampling (Murray, Adams &
  MacKay 2010) for targets `p(f) ∝ N(f; 0, LL') · exp(loglik(f))`: gradient-free
  and tuning-free, the standard tool for latent Gaussian-process functions
  under non-Gaussian likelihoods. Pair it with `gp_cholesky` for the prior
  factor — see [Modeling — Gaussian processes](modeling.md#gaussian-processes).
- Model comparison and predictive helpers: `waic`, `psis_loo` / `loo`,
  `pointwise_loglikelihood`, and `predict`, plus [`prior_predictive`](@ref)
  for the prior-workflow check before conditioning (does the model generate
  data on the right scale?).

## Diagnostics and export

`summarize`, `rhat`, `ess`, `check_diagnostics`, and `has_warnings` report
convergence and sampling health. `rhat(chains; method=:rank)` computes the
rank-normalized split-R̂ of Vehtari et al. (2021) — the max of the bulk and
tail-sensitive folded statistics, robust to heavy tails and nonlinear scale,
now the standard in Stan/ArviZ (`method=:split`, the default, is the classical
split-chain statistic). `posterior_array`, `parameter_names`,
`to_arviz_dict`, and `to_mcmcchains` (via the MCMCChains extension) hand draws to
the wider Julia and Python ecosystems.

See the [Eight Schools example](generated/eight_schools.md) for a full run with
real posterior output.

## Working with results

### Plotting

The plotting route goes through [`to_mcmcchains`](@ref): loading `MCMCChains`
activates UncertainTea's package extension, and `MCMCChains.Chains` objects
carry [StatsPlots](https://github.com/JuliaPlots/StatsPlots.jl) recipes for
trace plots, densities, autocorrelation, corner plots, and more:

```julia
using UncertainTea, UncertainTea.Inference
import MCMCChains
using StatsPlots

chains = nuts_chains(model, args, constraints; num_chains=4)
mc = to_mcmcchains(chains)

plot(mc)             # trace + density per parameter
autocorplot(mc)
corner(mc)
```

The `:internals` section of the converted object carries the per-draw sampler
statistics (`:lp`, `:diverging`, `:energy`, `:tree_depth`,
`:acceptance_rate`), so e.g. `plot(mc[:, [:lp], :])` shows the log-density
trace. Note the qualified-call caveat in the [`to_mcmcchains`](@ref)
docstring: `MCMCChains` also exports `summarize`/`ess`/`rhat`, so call the
UncertainTea versions qualified once both packages are loaded.

### Tabular access (Tables.jl / DataFrames.jl)

`HMCChains` and `PredictiveDraws` implement the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface (loading
`Tables` — which `DataFrames` does automatically — activates the
`UncertainTeaTablesExt` extension), so they slot into any Tables.jl sink:

```julia
using DataFrames

df = DataFrame(chains)     # chain, draw, then one column per parameter
pp = DataFrame(predict(model, args, chains; num_draws=200))  # draw + one column per address
```

The chains table is wide and chain-major (chain 1's draws in order, then
chain 2's, ...), with parameter columns named as in
[`parameter_names`](@ref) (vector latents flatten to `v[1]`, `v[2]`, ...).

### Persisting results across sessions (JLD2)

Result structs such as `HMCChains` embed the model, and a `@tea` model
contains compiled closures — which JLD2 stores **by name only**. Saving the
whole result works within a session, but reloading in a fresh session cannot
reconstruct the model and yields a warning plus an unusable stub. Persist the
**draws and names** instead of the result struct:

```julia
using JLD2

jldsave("fit.jld2";
    draws=posterior_array(chains),          # (num_samples, num_chains, num_params)
    names=parameter_names(chains),
    stats=to_arviz_dict(chains)["sample_stats"],  # optional: per-draw sampler stats
)

# fresh session — no model definition needed:
using JLD2
fit = load("fit.jld2")
fit["draws"], fit["names"]
```

Everything saved this way is plain arrays/strings/dicts, so it reloads
without UncertainTea installed at all. To resume model-coupled workflows
(`predict`, `loo`, `summarize`), re-run the `@tea` definition in the new
session and pair it with the reloaded draws; for archival interchange,
`to_arviz_dict` output round-trips through NPZ/JSON to Python ArviZ.

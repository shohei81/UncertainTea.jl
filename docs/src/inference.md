# Inference Overview

UncertainTea ships a family of gradient-based and Monte Carlo samplers, plus
variational and optimization-based approximations. They share one design theme:
keep model structure static so many chains or particles can run together over
dense layouts, first on the CPU reference runtime and then on GPU-friendly
device kernels.

This page is a map of the samplers and the batched/device architecture. The
deeper design rationale lives in the [Design Notes](design-notes.md).

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
  batched variational inference and particle methods.
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
`device_batched_logjoint_gradient`, and their in-place variants) are exported for
direct use.

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
- Model comparison and predictive helpers: `waic`, `psis_loo` / `loo`,
  `pointwise_loglikelihood`, and `predict`.

## Diagnostics and export

`summarize`, `rhat`, `ess`, `check_diagnostics`, and `has_warnings` report
convergence and sampling health. `posterior_array`, `parameter_names`,
`to_arviz_dict`, and `to_mcmcchains` (via the MCMCChains extension) hand draws to
the wider Julia and Python ecosystems.

See the [Eight Schools example](generated/eight_schools.md) for a full run with
real posterior output.

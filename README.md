# UncertainTea

[![CI](https://github.com/shohei81/UncertainTea.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/shohei81/UncertainTea.jl/actions/workflows/ci.yml)

UncertainTea is an experimental Julia probabilistic programming package with a
Gen-like frontend and a static execution model designed for GPU-friendly
backends.

The project is built around one constraint: keep model structure static enough
that `logjoint`, batched chains or particles, and parameter transforms can run
over dense layouts and backend-friendly control flow. The CPU reference runtime
is available today; the device backend (KernelAbstractions kernels with a Metal
extension) is under active development.

## Status

UncertainTea `0.2.0` is an experimental release.

- The static DSL, CPU evaluation path, and several inference algorithms are
  implemented.
- GPU work currently focuses on backend lowering, support checks, and
  device-resident kernels (via KernelAbstractions) for a supported static
  subset.
- APIs and model restrictions may change as the static IR and backend contract
  continue to converge.

## What It Provides

- Gen-like modeling with `@tea` and `@tea (static)`, tilde syntax, explicit
  addresses, hierarchical addresses, and external conditioning via `choicemap`
- Static model introspection through `modelspec`, `parameterlayout`,
  `executionplan`, and backend reports
- CPU reference evaluation with `generate`, `assess`, `logjoint`,
  unconstrained transforms, and batched logjoint/gradient APIs
- Inference methods including `hmc`, `nuts`, `hmc_chains`, `nuts_chains`,
  `batched_hmc`, `batched_nuts`, `batched_chees` (ChEES-HMC), `batched_meads`
  (MEADS), `gibbs`, `batched_advi`, `batched_svgd`,
  `batched_importance_sampling`, `batched_sir`, `batched_smc`, and
  `nested_sampling`
- Warm-start and point estimation: `pathfinder`, `map_estimate`,
  `laplace_approximation`
- Diagnostics and model comparison: `sbc` (simulation-based calibration),
  `waic`, `psis_loo` / `loo`, and posterior-predictive `predict`
- Experimental GPU-oriented lowering and device execution: support checks via
  `backend_report` and `backend_execution_plan`, plus a KernelAbstractions
  device backend (`device_batched_logjoint`, `device_batched_logjoint_gradient`,
  and device-resident batched HMC/ADVI inner loops) with a Metal extension

## Installation

UncertainTea currently targets Julia 1.10+.

```julia
using Pkg
Pkg.add(url="https://github.com/shohei81/UncertainTea.jl.git")
```

For local development:

```julia
using Pkg
Pkg.develop(path="/path/to/uncertaintea")
```

## Quick Start

The public API is namespaced (issue #329): `using UncertainTea` brings in the
modeling language only (the `@tea` DSL, `choicemap`, `generate`, `logjoint`,
and the distribution constructors); the samplers live in
`UncertainTea.Inference`, the chain summaries and model-comparison tools in
`UncertainTea.Diagnostics`, and the device density APIs in
`UncertainTea.Device`.

```julia
using Random
using UncertainTea               # the @tea DSL, choicemap, distributions
using UncertainTea.Inference     # hmc_chains, nuts_chains, parameter_vector, ...
using UncertainTea.Diagnostics   # summarize, rhat, ess, ...

@tea (static) function gaussian_mean()
    mu ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(mu, 1.0f0)
    return mu
end

constraints = choicemap((:y, 0.3f0))

trace, logw = generate(gaussian_mean, (), constraints; rng=MersenneTwister(1))
params = parameter_vector(trace)
joint = logjoint(gaussian_mean, params, (), constraints)

chains = hmc_chains(
    gaussian_mean,
    (),
    constraints;
    num_chains=4,
    num_samples=100,
    num_warmup=100,
    step_size=0.2,
    num_leapfrog_steps=8,
    rng=MersenneTwister(2),
)

summary = summarize(chains)
println(trace[:mu])
println(logw)
println(joint)
println(summary.parameters[1].mean)
```

## Current Direction

UncertainTea is intentionally not centered on Turing compatibility or
unrestricted dynamic traces. The main path is:

- Gen-like surface syntax
- static semantics
- dense parameter layouts
- CPU reference first, GPU backends second

The current built-in distribution set includes the continuous scalar families
`normal`, `lognormal`, `laplace`, `exponential`, `gamma`, `inversegamma`,
`weibull`, `beta`, `studentt`, `cauchy`, `halfnormal`, `halfcauchy`, `uniform`,
`logistic`, `gumbel`, `pareto`, `frechet`, `rayleigh`, and `inversegaussian`;
their `truncatednormal` / `truncatedstudentt` truncations; the discrete families
`bernoulli`, `bernoullilogit`, `binomial`, `betabinomial`, `geometric`,
`negativebinomial`, `poisson`, `categorical`, `discreteuniform`, and
`multinomial`, `orderedlogistic` (ordinal outcomes), `vonmises` (circular), and `zeroinflatedpoisson` / `zeroinflatednegativebinomial` (excess-zero counts); the vector/structured families `dirichlet`, diagonal `mvnormal`,
dense `mvnormaldense`, `mvstudentt` / `mvstudenttdense`, `lkjcholesky` (with the
`scale_cholesky` helper), `gaussianprocess` regression (isotropic or ARD kernel; with
the `gp_cholesky` helper for direct latent-function inference under non-Gaussian
likelihoods, and `sparsegaussianprocess` for the `O(NM²+M³)` FITC inducing-point
approximation), `hmm` (Gaussian-emission hidden Markov models via the forward
algorithm), and finite `mixture`s; plus `iid` vectors and
user-defined families via `register_distribution` (`AbstractTeaDistribution`).

## Documentation

- **[Rendered documentation site](https://shohei81.github.io/UncertainTea.jl/)** — getting started, inference guide, executable examples, API reference
- [Documentation index](docs/README.md)
- [Research notes](docs/research.md)
- [Architecture direction](docs/architecture.md)
- [Minimal DSL proposal](docs/dsl.md)
- [Batched inference design](docs/batched-inference.md)
- [GPU-native NUTS notes](docs/gpu-native-nuts.md)
- [Vector backend lowering notes](docs/vector-backend-lowering.md)
- [Repository agent guide](AGENTS.md)

## License

Apache 2.0

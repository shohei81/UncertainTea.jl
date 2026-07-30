# UncertainTea

UncertainTea is an experimental Julia probabilistic programming package with a
Gen-like frontend and a **static execution model** designed for GPU-friendly
backends.

The project is built around one constraint: keep model structure static enough
that `logjoint`, batched chains or particles, and parameter transforms can run
over dense layouts and backend-friendly control flow. The CPU reference runtime
is available today; the device backend (KernelAbstractions kernels with a Metal
extension) is under active development.

!!! warning "Experimental"
    UncertainTea `0.1.0` is an experimental release. The static DSL, CPU
    evaluation path, and several inference algorithms are implemented. APIs and
    model restrictions may change as the static IR and backend contract
    continue to converge.

## What it provides

- Gen-like modeling with `@tea` and `@tea static`, tilde syntax, explicit and
  hierarchical addresses, and external conditioning via `choicemap`.
- Static model introspection through `modelspec`, `parameterlayout`,
  `executionplan`, and backend reports.
- CPU reference evaluation with `generate`, `assess`, `logjoint`, unconstrained
  transforms, and batched logjoint/gradient APIs.
- Inference methods including [`hmc`](@ref), [`nuts`](@ref),
  [`hmc_chains`](@ref), [`nuts_chains`](@ref), [`batched_hmc`](@ref),
  [`batched_nuts`](@ref), [`batched_chees`](@ref), `batched_advi`,
  `pathfinder`, `gibbs`, importance sampling / SIR / SMC, and `sbc`.
- Experimental GPU-oriented lowering and a KernelAbstractions device backend
  (`device_batched_logjoint`, `device_batched_logjoint_gradient`, and
  device-resident batched HMC/NUTS/ADVI inner loops) with a Metal extension.

## Installation

UncertainTea targets Julia 1.10+.

```julia
using Pkg
Pkg.add(url="https://github.com/shohei81/uncertaintea.git")
```

For local development:

```julia
using Pkg
Pkg.develop(path="/path/to/uncertaintea")
```

## Quick start

```julia
using Random
using UncertainTea

@tea static function gaussian_mean()
    mu ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(mu, 1.0f0)
    return mu
end

constraints = choicemap((:y, 0.3f0))

chains = nuts_chains(
    gaussian_mean,
    (),
    constraints;
    num_chains=4,
    num_samples=200,
    num_warmup=200,
    rng=MersenneTwister(1),
)

summary = summarize(chains)
println(summary.parameters[1].mean)   # posterior mean of mu
```

## Where to go next

- [Getting Started](getting-started.md) — the `@tea` DSL and static semantics.
- [Inference Overview](inference.md) — samplers, batching, and the device path.
- [Eight Schools example](generated/eight_schools.md) — a runnable hierarchical
  model with real posterior output.
- [API Reference](api.md) — the exported surface.
- [Design Notes](design-notes.md) — the deeper architecture and design docs.

## License

Apache 2.0.

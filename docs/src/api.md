```@meta
CurrentModule = UncertainTea
```

# API Reference

This page auto-renders the docstrings attached to UncertainTea's **public**
surface, which as of issue #329 is organized into a minimal modeling-language
top level plus three facade submodules:

- `using UncertainTea` — the modeling language: the `@tea` DSL, traces and
  choicemaps, `generate`/`assess`/`logjoint`, and the distribution
  constructors.
- `using UncertainTea.Inference` — samplers and fitters, their result types
  and accessors, and the density/parameter machinery.
- `using UncertainTea.Diagnostics` — chain summaries, convergence checks,
  draw export, model comparison, and predictive checks.
- `using UncertainTea.Device` — the KernelAbstractions device-resident
  batched density APIs.

Every public name is also reachable qualified as `UncertainTea.<name>`; the
submodules are re-exporting facades, not separate implementations.

!!! note "Public vs. internal (issue #283)"
    The IR, execution-plan, transform, and lowering-introspection types
    (`ExecutionPlan`, `ChoicePlanStep`, `DistributionSpec`, `IdentityTransform`,
    `backend_report`, `StaticMode`, …) are **implementation details** and are no
    longer exported as of `v0.2.0`. They remain reachable qualified —
    `UncertainTea.ExecutionPlan`, `UncertainTea.backend_report`, … — for
    white-box tests and power users, but carry **no semver stability guarantee**.
    Only the names shown below are covered by the versioning contract.

!!! note "Docstring coverage is a work in progress"
    The build runs with `checkdocs=:none` because the exported surface is only
    partially docstringed today. Everything that currently has a docstring is
    shown below; the headline samplers ([`hmc`](@ref), [`nuts`](@ref),
    [`hmc_chains`](@ref), [`nuts_chains`](@ref), [`batched_hmc`](@ref),
    [`batched_nuts`](@ref), [`batched_chees`](@ref)) carry concise docstrings.
    Completing coverage across the rest of the surface is tracked with the
    doc-currency issues #213–#217 and should be folded in as content migrates.

```@index
```

## Modeling (`using UncertainTea`)

The `@tea` DSL, trace/choicemap types, core evaluation entry points, and the
distribution constructors — the names exported from the top-level module.

```@autodocs
Modules = [UncertainTea]
Private = false
Order = [:macro, :function, :type, :constant]
```

## Inference (`UncertainTea.Inference`)

```@docs
UncertainTea.Inference
```

```@autodocs
Modules = [UncertainTea]
Private = true
Order = [:function, :type, :constant]
Filter = o -> (o isa Function || (o isa Type && !(o isa Union))) && nameof(o) in names(UncertainTea.Inference)
```

## Diagnostics (`UncertainTea.Diagnostics`)

```@docs
UncertainTea.Diagnostics
```

```@autodocs
Modules = [UncertainTea]
Private = true
Order = [:function, :type, :constant]
Filter = o -> (o isa Function || (o isa Type && !(o isa Union))) && nameof(o) in names(UncertainTea.Diagnostics)
```

## Device (`UncertainTea.Device`)

```@docs
UncertainTea.Device
```

```@autodocs
Modules = [UncertainTea]
Private = true
Order = [:function, :type, :constant]
Filter = o -> (o isa Function || (o isa Type && !(o isa Union))) && nameof(o) in names(UncertainTea.Device)
```

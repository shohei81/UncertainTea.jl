# API Reference

This page auto-renders the docstrings attached to UncertainTea's **exported**
surface. The exported surface **is** the supported public API: the `@tea` DSL,
the samplers, the distribution families, the diagnostics, and the result types.

!!! note "Public vs. internal (issue #283)"
    The IR, execution-plan, transform, and lowering-introspection types
    (`ExecutionPlan`, `ChoicePlanStep`, `DistributionSpec`, `IdentityTransform`,
    `backend_report`, `StaticMode`, …) are **implementation details** and are no
    longer exported as of `v0.2.0`. They remain reachable qualified —
    `UncertainTea.ExecutionPlan`, `UncertainTea.backend_report`, … — for
    white-box tests and power users, but carry **no semver stability guarantee**.
    Only the exported names shown below are covered by the versioning contract.

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

```@autodocs
Modules = [UncertainTea]
Private = false
Order = [:function, :type, :constant]
```

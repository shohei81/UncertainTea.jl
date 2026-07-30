# API Reference

This page auto-renders the docstrings attached to UncertainTea's **exported**
surface.

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

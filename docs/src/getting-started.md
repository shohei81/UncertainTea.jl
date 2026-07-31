# Getting Started

This page introduces the `@tea` modeling DSL and the static execution model that
UncertainTea is built around. It is adapted from the fuller
[DSL design notes](design-notes.md); see there for the complete rationale.

## The `@tea` DSL

Models are written with `@tea`, using tilde syntax for random choices, explicit
choice addresses in `{...}`, and hierarchical addresses with `=>`. The syntax is
recognizably close to Gen, while compiling to a static, GPU-friendly execution
plan.

```julia
using UncertainTea

@tea static function gaussian_mean()
    mu ~ normal(0.0, 1.0)     # random choice bound to `mu`, address :mu
    {:y} ~ normal(mu, 1.0)      # explicitly addressed choice :y
    return mu
end
```

- `mu ~ normal(...)` creates a random choice, binds it to `mu`, and uses `:mu`
  as the implicit address.
- `{:y} ~ normal(...)` creates an explicitly addressed choice without a named
  binding. You can also capture it: `y = ({:y} ~ normal(mu, sigma))`.
- Deterministic local computation (`sigma = exp(log_sigma)`) is allowed and runs
  as part of the plan.

## Conditioning with `choicemap`

A random choice is an **observation** if and only if its address is present in
the constraints supplied at inference time; otherwise it is a **latent** that
receives a dense parameter slot. Binding a value with `x ~ dist` does not, by
itself, make a choice a latent or an observation.

```julia
constraints = choicemap((:y, 0.3))
trace, logw = generate(gaussian_mean, (), constraints)
```

Because the latent set depends on what is constrained, the parameter-vector
length is a function of the *conditioning signature* (the set of constrained
addresses), not of the model alone — exactly as Stan's parameter block depends
on its data block. `observation_addresses(model, args, constraints)` returns the
constrained-and-present choice addresses under this rule.

## Repeated structure and data

Data is supplied through constraints, typically over hierarchical addresses
produced in a `for` loop:

```julia
@tea static function iid_model(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0)
    end
    return mu
end

ys = [0.1, -0.2, 0.4]
constraints = choicemap((:y => i, ys[i]) for i in eachindex(ys))
trace, logw = generate(iid_model, (length(ys),), constraints)
```

In the static path the loop extent must be compile-time constant or part of the
shape-specialized execution cache.

## Static semantics

The `static` subset (`@tea static function ...`) requires that model structure
stays fixed:

- the set of choices is fixed for a compiled execution plan,
- each choice has a fixed shape and each address is statically enumerable,
- loop bounds are static or shape-specialized,
- no recursion, no trans-dimensional structure, no data-dependent creation of
  new addresses.

**Branchful control flow is rejected in static models.** `if`/`elseif`/`else`
and the ternary `?:` operator raise an `ArgumentError` at macro-expansion time,
because the linear execution plan has no IR for branches. Supported
alternatives:

- `ifelse(cond, a, b)` for deterministic value selection (both sides evaluate;
  the set of choices does not change),
- moving a data-dependent branch outside the model and passing its result in as
  a model argument,
- static `for` loops for repeated structure.

Dynamic-mode models (`@tea function ...` without `static`) may contain
`if`/`else`, but they do not compile to the static GPU-friendly plan.

## Distributions

The built-in distribution set includes `normal`, `lognormal`, `laplace`,
`exponential`, `gamma`, `inversegamma`, `weibull`, `beta`, `dirichlet`,
diagonal `mvnormal`, `mvnormaldense`, `bernoulli`, `bernoullilogit`, `binomial`,
`geometric`, `negativebinomial`, `poisson`, `studentt`, and `categorical`, plus
`truncatednormal`/`truncatedstudentt`, `mixture`, `lkjcholesky`, and the `iid`
combinator (with an optional `reparam=:noncentered` reparameterization). Custom
families can be added via `register_distribution`.

## Next steps

- [Inference Overview](inference.md) — how to sample these models.
- [Eight Schools example](generated/eight_schools.md) — a full hierarchical
  model end to end.

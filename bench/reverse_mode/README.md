# Reverse-mode AD prototype (issue #268, follow-up to RFC #263)

A self-contained prototype of the RFC #263 recommendation — **Option 1: a host
reverse-mode AD fallback (Enzyme.jl)** for the models the analytic/fused
gradient path (GLM, iid suffstats) does not cover, where the forward-mode
(ForwardDiff) gradient ceiling is quadratic.

Enzyme lives **only in this benchmark environment** (`bench/reverse_mode/Project.toml`),
never in the core `UncertainTea` dependency graph — the clean boundary the RFC
asked for.

## Run

From the repository root:

```sh
julia --project=bench/reverse_mode -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=bench/reverse_mode bench/reverse_mode/enzyme_scaling.jl
```

## What it measures

1. **Scaling + correctness** on a pure high-`P` logjoint mirroring the RFC's
   coupled model (`x ~ iid(normal,P)` + per-element `y[i] ~ normal(tanh(x[i]) +
   0.5·x[i+1], 0.3)` — nothing fuses). Reverse-mode matches ForwardDiff to
   ~`1e-15` and flattens the quadratic curve to linear.
2. **GP hyperparameter gradient** (the RFC's stated first target): a pure
   `UncertainTea.logpdf(gp, y)` through the dense Cholesky, differentiated by
   Enzyme vs ForwardDiff.
3. **The integration blocker**: Enzyme applied to UncertainTea's *real* batched
   per-column objective.

## Measured results (M-series, Julia 1.12.2, Enzyme 0.13.196)

### (1) pure high-P coupled logjoint

| P | ForwardDiff | Enzyme reverse | speedup | max\|Δ\| |
|---|---|---|---|---|
| 10 | 0.0004 ms | 0.0001 ms | 4.9× | 8.9e-16 |
| 50 | 0.0038 ms | 0.0003 ms | 11.6× | 1.8e-15 |
| 100 | 0.0136 ms | 0.0007 ms | 18.9× | 1.8e-15 |
| 200 | 0.0473 ms | 0.0015 ms | 32.1× | 2.7e-15 |
| 400 | 0.1810 ms | 0.0031 ms | 58.4× | 3.6e-15 |
| 800 | 0.7119 ms | 0.0066 ms | 107.7× | 3.6e-15 |

ForwardDiff is **~O(P²)** (each doubling of `P` is ~3.8×); Enzyme reverse is
**~O(P)** (each doubling ~2.1×), so the speedup grows without bound with `P` —
the quadratic ceiling and the ForwardDiff chunk-size cliff from #263 are both
gone.

### (2) GP hyperparameter gradient

Enzyme reverse matches ForwardDiff to `max|Δ| = 1.07e-14` (**MATCH**) — the
textbook reverse-mode-through-Cholesky win is numerically exact.

### (3) integration blocker — RESOLVED (shipped in #270–#272)

At prototype time Enzyme could not differentiate UncertainTea's real batched
per-column objective (type-unstable compiled-plan machinery + workspace
mutation). That blocker was resolved by the **generated scorer**: a pure,
type-stable per-column log-density function generated from the execution plan,
which is exactly what Enzyme differentiates today.

What shipped from this prototype's conclusions:

- **#270** — `UncertainTeaEnzymeExt`: the package extension wiring Enzyme
  reverse-mode through pure GP/`logpdf` objectives (the prototype's stated
  first target).
- **#271/#272** — reverse-mode through the generated scorer on the batched
  gradient path, selectable with `adtype=:reverse` and picked automatically by
  `adtype=:auto` for supported models with ≥ 24 parameters (threshold measured
  in #278).

Enzyme still lives outside the core dependency graph — loading `Enzyme`
activates the extension; nothing changes for users who never load it. This
directory remains useful as the standalone scaling benchmark for the
forward-vs-reverse crossover.

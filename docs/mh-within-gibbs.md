# MH-within-Gibbs for discrete latent sites

Design for issue #13, track 2: a compositional sampler that alternates
single-site Metropolis-Hastings updates on discrete latent sites with NUTS
updates on the continuous block. This is the companion to
docs/discrete-enumeration.md (track 1): enumeration is the right tool for
small finite supports; this sampler covers unbounded and large supports
(poisson, geometric, negativebinomial, binomial) and any finite-support
site the user prefers not to marginalize.

## Problem

A discrete latent without `marginalize=:enumerate` has no parameter slot and
no gradient, so every existing sampler either errors (the compiled logjoint
requires a provided value) or needs the latent summed out. Enumeration blows
up on unbounded supports. The standard remedy (Turing's compositional Gibbs)
is to sample discrete sites by MH conditional on the continuous block and
the continuous block by HMC/NUTS conditional on the discrete values.

## What the machinery already gives us

Three facts from the reconnaissance shape the whole design:

1. **The NUTS transition is already a standalone primitive.**
   `_nuts_proposal(target, position, logjoint, gradient, inverse_mass_matrix,
   step_size, max_tree_depth, divergence_threshold, rng)` allocates its own tree
   workspace per call and persists nothing, so freezing a `WarmupDriver`'s
   step size and metric and calling it in a loop is fully supported.
2. **Conditioning flows through a shared mutable `ChoiceMap`.**
   `ModelDensityTarget` and the ForwardDiff gradient cache close over the
   constraints object by reference, and `ChoiceMap` upserts in place via
   `_pushchoice!`. Holding ONE persistent map (observations plus the current
   discrete values) and mutating single entries between transitions means
   the NUTS side sees updated discrete values with zero rebuild cost.
3. **Scoring a site's conditional is a plain `logjoint_unconstrained` call.**
   There is no incremental scoring; each MH evaluation re-walks the compiled
   plan. That prices a site update at one full logjoint per proposal (the
   current state's value is cached across the sweep), which is acceptable
   for the handful-of-sites models this targets.

## API

```julia
chain = gibbs(model, args, observations;
              num_samples, num_warmup=0, rng=...,
              # NUTS block options, forwarded:
              step_size, max_tree_depth, target_accept,
              adapt_step_size, adapt_mass_matrix, metric, ...)
```

- **Site discovery is automatic**, mirroring the SBC observation detection:
  a prior `generate` trace's addresses, minus the continuous parameter-slot
  addresses (`parameterchoicemap`), minus the user-supplied observation
  addresses, minus sites flagged `marginalize=:enumerate` (those stay
  marginalized inside the logjoint and never appear as Gibbs sites; their
  spec addresses are static by construction). What remains — including
  loop-scoped `{:z => i}` addresses, which the trace enumerates concretely —
  is the discrete Gibbs site set, in trace order.
- A model with no discrete sites degrades to plain `nuts` semantics; a model
  with no continuous slots skips the NUTS block and becomes pure single-site
  MH.
- The result is a `GibbsChain`: the continuous block's samples and
  adaptation fields (mirroring `HMCChain`), plus
  `discrete_samples::Dict{address,Vector}` and per-site MH acceptance
  rates. `_split_ess`/`_split_rhat` consume plain matrices, so discrete
  sample vectors get diagnostics for free.

## The sweep

One iteration = one full Gibbs sweep:

1. **Discrete pass** (sites in fixed trace order). For each site, propose
   `z'` from a symmetric proposal (below), evaluate
   `logjoint_unconstrained(model, position, args, merged)` with only that
   site's entry mutated, and accept with probability
   `min(1, exp(L(z') - L(z)))`. The current sweep's `L` is carried along:
   one plan evaluation per proposal, plus one refresh at sweep start (the
   NUTS block moved `position` since the last pass). Working in
   unconstrained space is safe because the log-abs-det term is constant in
   the discrete values and cancels from every ratio.
2. **Continuous pass.** Refresh `current_logjoint`/`current_gradient` at the
   (unchanged) position — the discrete pass changed the conditioning — then
   run one `_nuts_proposal` with the frozen (or still-adapting, during
   warmup) step size and metric.

During warmup, both passes run and the `WarmupDriver` adapts across sweeps,
so the step size and mass matrix adapt to the MIXTURE of conditionals the
chain actually visits (the Turing-style pragmatic choice); after
`warmup_finalize!` they freeze. Samples are recorded post-warmup only.

### Symmetric proposals (no proposal-density machinery)

Prior-conditional proposals would need the site's standalone prior density,
which no current primitive exposes without re-deriving per-family logpdfs of
runtime-built distributions. Symmetric proposals sidestep the correction
term entirely — `log alpha = L(z') - L(z)`:

- integer supports (poisson, geometric, negativebinomial, binomial):
  `z' = z ± s` with random sign and step `s = 1` by default; an optional
  heavier-tailed symmetric step (`s = 1 + Geometric(p)`) helps large-count
  posteriors and stays correction-free.
- bernoulli: deterministic flip.
- categorical(K): uniform over the K-1 other categories. The probability
  vector must be a literal so K is known — a dynamic vector may contain
  zero-probability categories between positive ones, which a local walk
  cannot cross, so those sites are rejected honestly (marginalize instead).

The family (and K, for categorical) comes from the model spec choice
matching the site's address; loop-scoped sites share their template's spec.

Out-of-support proposals must be rejected BEFORE the logjoint evaluation
where possible: the compiled scorer binds the provided value and keeps
walking even past a `-Inf` pmf, so a suffix consumer can throw on an invalid
value (`z ~ poisson(...); {:y} ~ binomial(z, p)` with a proposed `z = -1`
dies in the binomial constructor, not in the pmf). Concretely:

- proposals below the family's static lower bound (0 for the count families)
  reject immediately without scoring — this covers the entire ±s hazard for
  poisson/geometric/negativebinomial;
- bernoulli/categorical proposals are in-support by construction;
- dynamic upper bounds (binomial trials) cannot be pre-checked statically,
  so the PROPOSAL's logjoint evaluation is wrapped in a catch that converts
  a throw into a rejection. This is MH-correct — unevaluable states are
  zero-density regions, and the chain invariant guarantees the CURRENT
  state always evaluates (its own scoring is never caught, so genuine model
  errors still surface loudly at initialization or in the current-state
  refresh).

## Acceptance

Track 1 supplies the oracle: the same indicator-mixture model written with
`marginalize=:enumerate` and sampled by NUTS is exact up to MC error, so

1. a bernoulli/categorical indicator model sampled by `gibbs` (site NOT
   marginalized) must match the enumerated NUTS posterior of the continuous
   parameters, and the site's posterior mean must match the enumerated
   responsibilities;
2. a pure-discrete model (`z ~ poisson(3.0)` with a `normal(z, sigma)`
   observation, no continuous slots) must match the posterior computed by
   direct truncated enumeration of the poisson support;
3. a mixed model (continuous location plus a poisson count latent) recovers
   within MC tolerances against a long-run reference, with finite
   diagnostics and reproducible seeded runs.

## PR sequence

- **PR-1** — this document.
- **PR-2** — the sampler: site discovery (+ marginalized-site exclusion),
  symmetric proposals per family, the sweep loop composing the discrete
  pass with `_nuts_proposal`, warmup across sweeps, `GibbsChain`, the
  degenerate cases (no discrete sites / no continuous slots), and the
  acceptance tests above (inference shard).
- **PR-3** — polish: docs/dsl.md user section, per-site ess/rhat surfacing,
  the optional heavier-tailed integer step, and SBC-based calibration of
  the combined kernel.

## Non-goals

- Trans-dimensional models: when the SET of Gibbs sites depends on any
  sampled quantity — a Gibbs site's own binding or a continuous parameter
  flowing into a loop bound or dynamic address around a site — moves would
  require missing choices or leave stale ones behind (reversible-jump
  territory). The sampler rejects such models at construction via a
  binding-taint walk over the plan.
- Batched/multi-chain Gibbs (single-chain first; the batched NUTS machinery
  conditions all columns on one constraint layout and would need per-column
  discrete states).
- Adaptive discrete proposals or per-site tuning beyond the fixed symmetric
  steps.
- Block updates of correlated discrete sites (single-site only).
- Device execution (the discrete pass is inherently sequential and cheap).

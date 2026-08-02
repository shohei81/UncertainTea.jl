# Constraint-Driven Latent/Observation Classification

Date: 2026-07-21
Status: design (issue #95); implementation staged over the PRs below.

## Motivation

Today a choice is classified as a *latent* (gets a dense parameter slot) or an
*observation* (value must be supplied through constraints) purely by its
**syntax** at macro-expansion time. `_parameterize_step` (`src/ir.jl`) grants a
parameter slot when a choice is *bound*, has a static address, is unscoped, and
carries a parameter transform; otherwise the choice has no slot and is treated
as an observation whose value is read from the `choicemap` constraints.

Binding is therefore the switch:

```julia
mu ~ normal(0.0, 1.0)          # bound  -> slot   -> latent
{:y} ~ normal(mu, 1.0)         # unbound -> no slot -> observation (must constrain)
y = ({:y} ~ normal(mu, 1.0))   # bound  -> slot   -> latent (even if the user
                               #                     meant to observe :y)
```

This produces the silent inconsistency reported in issue #95: `generate` and
`assess` condition on *any* constrained address (the runtime treats a choice as
observed iff its address is in the constraints), while `logjoint`, all gradient
paths, and every unconstrained-space inference API silently *ignore* a
constraint that targets a slotted (bound) address, scoring it as a free latent
instead. A constrained `y = ({:y} ~ ...)` gives a correct answer from
`generate`/`assess` and a wrong posterior from HMC/NUTS, with no error.

The root cause is that **whether an address is observed is a property of the
conditioning (which addresses are constrained), not of the syntax**. The
runtime already uses the conditioning rule; the compiled scoring layer uses the
syntactic rule; the two disagree.

## Decision

Adopt the conditioning rule everywhere (full Gen-style semantics):

> A random choice is an **observation** iff its address is present in the
> constraints supplied at inference time. Otherwise it is a **latent**.
> Binding (`x ~ dist` or `x = ({:a} ~ dist)`) is **orthogonal** to this
> classification: it only names the value for downstream use.

`generate`, `assess`, `logjoint`, gradients, batched, device, and the
diagnostics/predictive APIs all derive the latent/observation split from the
same rule, so they can never disagree again.

This is the largest of the three options considered (the others were: reject a
constraint on a bound address with an explicit error; or honor the constraint
only in the compiled path). It is chosen because it makes one rule true across
the whole system and matches the project goal of "observations supplied
externally through choicemap-style constraints" (`docs/architecture.md`) rather
than an `@observe`-centric language.

## Semantics

For a choice site `c` with address `a`, distribution `d`, optional binding `b`:

- **Observed** (`a` in constraints): the constraint value `v` is scored
  (`logpdf(d, v)` in constrained space, plus the loop/broadcast accounting that
  already exists) and, if `c` is bound, `b = v` is made available to downstream
  expressions. No parameter slot is allocated for `c`.
- **Latent** (`a` not in constraints): `c` occupies its parameter slot; the
  unconstrained parameter is transformed to the constrained value `theta`
  (existing slot transforms / Jacobian accounting), `logpdf(d, theta)` is scored
  as a prior term, and, if `c` is bound, `b = theta` is available downstream. An
  **unbound latent** is a perfectly valid sampled quantity that simply is not
  named locally (e.g. a prior-only draw); it still gets a slot.

Binding's only remaining job is downstream value resolution, and it resolves to
the observed value or the latent value depending on the classification above.

### The conditioning signature

The consequence for the dense/static layout is explicit and central to this
design:

> The parameter layout is a function of the **conditioning signature** -- the
> set of constrained addresses -- not of the model alone.

This is not a regression of the static/GPU-first direction; it is the honest
statement of it. You cannot know the parameter vector until you know what is
observed, exactly as Stan cannot build its parameter block without the data
block. **Once the conditioning signature is fixed, the layout is fully static,
dense, and GPU-friendly** -- identical in kind to today's layout. We therefore:

- compute `ParameterLayout` (and the compiled execution/lowering plans) as a
  function of `(model, conditioning_signature)`, where `conditioning_signature`
  is the canonicalized set of constrained addresses (address identities only,
  not values);
- memoize by `(model, conditioning_signature)` so repeated inference with the
  same conditioning pays the classification/layout cost once, preserving the
  "compile once, run dense" property the batched/device paths rely on.

Values never enter the signature, so re-running with new data at the same
observed addresses reuses the cached layout and compiled plan.

## Design

### 1. Classification and parameter layout (`src/ir.jl`)

`_parameterize_step` no longer decides latent/observed from `binding`. Instead
the parameterization pass takes the conditioning signature and, for each choice
step:

- if the address is in the signature -> observation step (no slot);
- else -> latent step (allocate a slot). This requires every choice site,
  including today's unbound `{:a} ~ dist`, to be able to carry a parameter
  transform. Slots are keyed by **choice index / address**, not by binding, so
  unbound latents get slots too. `reparam=:noncentered` eligibility keeps its
  existing structural requirements (static address, unscoped) but is evaluated
  against the latent set for the current signature.

`ParameterLayout`, `parametercount`, `parametervaluecount`, and the
slot-lookup helpers become results of `(model, signature)`.

### 2. Value resolution in the evaluator (`src/evaluator.jl`)

The environment fold resolves a bound symbol to:

- the constraint value when the site is observed, or
- the transformed parameter value when the site is latent.

Both branches already exist in pieces (bound latents resolve from params today;
observations-used-downstream is the gap). Unify them behind the classification
so downstream deterministic steps, `reparam=:noncentered` loc/scale
expressions, and loop bodies all see the right value regardless of
observed/latent status. (This subsumes part of issue #100, whose eager-eval
fragility lives in the same walk -- see Interactions.)

### 3. Observation staging and device lowering

Device staging (`src/device/staging.jl`) and `device_lowering_report`
(`src/device/plan/lowering.jl`) already consume `(args, constraints)`; they switch from
"a step is observed iff it has no parameter slot" to "a step is observed iff its
address is in the signature", which is the same source of truth the CPU layout
now uses. The dense observed matrix and the latent slot spans are both derived
from the signature-specific layout, so CPU/backend/device stay consistent by
construction.

### 4. Public API surface

APIs that take a raw parameter vector (`logjoint`, `logjoint_unconstrained`,
gradients, `transform_to_constrained`, batched/device entry points) validate
the vector length against the **signature-specific** `parametercount`. The
length now depends on what is constrained; the error message names the
conditioning signature so a mismatch is self-explanatory.
`observation_addresses` returns exactly the constrained-and-present choice
addresses.

## Interactions

- **#100 (noncentered eager-eval):** the unified value resolution in the
  evaluator walk is the right home for the reachability fix; #100 should land
  *after* this change and reuse the same walk. Until then #100 stays a separate
  targeted fix.
- **#88 (pointwise likelihood vs marginalized latents):** pointwise-likelihood
  observation classification must use this rule (constrained-and-present), and
  a `marginalize=:enumerate` latent is, by definition, never an observation.
  #88's correct-marginal implementation is built on top of this classification;
  sequence it after.
- **marginalize=:enumerate:** an enumerated discrete site is a latent that is
  summed out; it is never in the observed set. Its slotless treatment is
  unchanged, only re-expressed against the signature.
- **Vector latents (`iid`, `dirichlet`, diagonal `mvnormal`), loops, and
  scoped/tuple addresses:** a scoped/loop family conditioned only in part
  remains unsupported and must error explicitly (partial conditioning of a
  synchronized address span is out of scope for v1 -- see Non-goals). Fully
  observed or fully latent spans behave per the rule.
- **Broadcast observations:** unchanged; a dot-call observation constrained with
  one dense vector value stays a single observed span.

## Non-goals (v1)

- Partial conditioning of a synchronized/loop address span (some indices
  observed, others latent in one `{:y => i}` family). Detect and reject with a
  clear message; revisit later.
- Changing `generate`/`assess` runtime semantics -- they already implement the
  target rule; this work makes the compiled paths match them, not the reverse.
- Dynamic-trace / non-static control flow, which remains rejected in static
  models (issue #68).

## Behavior changes and migration

Two observable changes; both replace a wrong or error result with the intended
one, so no correct program regresses in its answer:

1. **Constraining a bound address now conditions on it** everywhere (was:
   silently ignored by compiled scoring). This fixes #95's wrong posterior.
2. **An unbound `{:a} ~ dist` left unconstrained is now a latent** (was: an
   error demanding a constraint). Programs that previously *required* a
   constraint on such a site still work; programs that previously errored now
   sample it.

Parameter-vector length becomes conditioning-dependent. Existing calls that
pass a full parameter vector for a fixed conditioning are unaffected; calls that
relied on the old syntactic split (e.g. binding an observation to reuse its
value while also constraining it) now get the honored-constraint behavior. The
migration note goes in `docs/dsl.md` and `docs/architecture.md`.

## Implementation plan (staged PRs)

1. **PR-1 (this doc).** Design + the conditioning-signature contract.
2. **PR-2:** signature-keyed `ParameterLayout` and classification in `src/ir.jl`,
   with memoization by `(model, signature)`; CPU `logjoint`/`assess`/`generate`
   made to agree on the #95 repro. Unbound-latent slot support.
3. **PR-3:** unified value resolution in the evaluator (observed-used-downstream
   and latent-used-downstream through one path); scalar gradients.
4. **PR-4:** batched + device layout/staging keyed off the signature; CPU vs
   backend vs device parity on mixed observed/latent models.
5. **PR-5:** API length validation with signature-aware messages;
   `observation_addresses` and predictive/diagnostics alignment; docs.

Each PR carries the regression tests below and keeps the full suite green.

## Testing

- The #95 repro: `y = ({:y} ~ normal(mu, 1.0))` with `:y` constrained gives the
  same conditioned posterior from `generate`/`assess`, `logjoint`, and HMC/NUTS
  (closed-form check on a conjugate model, not just agreement between paths).
- Same model, `:y` unconstrained: `y` is a latent; `parametercount` includes it;
  `logjoint` scores its prior. Constrained vs unconstrained differ by exactly
  the observation's log-density contribution.
- An unbound `{:a} ~ dist` left unconstrained samples/scores as a latent.
- CPU/backend/device parity on a model with the same address observed in one run
  and latent in another (two conditioning signatures, two cached layouts).
- Length-mismatch error names the signature.
- Signature memoization: repeated inference with new data at the same observed
  addresses reuses the compiled plan (no recompute).

## Open questions

- Canonical form of the conditioning signature (sorted address tuples vs a
  hashed set) and the memoization cache's eviction policy.
- Whether to expose a way to *pin* a conditioning signature (precompile a layout
  for a known observation set) for latency-sensitive callers.
- Error vs. best-effort for partial conditioning of a synchronized span -- v1
  errors; is there a real use case that justifies supporting it later?

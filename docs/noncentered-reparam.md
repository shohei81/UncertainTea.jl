# Non-Centered Reparameterization with Address Preservation

Date: 2026-07-09
Status: design (issue #19); implementation staged over PRs 2-6 below.

## Motivation

Hierarchical models written naturally (centered: `theta ~ normal(mu, tau)`
with latent `mu`, `tau`) produce funnel geometry that defeats HMC/NUTS at
small data sizes. Experienced users rewrite these non-centered by hand
(`z ~ normal(0, 1); theta = mu + tau * z`), which obfuscates the model and
renames the quantity of interest. The goal:

```julia
theta ~ normal(mu, tau; reparam=:noncentered)
```

samples the standardized `z` under the hood while the user keeps seeing
`theta`: in traces, choicemaps, constrained samples, and summaries.

## Semantics

For an eligible latent choice with `reparam=:noncentered`:

- The NUTS/HMC **parameter slot holds `z`** (standardized, real line).
- The **trace, choicemap, constrained sample vector, and summaries hold
  `theta = loc + scale * z`** at the choice's own address -- exactly the
  contract the existing constrained-latent transforms already provide, only
  with a transform that depends on the other latents feeding `loc`/`scale`.
- `generate`/`assess` semantics are unchanged (the runtime path keeps drawing
  and scoring `theta ~ normal(loc, scale)` directly; only gradient-based
  geometry changes).
- The change of variables contributes `log|scale|` to the unconstrained
  logjoint, the same accounting slot transforms use today. Equivalently, a
  layer may score `logpdf(normal(0, 1), z)` directly and skip the Jacobian
  term; both forms are identical because
  `N(z; 0, 1) = N(theta; loc, scale) * scale`.

Eligible sites (v1):

- families: `normal`, `studentt`, `laplace` (real-line location-scale; the
  reparameterized draw stays exact). `lognormal` (affine in log space)
  is a v2 follow-up.
- shapes: a **static-address scalar latent**, or the base call of an `iid`
  vector latent (`theta ~ iid(normal(mu, tau), n; reparam=:noncentered)`,
  one n-wide `z` slot). Loop-addressed latents (`{:theta => i} ~ ...`) have
  no parameter slots at all today (ir.jl `_parameterize_step` skips scoped
  steps), so they are out of scope here as everywhere else.
- Observations and constrained choices reject the flag (`ArgumentError` at
  macro expansion where detectable, at model definition otherwise).

## Why this cannot be a new `AbstractParameterTransform`

The transform pipeline (src/parameters.jl) is a **pre-pass over the flat
parameter vector**: `transform_to_constrained*` loop over `layout.slots` and
each `_transform_slot_to_constrained!` reads only its own slot's entries --
no model, no environment, no other slots' constrained values. `loc`/`scale`
are arbitrary expressions over earlier latents that only materialize inside
the execution plan's environment. A dependent transform therefore has to run
**inside the plan walk**, where the choice step already has `loc`/`scale` in
scope in all four engines:

- scalar evaluator: `_score_plan_step!` evaluates argument expressions from
  `PlanEnvironment` right before scoring (evaluator.jl);
- batched backend: `_score_backend_step!` evaluates `step.mu`/`step.sigma`
  from `BatchedPlanEnvironment` (backend/scoring/continuous.jl);
- batched gradients: same expressions with slot-gradient propagation
  (batched/gradients/continuous.jl);
- device kernel: `_device_score_step` evaluates `step.mu`/`step.sigma` from
  `slots` and already applies the slot transform **inline at the step**
  (`_device_transform`, device/logjoint_kernel.jl) -- the shape the other
  layers move toward.

## Architecture

### IR and layout

- `DistributionSpec` gains `reparam::Symbol` (default `:centered`), following
  the `builder` field precedent: additive field plus a positional
  compatibility constructor, threaded through `_substitute_rhs` and every
  frontend `DistributionSpec(...)` emission site.
- `ParameterSlotSpec.transform` for a reparameterized site becomes a new
  marker type `NoncenteredTransform` (scalar) / `VectorNoncenteredTransform(n)`
  (iid): it declares "identity in unconstrained space, value materialized at
  the plan step". The layout also gets a cached
  `has_dependent_transforms::Bool` so every existing consumer keeps its
  current fast path when the feature is unused.

### Frontend

`_rhs_spec_expr` / `_rewrite_tea_expr` currently do not handle keyword
arguments on the RHS of `~` at all -- `normal(mu, tau; reparam=...)` parses
as an `Expr(:parameters, ...)` that would be spliced in as a bogus positional
argument. PR-2 adds explicit kwarg parsing:

- `reparam=:noncentered` (a literal `QuoteNode`) is recognized, stripped from
  the positional arguments, validated (eligible family, latent-shaped site),
  and stored on the emitted `DistributionSpec`.
- any other keyword on a distribution call raises a macro-time
  `ArgumentError` (turning today's silent corruption into an error is a
  robustness fix in its own right).
- the runtime body rewrite keeps the centered call (semantics of
  `generate` are unchanged), so only the spec/plan carry the flag.

### Transforms as plan walks

Two new internal entry points (parameters.jl + evaluator/noncentered_walk.jl), used only when
`has_dependent_transforms`:

- **forward** `_transform_to_constrained_via_plan!(dest, model, params)`:
  walk the execution plan evaluating deterministic steps into an environment;
  at each choice step compute the constrained value -- `loc + scale * z` for
  reparameterized sites (accumulating `log|scale|` for the
  `with_logabsdet` variant), the existing slot transform otherwise -- write
  it to `dest`, and bind it into the environment so later expressions see it.
  Slots are already stored in execution order (ir.jl
  `_parameterize_plan_steps`), so the walk visits dependencies first.
- **inverse** `_transform_to_unconstrained_via_plan!(dest, model, values)`:
  same walk; the full constrained vector is available up front, so `loc` and
  `scale` evaluate from already-known constrained bindings and
  `z = (theta - loc) / scale`.

`transform_to_constrained`, `transform_to_constrained_with_logabsdet`,
`transform_to_unconstrained`, `parameter_vector`, `parameterchoicemap`, and
`initialparameters` keep their signatures; they branch on
`has_dependent_transforms`. The call-site census (samplers, ADVI, SMC,
pathfinder, optimize, SBC, predictive) shows every caller passes either a
model+full-vector or a trace -- both sufficient for the plan walk -- so no
public API changes.

### Per-engine scoring

- **Scalar evaluator**: unchanged structurally -- `logjoint_unconstrained`
  already composes `transform_to_constrained_with_logabsdet` + constrained
  `logjoint`; the plan-walk transform supplies the dependent values and
  `log|scale|` terms. The ForwardDiff gradient differentiates through the
  walk unchanged.
- **Backend lowering**: a reparameterized normal site lowers its **value as
  a backend expression** `step.mu + step.sigma * SlotExpr(z)` bound to the
  binding slot, and its score as the existing normal machinery against
  literal `(0, 1)` on the raw slot value -- no Jacobian term needed in this
  form and, critically, **no new gradient math**: the expression evaluator
  already propagates slot gradients through arbitrary lowered expressions,
  so `d logp/dz`, `d/dmu`, `d/dsigma` all emerge from machinery the
  cross-check suite (#9) already exercises. The batched transform pre-pass in
  workspace.jl treats the slot as identity.
- **Device**: same lowering shape as the backend (value expression + standard
  normal score). `_device_transform` stays untouched; the reparameterized
  step is a new step struct whose value source is an expression rather than a
  raw `value_source` slot read.

### What stays consistent

- `batched_logjoint_unconstrained` == per-column `logjoint_unconstrained`
  (value parity) and analytic gradients == finite differences: both enforced
  automatically once the reparameterized forms register in
  `gradient_crosscheck.jl`.
- Posterior over `theta` is invariant between `reparam=:centered` and
  `:noncentered` -- checked by SBC (#18) on a funnel and by comparing NUTS
  posteriors against the manual `z`-plus-deterministic rewrite.

## PR sequence

1. **PR-1 (this document).**
2. **PR-2 -- scaffolding, no semantics**: RHS keyword parsing with honest
   errors; `DistributionSpec.reparam`; eligibility validation;
   `NoncenteredTransform` marker + `has_dependent_transforms`; every engine
   REJECTS the flag with a clear "not implemented yet" error. Small and safe.
3. **PR-3 -- CPU semantics**: plan-walk transforms; scalar evaluator support;
   `generate`/`assess`/`parameter_vector` round-trips; acceptance on the
   funnel and iid eight-schools via single-chain `nuts` (CPU gradients):
   near-zero divergences, posterior matches the manual non-centered rewrite,
   SBC passes; `backend_report` reports reparameterized models unsupported.
4. **PR-4 -- backend + batched**: expression-valued lowering, batched
   scoring/gradients, cross-check registration, `batched_nuts` funnel run.
5. **PR-5 -- device**: device step lowering + Metal smoke parity.
6. **PR-6 (optional)**: `reparam=:auto` heuristic (location-scale family with
   non-constant latent-referencing arguments), `lognormal` log-space variant.

## Risks and mitigations

- **Plan-walk transform cost**: only paid by models that opt in
  (`has_dependent_transforms` gate); centered models keep the slot loop.
- **Contract drift between engines**: PR-3 lands the CPU reference first;
  PR-4/5 are held to parity tests against it plus the cross-check suite,
  the same discipline as every backend/device family port.
- **RNG stream stability**: the runtime `generate` path is untouched, so
  existing seeded tests are unaffected until a model opts in.
- **Kwarg parsing regression risk**: PR-2 turns a currently-silent
  malformation into explicit errors; existing models cannot be using RHS
  kwargs successfully today, so this cannot break working code.

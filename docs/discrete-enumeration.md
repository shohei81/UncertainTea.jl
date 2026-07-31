# Automatic enumeration of finite-support discrete latents

Design for issue #13, track 1: marginalize `bernoulli` / `categorical`
latents out of the logjoint via logsumexp so HMC/NUTS sample only the
continuous parameters. Track 2 (MH-within-Gibbs for unbounded discrete
latents) is out of scope here and stays on issue #13.

**Update (issue #264): the annotation is now optional.** An *unobserved*
finite-support discrete latent (`bernoulli`, or `categorical` with a literal
probability vector) that carries no `marginalize=` keyword is auto-marginalized
by exactly the machinery below — there is no valid way to HMC a discrete latent,
so this replaces the "requires a provided value" error with the marginal density.
It is bounded by a support-product cap (`AUTO_MARGINALIZE_SUPPORT_LIMIT = 32`):
past the cap, auto-marginalization declines (all-or-nothing) and the choice stays
an error, so a model with many discrete latents cannot silently create an
intractable `K^n` enumeration. The explicit `marginalize=:enumerate` annotation
remains valid and is required only to opt a past-cap model in deliberately.

## Problem

Every sampler is gradient-based, and a discrete latent has no parameter
slot: `_supports_parameter_slot` returns `false` for discrete families
(frontend.jl), so `z ~ bernoulli(0.5)` is invisible to `parameter_vector`
and the compiled `logjoint` treats it as an implicit observation — it
throws `"requires a provided value for choice"` unless `z` is constrained
(evaluator.jl `_score_plan_step!`). Models with a mixture indicator or a
switch latent therefore have no posterior inference story at all; the
poison-value walk from the noncentered work already names "a discrete
latent" as the case it must reject.

`mixture` handles the one-site special case by marginalizing the component
index inside a single density. This design generalizes that to a *named*
discrete latent whose value is referenced by arbitrary downstream steps.

## API

```julia
@tea static function indicator_mixture()
    m1 ~ normal(-2.0f0, 1.0f0)
    m2 ~ normal(2.0f0, 1.0f0)
    z ~ bernoulli(0.5f0; marginalize=:enumerate)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5f0)
    return z
end
```

- New `~` RHS keyword `marginalize` with the single accepted value
  `:enumerate` (a literal QuoteNode, exactly like `reparam=`; the keyword
  namespace stays parse-or-error). Mirrors the `reparam` plumbing:
  `DistributionSpec` gains a `marginalize::Symbol` field defaulting to
  `:none`, the kwarg is stripped from the runtime body so `generate`
  still forward-samples `z`, and `_normalized_rhs_call` /
  `_reject_nested_reparam`-style guards keep the layout pre-pass and the
  emitted spec in sync.
- Eligible families: `bernoulli` (support `(false, true)`, K = 2) and
  `categorical` (support `1:K`). For `categorical` the *length* of the
  probability argument must be a macro-time literal (`Expr(:vect, ...)` /
  tuple), the same literal-shape rule as `lkjcholesky`'s dimension and
  `mixture`'s component count; the probability *values* stay free
  expressions (latent-dependent probabilities are differentiated through).
- Ineligible and rejected at macro time: any other family, loop-scoped
  addresses (`{:z => i} ~ ...` inside `for` — the plan-suffix semantics
  below do not extend into loop bodies in v1), and non-literal categorical
  lengths.

### Semantics per entry point

- `generate` / forward simulation: unchanged — `z` is sampled and lands in
  the trace like today (the kwarg is stripped from the runtime body).
- `logjoint` / `logjoint_unconstrained` / gradients / batched:
  if the constraints provide a value for `z`, the step scores that branch
  exactly as today (conditioning on `z` stays free); otherwise the step
  marginalizes — the returned density is the *marginal* over `z`'s
  support. `assess` is exempt: it scores the full joint of the choices it
  is given through the interpreted runtime, so it keeps requiring `z` like
  any slotless choice (its semantics are "density of these choices", which
  includes the discrete one). Since `z` never had a parameter slot, `parametercount`,
  `initialparameters`, and every sampler are untouched: NUTS samples the
  continuous vector against the marginalized density.
- Posterior samples *of* `z` (responsibility extraction / `infer_discrete`)
  are a non-goal for track 1; the responsibilities fall out of the
  logsumexp and can be surfaced later.

## Design

### Compiled CPU evaluator: marginalization as a fold over the suffix

`_score_compiled_steps` is a right fold — `score(step) + score(tail)` —
so a marginalized choice composes as a *continuation*:

```
score(marginalized z, tail) =
    logsumexp_v( logpdf(dist_z, v) + score(tail with env[z] = v) )
```

for `v` in the compile-time support. Concretely:

- `CompiledChoicePlanStep` gains a `marginalize` marker (compile-time,
  like `noncentered`); `_score_compiled_steps` branches when the head
  step carries it: for each support value, `_environment_set!` the
  binding slot, score the *entire remaining tuple* of steps, and combine
  with the max-shifted logsumexp from `MixtureDist.logpdf`; then
  `_environment_restore!`. The save/re-score/restore idiom already exists
  verbatim in the `CompiledLoopPlanStep` handler — the only new part is
  the combiner (logsumexp instead of `+`) and that the step owns the
  suffix, not a body.
- The probability arguments of `z`'s own distribution are evaluated from
  the environment once per support value (they may depend on earlier
  continuous latents).
- Multiple marginalized latents nest naturally through the recursion: the
  inner one is part of the outer one's suffix, giving exhaustive product
  enumeration (cost `K1 * K2 * ...` suffix evaluations — documented, and
  acceptable for the small indicator counts this targets).
- A `_has_marginalized_choices` gate (mirroring `_has_dependent_transforms`)
  keeps every existing model on the untouched fold.
- ForwardDiff differentiates through the walk for free (as it does through
  `_dependent_transform_walk!`), so the single-chain gradient cache and the
  per-column batched fallback work from day one of the CPU phase.
- Conditioning: the step first does `_choice_tryget_normalized`; a found
  value short-circuits to the existing single-branch scoring.

Cost note: a marginalized site multiplies the *suffix* cost by K, so
site placement matters (`z` late in the model is cheaper). This is
inherent to enumeration and is called out in the user docs rather than
optimized (no variable elimination in v1).

### Backend expression IR: a suffix-owning marginalize step

No existing backend construct expresses "score the remaining steps K
times under different bindings and logsumexp the totals" —
`BackendLoopPlanStep` has the save/iterate/restore shape but sums, and
the mixture step logsumexps a single self-contained density. So this is
new IR:

```julia
struct BackendMarginalizeChoicePlanStep{P<:Tuple,S<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int} # the enumerated latent's binding
    address::AD
    parameter_slot::Union{Nothing,Int}   # always nothing (no slot) — kept for the choice-step field convention
    probabilities::P                 # K numeric exprs (bernoulli: (1-p, p))
    body::S                          # the lowered plan SUFFIX
end
```

Subtyping `BackendChoicePlanStep` (there is no `BackendPlanStep` root —
the hierarchy is `AbstractBackendPlanStep` / `BackendChoicePlanStep`) is
deliberate: the device's generic choice fallback dispatches on
`BackendChoicePlanStep`, so the new step stays device-honest-unsupported
with no extra method, exactly like the lkjcholesky step (#49/#57).

- Lowering: on a `marginalize=:enumerate` choice, lower the remaining
  steps recursively into `body` and stop the outer loop — the plan's tail
  becomes the step's continuation, so nested marginalized latents nest as
  nested steps. Slot kinds follow the value's type on the CPU path:
  a `bernoulli` binding is marked *numeric* (`false/true` lower as `0/1`
  feeding arithmetic), but a `categorical` binding must stay an *index*
  slot — the branch value is an `Int` on the CPU path and existing index
  evaluators (loop bounds, `binomial` trials, address parts) require
  `Integer`, while index slots can still feed numeric expressions.
  Storing categories in `numeric_values` as floats would break integer
  consumers like `{:y} ~ binomial(z, p)`.
- Scalar and batched scoring: for each `v`, bind, `_score_backend_steps`
  the body into a per-branch total, restore; combine per column with the
  max-shifted logsumexp. K per-branch batched totals need K scratch
  vectors (K is compile-time; cap the *product* of nested supports at a
  documented constant, e.g. 32, with an honest lowering rejection above).
- Conditioning parity: like the CPU path, a constrained `z` short-circuits
  to scoring the body once with the provided value plus its own logpmf.

### Analytic batched gradient: responsibility-weighted branch gradients

With branch totals `t_v` and responsibilities
`r_v = exp(t_v - logsumexp(t))` (the mixture accumulator's softmax
weights), the gradient is

```
∇ logsumexp_v(t_v) = Σ_v r_v · ∇t_v
```

where `∇t_v` = the branch's own probability-expression gradient (through
`log p_v`) plus the suffix gradient under `env[z] = v`. Implementation:
score each branch into its own totals *and* gradients scratch (K
gradient-matrix scratches — K is small and compile-time), then combine.
This is `_accumulate_mixture_normal_gradient!` generalized from "K
densities" to "K plan suffixes"; the existing per-step gradient drivers
run unchanged inside each branch. Until this lands, the
`_backend_gradient_supported` gate simply excludes plans containing the
marginalize step, and gradients ride the per-column ForwardDiff fallback
(correct, just slower) — the same honest-tier story every other family
started with.

### Device

Deferred (a follow-up issue once the backend step exists, like the
lkjcholesky mirror #57): the kernel-side shape is a compile-time-unrolled
K-way branch fold with register-resident per-branch totals, but suffix
re-execution multiplies kernel body size by K, so the Metal compile-time
budget needs measuring first. The backend step keeps the device honestly
unsupported through the generic step fallback until then.

## Acceptance (issue #13)

A 2-component mixture written with an explicit indicator recovers the
same posterior as the equivalent `mixture` model via NUTS, without
user-side marginalization:

1. Density parity: `logjoint_unconstrained` of the indicator model equals
   the `mixture` model's at matched continuous parameters (exact, both
   are the same logsumexp).
2. Gradient parity: compiled ForwardDiff and (once the backend lands)
   analytic batched gradients against central finite differences.
3. Sampler parity: NUTS posterior moments of the continuous parameters
   agree between the two spellings within MC error.
4. Conditioning: providing `z` in the constraints reproduces today's
   joint scoring exactly.

## Phases

- **PR-1** — this document.
- **PR-2** — frontend scaffolding: `marginalize=` kwarg parse +
  `DistributionSpec.marginalize` + runtime-body strip + macro-time
  eligibility (family whitelist, literal categorical length, no loop
  scope) + honest rejections everywhere else (backend_report issue,
  poison-walk message update). No semantics change yet.
- **PR-3** — compiled CPU semantics: suffix-logsumexp in
  `_score_compiled_steps`, `_has_marginalized_choices` gate,
  conditioning short-circuit, acceptance tests 1/2(FD)/3/4 on the CPU
  path (batched rides the per-column fallback).
- **PR-4** — backend/batched scoring: `BackendMarginalizeChoicePlanStep`
  + lowering + scalar/batched logsumexp scoring; gradient stays on the
  fallback via the support gate.
- **PR-5** — analytic batched gradient (responsibility-weighted branch
  gradients) + crosscheck registration.
- **Follow-ups (separate issues)** — device mirror; responsibility
  extraction (`infer_discrete`-style posterior for `z`); track 2
  MH-within-Gibbs for unbounded supports.

## Pointwise log-likelihood (issue #88)

`pointwise_loglikelihood` / `observation_addresses` (and the WAIC / PSIS-LOO
model-comparison APIs that consume them) treat a `marginalize=:enumerate`
latent as what it is: a latent that is summed out, never an observation. For
each observation in the enumerated latent's suffix the recorded pointwise
value is the *marginal* likelihood, integrating the discrete latent out with
the same logsumexp the marginalized logjoint uses. For the repro
`z ~ bernoulli(0.3; marginalize=:enumerate); {:y} ~ normal(mu + z, 1)` the
recorded value for `:y` is `log(0.7 * N(y; mu, 1) + 0.3 * N(y; mu + 1, 1))`.

Concretely, writing the observations in the latent's suffix as `y_k`,

```
p(data | theta) = sum_z p(z) * prod_k p(y_k | z)
                = [prod_{k in I} p(y_k)] * sum_z p(z) prod_{k in D} p(y_k | z)
```

where `I` are the observations whose density does not depend on `z` and `D`
are those that do. The z-independent factors decompose per observation
directly; the dependent block is one joint term over all of `D` and
decomposes into per-observation pointwise terms only when `|D| <= 1`.

**Supported (the pointwise marginal factorizes):**

- An enumerated latent local to a single observation (`|D| == 1`): that
  observation gets `log sum_z p(z) p(y | z)`; any z-independent observations
  are recorded as their own logpdf. This is the common case.
- Independent enumerated latents each feeding their own observation: the
  columns are conditionally independent, so each factorizes.
- Several enumerated latents all feeding a *single* observation (nested /
  product enumeration): that one column gets its full multi-sum marginal.
- A *constrained* enumerated latent: it is an ordinary fixed observation,
  recorded and scored once at the provided value.

**Rejected with a specific error (genuinely non-factorizing):**

- One enumerated latent shared across two or more observations that both
  depend on it (`|D| >= 2`): `log sum_z p(z) prod_i p(y_i | z)` does not
  factorize into independent per-observation terms.
- A free (slotted) continuous latent inside the enumeration suffix: its prior
  may depend on the enumerated value and reweight the enumeration in a way the
  per-observation decomposition cannot represent. Place the enumerated site
  *after* all continuous latents so its suffix scores only observations.
- An enumerated latent whose value changes the *set* of downstream
  observations.

The classification uses the constraint-driven rule
(`docs/constraint-driven-conditioning.md`): the enumerated latent is slotless
and unconstrained, hence never in the observed set.

## Non-goals

- Variable elimination / smarter enumeration orders (NumPyro-style):
  v1 is exhaustive product enumeration with the cost documented.
- Loop-scoped marginalized latents (`{:z => i} ~ ...` in `for`): the
  suffix semantics inside loop bodies (remaining iterations belong to the
  suffix) need their own design; rejected at macro time in v1.
- Unbounded discrete latents (poisson, geometric, ...): track 2.
- Posterior samples of the marginalized latent (future follow-up).
- Device lowering of the marginalize step (follow-up after PR-4/5).

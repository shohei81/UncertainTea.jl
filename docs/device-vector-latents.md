# Device backend: vector latents design (issue #12, group 3)

Design for lowering vector-latent and vector-observation families — `mvnormal`
(diagonal), `dirichlet`, `mvnormaldense` — onto the device (KernelAbstractions)
backend. This is the plan-layout extension that `docs/device-backend.md` lists
as the known limitation. Written before implementation; each phase below is one
PR.

## Current state (what blocks vector latents today)

Three independent walls, from the bottom up:

1. **Device value source is one scalar row.** A device choice step carries
   `value_source::Int32` — `> 0` means "read ONE unconstrained parameter row and
   apply ONE scalar transform code" (`Identity`/`Log`/`Logit`), `< 0` means
   "read ONE staged observation row". `_device_choice_value_source`
   (src/device/plan/lowering.jl) rejects `slot.value_length != 1 || slot.dimension != 1`,
   and `_device_transform_code` rejects every vector transform.
2. **Device staging emits one observation row per choice.**
   `_stage_observed_row!` (src/device/staging.jl) resolves observations through
   the scalar `_batched_choice_numeric_values!`, whose
   `_batched_backend_observed_value` throws on any non-`Real` constraint. A
   vector observation never becomes rows.
3. **Backend rejects latents that FOLLOW a dimension-changing latent**
   (issue #36). Scalar backend steps carry a single `parameter_slot` field used
   both as the constrained VALUE row and as the gradient SEED row
   (`_backend_scalar_parameter_row`, src/backend/lowering/core.jl). The two
   agree only while every preceding slot has `dimension == value_length`, so a
   simplex (K−1 → K) or Cholesky (d(d−1)/2 → d(d+1)/2) latent placed before any
   other latent forces a fallback today — on the CPU batched path as well as
   the device path.

Facts the design leans on (verified against main):

- `ParameterSlotSpec` already carries everything lowering needs: `index` +
  `dimension` (unconstrained rows) and `value_index` + `value_length`
  (constrained rows), with `parameterindices`/`parametervalueindices` accessors.
- Backend vector steps pass vector arguments as **tuples of per-element
  expressions with statically known arity** (`mu::Tuple`, `alpha::Tuple`), and
  their lowering already validates `slot.value_length == length(tuple)`. The
  device categorical step proved the matching device pattern: an `NTuple` of
  device expressions evaluated by a compile-time `_device_eval_args` walk.
- The CPU vector transforms to mirror are `_to_constrained_simplex!` +
  `_simplex_logabsdet` (shifted softmax with an implicit last logit of 0;
  log-abs-det `Σᵢ log pᵢ` over all K entries) and the `VectorIdentityTransform`
  pass-through (log-abs-det 0). Both are branch-and-arithmetic only —
  device-safe as they stand.
- `lkjcholesky` had **no backend lowering, scoring, or gradient at all** when
  this was designed; the device cannot get ahead of the backend, so it was out
  of scope here (see Non-goals). Backend-native support has since landed
  (issue #49); the device mirror remains open.

## Design

### 1. Compile-time dimension, register-resident transform

Vector dimensions are static by construction (backend lowering fixed the tuple
arity, and the frontend required static sizes to create the slot). So the
device step encodes the dimension **in the type**, and the kernel materializes
the constrained vector as an `NTuple` in registers — no slots-matrix rows, no
extra device buffers, no isbits violations:

```julia
struct DeviceMvNormalChoiceStep{D,M<:Tuple,S<:Tuple} <: AbstractDeviceChoiceStep
    mu::M                       # NTuple{D} of device exprs
    sigma::S                    # NTuple{D} of device exprs
    value_source::Int32         # > 0: first unconstrained row; < 0: observed
    binding_slot::Int32         # 0 only in phase 1 (vector bindings rejected)
end

struct DeviceDirichletChoiceStep{K,A<:Tuple} <: AbstractDeviceChoiceStep
    alpha::A                    # NTuple{K} of device exprs
    value_source::Int32         # > 0: first unconstrained row (K-1 rows read)
    binding_slot::Int32
end
```

There is no per-step scalar `transform::Int32` — the family implies the
transform (`mvnormal` ⇒ VectorIdentity, `dirichlet` ⇒ Simplex), so the kernel
calls the matching constrain function directly:

```julia
# math.jl — mirrors _to_constrained_simplex! / _simplex_logabsdet exactly
_device_simplex_constrain(z::NTuple{Km1,T}) -> (p::NTuple{K,T}, logabsdet::T)
```

The value-resolution contract generalizes from "(value, logabsdet, cursor)" to
the same tuple with `value::NTuple{N,T}`:

- **latent**: read rows `value_source .. value_source + dimension - 1` of
  `params`, constrain in registers, return the K-tuple and its log-abs-det.
- **observed**: read `value_length` consecutive staged rows, advance the cursor
  by `value_length`, log-abs-det zero.

The scoring step then folds the closed form over the tuple exactly like the
CPU: mvnormal is a compile-time-unrolled sum of scalar normal logpdfs;
dirichlet is `loggamma(Σα) − Σ loggamma(αᵢ) + Σ (αᵢ−1)·log(pᵢ)` (the simplex
validity check is only needed on the observed path — a constrained latent is
on-simplex by construction).

### 2. Vector observations in staging

`_stage_step!` gains a vector branch keyed off the backend step type (the
vector steps carry `value_length`): resolve the observation through the
existing `_batched_choice_vector_values!` fan-out (the batched CPU helper that
already scatters a Tuple/Vector constraint into per-component buffers) and push
`value_length` rows in component order. The kernel-side cursor advances by the
step's compile-time dimension, so pre-order alignment is preserved by
construction — same invariant as today, just with a stride.

### 3. Gradient seeding for vector rows

The dual value resolution seeds component `i` of a latent vector with
derivative `ifelse(value_source + i - 1 == pidx, 1, 0)` — the direct
generalization of the scalar seed. The constrain functions are written
`where {T}` like all device math, so forward-mode duals flow through the
softmax/log-abs-det arithmetic and reproduce the CPU analytic gradients
(the same mechanism groups 1–2 validated at 1e-10 parity; the simplex
Jacobian is smooth arithmetic, no special dual rules needed).

`promote` over heterogeneous tuples follows the categorical precedent: each
component promotes independently inside the fold, no whole-tuple promote.

### 4. What stays rejected (honest report messages)

- **Reads of a vector binding** (`theta ~ dirichlet(...)` then `theta` used in
  a later expression): the slots matrix holds one scalar per symbol, and CPU
  vector bindings live in generic per-column storage the kernel does not model.
  Note every latent slot is symbol-bound by construction (the frontend only
  creates parameter slots for bound choices), so the vector steps MUST accept a
  non-zero `binding_slot` — they simply never write it. The read/write slot
  audit then polices downstream reads, but NOT for free: today
  `_device_collect_written_slots!` counts EVERY lowered step with
  `binding_slot > 0` as written, so the implementation must teach it that
  vector choice steps do not materialize their binding (e.g. dispatch on the
  vector step types, or an `writes_binding(step)` trait). With that one audit
  change, a downstream read is rejected with the audit's honest
  "not materialized on the device" message, while a latent that is only scored
  (the common case) lowers fine. Materializing vector bindings later means
  reserving `value_length` slots-matrix rows per binding (a self-contained
  follow-up).
- **Dimension caps.** Group 2 taught us fused-kernel compile time is a real
  budget (multiple inlined loop bodies stalled MTLCompilerService for minutes).
  Compile-time-unrolled tuple folds scale the kernel body by K per step, and
  the 2D gradient kernel multiplies that again. Lowering rejects vector steps
  above a conservative cap (start at `K ≤ 16`; revisit against measured Metal
  compile times) with a message naming the cap.
- **`lkjcholesky`** (backend lowering/scoring/gradients landed with issue #49;
  the device mirror landed with issue #57 — a register-resident tanh/stick
  constrain sharing the dense dimension cap) and
  **`mixture`** weights-as-latents (already backend-rejected). Note the mixture VALUE is scalar: lowering
  `BackendMixtureNormalChoicePlanStep` to the device is a group-1-shaped
  addition (a weights/mus/sigmas tuple step with a log-sum-exp fold), *not* a
  vector-latent problem — it can ship independently any time.

## Phasing (one PR each)

**Phase 0 — finish issue #36 on the backend (prerequisite for general mixing).**
Split the conflated scalar `parameter_slot` into `(value_row, seed_row)` — the
two-field shape the dirichlet step already has — set them from
`(slot.value_index, slot.index)`, drop the `index != value_index` rejection in
`_backend_scalar_parameter_row`, and update `_fill_choice_gradient!` callers to
seed with `seed_row`. The device's `findfirst(s -> s.value_index == ...)` slot
recovery moves to matching on the explicit pair. This unblocks "scalar latent
after a simplex" on the CPU batched path and is independently valuable; the
crosscheck suite already has the mixed-model fixture to flip on.

**Phase 1 — `mvnormal` (diagonal).** Dimension-preserving
(`dimension == value_length`), so it composes with everything today **without
Phase 0**. Delivers: the `NTuple` step + register transform machinery, vector
observed staging, vector gradient seeding, `dev_`/`devg_` parity + Metal smoke.
Smallest end-to-end slice of every new mechanism.

**Phase 2 — `dirichlet`.** Adds the simplex constrain/log-abs-det device math
and the first `dimension != value_length` stride. Ships after Phase 0 (general
placement) or before it with a lowering guard that the dirichlet slot is the
LAST latent (the only placement the backend accepts today anyway — a
dimension-changing latent already rejects everything after it).

**Phase 3 — `mvnormaldense`.** The open question is where the constant
`scale_tril` lives. Options: (a) pack the lower triangle into the step as an
`NTuple{d(d+1)/2}` of literals when the slot's value is statically known —
simplest, but the matrix usually arrives as model data, not literals; (b) stage
the matrix into a dedicated device buffer at workspace construction (the
argument-staging precedent from issue #38) and give the step an offset —
handles data matrices, costs a new buffer + plumbing. Decide by what the
existing tests/models actually pass for `scale_tril`; (b) is the likely
answer. Forward substitution is a d² compile-time-unrolled fold — the K-cap
applies with d² weight.

## Acceptance (mirrors groups 1–2)

Per phase: `device_lowering_report` stays honest for everything not yet
supported; `dev_`/`devg_` value+gradient parity vs the CPU batched path
(Float64 rtol 1e-12 / 1e-10, Float32 tolerance) including observed AND latent
forms; out-of-support observed vectors → −Inf on both paths; Metal smoke run
locally, with compile time watched (the group-2 lesson: a slow
MTLCompilerService is a finding, not an inconvenience).

## Non-goals

- `lkjcholesky` device support (the backend prerequisite landed with issue
  #49; the device mirror has since landed with issue #57).
- Vector `binding_slot` materialization in the kernel slots matrix.
- Dynamic (runtime-sized) vector dimensions — everything here rides on the
  static sizes the frontend already requires for slot creation.
- On-device warmup adaptation / `per_chain_adaptation=true` (tracked in
  docs/device-backend.md as existing follow-ups).

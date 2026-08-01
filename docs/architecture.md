# Architecture Direction

Date: 2026-03-10

## Core Thesis

UncertainTea should look much closer to Gen on the surface than the previous draft did.
The key idea is:

- surface syntax close to official Gen
- implementation strategy centered on a static GPU-friendly IR

In other words, the user-facing model should feel like a generative-function system,
while the compiled runtime behaves more like a static log-density compiler.

## Surface Syntax Direction

The frontend should mirror official Gen patterns wherever practical:

- `@tea` or `@tea (static)` for model definitions
- tilde syntax for random choices
- explicit addresses with `{...}`
- hierarchical addresses with `=>`
- observations supplied externally through `choicemap`-style constraints

This is a better fit for the stated goal than an `@observe`-centric DSL.

## Architectural Layers

### 1. Frontend

Responsibilities:

- parse `@tea` and `@tea (static)` definitions
- resolve tilde expressions and choice addresses
- distinguish the static GPU subset from future dynamic features
- lower surface syntax into an internal execution plan

Design rules:

- stay close to Gen terminology: generative function, choice, trace, choicemap
- reject or reroute unsupported dynamic behavior early
- keep a clear split between "Gen-like syntax" and "GPU-compatible semantics"

### 2. Static IR

Minimum internal components:

- `ModelSpec`: static specification of a compiled model
- `ChoiceSpec`: metadata for each random choice
- `AddressSpec`: normalized address representation
- `ParameterLayout`: dense unconstrained parameter layout
- `ExecutionPlan`: compiled `logjoint`, replay, and initialization plan

Core invariants:

- the set and order of choices are fixed for a compiled execution plan
- each choice has a fixed shape
- each address is statically enumerable
- the hot path can be evaluated with dense arrays and backend-friendly control flow

### 3. Backends

CPU backend:

- reference implementation
- correctness oracle
- fallback path for unsupported dynamic behavior

GPU backend:

- start with `CUDA.jl` and `Metal.jl`
- use high-level array operations first
- use `KernelAbstractions.jl` or backend-specific kernels only for hotspots
- make chain, particle, and minibatch dimensions first-class
- lower the static execution plan into a symbolic primitive-only backend plan

Current backend-lowering subset:

- distribution families: `normal`, `lognormal`, `laplace`, `exponential`,
  `gamma`, `inversegamma`, `weibull`, `beta`, `bernoulli`, `binomial`,
  `geometric`, `negativebinomial`, `poisson`, `studentt`, `categorical`
- built-in CPU/reference distributions also include `dirichlet` and a
  restricted diagonal `mvnormal`; both use vector-valued latent slots with
  flattened constrained/unconstrained parameter spans, and both now lower to
  the backend-native subset for their restricted static forms
- primitive calls: `:`, `=>`, `+`, `-`, `*`, `/`, `^`, `%`, `exp`, `log`,
  `log1p`, `sqrt`, `abs`, `min`, `max`, `clamp`, and (issue #286) `sin`, `cos`,
  `tan`, `tanh`, `sinh`, `cosh`, `atan`, `expm1`
- batched backend execution assumes synchronized loop iterables across the batch
- backend environments separate numeric slots, index slots, and generic slots
- backend score evaluation uses direct family kernels for supported distributions
- address and loop-index evaluation use an integer-specific backend path
- distribution arguments and numeric deterministic assignments use a dedicated
  numeric backend evaluator instead of the generic expression path
- supported distribution choices lower to family-specific backend plan nodes
  rather than a symbol-dispatched generic choice node
- compiled and backend evaluators now build concrete addresses through tuple
  recursion and use normalized direct `ChoiceMap` lookup on the hot path
- `ChoiceMap` now maintains an address index so normalized lookup no longer
  scans linearly through stored entries
- synchronized backend loops with a single observed choice whose address depends
  only on the loop iterator now use a loop-local fast path instead of the
  generic body scorer
- batched backend workspaces now reuse their environment and temporary buffers
  across repeated evaluations with the same batch shape
- batched gradient caches now first try a plan-aware backend gradient evaluator
  for the differentiable backend subset and only fall back to the flat
  `ForwardDiff` objective or the older column-wise cache when that subset does
  not apply
 - the differentiable backend path now covers `normal`, `lognormal`,
  `laplace`, `exponential`, `gamma`, `inversegamma`, `weibull`, `beta`,
  `studentt`, observed `bernoulli`, observed `binomial`, observed
  `geometric`, observed `negativebinomial`, observed `categorical`,
  observed `poisson`, and numeric deterministic assignments
- vector-valued latent slots are now supported in the CPU reference path,
  including simplex-valued `dirichlet` latents with flattened constrained
  sample storage and lower-dimensional unconstrained inference state, and
  diagonal `mvnormal` latents with equal-dimensional constrained and
  unconstrained storage
- backend-native vector support has now started with restricted diagonal
  `mvnormal` and simplex-valued `dirichlet`; both the scoring path and the
  batched manual backend gradient path are native for the current vector
  subset
- the differentiable backend primitive subset now includes `abs`, `min`,
  `max`, `clamp`, `%` with a literal divisor, and `^` with a literal exponent
  in addition to the earlier arithmetic and log/exp primitives; issue #286
  added the trig/hyperbolic set (`sin`, `cos`, `tan`, `tanh`, `sinh`, `cosh`,
  `atan`, `expm1`) with hand-derived chain rules and device dual rules, so link
  functions and periodic features stay on the fast paths
- batched HMC now keeps a sampler workspace for momentum, proposals,
  diagnostics, and constrained-position scratch so repeated sampling no longer
  rebuilds those buffers every iteration
- batched HMC now also carries the current gradient state across accepted
  proposals and uses a combined value+gradient path at the final leapfrog
  position when the backend-plan-aware gradient evaluator is available
- batched HMC warmup now mirrors the single-chain structure more closely:
  dual-averaging step-size adaptation, optional reasonable-step-size search,
  divergence-aware acceptance aggregation, and windowed shared diagonal mass
  adaptation from pooled non-divergent chain positions using winsorized online
  variance updates whose clip threshold anneals from a loose early-window value
  to a tighter steady-state value, with acceptance-aware weights so rejected
  states do not count like fully refreshed samples; both schedules are now
  explicitly tied to the current slow-window length, and each slow-window
  update is surfaced as a diagnostic summary after sampling and carried into
  the high-level `summarize` output, including a richer plain-text display for
  interactive inspection
- batched NUTS now exists as a CPU reference state machine with `param x chain`
  storage and pooled warmup adaptation; the first trajectory doubling step now
  uses a batched leapfrog-and-selection step, and its result now initializes
  each chain's reusable continuation state directly; single-chain `nuts`
  initializes its continuation state through the same first-step helper, and
  both paths now seed that transition from the same reusable subtree scratch
  states; on the batched path, continuation frontier/proposal vectors and
  continuation control metadata now live in batched workspace buffers, and the
  per-chain subtree scratch itself is now view-backed by those batch-owned
  matrices; per-chain gradient caches on that path also now target
  workspace-backed gradient columns, and they reuse a shared
  `ForwardDiff` objective/config when batch inputs are homogeneous; continuation
  subtrees now also run through the batched leapfrog and batched value+gradient
  path in same-depth cohorts before falling back, with subtree energy,
  acceptance, and log-weight bookkeeping now stored in batch-owned scratch, and
  the continuation merge path now keeps proposal-selection/log-weight scratch
  there as well; turning checks on that path now also run through batched
  helpers over the workspace matrices, with frontier/proposal column copies now
  routed through masked matrix-copy helpers, and subtree start-state
  initialization now uses that same masked-copy path instead of per-chain state
  cloning; logjoint values for those batched subtree/continuation states now
  also have workspace-owned vector storage with explicit sync helpers; the
  first-step batched NUTS initialization now likewise loads `current/next`
  tree states through batched helpers and restores rejected/divergent proposal
  columns through masked batch copies before the remaining chain-local
  selection logic continues; final batched proposal summary statistics now also
  run through batch helpers over workspace vectors rather than a per-chain
  epilogue loop, and the single-chain NUTS path now uses the same acceptance /
  energy / moved-summary helper logic, with continuation proposal energy and
  energy-error values now carried as state on both paths; the remaining
  chain-local batched fallback now also records subtree summary and selected
  subtree proposal energy/error in workspace vectors before continuation merge,
  and its frontier/proposal copies plus turning checks now also go through
  one-chain masked batch helpers; the single-chain subtree builder now also
  carries its metadata through a reusable scratch summary object attached to
  `NUTSSubtreeWorkspace`, and scalar continuation merge now consumes that
  scratch through a dedicated subtree-merge helper while scalar frontier /
  proposal copies and turning updates also run through dedicated helpers, and
  direction sampling / active checks / subtree-start selection are now shared
  helper concepts across scalar and batched continuation loops; the batched
  depth-cohort scheduler now also runs through explicit reset / select /
  activate / initialize / advance / merge helpers, with continuation gating
  masks and selected cohort depth/count now stored in a dedicated scheduler
  state object on the batched workspace, and the scheduler itself now advances through explicit
  `idle/expand/merge/done` phases with a remaining-step counter; accepted /
  divergent flags, step directions, tree depths, and integration counters now
  also live in a dedicated batched NUTS control object instead of as top-level
  workspace fields, and that control object now lowers to a small
  `IdleIR`/`ExpandIR`/`MergeIR`/`DoneIR` snapshot that the scheduler step can
  dispatch through directly, with `ExpandIR`/`MergeIR` reloading the active
  control masks they need from the IR payload itself through an intermediate
  executable control-block layer, and then through a 1-step descriptor that
  names the phase-local scratch masks touched by expand/merge; a final
  step-state object then bundles that descriptor with the numeric subtree
  energy/log-weight scratch for the current batched scheduler step, and a
  kernel-frame object then makes the concrete matrix/vector buffers for that
  step explicit as well; the current CPU path now also flattens that frame
  into a phase-local kernel-access object whose fields directly expose the
  buffers touched by each step, and wraps that access layer in a small kernel
  program with a fixed per-phase op sequence that the scheduler step executes
  directly. (An earlier descriptive lowering chain — dataflow descriptors,
  schedules, resource/barrier plans, and the string-emission GPU
  bundle/package substrate built on top of them — has been removed; the live
  GPU path is the KernelAbstractions device backend, see
  [device-backend.md](device-backend.md).) The
  remaining chain-local subtree builder also reuses a per-chain
  current/next/left/right/proposal scratch workspace, but deeper tree growth
  is still performed chain-by-chain rather than through a backend-lowered
  batched tree kernel, and subtree expansion now returns metadata while leaving
  mutable frontier state in that scratch workspace; the continuation loop
  itself now mutates an explicit reusable continuation-state object
- NUTS trajectory selection now follows canonical multinomial NUTS (Stan
  `base_nuts`) semantics on every path (scalar, batched hybrid/masked, tempered
  SMC moves, device masked): a final doubling whose subtree contains an
  internal U-turn or divergence is discarded whole -- no weight merge, no
  proposal swap, no frontier update -- keeping only leapfrog/accept-stat
  accounting and the termination flags; U-turn checks (frontier, batched, and
  dyadic checkpoint) are metric-aware, projecting momenta through the inverse
  mass matrix (`M^{-1} p` velocities) for diagonal and dense metrics; and the
  shared batched reasonable-step-size search keeps its initially chosen
  direction so it stops at the first crossing of the 0.5 acceptance target
- supported backend numeric expressions now evaluate over the whole batch using
  reusable scratch vectors, reducing per-column recursive interpretation on the
  batched path
- backend index expressions and dynamic address parts now use reusable integer
  scratch buffers, so batched address evaluation no longer reinterprets each
  index expression independently for every column
- synchronized backend loops now derive their reference iterable from
  batch-wide evaluated range endpoints instead of building one iterable per
  batch element before comparison
- observed-loop fast paths now batch constraint lookup per synchronized address
  and reuse a shared observed-value buffer instead of re-querying each
  `ChoiceMap` inside the innermost scoring loop
- supported backend choice steps now preload their batched choice values into a
  shared numeric buffer, so `normal`, `lognormal`, and `bernoulli` score
  evaluation no longer performs value lookup inside the per-column scoring loop
- tempered SMC `:nuts` rejuvenation now also owns a dedicated persistent move
  workspace, so repeated rejuvenation steps reuse batched cohort scratch,
  initial trajectory buffers, and per-particle tree workspaces instead of
  rebuilding them every time
- that workspace now also separates cohort control metadata from numeric
  subtree scratch, mirroring the broader batched NUTS direction of keeping
  scheduler/control state explicit
- the tempered SMC NUTS cohort scheduler now also runs through an explicit
  `Expand/Merge/Done` control IR, so its control path can evolve toward the
  same backend-lowered style as the main batched NUTS implementation
- that path now also exposes `IR -> control block -> descriptor -> execution`,
  so the scheduler control surface is no longer coupled directly to the
  numeric expand/merge routines
- the string-emission substrate that once mirrored these control descriptors
  into generated GPU package stubs (for static models, batched NUTS, and
  tempered SMC NUTS alike) has been removed in favor of the KernelAbstractions
  device backend
- unsupported expressions fall back to the compiled CPU evaluator on the
  batched path

### 4. Inference

Recommended order:

1. `ADVI/SVI`
2. importance sampling
3. `SIR/SMC`
4. batched `HMC`
5. CPU reference iterative `NUTS`
6. batched / GPU-oriented `NUTS` only after the static path is mature

Why:

- the first three are naturally batch-oriented
- batched `HMC` has regular state updates
- `NUTS` brings dynamic tree-building and adaptation complexity, so the first
  version should be an iterative CPU reference path instead of a direct GPU
  target
- the current runtime now has CPU reference `batched_advi`,
  `batched_importance_sampling`, `batched_sir`, and adaptive tempered
  `batched_smc`, including optional batched rejuvenation moves between
  resampling stages, with random-walk, fixed-step tempered HMC, and
  CPU-reference tempered NUTS move kernels, so the remaining work is to widen
  the backend-native subset and lower those same batch APIs to GPU
- the current tempered `:nuts` rejuvenation path already batch-shares its
  initial trajectory and then advances shared depth cohorts before any fully
  chain-local remainder, which is the same structural split the future GPU
  path will want

## Recommended Execution Modes

### `@tea (static)`

This should be the primary mode and the main GPU target.

Requirements:

- the choice set is fixed for a compiled execution plan
- loop extents are static or shape-specialized
- addresses are statically enumerable
- distribution parameters are backend-compatible

### `@tea`

This can exist as a future, more dynamic mode.

Possible future use cases:

- data-dependent trace growth
- richer programmable inference features
- experimental dynamic generative functions

This mode should start as CPU-only fallback, not as a GPU target.

## Conditioning Model

To stay close to Gen, conditioning should be external:

- models define random choices
- observed values are supplied through a `choicemap`-like structure
- the runtime provides `generate`, `assess`, replay, and later convenience wrappers

This avoids baking a separate observed-site syntax into the core language.

### Constraint-driven classification (issue #95)

A choice is an **observation** iff its address is present in the constraints at
inference time; otherwise it is a **latent** with a dense parameter slot.
Binding (`x ~ dist` versus `y = ({:a} ~ dist)`) is orthogonal -- it only names
the value downstream. This single rule is now shared by `generate`/`assess`, the
compiled `logjoint`/gradient paths, the batched and device layouts, and the
diagnostics/predictive APIs, so no two layers can disagree about what is
observed. See
[docs/constraint-driven-conditioning.md](constraint-driven-conditioning.md).

Two behavior changes replaced the earlier syntactic rule (latent iff bound):

- constraining a bound address now conditions on it everywhere (previously the
  compiled scoring paths silently ignored it, producing wrong posteriors);
- an unbound `{:a} ~ dist` left unconstrained is now a latent that is sampled
  and scored (previously an error).

The key architectural consequence: **the parameter layout is a function of the
conditioning signature** (the canonicalized set of constrained addresses), not
of the model alone. This is not a regression of the static/GPU-first direction
-- once the conditioning is fixed the layout is fully static and dense, and the
signature-specific `ParameterLayout` and compiled plans are memoized by
`(model, signature)`. Consequently the parameter-vector length is
conditioning-dependent, and the raw-parameter-vector APIs validate it against
the signature-specific count with an error that names the conditioning.

## Memory and Parameter Layout

Preferred:

- a single unconstrained vector or dense `param x chain` tensor
- normalized address metadata compiled once per execution plan
- one dense layout per conditioning signature (see the constraint-driven
  classification note above), memoized by `(model, signature)`
- dense observed-data layouts for batch-heavy likelihoods
- the same dense `param x particle` layout now also underpins the current
  `batched_advi`, importance-sampling, and SIR reference paths

Avoid:

- dynamic dictionaries in the GPU hot path
- heterogeneous trace containers in performance-critical kernels
- scalar-loop-heavy `logpdf` implementations

## Proposed Package Structure

Initial direction:

- `src/UncertainTea.jl`
- `src/frontend/`
- `src/ir/`
- `src/constraints/`
- `src/choicemaps/`
- `src/backends/cpu/`
- `src/backends/cuda/`
- `src/backends/metal/`
- `src/inference/`
- `src/distributions/`
- `test/`
- `docs/`

## Near-Term Deliverables

1. define `ModelSpec`, `ChoiceSpec`, and `AddressSpec`
2. define the `choicemap` and `generate` API surface
3. implement CPU-side lowering and `logjoint`
4. implement the minimal Gen-like static DSL
5. add a batched `logjoint` reference path with a stable public API
6. lower that batched path into a compiled evaluator before GPU backend work

## Non-Goals for v0

- full Gen compatibility
- Turing-compatible runtime semantics
- unrestricted dynamic trace manipulation on the GPU
- trans-dimensional inference

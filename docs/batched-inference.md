# Batched Inference Design

Date: 2026-03-10

## Motivation

UncertainTea now has a good CPU reference path for single-chain `logjoint`,
gradients, HMC, multi-chain HMC, and summaries.

The next architectural step is not `NUTS`.
It is a batched execution path that makes `chain`, `particle`, or `minibatch`
dimensions explicit in the evaluator.

This is the bridge from the current CPU implementation to a future GPU-native
backend.

## Design Goal

Add a batched evaluator in phases:

1. CPU reference batching with the final public API
2. compiled batched execution on CPU
3. backend-specific lowering for `CUDA.jl` and `Metal.jl`

The public API should stabilize in phase 1 even if the implementation is still
loop-based internally.

## Batch Axis Convention

The primary convention is:

- parameter arrays use shape `num_params x batch`
- outputs use shape `batch`
- gradient outputs use shape `num_params x batch`

This matches the long-term target layout for:

- `theta[param, chain]`
- `grad[param, chain]`
- `weight[particle]`

## Phase 1 API

The initial API should be:

- `batched_logjoint(model, params, args, constraints)`
- `batched_logjoint_unconstrained(model, params, args, constraints)`
- `batched_logjoint_gradient_unconstrained(model, params, args, constraints)`
- `batched_advi(model, args, constraints; ...)`
- `batched_importance_sampling(model, args, constraints; ...)`
- `batched_sir(model, args, constraints; ...)`
- `batched_smc(model, args, constraints; ...)`
- `batched_hmc(model, args, constraints; ...)`
- `batched_nuts(model, args, constraints; ...)`
- `batched_chees(model, args, constraints; ...)` — ChEES-HMC, fixed-length
  jittered HMC whose trajectory length is adapted from cross-chain statistics
  (issue #161); see [chees-hmc.md](chees-hmc.md)
- `batched_meads(model, args, constraints; ...)` — MEADS, the generalized-HMC
  ensemble adaptation of step size, damping, and mass (issue #161)

For repeated gradient evaluations on a fixed batch shape, phase 1 also supports
an explicit cache:

- `BatchedLogjointGradientCache(model, params, args, constraints)`
- `batched_logjoint_gradient_unconstrained!(cache, params)`

### Gradient backends

The batched gradient path picks, per model, between **hand-derived analytic**
gradients (the fused families: GLM, iid sufficient statistics, dense-covariance
Gaussians, LKJ) and a **forward-mode** (`ForwardDiff`) fallback for everything
else. Forward-mode is `O(P)` in the parameter count, so on genuinely
high-dimensional non-analytic models it is structurally slower than a
reverse-mode pass.

For those models a **host reverse-mode** entry point is available through an
optional Enzyme.jl package extension (issue #268, follow-up to the RFC #263):

- `reverse_mode_gradient(f, x)` — the reverse-mode gradient `∇f(x)` of a pure
  scalar objective (e.g. a GP marginal likelihood as a function of its
  hyperparameters).
- `reverse_mode_gradient(model, params, args, constraints)` — the reverse-mode
  analogue of `logjoint_gradient_unconstrained`: `∇` of a whole `@tea` model's
  unconstrained logjoint. It reuses the same type-stable generated scorer the
  forward path differentiates, so the result is identical — just computed the
  cheaper way. Models that fall back to the interpreter (e.g. scalar, non-loop
  observations) raise an `ArgumentError`; use forward-mode for those.

Both are active once `using Enzyme` loads `UncertainTeaEnzymeExt` (without it they
raise a `MethodError`); Enzyme is a weak dependency and never enters the core load
path. They match the forward-mode gradient to machine precision and turn the
non-analytic gradient cost from `O(P²)` into `O(P)` (measured 18.9× at `P=100`,
107× at `P=800`; see `bench/reverse_mode/`):

```julia
using UncertainTea, Enzyme
gp_nlml(h) = UncertainTea.logpdf(gaussianprocess(X, exp(h[1]), exp(h[2]), exp(h[3])), y)
g = reverse_mode_gradient(gp_nlml, [0.0, 0.0, -1.0])   # pure objective
g = reverse_mode_gradient(model, theta, args, constraints)  # whole model
```

Reverse-mode is also wired into the **batched sampler gradient path** as a
per-column tier (issue #268, part A2). Every batch column is a single chain over
the same posterior, so one generated-scorer objective is reused across the whole
batch, differentiated by an Enzyme `ReverseWithPrimal` pass per column (value and
gradient in one pass). It is selected through an `adtype` argument on the gradient-based samplers —
`batched_nuts`, `batched_advi`, `batched_svgd`, and `batched_smc` (the NUTS move
kernel) — and on `BatchedLogjointGradientCache`:

- `adtype=:auto` (default) — **guarded automatic**: uses reverse mode only when it
  is safe *and* beneficial (Enzyme loaded, the model is on the generated-scorer
  path, the batch shares one `args`/`constraints`, the parameter count clears a
  threshold, and a trial gradient compiles to a finite result), and forward mode
  otherwise. Any of those failing — including Enzyme not being loaded — falls back
  to the existing forward/analytic tiers, so behavior is unchanged for users who
  never load Enzyme.
- `adtype=:forward` — always forward-mode/analytic.
- `adtype=:reverse` — force reverse mode (host-only; incompatible with a device
  `backend`). If it cannot be built it still falls back to forward rather than
  failing the run.

The analytic (fused-family) tier is never preempted — reverse only replaces the
*forward-mode* fallback. End-to-end this turns the non-analytic high-`P` cost from
`O(P²)` into `O(P)` in actual sampling (measured **24.8× faster** `batched_nuts` on
a `P=60` non-analytic model, 4 chains, 200+200), with a posterior statistically
identical to forward mode.

```julia
using UncertainTea, Enzyme
chain = batched_nuts(model, args, constraints; num_chains=4, num_samples=1000,
                     num_warmup=1000, adtype=:auto)   # reverse when it helps, forward otherwise
```

The batched per-column *interpreter* objective (the pre-A2 fallback) is not
Enzyme-differentiable, which is why the reverse tier is built on the type-stable
generated scorer instead.

Accepted batching modes:

- shared `args::Tuple` for every batch element
- `Vector{<:Tuple}` for per-batch arguments
- shared `ChoiceMap` for every batch element
- `Vector{ChoiceMap}` for per-batch constraints

The current reference inference layer now includes:

- `pathfinder`, the Zhang et al. (2022) variational initialization: L-BFGS
  toward the mode with per-iterate Gaussian approximations from the inverse-
  Hessian estimate, ELBO selection, and optional multi-path importance
  resampling. A `PathfinderResult` passes directly as `initial_params` to
  `hmc`/`nuts`/`nuts_chains`/`batched_hmc`/`batched_nuts`; the sampler both
  starts each chain from a Pathfinder draw AND seeds the warmup initial
  inverse-mass metric from the Pathfinder covariance diagonal (issue #162), so
  adaptation begins from the approximate posterior scale rather than the
  identity
- `batched_advi`, which fits a Gaussian in unconstrained space and reuses
  `BatchedLogjointGradientCache`. `guide=:meanfield` (default, also the
  device-supported variant) uses a diagonal covariance; `guide=:fullrank`
  parameterizes a full Cholesky factor; `guide=:lowrank` with `lowrank_rank=k`
  uses `Sigma = D^2 + B B'`. The fitted covariance is exposed through
  `variational_covariance`
- `batched_importance_sampling`, which evaluates particle weights through the
  same batched unconstrained target density
- `batched_sir`, which is the current one-stage sampling-importance-resampling
  reference path
- `batched_smc`, which now uses adaptive tempering between the Gaussian
  proposal and the target density, with stage-wise ESS control and systematic
  resampling between intermediate targets
- those tempered stages can now optionally run batched random-walk
  rejuvenation moves after resampling, so particle diversity does not depend
  only on the initial proposal
- the current rejuvenation kernels are a simple symmetric random walk, a
  fixed-step tempered HMC move that reuses `BatchedLogjointGradientCache`, and
  a CPU-reference tempered NUTS move for higher-quality rejuvenation
- that tempered `:nuts` move now batch-shares the initial trajectory
  evaluation and then advances active particles in shared depth cohorts before
  falling back to fully chain-local behavior
- repeated tempered `:nuts` move steps inside one SMC run now reuse a
  persistent move workspace, so the cohort scratch and initial-trajectory
  buffers are not rebuilt at every rejuvenation step
- that move workspace now also carries an explicit cohort control/scheduler
  state, so active-depth selection and cohort activation are separated from
  the numeric subtree scratch
- the tempered cohort scheduler now also lowers to a small `Expand/Merge/Done`
  control IR, so the rejuvenation control skeleton is no longer a single
  monolithic helper
- that scheduler now also passes through explicit control-block and descriptor
  layers before execution, mirroring the staged batched-NUTS control lowering
- the tempered SMC NUTS scheduler can now also emit a shared GPU-backend
  package stub through the generic bundle/package substrate, so this control
  skeleton is no longer isolated from the repository's codegen path
- that same tempered scheduler now also exposes explicit dataflow
  descriptors, dependency tables, buffer lifecycles, resource groups, and
  barrier placements, so its staged control lowering is closer to the
  batched-NUTS backend plan than to an opaque SMC-only helper
- `batched_sir` and `batched_smc` now accept a `resampling::Symbol` keyword
  (`:systematic` default, plus `:stratified`, `:residual`, `:multinomial`); the
  default preserves the previous systematic behavior and RNG consumption
- `log_evidence(result)` reads the log-evidence estimate off an
  `ImportanceSamplingResult` or `SMCResult`, and `SMCResult` now prints a
  one-line summary
- `predict(model, args, chains::HMCChains; ...)`, `predict(model, args,
  ::ImportanceSamplingResult; ...)`, and `predict(model, args, ::SMCResult;
  ...)` produce posterior-predictive `PredictiveDraws`; `predict(model, args;
  ...)` produces prior-predictive draws. These use the dynamic `generate` path
  and are CPU-only by design
- `predict(model, args, ::SIRResult; ...)` draws from the particles the SIR
  step already selected (`num_samples` of them by default) instead of
  re-resampling the importance population

The first `batched_hmc` implementation is intentionally narrower than the
single-chain HMC, but it now supports the same basic warmup structure:

- fixed-step HMC only
- per-chain warmup adaptation (independent step size + diagonal mass per
  chain) is the DEFAULT on the host path (issue #137: a shared step size
  permanently strands prior-initialized chains in high-curvature starts);
  `per_chain_adaptation=false` selects the pooled shared adaptation below,
  and is what an unset value resolves to when a device `backend` is given
- with `per_chain_adaptation=false`: a shared diagonal mass matrix across
  the whole batch
- dual-averaging step-size adaptation during warmup
- windowed shared mass-matrix adaptation from pooled chain positions
- optional initial step-size search
- divergence-aware aggregation of per-chain acceptance when adapting step size
- divergence-aware pooled variance updates when adapting the shared mass matrix
- winsorized online variance updates so transient outlier positions do not
  dominate the shared diagonal mass estimate
- the winsorization threshold now starts loose within each warmup window and
  tightens toward the target clip scale as more pooled samples arrive
- pooled variance now uses acceptance-aware online weights, so rejected
  positions contribute fractionally and divergent proposals contribute zero
- both the clip threshold and the rejection-weight taper are now parameterized
  by each slow warmup window length instead of an implicit global sample-count
  heuristic
- window-level mass-adaptation diagnostics are now exposed through
  `massadaptationwindows(chain)` so warmup updates can be inspected after
  sampling

`batched_nuts` initialization is robust to non-finite starting points (issue
#162):

- `init_strategy=:prior` (default) draws each chain's start from the prior; `init_strategy=:uniform`
  uses the Stan/NumPyro `Uniform(-2, 2)` protocol on the unconstrained scale
  (ship alongside the per-chain step-size adaptation above, which is the default)
- a chain whose initial unconstrained logjoint or gradient is non-finite is
  re-drawn up to `init_max_retries` times (default 100, Stan's retry budget)
  instead of throwing outright — heavy-tailed priors (e.g. a HalfCauchy scale)
  can otherwise start a chain arbitrarily far out. Only re-drawable inits
  (prior/uniform/`PathfinderResult`) retry; an explicit `initial_params` array
  is a fixed value and so fails fast
- `summarize(chains)` now includes aggregated diagnostics, so acceptance,
  divergence, step-size, and warmup mass-adaptation windows can be inspected
  from one summary object
- the `text/plain` display for summaries and diagnostics now expands those
  fields into a compact multi-line report for interactive inspection
- CPU reference path built on the compiled batched evaluator

The current `batched_nuts` implementation is the analogous CPU reference step
for dynamic trajectory building:

- state is stored in `param x chain` matrices
- warmup uses the same pooled diagonal-mass adaptation as `batched_hmc`
- the first trajectory doubling step now executes as a true batched leapfrog
  and batched multinomial selection step over the whole chain matrix
- that first doubling step now initializes each chain's reusable continuation
  state directly, instead of staging separate batched frontier arrays first
- the single-chain `nuts` path now shares the same first-step continuation
  initialization logic, so batched and non-batched trajectories start from the
  same proposal-state transition before deeper tree growth diverges
- both paths now also load that first-step current/proposed state into the
  reusable subtree scratch workspace, so the remaining continuation logic can
  start from the same mutable state buffers rather than per-step `NUTSState`
  allocation
- on the batched path, reusable continuation frontiers, proposal vectors, and
  continuation control metadata now live in batched workspace buffers, so
  deeper chain-local continuation reuses batch-owned storage for
  `left/right/proposal` state and tree bookkeeping instead of per-chain owned
  vectors and scalars
- the remaining per-chain subtree builder scratch is now also view-backed by
  batched workspace matrices, so deeper tree expansion no longer owns separate
  vector storage per chain even though the control flow is still chain-local
- per-chain gradient caches on that path now write into workspace-backed
  `tree_next_gradient` columns, so the remaining chain-local cache objects are
  mostly `ForwardDiff` config/objective wrappers rather than owners of gradient

The current vector backend subset also flows through those batch APIs:

- restricted diagonal `mvnormal` now uses backend-native batched logjoint and
  manual backend gradients in `batched_advi`, batched HMC/NUTS, and particle
  weighting
- `dirichlet` now does the same through the simplex transform, so batched VI
  and SIR can already exercise one nontrivial constrained vector family
  output storage
- when batched chains share the same `args` and `constraints`, those remaining
  wrappers now also share a single `ForwardDiff` objective/config and differ
  only in which workspace-backed gradient column they write into
- after the initial batched first step, continuation subtrees now stay on the
  batched leapfrog/value+gradient path depth-cohort by depth-cohort, picking
  the largest active tree-depth group before falling back to per-chain subtree
  expansion
- subtree energy, acceptance probability, and log-weight bookkeeping for those
  cohort steps now also live in batch-owned scratch vectors rather than local
  scalar temporaries
- the continuation merge step now uses batch-owned scratch for candidate and
  combined log weights plus proposal-selection decisions, leaving only the
  actual proposal-state copies as column-wise work
- turning checks for both subtree growth and merged continuation frontiers now
  run through batched helpers over the workspace matrices instead of per-chain
  scalar calls
- those remaining proposal/frontier copies now also go through masked
  matrix-copy helpers, so the batched subtree path no longer performs repeated
  per-chain vector copies for position/momentum/gradient buffers
- subtree start-state initialization now also uses those masked matrix-copy
  helpers to load `left/right` continuation frontiers into the tree scratch
  buffers for the active cohort
- subtree and continuation frontier/proposal logjoint values now also live in
  workspace-owned vectors, with explicit sync helpers keeping the reference
  `NUTSState` scalars aligned for the remaining chain-local fallback paths
- the initial batched NUTS first step now also loads `current/next` tree
  states through a batched helper and restores rejected/divergent proposal
  columns with masked batch copies rather than per-chain buffer resets
- the final batched NUTS proposal summary now computes acceptance means,
  proposal energies, energy errors, and moved masks through batch helpers over
  the workspace vectors instead of a per-chain epilogue
- the scalar NUTS proposal return path now uses the same acceptance/energy/move
  summary helper family, so diagnostics math is aligned between single-chain
  and batched implementations
- both paths now treat continuation proposal energy and energy error as
  source-of-truth state, so the final epilogue mostly copies precomputed
  diagnostics instead of recomputing Hamiltonians again
- the remaining chain-local batched-NUTS fallback now also stages subtree
  proposal energy/error and subtree summary metadata in workspace vectors
  before merging into continuation state, so batched and fallback merge logic
  share the same scratch-backed bookkeeping path
- that fallback path now also routes one-chain frontier/proposal copies and the
  merged turning check through the same masked batch helpers used elsewhere, so
  even its state updates follow the batched workspace-copy model
- the scalar subtree builder now keeps its own mutable subtree-summary scratch
  object inside `NUTSSubtreeWorkspace`, so even the single-chain reference path
  accumulates subtree metadata in a reusable state object rather than locals
- single-chain continuation merge now also runs through a dedicated subtree
  merge helper that consumes that scratch metadata, making the scalar and
  batched continuation-merge shapes more directly comparable
- single-chain continuation frontier/proposal copies and merged-turning update
  now also run through dedicated helpers, so its expansion loop follows the
  same step-copy-merge structure as the batched fallback path
- direction sampling, continuation-active checks, and subtree start-state
  selection are now also helperized across scalar and batched continuation
  loops, so the control skeleton is more obviously shared
- the batched depth-cohort scheduler now likewise runs through helpers for
  scratch reset, active-depth selection, cohort activation, subtree
  initialization, cohort advancement, and continuation merge, so the
  scheduler body is mostly orchestration over explicit state-machine phases
- continuation gating and selected cohort metadata now also live in a dedicated
  scheduler state object on the workspace (`continuation_active`,
  `subtree_started`, selected depth/count) rather than transient local
  variables
- the subtree scheduler itself now carries an explicit phase and remaining-step
  counter (`idle -> expand -> merge -> done`), so batched continuation is
  driven by a small state machine rather than a monolithic helper body
- chain-level control arrays such as accepted/divergent flags, step direction,
  tree depth, and integration-step counters now also live under a dedicated
  batched NUTS control state, separating control/bookkeeping from numeric
  subtree scratch buffers
- that control state now also lowers to a small control-IR snapshot
  (`IdleIR`, `ExpandIR`, `MergeIR`, `DoneIR`), and the subtree scheduler step
  dispatches through that IR shape instead of branching directly on the raw
  phase enum; `ExpandIR` and `MergeIR` now also reload the scheduler's active
  masks and sampled directions from the IR payload before executing, via an
  explicit executable control-block layer built from each IR snapshot; the
  scheduler now then derives a 1-step descriptor from that block so the
  phase-local scratch masks used by expand/merge are explicit as well, and a
  final step-state layer now bundles those descriptors with the numeric
  energy/log-weight scratch touched by the batched subtree step itself; the
  current CPU path now also derives a kernel-frame object from that step state
  so the numeric matrix/vector buffers consumed by one batched subtree step are
  explicit too, then flattens that frame into a phase-local kernel-access
  object whose fields name the concrete buffers touched by the current step,
  and now further lowers each primitive kernel step into a typed dataflow
  descriptor with explicit logical read/write buffer sets, alias classes, and
  a fixed intra-program dependency table, then bundles those per-step
  descriptors into a phase-local schedule with derived buffer lifecycles,
  resource groups, and explicit barrier placements,
  and now further lowers that schedule into a backend execution block with
  concrete buffer bindings and per-stage barrier hints, and then into a
  device-plan skeleton with segment-local buffer slots and device-stage
  barrier hints, plus a target-plan layer that assigns target-specific
  allocation and barrier strategy for `:gpu`, `:metal`, and `:cuda`, and then
  into a launch-plan layer with concrete constant/device/shared bindings and
  per-stage executor skeletons, and then into a backend executor-plan layer
  with target-specific argument classes and kernel symbols, and then into a
  codegen-plan layer with backend module symbols, generated entry symbols, and
  per-stage generated-argument descriptors, and then into an artifact-plan
  layer with backend artifact symbols and generated-kernel artifact layouts,
  and now into a source-plan layer with generated stub source entrypoints and
  backend-specific argument declarations, and then into a module-plan layer
  with materialized source blobs and backend-specific filenames, and then into
  a bundle-plan layer that groups stage modules behind a backend bundle
  manifest, and then into a package-plan layer with a writeable root layout
  and concrete file entries, and now further into a generic GPU-backend
  bundle/package-layout substrate plus an on-disk emitter stub for
  materializing that layout under a target-specific output root,
  and then stages the step as a small kernel program with a
  fixed op sequence (`reload_control`, `leapfrog`, `hamiltonian`, `advance`,
  `transition_phase` for expand; `reload_control`, `activate_merge`, `merge`,
  `transition_phase` for merge); those op tuples now remain as declarative
  metadata while execution itself dispatches through phase-specialized program
  handlers rather than a generic kernel-op interpreter, with a typed primitive
  step table under each program plus explicit access and dataflow layers
  between frame and execution so per-phase read/write buffers are visible in
  the IR shape
- deeper chain-local subtree expansion now also reuses a per-chain
  current/next subtree scratch workspace, reducing per-step allocations inside
  the remaining CPU reference tree builder
- the chain-local subtree frontier and proposal states now reuse that same
  subtree workspace instead of allocating fresh copies on each integration step
- subtree growth now returns metadata only and leaves frontier/proposal state in
  that scratch workspace, which is closer to the mutable state-machine shape a
  batched backend will want
- chain-local continuation itself now runs from a reusable continuation state
  object, instead of passing frontier/proposal/log-weight scalars around as a
  long argument list
- deeper tree growth is still chain-local and iterative after that first
  batched doubling step
- this is a staging layer toward a future backend-lowered NUTS state machine,
  not the final GPU implementation

Phase 1 intentionally uses the existing single-item evaluator internally.
That gives us:

- a stable API
- clear shape conventions
- a correctness oracle for later optimized implementations

The current implementation already moves part of this work into a compiled
batched evaluator and a reusable gradient cache, but it is still a CPU
reference path rather than a GPU kernel design.

The next lowering layer is now explicit:

- `backend_execution_plan(model)` produces a symbolic primitive-only plan for
  the GPU-friendly subset
- `backend_report(model)` reports whether a model stays inside that subset
- backend-lowered models can now also lower into the same generic GPU-backend
  bundle/package substrate used by the NUTS lowering stack and be emitted as
  stub packages on disk; both paths now also share a generic codegen-bundle
  contract for stage source blobs, entry symbols, and manifest generation
- the static backend package emitter can now target `:gpu`, `:metal`, and
  `:cuda` stub source/module policies even though the symbolic lowering itself
  still comes from the current `:gpu` backend plan
- the shared GPU backend substrate now also owns a generic stub-source
  template layer, so NUTS and static backend packages generate module bodies
  through the same source helper instead of separate string builders
- that source helper now carries explicit stage-kind metadata as well, so both
  emitted manifests and generated stub modules expose the same stage taxonomy
- unsupported models keep working through the compiled CPU fallback path
- batched backend execution currently requires synchronized loop iterables across
  the batch; divergent loop shapes fall back to the compiled CPU path
- backend-plan environments now split numeric slots, index slots, and true
  generic slots so arithmetic-heavy bindings can live in dense `Float64`
  storage and loop/address state can live in dense integer storage
- backend-plan distribution scoring now uses family-specific direct score kernels
  instead of constructing distribution objects on the hot path
- address construction and `1:n` loop evaluation now use an integer-specific
  evaluator instead of the generic expression path
- distribution arguments and numeric deterministic assignments now use a
  numeric-specific evaluator instead of the generic expression path
- supported choices now lower to family-specific backend plan nodes, so hot
  scoring no longer branches on a distribution family symbol
- normalized address lookup now uses a direct `ChoiceMap` helper, and concrete
  addresses are assembled with tuple recursion instead of `Any[]` accumulation
- `ChoiceMap` itself now keeps an address index, so normalized lookup is no
  longer linear in the number of stored constraints
- synchronized repeated-observation loops now have a narrow fast path when the
  loop body is a single observed choice and the address depends only on the
  iterator
- batched backend evaluation now reuses its environment, totals buffer, and
  unconstrained temporary buffers when the batch shape stays fixed
- batched gradient caches now first try a backend-plan-aware gradient evaluator
  for a differentiable subset of the lowered GPU plan
- that manual path currently covers `normal` / `lognormal` / `laplace` /
  `exponential` / `gamma` / `inversegamma` / `weibull` / `beta` /
  `studentt` choices, observed `bernoulli` / `binomial` / `geometric` /
  `negativebinomial` / `categorical` / `poisson`, numeric deterministic
  assignments, and the primitive subset
  `+`, `-`, `*`, `/`, `exp`, `log`, `log1p`, `sqrt`, `abs`, `min`, `max`,
  `clamp`, `sin`, `cos`, `tan`, `tanh`, `sinh`, `cosh`, `atan`, `expm1`,
  `%` with a literal divisor, and `^` with a literal exponent
- backend-lowered models outside that differentiable subset fall back to the
  flat `ForwardDiff` objective over the whole `num_params x batch` state, and
  fully unsupported models still fall back to the older column-wise cache
- vector-valued latent slots are now supported in the CPU/fallback batched
  path, so models with built-in `dirichlet` and restricted diagonal
  `mvnormal` latents can use batched `logjoint`, unconstrained gradients, HMC,
  and NUTS regardless of whether they take the backend-native path
- restricted diagonal `mvnormal` and simplex-valued `dirichlet` now lower to
  the backend-native batched scoring path, and their batched gradients use the
  manual backend gradient evaluator instead of the flat backend `ForwardDiff`
  fallback
- batched HMC and batched NUTS workspace initialization now preserve that
  backend-aware gradient path for both vector families
- batched HMC now reuses sampler-local momentum, proposal, diagnostics, and
  constrained-position buffers instead of reallocating them on each iteration
- batched HMC now also keeps the current unconstrained gradient for each chain
  and updates it from accepted proposals, so the next leapfrog does not need to
  recompute the initial gradient from scratch
- batched HMC now consumes a combined value+gradient internal path, so the
  backend-plan-aware gradient subset can produce the final leapfrog gradient and
  proposed `logjoint` in one evaluator call
- backend numeric expressions for supported models now evaluate batch-wide into
  reusable scratch buffers instead of recursing separately for each column
- backend index expressions and dynamic address parts now use reusable integer
  scratch buffers on the batched path, reducing per-column address evaluation
- synchronized loop iterables now compare batch-wide evaluated range endpoints
  instead of materializing one iterable object per column before checking
- supported observed-loop scoring now batches `ChoiceMap` value lookup for each
  synchronized address and reuses a shared observed-value buffer across items
- batched supported choice scoring now fills a shared numeric choice-value
  buffer before scoring, removing per-column choice lookup calls from the hot
  scoring loop for `normal`, `lognormal`, `laplace`, `exponential`,
  `gamma`, `inversegamma`, `weibull`, `beta`, `bernoulli`, `binomial`,
  `geometric`, `negativebinomial`, `categorical`, `poisson`, and
  `studentt`

## Phase 2 Execution Strategy

After the public API lands, the next step is a compiled batched evaluator.

Requirements:

- no dynamic dictionaries on the hot path
- one compiled execution plan reused across the whole batch
- batched environments represented with dense arrays
- deterministic assignments and distribution arguments lowered once
- a symbolic backend plan that avoids arbitrary Julia callables on the hot path

The target abstraction is not "run the single evaluator in a loop on GPU".
The target abstraction is "evaluate one static execution plan over batched
storage".

## Phase 3 GPU Strategy

Once the batched evaluator exists on CPU, GPU support can follow by lowering the
same batched representation to:

- `CUDA.jl`
- `Metal.jl`
- possibly `KernelAbstractions.jl` for selected kernels

The initial GPU scope should stay narrow:

- continuous latent choices only
- fixed parameter dimension
- static execution plans only
- vectorized likelihoods first

## Non-Goals for the Batched Path

- dynamic trace growth
- trans-dimensional models
- arbitrary heterogeneous batch elements
- direct GPU support for the current single-trace runtime path

## Immediate Implementation Plan

1. add phase 1 batched evaluator functions in `src/`
2. add regression tests that compare batched outputs with per-column scalar calls
3. add batch-specific `args` and `ChoiceMap` tests
4. keep phase 1 implementation simple and correct
5. optimize only after the API and tests are stable

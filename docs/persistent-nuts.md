# Persistent per-chain tree kernel — true GPU-native NUTS (issue #154)

Authoritative design spec for the #154 epic. Increments 1–4 have shipped: the
on-device counter-based RNG (increment 1), the persistent per-chain tree kernel
(increment 2, `batched_nuts(...; tree_strategy=:persistent, backend=...)`), the
full sweep + summary refresh (increment 3, folded into
[benchmarks.md](benchmarks.md) via issue #223), and the wide in-kernel
`DeviceGradN` gradient (increment 4, extended to D > 16 GLMs by issue #221). The
threadgroup-per-chain intra-chain-parallel variant remains deliberately deferred
(issue #219); everything else described below is implemented.

## Goal

**One kernel launch per NUTS iteration.** Each threadgroup (or SIMD-group) owns
one chain and builds its OWN No-U-Turn tree entirely device-resident. The host
does only three things per iteration: upload the per-chain momentum scale (mass
matrix), run dual-averaging/mass adaptation from a small download, and record the
accepted draw. Concretely, the per-iteration cost collapses to roughly
`1 launch + 1 sync + O(C) download ≈ 1–3 ms`, versus today's per-tree-level host
round-trips.

The device-resident tree needs, inside the kernel:

- **On-device counter-based RNG (Philox).** The kernel draws its own momentum and
  its own slice/accept uniforms from a coordinate, with no host-supplied
  randomness. *(This is increment 1 — see below.)*
- **A threadgroup checkpoint stack** of `P × (max_depth + 1)` floats (the
  left/subtree endpoints the iterative NUTS recursion needs). Trivially small at
  the parameter counts `P` we target, so it lives in threadgroup memory.
- **An obs-tiled gradient within the threadgroup**: the leapfrog gradient is
  computed by the chain's own lanes cooperatively tiling the observation loop
  (reusing the #153 observation-parallel tiled-gradient machinery), so the whole
  trajectory stays on-device.

## Why the current async device NUTS cannot reach one-launch/iteration

After #152 (Tier-2 pre-drawn round RNG + device-side accept/select) and #153
(observation-parallel tiled gradient), the device masked NUTS is down to about
**one sync per doubling ROUND**. It gets there by having the host **pre-draw all
of a round's randomness** (`leaf_uniform` uploaded per round) and run the per-leaf
accept/select on the device. That pre-draw is only possible because the masked
design's round structure is **fixed and identical across chains** — every chain
executes the same doubling schedule in lockstep, so the host knows in advance
exactly which random numbers each round consumes.

A persistent per-chain tree is **data-dependent**: each chain U-turns or diverges
at a different depth, takes a different number of leapfrog steps, and therefore
consumes a different amount of randomness in a different order. The host cannot
pre-draw "the right" per-chain RNG because it does not know the tree shape until
the tree has been built. The only way to keep the tree on-device for a full
iteration is for **the kernel to draw its own randomness** — which requires an
on-device RNG. Until increment 1 there was NO on-device RNG anywhere in the
package: momenta and round uniforms were all host-drawn (`Random` on the host)
and uploaded. That missing prerequisite is what increment 1 adds.

## Honest risks (from the issue)

- **(a) RNG-semantics break vs the CPU bitwise reference.** On-device draws will
  not be bit-for-bit identical to the host `Random`-drawn draws, and — for any
  draw that goes through a transcendental (`log`/`cos` in the Gaussian momentum
  transform) — not even bit-identical across the CPU() and Metal backends,
  because their libm differ in the last bits (measured ~5e-7 Float32 drift on an
  Apple M4). **Mitigation:** validate by the #121 statistical-equivalence gate,
  NOT by a bitwise comparison. The integer RNG core and the uniform draw ARE
  bit-identical across backends; only the float transforms are merely
  statistically equivalent.
- **(b) Lane divergence across chains at different tree depths.** Chains that stop
  at different depths leave some lanes idle. Bounded by `max_tree_depth`, so the
  worst case is fixed and small.
- **(c) Long-running kernels vs the Metal watchdog.** A full multi-round tree in
  one launch at large `C × depth` risks the GPU watchdog timeout. **Mitigation:**
  launch per doubling round first (still ~10× fewer syncs than today), then fuse
  further once stable.
- **(d) Tree logic duplicated in kernel code.** The NUTS recursion is
  re-expressed in device code. Mitigated by the plan being `isbits` and the tree
  logic being ~200 lines.

## Increment plan

### (1) On-device counter-based RNG — **this PR**

A KernelAbstractions-compatible, pure, stateless, allocation-free
**Philox4x32-10** counter-based RNG usable inside `@kernel` device code
(`src/device/rng.jl`, included from `src/device.jl`).

- `philox4x32(key::NTuple{2,UInt32}, counter::NTuple{4,UInt32}) -> NTuple{4,UInt32}`
  — the pure bijection. Validated bit-exactly against the Random123 /
  Salmon et al. (2011) known-answer test vectors (and cross-checked against
  Random123.jl v1.7.1 over 200k random inputs during development; Random123 is
  NOT a package dependency).
- `device_rand_uniform(::Type{T}, chain_id, iteration, stream_id, draw_index)::T`
  — a `[0, 1)` uniform of precision `T` (Float32 for Metal, Float64 for CPU()).
- `device_rand_normal(::Type{T}, chain_id, iteration, stream_id, draw_index)::T`
  — a standard normal via Box-Muller (the momentum-draw primitive).

The coordinate `(chain_id, iteration, stream_id, draw_index)` maps to
`key = (chain_id, iteration)`, `counter = (stream_id, draw_index, 0, 0)`, giving
each chain a reproducible, independent stream with **no host coordination**:
Philox4x32's 2-word key + 4-word counter absorb the genuinely 4-dimensional
coordinate with zero folding/collision. `stream_id` separates purposes (e.g.
momentum vs slice/accept) so those draws never correlate.

Chosen over Philox2x32/Threefry because the coordinate is 4-D and Philox4x32's
wider counter maps it cleanly while emitting four 32-bit words per call (a whole
momentum pair, or four uniforms). Nothing calls it yet — this increment is purely
additive infrastructure. Tests: `test/uncertaintea/core/device_rng.jl` (KAT
bit-exactness, CPU()-kernel-vs-host determinism, uniform/normal distributional
sanity, stream independence) and a Metal smoke leg in `test/gpu/runtests.jl`.

### (2) Per-chain persistent tree kernel — **SHIPPED (this PR)**

`batched_nuts(...; tree_strategy=:persistent, backend=...)` builds the ENTIRE
NUTS tree for one iteration in a **single kernel launch** (the one-launch target,
not the per-round fallback), each grid lane owning one chain. Implemented in
`src/device/persistent_nuts.jl`, wired as a `batched_nuts` device strategy.

**What shipped, and the design chosen.**

- **One thread per chain, one launch per iteration.** Rather than a threadgroup
  per chain with `@localmem` + `@synchronize` (which the sketch also floats), the
  shipped kernel gives each chain a single grid lane and keeps its per-chain
  vector state (position/momentum/gradient of length `P` and the checkpoint stack)
  in a per-chain COLUMN of a device buffer indexed by the chain id — exactly like
  the gradient kernel's `slots[:, pidx, b]`. This deliberately avoids
  threadgroup-shared memory and barriers (and KA's group-indexing hazards), so the
  identical kernel compiles on **CPU()** and **Metal**. It is the right shape for
  the epic's headline regime — thousands of chains — where one lane per chain
  already saturates the GPU. The obs-tiled / P-parallel **threadgroup-per-chain**
  variant (which helps the few-chains / large-`N` regime) is deferred to increment
  3; see "Deviations" below.
- **Checkpoint stack** lives in a packed `P × (max_depth+1) × 2 × C` device buffer
  (position + momentum planes), indexed by the chain column — the register/
  threadgroup stack the sketch calls for, realized as a per-chain buffer slice.
- **In-kernel gradient.** The leapfrog gradient is computed inside the kernel by
  looping the differentiation target over `P` and walking the lowered device plan
  in forward-mode duals — the SAME `_device_grad_score_steps` the grid gradient
  kernel uses (`src/device/gradient_kernel.jl`). It is NOT yet observation-tiled
  (#153): each lane scans the full observation loop. Correct for any device-
  lowerable model; obs-tiling is an increment-3 optimization.
- **On-device RNG.** Momentum (stream 0), doubling direction (stream 1), within-
  subtree multinomial slice (stream 2), and biased-progressive merge (stream 3)
  draws all come from increment 1's Philox via the coordinate `(chain_id,
  iteration, stream_id, draw_index)`. A per-run seed drawn from the host `rng` is
  folded into the iteration coordinate so different seeds give different (but
  reproducible) draws.
- **Ported CPU reference:** `_build_nuts_subtree`, `_continue_nuts_proposal!`,
  `_initialize_nuts_first_step!`, `leapfrog_step!`, `_is_turning` /
  `_dyadic_turning` / `_logaddexp` / `_advance_tree_leaf` / `_merge_subtree_stats`.

**Metal argument-buffer packing (risk beyond the sketch).** Metal caps a kernel at
31 indirect argument-buffer resources; the naive per-chain buffer set is 37. The
shipped kernel PACKS related buffers into multi-plane arrays (each frontier packs
pos/mom/grad into `P×3×C`, the subtree proposal packs pos/grad into `P×2×C`, the
checkpoints pack pos/mom, the working sub-tree logjoints and the scalar diagnostics
pack into small `k×C` matrices), landing at ~22 buffers.

**Validation (the #121 gate).** Passes on the conjugate gauss and the two-parameter
location/log-scale model on **CPU() Float64** AND **Metal Float32**:
rank-normalized split R-hat < 1.01 (< 1.02 Float32), analytic mean/sd within a few
MCSE, divergence-free, min-ESS within a few percent of the masked device path,
matching tree-depth distribution (depth ≥ 3–5 reached). Tests:
`test/uncertaintea/core/device_persistent_nuts.jl` (CPU()) and a Metal smoke in
`test/gpu/runtests.jl`. Statistically — not bitwise — equivalent (risk (a)), as
designed.

**Measured Metal speedup (gauss, Apple M4, 300 warmup + 300 sample).** Persistent
vs the masked device path: **~8× at 64 chains, ~7× at 512, ~6× at 4096**; the
4096-chain run completes in ~4.2 s total (~7 ms/iteration), inside the epic's
~2–5 s / ~1–3 ms-per-draw target and versus ~26 s for the masked device path and
the ~228 s historical baseline.

**Deviations / limitations (honest scope).**

- No watchdog fallback was needed: the one-launch design validated on Metal up to
  4096 chains without a per-round split (risk (c) did not bite at these sizes). Very
  large `C × max_tree_depth` may still approach the watchdog; the per-round fallback
  in the plan remains the mitigation if it does.
- Diagonal (pooled) mass only, matching the masked device path.
- Gradient is not observation-tiled and the kernel is one-thread-per-chain, so the
  few-chains / large-`N` regime does not yet get intra-chain parallelism — that is
  increment 3, along with obs-tiling and the SBC-on-device wiring.
- Increment 3's remaining item is generalization + the full 64–16384-chain bench
  sweep; the strategy is already promoted to a `batched_nuts` `tree_strategy` here.

### (3) Generalize + wire as a `batched_nuts` device strategy

Promote the kernel to a general `tree_strategy` on the device path, then bench the
4096-chain Gaussian Metal leg — target ~2–5 s total versus today's ~228 s (and
NumPyro-CPU's ~171 s), the point at which "GPU-native" is demonstrably true at
every chain count ≥ 64. Validate across 64–16384 chains with the #121 gates.

**Bench integration (done).** The cross-PPL harness gained a `batched-metal-persistent`
variant (Metal Float32) plus a `batched-cpu-persistent` Float64 correctness-gate leg
(`bench/crossppl/julia/run.jl`; `./run_all.sh persistent` runs the gauss sweep). Both
clear the #121 gate (rank-normalized split R-hat < 1.01, mean/quantile within combined
MCSEs) with 0 divergences. Measured on the gauss Metal sweep (min-bulk ESS/s; local
`reps=1`): persistent **2,583** @64 and **17,017** @512 chains vs the `:masked`
`batched-metal` rows (~1,261–2,411 @64, ~6,195–9,648 @512) — **~2× ESS/s and ~3–4×
faster wall-clock** in the canonical harness. gauss is a trivial target, so this is a
best case; the full `reps=3` 64–16384 sweep and the summary refresh are a maintainer
step. Remaining increment-3/4 work: obs-tiling the in-kernel gradient for
heavy-per-gradient models (e.g. `logistic_large`), threadgroup-per-chain for the
few-chains/large-N regime, and on-device SBC.

### (4) Heavy-per-gradient in-kernel gradient — **SHIPPED (this PR)**

The increment-2 kernel evaluates the leapfrog gradient with **P serial scalar-dual plan
walks** (one per parameter; `_persist_eval_grad!` over `DeviceDual`, `dual.jl`). Each
walk is O(N) over the observations and each **recomputes the entire logjoint value** —
every transcendental in every logpdf — only to read off one partial derivative. On a
heavy-per-gradient model (`logistic_large`: P=16 coefficients + intercept, N=8000) that
is O(P·N) serial per leaf and, worse, evaluates `N·(P+1)` transcendentals per gradient
where only `N` are mathematically distinct. This increment removes that redundancy.

**What shipped: a P-partial "wide" forward-dual, single plan walk (lane-per-chain).**
`src/device/grad_wide.jl` adds `DeviceGradN{N,T}` — a forward-mode dual carrying a value
and all `N == P` directional derivatives at once (`partials::NTuple{N,T}`). One walk of
the SAME lowered device plan then produces the whole gradient: the `.partials` ARE
`gradients[:, b]` and `.value` is the logjoint. Each transcendental is evaluated **once**
and its derivative fanned across the `N` partials by cheap multiply-adds, collapsing the
`N·(P+1)` transcendentals to `N`. It is `isbits`, allocation-free and exception-free, so
the identical code runs on **CPU()** (Float64) and **Metal** (Float32), exactly like
`DeviceDual`. The step family in `gradient_kernel.jl` was made dual-width-agnostic by
routing its two differentiation-seed sites through shared `_seed_latent` / `_seed_obs`
primitives (scalar seeding is byte-for-byte the pre-refactor expression, so the masked
and grid gradient paths are unchanged — verified by the full device regression suite).

**Why this and NOT the threadgroup-per-chain tree the #154 sketch names as the
endgame.** The sketch's headline is a threadgroup per chain that tiles the gradient
across a group's lanes with `@localmem` + `@synchronize`. In KernelAbstractions 0.9,
`@synchronize`/`@localmem` are usable ONLY in the `@kernel` body — empirically they
cannot be captured inside a called `@inline` device function (`@synchronize used outside
kernel or not captured`). The persistent tree's gradient is evaluated from deep inside
its data-dependent doubling/leaf loops, so a threadgroup variant would have to **inline
the whole ~250-line tree into one kernel with barriers threaded through those loops** —
an untested Metal deadlock risk (divergent `@synchronize`) with no `@localmem`/
`@synchronize` precedent anywhere in this package. The #154 issue explicitly sanctions
the **P-partial forward-dual single walk as the acceptable fallback**; that is what
shipped. It stays **lane-per-chain**: no barriers, no shared memory, and the on-device
Philox RNG stream is byte-for-byte the increment-2 kernel's (draws are still a pure
function of `(chain, iteration, stream, draw)`, independent of any parallelization), so
the wide path is a drop-in change to *how the gradient is computed*, nothing else. The
threadgroup-per-chain tree remains genuine future work.

**Auto-selection heuristic.** The dual width is chosen at workspace construction from the
unconstrained parameter count `P`:

- `P >= 8` (`_PERSIST_WIDE_MIN_PARAMS`) → **wide** (`DeviceGradN{P}` single walk). Heavy
  GLMs land here (`logistic` P=9, `logistic_large` P=17) and get the transcendental
  collapse.
- `P < 8` → **scalar** (the proven increment-2 per-parameter walk). Every small model
  (gauss P≤2, the two-parameter location/log-scale model, eight-schools) stays on the
  unchanged default path, where the wide dual's fixed N-wide-`NTuple` arithmetic per op
  would not pay for itself and would only waste work.

`batched_nuts(...; persistent_gradient = :auto | :scalar | :wide)` overrides the
heuristic per call (`:auto` is the default). Small-P / many-chains therefore keeps using
the increment-2 kernel exactly, so increments 2/3 do not regress.

**Cost model (honest).** The wide walk is the SAME O(P·N) FLOP class as the scalar walk —
the partial propagation is O(P) per operation — so it is never slower, and it is strictly
faster only where a logpdf spends real work in a transcendental (the redundant `N·P`
transcendentals become `N`). It is therefore targeted at, and auto-selected for, the
heavy-GLM regime; it is not a universal win and is deliberately not used at small P.

**Validation (the #121 gate, CPU() Float64).** On the heavy logistic GLM (D=8, N=600,
16 chains) the wide path clears the gate: rank-normalized split R-hat < 1.02, 0
divergences, posterior mean/sd agreeing with the `:masked` device path within a few MCSE,
min-ESS within a few percent of masked. `DeviceGradN`'s derivative channel is validated
to equal `DeviceDual`'s exactly across the full op set (`+,-,*,/,^,exp,log,log1p,sqrt,
tanh,…`), and the wide persistent NUTS samples the SAME posterior as the scalar
persistent NUTS (distributional agreement; the two are NOT expected to trace identical
trajectories because NUTS's multinomial slice/merge makes discrete `log(u) < …` decisions
that a ~1e-12 partial-sum reassociation can flip). gauss and the two-parameter model
(auto → scalar) still pass unchanged. Tests:
`test/uncertaintea/core/device_persistent_nuts_tiled.jl` (78 assertions) plus a Metal
Float32 smoke leg in `test/gpu/runtests.jl`.

**Measured heavy-model speedup (heavy logistic GLM, D=8 → P=9, N=8000, CPU() Float64,
Apple M4; end-to-end wall-clock incl. warmup, 200 warmup + 200 sample).** Persistent wide
vs persistent scalar: **3.55× at 16 chains** (17.8 s vs 63.0 s) and **3.69× at 32 chains**
(36.0 s vs 132.6 s), both at R-hat 1.0, 0 divergences, and identical posterior mean — the
wide walk is the scalar path's work minus the redundant `N·P` transcendentals. The wide
kernel itself compiled in 0.7 s here (the scalar persistent kernel 6.5 s), and the full
`logistic_large` D=16 (P=17) wide kernel compiles + runs in ~7.5 s — no compile-time
blowup with `P`. The speedup grows with `P` (more redundant transcendentals collapsed),
so `logistic_large`'s P=17 shape gets an even larger win than the P=9 numbers above.
(N.B. building the `logistic_large` observation `choicemap` by SPLATTING an 8000-element
generator, `choicemap((… for i in 1:8000)...)`, is a many-minute type-inference trap in
the harness unrelated to this kernel; pass the generator WITHOUT `...` so `choicemap`
takes its iterating path.)

**Deviations / limitations (honest scope).**

- The **threadgroup-per-chain** tree (true intra-chain lane parallelism for the
  few-chains / large-N regime) is NOT shipped; the wide dual is the issue's sanctioned
  P-partial fallback and is a lane-per-chain change. The threadgroup tree, blocked on the
  `@synchronize`-in-kernel-body constraint above, remains future work.
- The wide walk shares the FLOP class of the scalar walk; the win is the transcendental
  collapse, so it helps heavy-per-gradient models and is auto-gated to `P >= 8`.
- Diagonal (pooled) mass only, as the rest of the device path.

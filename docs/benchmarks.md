# Cross-PPL Benchmarks

Reproducible comparison of UncertainTea against NumPyro (JAX) and Stan
(CmdStan) on correctness first, then performance — produced by the
`bench/crossppl/` harness (issue #121). Turing.jl and Gen.jl are planned
follow-up targets.

**Never quote a timing from this document without its correctness gate.**
Every headline number below comes from a run whose draws passed the shared
gate — rank-normalized split R-hat < 1.01 on every parameter, posterior means
and 5%/95% quantiles within 4 combined MCSEs of the CmdStan reference —
computed by ONE implementation (ArviZ 0.x) over every framework's raw draws.
Rows marked FAIL keep their timings visible for context only.

## Provenance (current)

- **Date:** 2026-07-28 (device legs re-measured at the new default after the
  #137 device fix; CPU/host legs carried over from the 2026-07-24 pass, which
  supersedes the 2026-07-23 baseline preserved at the bottom of this file).
- **Hardware:** Apple M4 (10 cores, 32 GB), Metal 4; macOS 26.5.1. All
  frameworks measured natively on this one machine.
- **Software:** Julia 1.12.2, UncertainTea @ `c404ee2` (main, after the first
  performance wave PRs #164–#173 + #170 and the device per-chain-step fix
  #176), NumPyro 0.19 / JAX 0.9 (pinned in `bench/crossppl/python/uv.lock`),
  CmdStan 2.36.0, Python 3.12.
- **Sampler settings:** NUTS everywhere, `target_accept=0.8`,
  `max_tree_depth=10`, diagonal metric, framework-default warmup schedules.
  Correctness pass: 4 chains × 1000 warmup + 1000 draws × 3 repetitions
  (± is std over repetitions). Scaling sweep: 200 warmup + 500 draws at
  64–4096 chains (3 reps ≤512, 1 rep at 4096).
- **Metric:** min-over-parameters bulk ESS per second of pure sampling time.
  Warmup and compile/TTFX are excluded and reported in their own columns.

## Key findings (current)

1. **On the many-chains vectorized workload UncertainTea now leads NumPyro
   and is second only to single-chain Stan.** On the `gauss` scaling sweep the
   host batched CPU backend hits 53,430 bulk ESS/s at 512 chains and 36,879 at
   4096, versus NumPyro-vectorized's 28,378 and 7,687 — all gate-passing. This
   is the payoff of the batched-gradient rework (observed-loop fast path #166,
   sufficient-statistics fusion #146) plus per-chain adaptation becoming the
   host default (#137), which together took the 512-chain leg from a
   gate-failing 72 s to a passing 2.0 s. Only Stan's single-chain C++
   (171,372 ESS/s on this 2-parameter model) is faster.
2. **The device paths now pass the gate by default (no init workaround).**
   Issue #137 is fully closed: per-chain step-size adaptation now works on the
   device masked path via a pooled-mass + per-chain-step warmup driver (PR
   #176). With plain prior-draw init the Metal and KernelAbstractions-CPU
   backends go from ~6% divergence / R-hat ≈ 11.7 (shared adaptation,
   stranded chains) to **0% divergence / R-hat ≈ 1.003** — verified on the M4
   Metal GPU. Every device row in the sweep below is now a real
   default-configuration result; the `-pinned-init` diagnostic workaround is
   retired.
3. **Metal beats NumPyro at 4096 chains but the CPU backend still dominates;
   device per-draw cost is the next target.** At default init Metal reaches
   18,104 ESS/s at 4096 chains (PASS) — ahead of NumPyro-vectorized (7,687) —
   but the host `batched-cpu` backend leads everywhere (36,879 at 4096,
   53,430 at 512). Metal's per-draw cost is still high because the masked path
   runs a host gradient every iteration (#151, not yet fixed) and pays
   full-width work on finished lanes (#160); these are the levers to make the
   GPU genuinely win. The earlier `-pinned-init` diagnostic overstated the
   device story (best-case init, shorter trees) — the default numbers here are
   the honest ones. Warmup cost also rose with per-chain adaptation (#158
   tracks recovering it via pooled-mass — whose core machinery #176 already
   built).
4. **GLMs still trail.** `logistic` batched-cpu passes the gate now (156
   ESS/s, was gate-marginal) but stays far behind NumPyro (4,185) because the
   bernoulli-logit + covariate observation does not lower to the analytic
   batched path yet (#150/#134/#135). Single-chain `logistic` improved 83 →
   406 ESS/s from the interpreter rework (#145).

Open issues from the audit still shaping these numbers: #150/#134/#135 (GLM /
device lowering — the `logistic` gap), #151/#152/#153/#160 (device
engineering — the `batched-cpu`-vs-Metal gap), #158 (pooled-mass host default
+ nutpie, to recover warmup cost), #144 (generated type-stable scorer, the
remaining single-chain gap vs Stan). #137 (per-chain step-size adaptation) is
now closed on both host and device.

## Scaling sweep — gauss (mean/scale, N=1000; 200 warmup + 500 draws)

| framework | chains | precision | correct | min bulk ESS/s | sampling s | warmup s | div |
|---|---|---|---|---|---|---|---|
| stan (single chain, 1000 draws) | 4 | f64 | PASS | 171,372 ± 32,742 | 0.021 | 0.015 | 0 |
| uncertaintea-batched-cpu | 4 | f64 | PASS | 38,708 ± 2,955 | 0.05 | 0.086 | 0 |
| uncertaintea-batched-cpu | 64 | f64 | PASS | 57,519 ± 17,018 | 0.26 | 0.62 | 0 |
| uncertaintea-batched-cpu | 512 | f64 | PASS | **53,430 ± 5,336** | 2.05 | 4.68 | 0 |
| uncertaintea-batched-cpu | 4096 | f64 | PASS | **36,879** | 23.2 | 38.7 | 0 |
| numpyro-vectorized | 64 | f32 | PASS | 14,279 ± 1,205 | 1.70 | 1.63 | 0 |
| numpyro-vectorized | 512 | f32 | PASS | 28,378 ± 1,773 | 7.08 | 4.23 | 0 |
| numpyro-vectorized | 4096 | f32 | PASS | 7,687 | 205 | 93.6 | 0 |
| uncertaintea-batched-cpu-ka | 64 | f64 | PASS | 5,259 ± 140 | 3.20 | 2.86 | 0 |
| uncertaintea-batched-cpu-ka | 512 | f64 | PASS | 2,916 ± 620 | 46.0 | 33.5 | 0 |
| uncertaintea-batched-cpu-ka | 4096 | f64 | PASS | 6,282 | 159 | 178 | 0 |
| uncertaintea-batched-metal | 64 | f32 | PASS | 648 ± 4.2 | 25.9 | 18.9 | 0 |
| uncertaintea-batched-metal | 512 | f32 | PASS | 2,770 ± 53 | 46.9 | 37.9 | 0 |
| uncertaintea-batched-metal | 4096 | f32 | PASS | **18,104** | 55.3 | 72.3 | 0 |

All device rows are now default-configuration (prior-draw init, per-chain
adaptation the device default) and gate-passing after #176 — no `-pinned-init`
workaround. `batched-cpu` dominates the device backends on this shape because
it uses the fused analytic gradient (#146/#166) while the masked
KernelAbstractions/Metal paths run the (now-fused but still host-side)
gradient every iteration (#151) and pay full-width work on finished lanes
(#160). Metal overtakes NumPyro only at 4096 chains; closing the gap to
`batched-cpu` is the device-engineering work in #151/#152/#153/#160.

## Correctness pass (4 chains × 1000 warmup + 1000 draws)

### eight_schools_noncentered — all PASS

| framework | min bulk ESS/s | sampling s | div rate |
|---|---|---|---|
| stan | 52,567 ± 3,823 | 0.02 | 0.0002 |
| uncertaintea-cpu | 9,554 ± 2,393 | — | ~0 |
| numpyro-parallel | 1,978 ± 120 | — | ~0 |
| uncertaintea-batched-cpu | 1,640 ± 330 | — | ~0 |

Single-chain UncertainTea improved 3,354 → 9,554 ESS/s from the biased-merge
tree change (#159, +54% ESS/gradient) and the interpreter rework (#145). The
batched path's 4-chain number is lower because per-chain adaptation (now
default) spends more warmup per chain at tiny chain counts — the batched path
is built for the many-chains regime above, not 4 chains.

### logistic (N=500, D=8) — all PASS

| framework | min bulk ESS/s | div rate |
|---|---|---|
| stan | 23,628 ± 1,951 | 0 |
| numpyro-parallel | 4,185 ± 500 | 0 |
| uncertaintea-cpu | 406 ± 32 | 0 |
| uncertaintea-batched-cpu | 156 ± 6 | 0 |

GLM gap persists (no analytic/​device lowering yet — #150/#134/#135);
single-chain improved 83 → 406 from #145.

### eight_schools_centered — all FAIL (funnel; expected)

Every framework, Stan included, exceeds R-hat 1.01 with 2–3% divergences at
`target_accept=0.8` — the canonical centered-parameterization pathology. The
gate rejecting all four implementations equally is evidence it works; this
model stays in the suite as the honesty check.

## What changed since the 2026-07-23 baseline

| leg | 2026-07-23 | now | driver |
|---|---|---|---|
| gauss batched-cpu 512 chains | 72 s / FAIL | 2.0 s / **PASS**, 53k ESS/s | #146, #166, #137, #142 |
| gauss batched-cpu 4096 chains | 763 s / FAIL | 23.2 s / **PASS**, 37k ESS/s | same |
| gauss Metal 4096 (default) | FAIL (~6% div) | **PASS**, 18k ESS/s (55 s) | #137 device (#176) + #146 |
| gauss Metal / ka default gate | FAIL, R-hat ≈ 11.7 | **PASS**, R-hat ≈ 1.003, 0% div | #176 |
| gauss cpu (single chain) | 173 ESS/s | 509 ESS/s | #145, #159 |
| eight-schools-nc cpu | 3,354 ESS/s | 9,554 ESS/s | #145, #159 |
| logistic cpu | 83 ESS/s | 406 ESS/s | #145 |

## Models

| model | shape | exercises |
|---|---|---|
| `eight_schools_centered` | hierarchical, funnel | divergence behaviour, gate honesty |
| `eight_schools_noncentered` | hierarchical | `reparam=:noncentered`, `iid` latents |
| `logistic` | GLM, N=500, D=8 | loop-addressed discrete observations |
| `gauss` | mean/scale, N=1000 | device path; chain-count scaling sweep |

Identical joint densities across frameworks; priors in
`bench/crossppl/julia/models.jl` and `bench/crossppl/python/stan/*.stan`. A
discrete-latent model (`marginalize=:enumerate` vs Stan's hand-marginalization)
and an `lkjcholesky` model are planned additions; the scaling model is a
loop-addressed gaussian rather than a regression because of #134/#135.

## Methodology notes

- Sampling time is isolated from warmup per framework: UncertainTea runs
  (warmup, 1 draw) and (warmup, S draws) from the same RNG seed (identical
  warmup trajectories) and differences them; NumPyro's `warmup()`/`run()`
  are timed separately; CmdStan's own per-chain elapsed report is used (max
  over concurrently running chains).
- All checked-in numbers are native macOS, one machine, so the GPU/CPU
  crossover axis is not distorted by a VM. `bench/crossppl/docker/`
  provides a single pinned Linux environment for portable CPU-only reruns.
- Precision differs by leg and is disclosed per row (Metal requires f32;
  the NumPyro sweep matches it; everything else is f64).

## Refresh procedure

```bash
cd bench/crossppl
./run_all.sh cpu && ./run_all.sh metal && ./run_all.sh analyze
```

The device legs (`metal`, and the `-ka` legs inside `cpu`) now pass the gate
at their default configuration, so the `pinned` mode is no longer needed to
report device numbers — it remains only as an init-sensitivity diagnostic.
Update the "current" sections above from `results/summary.md` with the date,
hardware, and commit; move the previous numbers into the history table; keep
the correctness-gate framing intact.

---

## Baseline archive — 2026-07-23 (pre-optimization, UncertainTea @ `6df3064`)

The original first-cut measurement, kept for the history table above. At that
commit UncertainTea's batched paths failed the gate at ≥64 chains (#137
undiagnosed as default), the batched CPU gradient re-fetched observations per
call (#138), and Metal ran an unfused host gradient every iteration (#151):
gauss batched-cpu 512 chains took 72 s (FAIL), Metal 4096 chains 228 s at
3,648 ESS/s (pinned), single-chain gauss 173 ESS/s, and NumPyro-vectorized led
the gate-passing scaling sweep at every chain count. See the git history of
this file (`docs/benchmarks.md` at `6df3064`) for the full original tables.

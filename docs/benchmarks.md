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

- **Date:** 2026-07-29. Full re-measure of every leg (single-chain,
  host `batched-cpu`, KernelAbstractions-CPU, and Metal device) from a clean
  regenerated `results/summary.md`. The new `logistic_large` model (#193) is
  in the sweep. Supersedes the 2026-07-28 pass; the 2026-07-23 first-cut
  baseline is preserved at the bottom of this file.
- **Hardware:** Apple M4 (10 cores, 32 GB), Metal 4; macOS 26.5.1. All
  frameworks measured natively on this one machine.
- **Software:** Julia 1.12.2, UncertainTea @ `4df60b0` (main). The
  optimization stack now included, relative to the 2026-07-23 baseline:
  chain-thread parallelism (#136), the batched observed-loop fast path (#140/
  #141) and iid sufficient-statistics fusion (#146), allocation-free batched
  leapfrog (#142), the single-chain interpreter rework (#145) and Stan-style
  biased-merge NUTS (#159), per-chain step-size adaptation as the host/device
  default (#137) with pooled-mass unification and a vectorized warmup step
  search (#158), the GLM analytic lowering on host (#149/#150) and device
  (#135), batched-gradient threading (#143), the type-stable generated
  single-chain scorer (#144) with single-chain O(1) suffstats fusion (#189)
  and a value-path observation cache (#188/#190), the large-GLM probe (#193),
  and the device masked-NUTS engineering waves #152 (leaf micro-kernel fusion,
  pre-drawn round RNG, device-side accept/select) and #153 (observation-
  parallel tiled device gradient + shared-observation staging).
  NumPyro 0.19 / JAX 0.9 (pinned in `bench/crossppl/python/uv.lock`),
  CmdStan 2.36.0, Python 3.12. Host batched legs measured at `-t auto`
  (4 performance cores).
- **Sampler settings:** NUTS everywhere, `target_accept=0.8`,
  `max_tree_depth=10`, diagonal metric, framework-default warmup schedules.
  Correctness pass: 4 chains × 1000 warmup + 1000 draws × 3 repetitions
  (± is std over repetitions). Scaling sweep: 200 warmup + 500 draws at
  64–4096 chains (3 reps ≤512, 1 rep at 4096).
- **Metric:** min-over-parameters bulk ESS per second of pure sampling time.
  Warmup and compile/TTFX are excluded and reported in their own columns.

## Key findings (current)

1. **Single-chain `gauss` now measures higher ESS/s than the CmdStan
   reference on this model.** `uncertaintea-cpu` reaches 546,365 bulk ESS/s
   on `gauss` (≈0.9 µs per draw; 4 chains × 1000 draws in 3.7 ms of sampling),
   versus Stan's 278,037 on the same model. This is a measured result, with an
   honest caveat: `gauss` is a 2-parameter model whose iid-observation
   likelihood fully sufficient-statistics-fuses (#146/#189), an ideal case
   that does not generalize to all models — single-chain `logistic`, for
   example, is 8,869 versus Stan's 61,078. The single-chain jump is driven by
   the type-stable generated scorer (#144), single-chain O(1) suffstats fusion
   (#189), and the value-path observation cache (#190). Single-chain
   `eight_schools_noncentered` reaches 37,560 ESS/s — higher than
   NumPyro-parallel (4,265) and within ~3× of Stan (112,647).
2. **On the many-chains `gauss` sweep the host `batched-cpu` backend measures
   higher ESS/s than NumPyro-vectorized at every chain count.** 64/512/4096
   chains give 101,537 / 60,481 / 58,697 bulk ESS/s for `batched-cpu` versus
   23,024 / 45,922 / 10,078 for NumPyro-vectorized — all gate-passing. This is
   the payoff of the batched observed-loop fast path (#140/#141), iid
   suffstats fusion (#146), and per-chain adaptation as the host default
   (#137).
3. **The Metal device path now passes the gate by default at every measured
   size, and the #152/#153 work made it substantially faster.** With plain
   prior-draw init the device backends produce 0% divergence and R-hat < 1.01
   at every size in the sweep — no `-pinned-init` workaround. The recent
   engineering is a large step change: `logistic_large` at 64 chains did not
   finish within 35 minutes on the pre-#153 serial-scan path and now completes
   in ~32 s of sampling; #152 cut per-draw device synchronizations from
   ~45–67 per draw to ~4 per NUTS round. Being honest about where the device
   sits today: on the current benchmark models the host `batched-cpu` backend
   is still faster than Metal. The device pays a per-draw dispatch floor,
   these models do not saturate the GPU, and the device coefficient-vector
   dimension is capped at 16 (`DEVICE_MAX_VECTOR_DIMENSION`). Metal is
   competitive with `batched-cpu` only on the larger `logistic_large` (Metal
   64 chains 2,277 vs `batched-cpu` 4 chains 1,531). Frame the device path as
   correct-by-default and much improved, not yet the fastest backend at these
   sizes.
4. **A new `logistic_large` model (#193, D=16, N=8000) is in the sweep** as
   the device / large-model probe: a genuinely heavy per-gradient GLM that
   exercises the device analytic path (#135) without being dominated by device
   dispatch overhead. D is 16 because the device coefficient-dimension cap is
   16; N is raised to 8000 to keep the per-gradient work budget large while the
   model still lowers to the device.

Open issues from the audit still shaping these numbers: #151 (host-gradient on
the device masked path — deprioritized), #154 (persistent-kernel device epic),
#160 (lane compaction on finished chains), #161 (ChEES/MEADS adaptation), and
#16 (CUDA backend). Now closed and reflected above: #135 (device GLM
lowering), #149/#150 (host GLM analytic lowering), #137 (per-chain step-size
adaptation), #143 (batched-gradient threading), #158 (pooled-mass unification
+ vectorized warmup step search), #138/#188 (single-chain suffstats fusion and
value-path observation cache), and #152/#153 (device masked-NUTS engineering).

## Single-chain correctness pass (4 chains × 1000 warmup + 1000 draws)

Min bulk ESS/s per model, all frameworks measured at the same 4-chain
configuration. FAIL rows (see `eight_schools_centered`) are shown for context
and must not be quoted as headline throughput.

| model | gate | uncertaintea-cpu | uncertaintea-batched-cpu (4ch) | stan | numpyro-parallel |
|---|---|---|---|---|---|
| gauss | PASS | **546,365 ± 99,595** | 52,795 ± 24,342 | 278,037 ± 930 | 7,346 ± 480 |
| eight_schools_noncentered | PASS | **37,560 ± 5,798** | 4,896 ± 370 | 112,647 ± 16,135 | 4,265 ± 440 |
| logistic (N=500, D=8) | PASS | 8,869 ± 430 | 11,067 ± 1,113 | 61,078 ± 1,162 | 10,386 ± 700 |
| logistic_large (N=8000, D=16) | PASS | 170 ± 12 | 1,531 ± 79 | 5,538 ± 530 | 6,755 ± 560 |
| eight_schools_centered | FAIL (funnel) | 1,311 ± 640 | 110 ± 41 | 5,404 ± 2,615 | 185 ± 92 |

`gauss` single-chain leads the CmdStan reference on this 2-parameter,
fully-fusible model (see key finding 1 for the caveat). `logistic` and
`logistic_large` single-chain remain below Stan — their per-observation
covariate dot products cannot be sufficient-statistics-fused. On
`logistic_large` the host `batched-cpu` path (1,531) is far ahead of
single-chain `uncertaintea-cpu` (170).

## Scaling sweep — gauss (mean/scale, N=1000; 200 warmup + 500 draws)

| framework | chains | precision | correct | min bulk ESS/s | sampling s | warmup s | div |
|---|---|---|---|---|---|---|---|
| uncertaintea-cpu (single chain, 1000 draws) | 4 | f64 | PASS | 546,365 ± 99,595 | 0.0037 | 0.0062 | 0 |
| stan (single chain, 1000 draws) | 4 | f64 | PASS | 278,037 ± 930 | 0.013 | 0.010 | 0 |
| uncertaintea-batched-cpu | 4 | f64 | PASS | 52,795 ± 24,342 | 0.055 | 0.033 | 0 |
| uncertaintea-batched-cpu | 64 | f64 | PASS | **101,537 ± 2,101** | 0.165 | 0.082 | 0 |
| uncertaintea-batched-cpu | 512 | f64 | PASS | **60,481 ± 22,963** | 2.67 | 1.38 | 0 |
| uncertaintea-batched-cpu | 4096 | f64 | PASS | **58,697** | 17.1 | 7.9 | 0 |
| numpyro-vectorized | 64 | f32 | PASS | 23,024 ± 1,888 | 1.05 | 0.918 | 0 |
| numpyro-vectorized | 512 | f32 | PASS | 45,922 ± 2,797 | 4.37 | 2.82 | 0 |
| numpyro-vectorized | 4096 | f32 | PASS | 10,078 | 156 | 70.1 | 0 |
| uncertaintea-batched-cpu-ka | 64 | f64 | PASS | 6,277 ± 310 | 2.64 | 1.77 | 0 |
| uncertaintea-batched-cpu-ka | 512 | f64 | PASS | 6,532 ± 61 | 19.9 | 17.4 | 0 |
| uncertaintea-batched-metal | 64 | f32 | PASS | 1,261 ± 11 | 13.0 | 7.08 | 0 |
| uncertaintea-batched-metal | 512 | f32 | PASS | 6,195 ± 76 | 20.8 | 10.6 | 0 |

All device rows are default-configuration (prior-draw init, per-chain
adaptation the device default) and gate-passing — no `-pinned-init`
workaround. `batched-cpu` leads NumPyro-vectorized at every chain count and
leads the device backends on this shape: it uses the fused analytic gradient
(#146) while the masked KernelAbstractions/Metal paths pay a per-draw dispatch
floor and full-width work on finished lanes (#160), and this small model does
not saturate the GPU.

## Correctness pass (4 chains × 1000 warmup + 1000 draws)

### eight_schools_noncentered — all PASS

| framework | min bulk ESS/s | sampling s | div rate |
|---|---|---|---|
| stan | 112,647 ± 16,135 | 0.019 | 0.0003 |
| uncertaintea-cpu | 37,560 ± 5,798 | 0.050 | 0.0009 |
| uncertaintea-batched-cpu | 4,896 ± 370 | 0.40 | 0.0006 |
| numpyro-parallel | 4,265 ± 440 | 0.53 | 0.0002 |

Single-chain `uncertaintea-cpu` (37,560) is higher than NumPyro-parallel and
within ~3× of Stan on this hierarchical model. The batched path's 4-chain
number is lower because per-chain adaptation (the default) spends more warmup
per chain at tiny chain counts — the batched path is built for the many-chains
regime, not 4 chains.

### logistic (N=500, D=8) — all PASS

| framework | chains | min bulk ESS/s | div rate |
|---|---|---|---|
| stan | 4 | 61,078 ± 1,162 | 0 |
| uncertaintea-batched-cpu | 4 | 11,067 ± 1,113 | 0 |
| numpyro-parallel | 4 | 10,386 ± 700 | 0 |
| uncertaintea-cpu | 4 | 8,869 ± 430 | 0 |
| uncertaintea-batched-metal | 512 | 9,648 ± 180 | 0 |
| uncertaintea-batched-metal | 64 | 2,411 ± 120 | 0 |

`logistic` rides the fused GLM analytic path on host (#149/#150) and device
(#135). At 4 chains `batched-cpu` (11,067) is close to NumPyro-parallel
(10,386); Stan leads (61,078). The Julia model uses
`bernoullilogit(alpha + sum(beta .* X[:, i]))`, matching the Stan
(`bernoulli_logit`) and NumPyro (`Bernoulli(logits=...)`) joint densities. The
Metal rows are default-configuration and gate-passing (0 divergence); at 512
chains Metal reaches 9,648.

### logistic_large (N=8000, D=16) — all PASS

| framework | chains | min bulk ESS/s | div rate |
|---|---|---|---|
| numpyro-parallel | 4 | 6,755 ± 560 | 0 |
| stan | 4 | 5,538 ± 530 | 0 |
| uncertaintea-batched-metal | 64 | 2,277 ± 570 | 0 |
| uncertaintea-batched-cpu | 4 | 1,531 ± 79 | 0 |
| uncertaintea-cpu | 4 | 170 ± 12 | 0 |

The large GLM is the device / large-model probe (#193). On the heavy
per-gradient work the host `batched-cpu` path (1,531 at 4 chains) is well ahead
of single-chain `uncertaintea-cpu` (170), and Metal at 64 chains (2,277) is
competitive with `batched-cpu` here — the one model in the suite where the
device is close to the host backend. This is also the leg that exercises the
#152/#153 device work: it did not finish within 35 minutes on the pre-#153
serial-scan device path and now completes in ~32 s of sampling.

### eight_schools_centered — all FAIL (funnel; expected)

Every framework, Stan included, exceeds R-hat 1.01 with 1–3% divergences at
`target_accept=0.8` — the canonical centered-parameterization pathology. The
gate rejecting all four implementations equally is evidence it works; this
model stays in the suite as the honesty check.

## What changed since the 2026-07-23 baseline

| leg | 2026-07-23 | now | driver |
|---|---|---|---|
| gauss cpu (single chain) | 173 ESS/s | **546,365 ESS/s** | #144, #189, #190 |
| eight-schools-nc cpu (single chain) | 3,354 ESS/s | **37,560 ESS/s** | #145, #159, #144 |
| logistic cpu (single chain) | 83 ESS/s | **8,869 ESS/s** | #145, #149/#150, #144 |
| gauss batched-cpu 512 chains | 72 s / FAIL | 2.7 s / **PASS**, 60,481 ESS/s | #140/#141, #146, #137 |
| gauss batched-cpu 4096 chains | 763 s / FAIL | 17.1 s / **PASS**, 58,697 ESS/s | same |
| gauss Metal / ka default gate | FAIL, R-hat ≈ 11.7 | **PASS**, R-hat < 1.01, 0% div | #137 device (#176) |
| logistic batched-cpu 4 chains | 156 ESS/s (gate-marginal) | **11,067 ESS/s** (PASS) | #149, #150 |
| logistic_large 64 chains (Metal) | did not finish in 35 min | **PASS**, ~32 s sampling, 2,277 ESS/s | #152, #153 |
| device masked-NUTS syncs | ~45–67 / draw | ~4 / NUTS round | #152 |

## Models

| model | shape | exercises |
|---|---|---|
| `eight_schools_centered` | hierarchical, funnel | divergence behaviour, gate honesty |
| `eight_schools_noncentered` | hierarchical | `reparam=:noncentered`, `iid` latents |
| `logistic` | GLM, N=500, D=8 | loop-addressed discrete observations |
| `logistic_large` | GLM, N=8000, D=16 | heavy per-gradient GLM; host + device analytic path; chain-count scaling sweep |
| `gauss` | mean/scale, N=1000 | device path; chain-count scaling sweep |

Identical joint densities across frameworks; priors in
`bench/crossppl/julia/models.jl` and `bench/crossppl/python/stan/*.stan`.
`logistic_large` is a larger GLM whose per-observation D-dim dot product over N
observations gives a genuinely heavy gradient — the case that exercises the
device analytic path (#135) and many-chains scaling without being dominated by
device dispatch overhead. Its D is 16 (rather than a larger value) because the
device coefficient-dimension cap is 16 (`DEVICE_MAX_VECTOR_DIMENSION`); N is
raised to 8000 to keep the per-gradient work budget large while the model still
lowers to the device. A discrete-latent model (`marginalize=:enumerate` vs
Stan's hand-marginalization) and an `lkjcholesky` model are planned additions.

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

The sweep now includes `logistic_large` alongside `gauss` and `logistic`. The
device legs (`metal`, and the `-ka` legs inside `cpu`) pass the gate at their
default configuration, so the `pinned` mode is no longer needed to report
device numbers — it remains only as an init-sensitivity diagnostic. Update the
"current" sections above from `results/summary.md` with the date, hardware, and
commit; move the previous numbers into the history table; keep the
correctness-gate framing intact.

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

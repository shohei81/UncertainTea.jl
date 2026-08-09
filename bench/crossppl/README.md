# Cross-PPL benchmark harness (issue #121)

Reproducible correctness + performance comparison of UncertainTea against
NumPyro and CmdStan on shared canonical models. Results and methodology live
in [docs/benchmarks.md](../../docs/benchmarks.md); this directory is the
harness that produces them.

## Layout

```
data/          shared datasets (JSON, fixed seed; generate_data.jl documents provenance)
julia/         UncertainTea runner (pinned Project.toml/Manifest.toml, UncertainTea dev'd by path)
python/        NumPyro + CmdStan runners, shared ArviZ diagnostics (pinned by uv.lock)
docker/        single pinned Linux environment for portable CPU comparisons
results/raw/   one .npz (draws) + .json (timings, settings, provenance) per run  [gitignored]
run_all.sh     orchestration: cpu | metal | analyze
```

## Quick start (native macOS)

```bash
cd bench/crossppl
./run_all.sh cpu       # correctness pass (Stan / NumPyro / UncertainTea) + CPU scaling sweep
./run_all.sh metal     # Metal scaling sweep (Apple GPU)
./run_all.sh analyze   # shared diagnostics -> results/summary.{json,md}
```

First runs bootstrap everything: `uv` creates `python/.venv` from `uv.lock`,
CmdStan 2.36.0 is compiled into `python/.cmdstan`, and the Julia project is
instantiated from the checked-in Manifest.

## Docker (pinned fair-comparison environment)

The container puts every CPU framework in one pinned Linux userland. Metal
legs cannot run inside it (no GPU in the VM on macOS), so GPU-vs-CPU crossover
curves should come from a single native machine instead; the container is for
portable/repeatable CPU-only comparisons.

```bash
cd bench/crossppl
docker build -t uncertaintea-crossppl -f docker/Dockerfile .
docker run --rm -v "$(git rev-parse --show-toplevel)":/work \
    -v uncertaintea-crossppl-depot:/opt/julia-depot \
    -w /work/bench/crossppl uncertaintea-crossppl ./run_all.sh cpu
```

## Design decisions

- **Identical joint densities.** Every framework states the same model
  (priors listed in `julia/models.jl`, `python/stan/*.stan`,
  `python/run_numpyro.py`); half-Cauchy is expressed as
  `truncatedstudentt(1, 0, 5, 0, Inf)` in UncertainTea, `cauchy(0,5)` +
  `<lower=0>` in Stan, `HalfCauchy(5)` in NumPyro — the same density up to a
  constant.
- **One diagnostic implementation.** `analyze.py` computes rank-normalized
  split R-hat and bulk/tail ESS with ArviZ for *every* framework's draws;
  no framework's built-in numbers are trusted.
- **Correctness gates timings.** A run only contributes ESS/second if every
  parameter has R-hat < 1.01 and its posterior mean and 5%/95% quantiles
  agree with the CmdStan reference within 4 combined MCSEs.
- **ESS/second excludes warmup and compile.** Warmup and TTFX/compile are
  reported in separate columns. The UncertainTea sampling time is isolated by
  running (warmup, 1 draw) and (warmup, S draws) from the same RNG seed —
  identical warmup trajectories — and differencing; NumPyro's
  `warmup()`/`run()` are timed separately; CmdStan's own per-chain elapsed
  report is used (max over concurrently running chains).
- **Chain parallelism differs by framework and is recorded per run**
  (`sampler.chain_parallelism`): CmdStan forks processes, NumPyro
  `parallel`/`vectorized`, UncertainTea `cpu` is sequential (issue #136) and
  `batched-*` vectorized.
- **Gradient tiers are measured explicitly, never mixed** (issue #317).
  `poisson_glm` states the model in every framework's natural vectorized form
  (UncertainTea's broadcast `{:y} ~ poisson.(exp.(a .+ b .* x))`, Stan's
  `poisson_log`, a NumPyro plate) and rides the analytic broadcast gradient.
  `schools_large` (J=32, P=34; noncentered, log-normal tau) is the
  reverse-mode leg: `julia/run.jl --adtype forward|reverse` pins the
  ForwardDiff vs Enzyme tier on the host batched path, labels the runs
  `batched-cpu-forward`/`batched-cpu-reverse`, and records
  `sampler.adtype`; Enzyme is loaded only for `reverse` runs so `auto` legs
  keep their default tier. The model states the canonical noncentered form
  (`reparam=:noncentered`, log-normal tau) — it was originally forced
  centered because the noncentered `reparam` machinery and the truncated-t
  tau used to fail Enzyme's type analysis and silently fall back to forward,
  but issue #326 fixed that (an explicit `adtype=:reverse` request now warns
  whenever it cannot engage), so the canonical shape measures the tiers
  (issue #365).

## Sampler settings

NUTS everywhere: `target_accept = 0.8`, `max_tree_depth = 10`, diagonal mass
matrix, framework-default warmup adaptation schedules (documented difference —
Stan's windowed adaptation is the reference design; UncertainTea and NumPyro
follow the same windowed scheme). Correctness pass: 4 chains × 1000 warmup +
1000 draws × 3 repetitions. Scaling sweep: 64–4096 chains × 200 warmup +
500 draws (3 repetitions at ≤512 chains, 1 above), float32 for
UncertainTea-Metal and the NumPyro sweep (`--no-x64`), float64 elsewhere.
The sweep stops at 4096 chains: larger points cost hours and add no new
information until issues #137/#138 are fixed.

`run_all.sh pinned` re-runs the sweep with every chain pinned to the
posterior mode — the issue #137 diagnostic. Those rows are labelled
`-pinned-init` and must never be quoted as default-configuration results.

## Models

| model | shape | exercises |
|---|---|---|
| `eight_schools_centered` | hierarchical, funnel | divergence behaviour, gate honesty |
| `eight_schools_noncentered` | hierarchical | `reparam=:noncentered`, `iid` latents |
| `logistic` | GLM, N=500, D=8 | loop-addressed discrete observations; host analytic path |
| `logistic_large` | GLM, N=8000, D=16 | heavy per-gradient GLM; host + device analytic path; chain-count scaling sweep |
| `gauss` | mean/scale, N=1000 | device path; chain-count scaling sweep |

`logistic_large` is a larger GLM whose per-observation D-dim dot product over N
observations gives a genuinely heavy gradient, so the device analytic path
(#135) and many-chains scaling are no longer dominated by device dispatch
overhead. Priors are identical to `logistic` (`alpha ~ N(0, 2.5)`,
`beta_d ~ N(0, 2.5)` written as a diagonal `mvnormal`). D is 16 because the
device coefficient-dimension cap is 16 (`DEVICE_MAX_VECTOR_DIMENSION`); N is
raised to 8000 to keep the per-gradient work budget large while the model still
lowers to the device (`device_lowering_report` is supported).

## Adding a model

1. Generate/check in the dataset in `data/` (extend `generate_data.jl`).
2. Add the model to `julia/models.jl` + `setup_model` in `julia/run.jl`,
   `python/stan/<name>.stan` + `PARAMS` in `run_stan.py`, and
   `build_model` in `run_numpyro.py` — with identical priors.
3. Add it to `MODELS` in `run_all.sh`.

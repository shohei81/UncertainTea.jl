"""CmdStan runner for the cross-PPL benchmark (issue #121).

Usage (from bench/crossppl):
    uv run --project python python/run_stan.py --model eight_schools_noncentered \
        --chains 4 --samples 1000 --warmup 1000 --seed 1 --reps 3

CmdStan is installed on demand into python/.cmdstan (pinned version below).
Chains run as parallel processes (CmdStan's normal mode).  Warmup and
sampling wall-clock come from CmdStan's own per-chain elapsed-time report
(max over chains, since chains run concurrently); model compilation is a
one-time cost reported as compile_s.
"""

import argparse
import os
import platform
import re
import time
from pathlib import Path

import numpy as np

CMDSTAN_VERSION = "2.36.0"
HERE = Path(__file__).resolve().parent
# The benchmark container pre-bakes CmdStan and points this at /opt/cmdstan.
CMDSTAN_DIR = Path(os.environ.get("CROSSPPL_CMDSTAN_DIR", HERE / ".cmdstan"))

# Canonical exported parameters per model (Stan column naming).
PARAMS = {
    "eight_schools_centered": ["mu", "tau"] + [f"theta[{i}]" for i in range(1, 9)],
    "eight_schools_noncentered": ["mu", "tau"] + [f"theta[{i}]" for i in range(1, 9)],
    "logistic": ["alpha"] + [f"beta[{i}]" for i in range(1, 9)],
    "logistic_large": ["alpha"] + [f"beta[{i}]" for i in range(1, 17)],
    "gauss": ["mu", "s"],
    "mixture": ["mu1", "log_gap", "s"],
    "poisson_glm": ["a", "b"],
    "schools_large": ["mu", "log_tau"] + [f"theta[{i}]" for i in range(1, 33)],
    # Omega[1] == 1 is a structural constant (zero variance) the gate skips.
    "lkj": ["Omega[1]", "Omega[2]", "Omega[3]", "tau[1]", "tau[2]"],
}


def ensure_cmdstan():
    import cmdstanpy

    installed = sorted(CMDSTAN_DIR.glob(f"cmdstan-{CMDSTAN_VERSION}"))
    if not installed:
        CMDSTAN_DIR.mkdir(parents=True, exist_ok=True)
        cmdstanpy.install_cmdstan(dir=str(CMDSTAN_DIR), version=CMDSTAN_VERSION)
    cmdstanpy.set_cmdstan_path(str(CMDSTAN_DIR / f"cmdstan-{CMDSTAN_VERSION}"))


def stan_data(model_name: str, data: dict) -> dict:
    if model_name.startswith("eight_schools"):
        return {"J": data["J"], "y": data["y"], "sigma": data["sigma"]}
    if model_name in ("logistic", "logistic_large"):
        return {"N": data["n"], "D": data["d"], "X": data["X"], "y": data["y"]}
    if model_name in ("gauss", "mixture"):
        return {"N": data["n"], "y": data["y"]}
    if model_name == "poisson_glm":
        return {"N": data["n"], "x": data["x"], "y": data["y"]}
    if model_name == "schools_large":
        return {"J": data["J"], "y": data["y"], "sigma": data["sigma"]}
    if model_name == "lkj":
        return {"N": data["n"], "y": data["y"]}
    raise ValueError(f"unknown model {model_name}")


def elapsed_times(fit) -> tuple[float, float]:
    """Max over chains of CmdStan's reported (warmup, sampling) seconds."""
    warmups, samplings = [], []
    for csv in fit.runset.csv_files:
        text = Path(csv).read_text()
        w = re.search(r"([0-9.eE+-]+) seconds \(Warm-up\)", text)
        s = re.search(r"([0-9.eE+-]+) seconds \(Sampling\)", text)
        if w and s:
            warmups.append(float(w.group(1)))
            samplings.append(float(s.group(1)))
    if not warmups:
        raise RuntimeError("could not parse CmdStan elapsed-time report")
    return max(warmups), max(samplings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PARAMS))
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    args = ap.parse_args()

    ensure_cmdstan()
    import cmdstanpy

    from common import git_commit, load_data, write_result

    data_name = "eight_schools" if args.model.startswith("eight_schools") else args.model
    data = stan_data(args.model, load_data(data_name))
    stan_file = HERE / "stan" / f"{args.model}.stan"

    # Compiled binaries are platform-specific: compile from a per-platform
    # copy of the .stan file so the host and the Linux container never reuse
    # each other's executables through the shared mount.
    build_dir = HERE / "stan" / "build" / f"{platform.system().lower()}-{platform.machine()}"
    build_dir.mkdir(parents=True, exist_ok=True)
    build_stan = build_dir / stan_file.name
    if not build_stan.exists() or build_stan.read_text() != stan_file.read_text():
        build_stan.write_text(stan_file.read_text())
    t0 = time.perf_counter()
    model = cmdstanpy.CmdStanModel(stan_file=str(build_stan))
    compile_s = time.perf_counter() - t0
    print(f"compile: {compile_s:.2f} s (cached if ~0)")

    params = PARAMS[args.model]
    for rep in range(1, args.reps + 1):
        seed = args.seed + rep
        t0 = time.perf_counter()
        fit = model.sample(
            data=data, chains=args.chains, parallel_chains=args.chains,
            iter_warmup=args.warmup, iter_sampling=args.samples, seed=seed,
            adapt_delta=args.target_accept, max_treedepth=args.max_tree_depth,
            show_progress=False,
        )
        wall_s = time.perf_counter() - t0
        warmup_s, sampling_s = elapsed_times(fit)

        raw = fit.draws(concat_chains=False)  # (draws, chains, cols)
        cols = list(fit.column_names)
        idx = [cols.index(p) for p in params]
        draws = np.transpose(raw[:, :, idx], (1, 0, 2)).astype(np.float64)
        div_rate = float(np.mean(raw[:, :, cols.index("divergent__")]))

        stem = write_result(
            model=args.model, framework="stan", variant=None,
            params=params, draws=draws,
            num_chains=args.chains, num_samples=args.samples, num_warmup=args.warmup,
            seed=seed, rep=rep,
            timings={"compile_s": compile_s, "warmup_s": warmup_s,
                     "sampling_s": sampling_s, "total_s": warmup_s + sampling_s,
                     "wall_s": wall_s},
            sampler={"kind": "nuts", "target_accept": args.target_accept,
                     "max_tree_depth": args.max_tree_depth, "metric": "diag",
                     "chain_parallelism": "process-parallel",
                     "precision": "float64"},
            env={"cmdstan": CMDSTAN_VERSION, "cmdstanpy": cmdstanpy.__version__,
                 "bench_commit": git_commit()},
            diagnostics={"divergence_rate": div_rate},
        )
        print(f"{stem}  warmup={warmup_s:.3f}s  sampling={sampling_s:.3f}s  div={div_rate:.4f}")


if __name__ == "__main__":
    main()

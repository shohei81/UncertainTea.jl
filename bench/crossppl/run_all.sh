#!/usr/bin/env bash
# Orchestrates the cross-PPL benchmark (issue #121).
#
#   ./run_all.sh cpu      correctness pass + CPU scaling legs
#                         (run natively, or inside the docker/ container for
#                         the pinned fair-comparison environment)
#   ./run_all.sh metal    Metal scaling legs (native macOS only)
#   ./run_all.sh pinned   diagnostic sweep with pinned initialization — the
#                         issue #137 workaround; rows are labelled
#                         "-pinned-init" and must never be quoted as
#                         default-configuration results
#   ./run_all.sh pathfinder  Pathfinder warm-start leg (issue #162): chains start
#                         from Pathfinder draws and the warmup initial inverse-mass
#                         metric is seeded from the Pathfinder covariance diagonal;
#                         rows labelled "-pathfinder-init"
#   ./run_all.sh analyze  shared ArviZ diagnostics -> results/summary.{json,md}
#
# All legs append to results/raw/; analyze reads everything found there.
set -euo pipefail
cd "$(dirname "$0")"

# Correctness pass settings (4 chains is the conventional cross-PPL setup).
CHAINS=4
SAMPLES=1000
WARMUP=1000
REPS=3
SEED=100

# Scaling sweep settings (short chains, many of them).  500 draws/chain so a
# healthy sampler clears the R-hat<1.01 gate (200 was inside its noise
# floor).  Capped at 4096 chains: the 16384 points cost hours and add no new
# information until issues #137/#138 are fixed.  Large-chain runs are minutes
# long, so timer dispersion is negligible and one repetition keeps the sweep
# tractable; short runs keep 3 repetitions.
SCALE_CHAINS=(64 512 4096)
SCALE_SAMPLES=500
SCALE_WARMUP=200
SCALE_SEED=200
# Unconstrained posterior-mode init for the gauss model (mu, log s) — the
# issue #137 diagnostic workaround used by the `pinned` mode.
GAUSS_PINNED_INIT="0.5,0.18232155679395463"
scale_reps() { if [[ "$1" -ge 4096 ]]; then echo 1; else echo 3; fi; }

MODELS=(eight_schools_centered eight_schools_noncentered logistic logistic_large gauss mixture lkj)

if [[ "${CROSSPPL_IN_CONTAINER:-0}" == "1" ]]; then
    PY=(uv run --project /opt/bench-python --no-sync python)
else
    PY=(uv run --project python python)
fi
# -t auto: only the batched-cpu-ka variant uses threads; harmless elsewhere.
JL=(julia -t auto --project=julia)

run_cpu() {
    "${JL[@]}" -e 'using Pkg; Pkg.instantiate()'
    for model in "${MODELS[@]}"; do
        echo "=== $model: correctness pass ==="
        "${PY[@]}" python/run_stan.py --model "$model" \
            --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
        "${PY[@]}" python/run_numpyro.py --model "$model" --chain-method parallel \
            --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
        "${JL[@]}" julia/run.jl --model "$model" --variant cpu \
            --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
        "${JL[@]}" julia/run.jl --model "$model" --variant batched-cpu \
            --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
    done
    echo "=== gauss: CPU scaling sweep ==="
    for k in "${SCALE_CHAINS[@]}"; do
        r=$(scale_reps "$k")
        "${PY[@]}" python/run_numpyro.py --model gauss --chain-method vectorized --no-x64 \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$r"
        "${JL[@]}" julia/run.jl --model gauss --variant batched-cpu \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$r"
        "${JL[@]}" julia/run.jl --model gauss --variant batched-cpu-ka \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$r"
    done
    # logistic_large (D=16, N=8000) is the device-story model: a heavy
    # per-gradient GLM that lowers to the device analytic path (issue #135), so
    # the device/many-chains legs are no longer dominated by dispatch overhead.
    echo "=== logistic_large: CPU scaling sweep ==="
    for k in "${SCALE_CHAINS[@]}"; do
        r=$(scale_reps "$k")
        "${PY[@]}" python/run_numpyro.py --model logistic_large --chain-method vectorized --no-x64 \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$r"
        "${JL[@]}" julia/run.jl --model logistic_large --variant batched-cpu \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$r"
        "${JL[@]}" julia/run.jl --model logistic_large --variant batched-cpu-ka \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$r"
    done
}

run_metal() {
    "${JL[@]}" -e 'using Pkg; Pkg.instantiate()'
    echo "=== gauss: Metal scaling sweep (native) ==="
    for k in "${SCALE_CHAINS[@]}"; do
        "${JL[@]}" julia/run.jl --model gauss --variant batched-metal \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$(scale_reps "$k")"
    done
    echo "=== logistic_large: Metal scaling sweep (native) ==="
    for k in "${SCALE_CHAINS[@]}"; do
        "${JL[@]}" julia/run.jl --model logistic_large --variant batched-metal \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$(scale_reps "$k")"
    done
}

run_persistent() {
    # Persistent per-chain tree kernel (issue #154): one device kernel launch per NUTS
    # iteration builds each chain's whole tree, GPU-native. Reported as its own
    # `uncertaintea-batched-metal-persistent` label alongside the `:masked`
    # `batched-metal` rows. The CPU()-Float64 leg is the debuggable correctness-gate
    # reference (the Metal leg is Float32, statistically -- not bitwise -- equivalent).
    # Scoped to gauss (the many-chain story model); observation-tiling the in-kernel
    # gradient for heavy-per-gradient models like logistic_large is increment 4.
    "${JL[@]}" -e 'using Pkg; Pkg.instantiate()'
    echo "=== gauss: persistent-NUTS Metal scaling sweep ==="
    for k in "${SCALE_CHAINS[@]}"; do
        "${JL[@]}" julia/run.jl --model gauss --variant batched-metal-persistent \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$(scale_reps "$k")"
    done
    echo "=== gauss: persistent-NUTS CPU() Float64 correctness-gate leg ==="
    "${JL[@]}" julia/run.jl --model gauss --variant batched-cpu-persistent \
        --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
}

run_chees() {
    # ChEES-HMC leg (issue #161): a different sampler (fixed-length jittered HMC
    # with cross-chain trajectory-length adaptation), reported as its own
    # `uncertaintea-chees` label rather than replacing NUTS. ChEES's optimal target
    # accept is 0.651 (vs NUTS's 0.8). The Stan/NumPyro/NUTS reference rows come
    # from the `cpu` leg.
    #
    # gauss is the many-chain GPU-story model; eight_schools_noncentered is the
    # funnel/heavy-tailed leg. The funnel was previously excluded (#203) because
    # `batched_hmc`/`batched_chees` THREW on it -- the noncentered-reparam
    # finite-check was not routed through the reject-invalid-parameters path inside
    # the batched HMC leapfrog gradient (a #157-class gap; `batched_nuts` was
    # unaffected). Fixed in #202: a non-finite loc/scale is now a rejected/divergent
    # proposal (-Inf log-joint), so ChEES samples the funnel robustly.
    "${JL[@]}" -e 'using Pkg; Pkg.instantiate()'
    echo "=== gauss: ChEES pass ==="
    "${JL[@]}" julia/run.jl --model gauss --variant chees --target-accept 0.651 \
        --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
    echo "=== eight_schools_noncentered: ChEES pass ==="
    "${JL[@]}" julia/run.jl --model eight_schools_noncentered --variant chees --target-accept 0.651 \
        --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
    echo "=== gauss: ChEES scaling sweep ==="
    for k in "${SCALE_CHAINS[@]}"; do
        "${JL[@]}" julia/run.jl --model gauss --variant chees --target-accept 0.651 \
            --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
            --seed $SCALE_SEED --reps "$(scale_reps "$k")"
    done
}

run_pinned() {
    "${JL[@]}" -e 'using Pkg; Pkg.instantiate()'
    echo "=== gauss: pinned-init diagnostic sweep (issue #137 workaround) ==="
    for v in batched-cpu-ka batched-metal; do
        for k in "${SCALE_CHAINS[@]}"; do
            "${JL[@]}" julia/run.jl --model gauss --variant "$v" \
                --init "$GAUSS_PINNED_INIT" \
                --chains "$k" --samples $SCALE_SAMPLES --warmup $SCALE_WARMUP \
                --seed $SCALE_SEED --reps "$(scale_reps "$k")"
        done
    done
}

run_pathfinder() {
    # Pathfinder warm-start leg (issue #162): each chain starts from a Pathfinder
    # draw AND the warmup initial inverse-mass metric is seeded from the Pathfinder
    # covariance diagonal. Rows are labelled "-pathfinder-init" and carry
    # sampler.init="pathfinder" so they are compared against, not mistaken for, the
    # default prior-draw rows from the `cpu` leg. gauss is the many-chain model;
    # eight_schools_noncentered is the funnel/heavy-tailed leg. Runs at the
    # correctness-gate chains/samples on the host per-chain (batched-cpu) path.
    "${JL[@]}" -e 'using Pkg; Pkg.instantiate()'
    for model in gauss eight_schools_noncentered; do
        echo "=== $model: Pathfinder warm-start pass (issue #162) ==="
        "${JL[@]}" julia/run.jl --model "$model" --variant batched-cpu \
            --init pathfinder \
            --chains $CHAINS --samples $SAMPLES --warmup $WARMUP --seed $SEED --reps $REPS
    done
}

case "${1:-}" in
    cpu) run_cpu ;;
    metal) run_metal ;;
    chees) run_chees ;;
    persistent) run_persistent ;;
    pinned) run_pinned ;;
    pathfinder) run_pathfinder ;;
    analyze) "${PY[@]}" python/analyze.py ;;
    *) echo "usage: $0 {cpu|metal|chees|persistent|pinned|pathfinder|analyze}" >&2; exit 1 ;;
esac

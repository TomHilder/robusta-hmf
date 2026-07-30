#!/usr/bin/env bash
# Run the full main-sequence Gaia RVS analysis (training grid + CV + outliers)
# on however many GPUs are visible. Safe to re-run: finished models are
# skipped, so an interrupted run resumes where it left off.
#
# USAGE:
#   ./run_full_ms.sh                 # use all visible GPUs
#   N_GPUS=1 ./run_full_ms.sh        # force a single GPU
#   ./run_full_ms.sh --ranks 5 10 15 --q-vals 3.0 5.0   # smaller grid
#
# Logs go to full_ms_shard<i>.log; the analysis stage logs to full_ms_analyse.log.

set -euo pipefail
cd "$(dirname "$0")"

EXTRA_ARGS=("$@")

# How many GPUs?
if [[ -z "${N_GPUS:-}" ]]; then
    if command -v nvidia-smi > /dev/null 2>&1; then
        N_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
    else
        N_GPUS=1
    fi
fi
echo "Using ${N_GPUS} GPU(s)"

# --- Stage 1: train the (K, Q) grid, one shard per GPU --- #
pids=()
for ((g = 0; g < N_GPUS; g++)); do
    echo "Launching shard ${g}/${N_GPUS} on GPU ${g} (log: full_ms_shard${g}.log)"
    CUDA_VISIBLE_DEVICES=${g} uv run python train_full_ms.py \
        --shard "${g}" --n-shards "${N_GPUS}" "${EXTRA_ARGS[@]}" \
        > "full_ms_shard${g}.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "${pid}" || fail=1
done
if [[ ${fail} -ne 0 ]]; then
    echo "ERROR: at least one training shard failed -- check full_ms_shard*.log"
    echo "Re-running ./run_full_ms.sh will resume from the completed models."
    exit 1
fi
echo "Training complete."

# --- Stage 2: CV scoring, best-model selection, outliers (single GPU) --- #
echo "Running analysis (log: full_ms_analyse.log)"
CUDA_VISIBLE_DEVICES=0 uv run python analyse_full_ms.py "${EXTRA_ARGS[@]}" \
    > full_ms_analyse.log 2>&1
tail -n 6 full_ms_analyse.log
echo "All done. Outputs: gaia_rvs_results/full_ms_* and plots_full_ms/"

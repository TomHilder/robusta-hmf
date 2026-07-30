#!/usr/bin/env bash
# Run the full main-sequence Gaia RVS analysis (training grid + CV + outliers)
# on however many GPUs are visible. Safe to re-run: finished models are
# skipped, so an interrupted run resumes where it left off.
#
# USAGE:
#   ./run_full_ms.sh                 # main-sequence sample, all visible GPUs
#   ./run_full_ms.sh --sample all    # the WHOLE matched RVS sample instead
#   N_GPUS=1 ./run_full_ms.sh        # force a single GPU
#   ./run_full_ms.sh --ranks 5 10 15 --q-vals 3.0 5.0   # smaller grid
#
# Extra args are passed to both the training and analysis stages, so --sample,
# --ranks, and --q-vals stay consistent across stages automatically.
# Logs go to full_ms_shard<i>.log; the analysis stage logs to full_ms_analyse.log.

set -euo pipefail
cd "$(dirname "$0")"

# Use the CUDA-enabled jax/jaxlib from the module python's system site-packages.
# The venv is deliberately synced without jax/jaxlib (they would be CPU-only
# wheels that shadow the system build), so never let uv re-add them:
#   uv sync --all-groups --no-install-package jax --no-install-package jaxlib
export UV_NO_SYNC=1

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

# Older jaxlib builds emit one of these per kernel compilation. Pure noise, but
# enough of it to bury the actual progress lines, so keep it out of the terminal
# (the log files still get everything).
NOISE='is not a recognized feature'

# Both stages below check PIPESTATUS[0] with errexit off rather than relying on
# pipefail: grep -v exits 1 when it filters out every line, which would
# otherwise be indistinguishable from the python process failing.

# --- Stage 1: train the (K, Q) grid, one shard per GPU --- #
fail=0
if [[ ${N_GPUS} -eq 1 ]]; then
    # One GPU means one shard, so there is nothing to interleave with and no
    # reason to background it: run in the foreground and mirror to the terminal.
    echo "Launching shard 0/1 on GPU 0 (log: full_ms_shard0.log, mirrored below)"
    set +e
    CUDA_VISIBLE_DEVICES=0 uv run python -u train_full_ms.py \
        --shard 0 --n-shards 1 "${EXTRA_ARGS[@]}" 2>&1 \
        | tee full_ms_shard0.log \
        | grep -v --line-buffered "${NOISE}"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || fail=1
    set -e
else
    # Mirror each shard too, tagged with its GPU so the interleaving is
    # readable. Process substitution, not a pipe, so $! stays the python PID
    # and wait below reports training's status rather than the filter's.
    pids=()
    for ((g = 0; g < N_GPUS; g++)); do
        echo "Launching shard ${g}/${N_GPUS} on GPU ${g} (log: full_ms_shard${g}.log)"
        CUDA_VISIBLE_DEVICES=${g} uv run python -u train_full_ms.py \
            --shard "${g}" --n-shards "${N_GPUS}" "${EXTRA_ARGS[@]}" \
            > >(tee "full_ms_shard${g}.log" \
                | grep -v --line-buffered "${NOISE}" \
                | sed -u "s/^/[gpu${g}] /") 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || fail=1
    done
fi
if [[ ${fail} -ne 0 ]]; then
    echo "ERROR: at least one training shard failed -- check full_ms_shard*.log"
    echo "Re-running ./run_full_ms.sh will resume from the completed models."
    exit 1
fi
echo "Training complete."

# --- Stage 2: CV scoring, best-model selection, outliers (single GPU) --- #
echo "Running analysis (log: full_ms_analyse.log)"
# Always a single process, so tee it straight through.
set +e
CUDA_VISIBLE_DEVICES=0 uv run python -u analyse_full_ms.py "${EXTRA_ARGS[@]}" 2>&1 \
    | tee full_ms_analyse.log \
    | grep -v --line-buffered "${NOISE}"
analyse_status=${PIPESTATUS[0]}
set -e
if [[ ${analyse_status} -ne 0 ]]; then
    echo "ERROR: analysis failed -- check full_ms_analyse.log"
    exit 1
fi
echo "All done. Outputs: gaia_rvs_results/full_ms_* and plots_full_ms/"

#!/usr/bin/env bash
#SBATCH --job-name=pier-e0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=pier-e0-%j.out

set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-${SLURM_SUBMIT_DIR:?submit from the Pier repository root}}"
cd "$PIER_ROOT"
[[ -f megatron/core/outer_sync/legacy.py ]] || { echo 'PIER_ROOT is not the updated repository'; exit 2; }
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PIER_GPUS_PER_NODE="${PIER_GPUS_PER_NODE:-4}"
world=$((SLURM_JOB_NUM_NODES * PIER_GPUS_PER_NODE))
if (( world < 4 || (world & (world - 1)) != 0 )); then
    echo 'E0 requires a power-of-two allocation of at least 4 GPUs'; exit 2
fi
export PIER_E0_RUN_DIR="${PIER_E0_RUN_DIR:-$PIER_ROOT/local/centered_outer/e0-${SLURM_JOB_ID}}"
# A new job must not inherit a previous successful report.
if [[ -e "$PIER_E0_RUN_DIR" ]]; then
    echo "Refusing to reuse output directory: $PIER_E0_RUN_DIR"; exit 2
fi
mkdir -p "$PIER_E0_RUN_DIR"
finish() {
    local run_status=$?
    trap - EXIT
    set +e
    "$PIER_PYTHON" experiments/centered_outer/summarize.py \
        --output-dir "$PIER_E0_RUN_DIR" --launcher-exit "$run_status"
    local summary_status=$?
    echo "E0 artifacts: $PIER_E0_RUN_DIR"
    if (( run_status != 0 )); then exit "$run_status"; fi
    exit "$summary_status"
}
trap finish EXIT
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
"$PIER_PYTHON" experiments/centered_outer/manifest.py --output-dir "$PIER_E0_RUN_DIR"
for phase in megatron-dp1 megatron-dp2 protocol; do
    echo "Starting $phase"
    srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
        bash experiments/centered_outer/node_entry.sh "$phase" \
        2>&1 | tee "$PIER_E0_RUN_DIR/$phase.log"
done

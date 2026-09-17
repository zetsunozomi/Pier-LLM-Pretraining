#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --job-name=pier-e0b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=pier-e0b-%j.out
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repository root}}"
cd "$PIER_ROOT"
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PIER_GPUS_PER_NODE=4
[[ "$SLURM_JOB_NUM_NODES" == 1 ]] || { echo 'E0b requires one four-GPU node'; exit 2; }
export PIER_E0B_RUN_DIR="${PIER_E0B_RUN_DIR:-$PIER_ROOT/local/centered_outer/e0b-${SLURM_JOB_ID}}"
[[ ! -e "$PIER_E0B_RUN_DIR" ]] || { echo 'Refusing to reuse an E0b output directory'; exit 2; }
mkdir -p "$PIER_E0B_RUN_DIR"
export PYTHONPATH="$PIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=Ring TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
finish() {
    local run_status=$?
    trap - EXIT
    set +e
    "$PIER_PYTHON" experiments/centered_outer/e0b_summarize.py \
      --output-dir "$PIER_E0B_RUN_DIR" --launcher-exit "$run_status"
    local summary_status=$?
    echo "E0b artifacts: $PIER_E0B_RUN_DIR"
    if (( run_status != 0 )); then exit "$run_status"; fi
    exit "$summary_status"
}
trap finish EXIT
"$PIER_PYTHON" experiments/centered_outer/manifest.py --stage E0b --output-dir "$PIER_E0B_RUN_DIR"
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PIER_PYTHON" experiments/centered_outer/manifest.py --node-only --output-dir "$PIER_E0B_RUN_DIR"
for phase in tp1-s1 tp1-s2 tp1-s4 tp1-host split resume tp2-s2 dp2-s2; do
    echo "Starting E0b $phase"
    srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
      bash experiments/centered_outer/e0b_node.sh "$phase" \
      2>&1 | tee "$PIER_E0B_RUN_DIR/$phase.log"
done

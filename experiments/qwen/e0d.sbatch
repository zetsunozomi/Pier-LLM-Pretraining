#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --job-name=pier-e0d-qwen3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=pier-e0d-%j.out
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PIER_ROOT"
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]] || {
    echo 'E0d requires one allocated node with four GPUs.' >&2; exit 2;
}
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PYTHONPATH="$PIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PIER_QWEN_SNAPSHOT="${PIER_QWEN_SNAPSHOT:-$PIER_ROOT/local/qwen/models/Qwen2.5-3B/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b}"
export PIER_QWEN_DATA_PREFIX="${PIER_QWEN_DATA_PREFIX:-$("$PIER_PYTHON" experiments/qwen/e0d_config.py)}"
: "${PIER_E0C_EVIDENCE:?set PIER_E0C_EVIDENCE to the passed E0c artifact directory}"
[[ -d "$PIER_QWEN_SNAPSHOT" && -d "$PIER_E0C_EVIDENCE" ]] || {
    echo 'Missing Qwen snapshot or E0c evidence directory.' >&2; exit 2;
}
for suffix in .bin .idx .manifest.json; do
    [[ -f "$PIER_QWEN_DATA_PREFIX$suffix" ]] || {
        echo 'Missing audited indexed data: run prepare_corpus.py or set PIER_QWEN_DATA_PREFIX.' >&2; exit 2;
    }
done
export PIER_E0D_RUN_DIR="${PIER_E0D_RUN_DIR:-$PIER_ROOT/local/qwen/e0d-${SLURM_JOB_ID:?missing Slurm job id}-$(date +%Y%m%d-%H%M%S)}"
[[ ! -e "$PIER_E0D_RUN_DIR" ]] || { echo 'Refusing to reuse an E0d output directory' >&2; exit 2; }
mkdir -p "$PIER_E0D_RUN_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=Ring TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=1
export SLURM_OVERLAP="${SLURM_OVERLAP:-1}"
export PIER_GPUS_PER_NODE=4
phases=(s2-device s1-host s2-host split resume)
phase=preflight
number=0
finish() {
    local run_status=$?
    trap - EXIT
    set +e
    if (( run_status != 0 )); then
        echo "[E0d $number/${#phases[@]}] FAILED $phase (exit=$run_status); remaining phases stopped." >&2
    fi
    "$PIER_PYTHON" experiments/qwen/e0d_evidence.py --summarize \
      --output-dir "$PIER_E0D_RUN_DIR" --launcher-exit "$run_status"
    local summary_status=$?
    echo "E0d artifacts: $PIER_E0D_RUN_DIR"
    echo 'Git returns JSON/log evidence only; checkpoint, oracle vectors and indexed data stay on scratch.'
    if (( run_status != 0 )); then exit "$run_status"; fi
    exit "$summary_status"
}
trap finish EXIT
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PIER_PYTHON" experiments/qwen/e0d_evidence.py --output-dir "$PIER_E0D_RUN_DIR" \
  --snapshot "$PIER_QWEN_SNAPSHOT" --data-prefix "$PIER_QWEN_DATA_PREFIX" --e0c-dir "$PIER_E0C_EVIDENCE"
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PIER_PYTHON" experiments/centered_outer/manifest.py --node-only --output-dir "$PIER_E0D_RUN_DIR"
echo 'E0d runs five short real-Qwen phases; split intentionally saves and exits at attempted step 5.'
for phase in "${phases[@]}"; do
    number=$((number + 1))
    echo "[E0d $number/${#phases[@]}] START $phase"
    srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
      "$PIER_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
      experiments/qwen/e0d_train.py --case "$phase" --output-dir "$PIER_E0D_RUN_DIR" \
      2>&1 | tee "$PIER_E0D_RUN_DIR/$phase.log"
    echo "[E0d $number/${#phases[@]}] DONE $phase (exit=0); evidence checked in final summary."
done

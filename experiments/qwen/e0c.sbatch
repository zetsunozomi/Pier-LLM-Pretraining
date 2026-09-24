#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --job-name=pier-e0c-qwen3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=pier-e0c-%j.out
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
# sbatch runs a spool copy, so its script directory may not be the repository.
# Direct bash prefers the actual script's repository even when an interactive
# allocation was requested from some other directory.
if [[ -z "${PIER_ROOT:-}" ]]; then
    script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    for candidate in "$script_root" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        if [[ -f "$candidate/pretrain_qwen.py" && -f "$candidate/experiments/qwen/e0c_evidence.py" ]]; then
            PIER_ROOT="$candidate"
            break
        fi
    done
fi
if [[ -z "${PIER_ROOT:-}" || ! -f "$PIER_ROOT/pretrain_qwen.py" || ! -f "$PIER_ROOT/experiments/qwen/e0c_evidence.py" ]]; then
    printf 'Cannot locate Pier repository (PIER_ROOT=%s, SLURM_SUBMIT_DIR=%s). Set PIER_ROOT to the repository root.\n' \
      "${PIER_ROOT:-unset}" "${SLURM_SUBMIT_DIR:-unset}" >&2
    exit 2
fi
cd "$PIER_ROOT"
export PIER_ROOT="$PWD"
source "$PIER_ROOT/experiments/qwen/log_output.sh"
pier_capture_output e0c
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]] || {
    echo 'E0c requires a Slurm allocation of one node / four GPUs (sbatch or an existing interactive allocation).' >&2
    exit 2
}
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
collection_args=()
case "${PIER_E0C_COLLECT_ALL:-0}" in
    0) ;;
    1) collection_args+=(--collect-numerical-mismatches) ;;
    *) echo 'PIER_E0C_COLLECT_ALL must be 0 or 1' >&2; exit 2 ;;
esac
export PIER_QWEN_SNAPSHOT="${PIER_QWEN_SNAPSHOT:-$PIER_ROOT/local/qwen/models/Qwen2.5-3B/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b}"
export PIER_E0C_RUN_DIR="${PIER_E0C_RUN_DIR:-$PIER_ROOT/local/qwen/e0c-${SLURM_JOB_ID:?missing Slurm job id}-$(date +%Y%m%d-%H%M%S)}"
printf '[E0c] Repository: %s\n[E0c] Snapshot: %s\n' "$PIER_ROOT" "$PIER_QWEN_SNAPSHOT"
[[ -d "$PIER_QWEN_SNAPSHOT" ]] || {
    printf 'Missing snapshot directory: %s\nSet PIER_QWEN_SNAPSHOT to the directory containing the eight pinned files.\n' \
      "$PIER_QWEN_SNAPSHOT" >&2
    exit 2
}
[[ ! -e "$PIER_E0C_RUN_DIR" ]] || { echo 'Refusing to reuse an E0c output directory' >&2; exit 2; }
mkdir -p "$PIER_E0C_RUN_DIR"
export PYTHONPATH="$PIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=Ring TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=1
export PIER_GPUS_PER_NODE=4
# Permit these sequential srun steps inside the allocation's interactive shell step.
export SLURM_OVERLAP="${SLURM_OVERLAP:-1}"
phase=preflight
finish() {
    local run_status=$?
    trap - EXIT
    set +e
    if (( run_status != 0 )); then
        echo "[E0c] FAILED $phase (exit=$run_status); remaining phases stopped." >&2
    fi
    "$PIER_PYTHON" experiments/qwen/e0c_evidence.py --summarize \
      --output-dir "$PIER_E0C_RUN_DIR" --launcher-exit "$run_status"
    local summary_status=$?
    echo "E0c artifacts: $PIER_E0C_RUN_DIR"
    echo 'Commit top-level JSON/logs only; reference-* tensors and downloaded models stay on scratch.'
    if (( run_status != 0 )); then exit "$run_status"; fi
    exit "$summary_status"
}
trap finish EXIT
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PIER_PYTHON" experiments/qwen/e0c_evidence.py \
  --output-dir "$PIER_E0C_RUN_DIR" --snapshot "$PIER_QWEN_SNAPSHOT" ${collection_args[@]+"${collection_args[@]}"}
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$PIER_PYTHON" experiments/centered_outer/manifest.py --node-only --output-dir "$PIER_E0C_RUN_DIR"
stage=0
for precision in fp32 bf16; do
    stage=$((stage + 1))
    phase="hf-$precision"
    echo "[E0c $stage/6] START $phase"
    srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
      "$PIER_PYTHON" experiments/qwen/e0c.py --phase reference --dtype "$precision" \
      --output-dir "$PIER_E0C_RUN_DIR" 2>&1 | tee "$PIER_E0C_RUN_DIR/$phase.log"
    echo "[E0c $stage/6] DONE $phase"
    for tp in 1 2; do
        stage=$((stage + 1))
        phase="native-$precision-tp$tp"
        echo "[E0c $stage/6] START $phase"
        srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
          "$PIER_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
          --nproc_per_node=4 --max_restarts=0 experiments/qwen/e0c.py \
          --phase native --dtype "$precision" --tp "$tp" --output-dir "$PIER_E0C_RUN_DIR" \
          ${collection_args[@]+"${collection_args[@]}"} \
          2>&1 | tee "$PIER_E0C_RUN_DIR/$phase.log"
        echo "[E0c $stage/6] FINISHED $phase; numerical verdict is in rank JSON and final summary."
    done
done

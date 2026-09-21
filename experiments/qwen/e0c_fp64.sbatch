#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --job-name=pier-e0c-fp64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=pier-e0c-fp64-%j.out

set -euo pipefail
if [[ -z "${PIER_ROOT:-}" ]]; then
    script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    for candidate in "$script_root" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        if [[ -f "$candidate/experiments/qwen/e0c_fp64.py" && -f "$candidate/pretrain_qwen.py" ]]; then
            PIER_ROOT="$candidate"
            break
        fi
    done
fi
[[ -n "${PIER_ROOT:-}" && -f "$PIER_ROOT/experiments/qwen/e0c_fp64.py" ]] || {
    echo 'Set PIER_ROOT to the updated Pier repository.' >&2; exit 2;
}
cd "$PIER_ROOT"
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]] || {
    echo 'Requires an existing one-node/four-GPU allocation or sbatch.' >&2; exit 2;
}
[[ -n "${PIER_E0C_FP32_RUN:-}" && -f "$PIER_E0C_FP32_RUN/native-fp32-tp1-rank0.json" ]] || {
    echo 'Set PIER_E0C_FP32_RUN to the original E0c output directory.' >&2; exit 2;
}
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PIER_E0C_FP64_RUN_DIR="${PIER_E0C_FP64_RUN_DIR:-$PWD/local/qwen/e0c-fp64-${SLURM_JOB_ID:?missing Slurm job id}-$(date +%Y%m%d-%H%M%S)}"
[[ ! -e "$PIER_E0C_FP64_RUN_DIR" ]] || { echo 'Refusing to reuse an output directory' >&2; exit 2; }
mkdir -p "$PIER_E0C_FP64_RUN_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=Ring TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=1
export SLURM_OVERLAP="${SLURM_OVERLAP:-1}"
phase=prepare
finish() {
    local run_status=$?
    trap - EXIT
    set +e
    if (( run_status != 0 )); then echo "[E0c FP64] FAILED $phase (exit=$run_status)" >&2; fi
    "$PIER_PYTHON" experiments/qwen/e0c_fp64.py --phase summary \
        --output-dir "$PIER_E0C_FP64_RUN_DIR" --launcher-exit "$run_status"
    local summary_status=$?
    echo "FP64 diagnostic artifacts: $PIER_E0C_FP64_RUN_DIR"
    echo 'Return top-level JSON/logs through Git; reference-fp64 tensors stay on scratch.'
    if (( run_status != 0 )); then exit "$run_status"; fi
    exit "$summary_status"
}
trap finish EXIT
printf '[E0c FP64] Original run: %s\n' "$PIER_E0C_FP32_RUN"
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    "$PIER_PYTHON" experiments/qwen/e0c_fp64.py --phase prepare \
    --fp32-run "$PIER_E0C_FP32_RUN" --output-dir "$PIER_E0C_FP64_RUN_DIR"
phase=hf
echo '[E0c FP64 1/2] START hf (one GPU)'
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    "$PIER_PYTHON" experiments/qwen/e0c_fp64.py --phase hf \
    --output-dir "$PIER_E0C_FP64_RUN_DIR" 2>&1 | tee "$PIER_E0C_FP64_RUN_DIR/hf.log"
phase=native
echo '[E0c FP64 2/2] START native TP1 (one GPU)'
srun --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    "$PIER_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    --max_restarts=0 experiments/qwen/e0c_fp64.py --phase native \
    --output-dir "$PIER_E0C_FP64_RUN_DIR" 2>&1 | tee "$PIER_E0C_FP64_RUN_DIR/native.log"

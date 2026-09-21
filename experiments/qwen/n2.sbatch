#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --job-name=pier-n2-qwen3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=pier-n2-%j.out

set -euo pipefail
if [[ -z "${PIER_ROOT:-}" ]]; then
    script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    for candidate in "$script_root" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        if [[ -f "$candidate/experiments/qwen/n2.py" && -f "$candidate/pretrain_qwen.py" ]]; then
            PIER_ROOT="$candidate"
            break
        fi
    done
fi
[[ -n "${PIER_ROOT:-}" && -f "$PIER_ROOT/experiments/qwen/n2.py" ]] || {
    echo 'Cannot locate Pier; set PIER_ROOT to the repository root.' >&2; exit 2;
}
cd "$PIER_ROOT"
export PIER_ROOT="$PWD"
source "$PIER_ROOT/experiments/qwen/log_output.sh"
pier_capture_output n2
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PYTHONPATH="$PIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Allows nested srun from an interactive, exclusive four-GPU shell.
export SLURM_OVERLAP="${SLURM_OVERLAP:-1}"
export PIER_N2_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
if [[ "$PIER_N2_NODES" != 1 ]]; then
    export PIER_N2_MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?missing allocation host list}" | head -n 1)"
fi
export PIER_N2_MASTER_PORT="${PIER_N2_MASTER_PORT:-29541}"
exec "$PIER_PYTHON" experiments/qwen/n2.py --output-dir "$PIER_OUT_DIR"

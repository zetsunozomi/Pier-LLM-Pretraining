#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-capacity40
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --array=0-4%1
#SBATCH --output=out/capacity40-%A_%a.out
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@90
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
if [[ -z "${PIER_ROOT:-}" ]]; then
    script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    for candidate in "$script_root" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        if [[ -f "$candidate/experiments/qwen/capacity.py" ]]; then
            PIER_ROOT="$candidate"
            break
        fi
    done
fi
cd "${PIER_ROOT:?Set PIER_ROOT to the Pier checkout}"
export PIER_ROOT="$PWD"
if [[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" != 8 ]]; then
    echo 'Capacity recipe requires eight nodes, four A100 40GB GPUs/node (TP2, K16).' >&2
    exit 2
fi
arms=(R G OS P2 P16 O W)
arm="${1:-${arms[${SLURM_ARRAY_TASK_ID:-0}]}}"
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PIER_CAPACITY_DIR="${PIER_CAPACITY_DIR:-$PIER_ROOT/out/capacity40-depth-v1}"
mkdir -p "$PIER_CAPACITY_DIR/$arm"
PIER_CAPACITY_DIR=$(cd "$PIER_CAPACITY_DIR" && pwd)
export PIER_CAPACITY_DIR
log="$PIER_CAPACITY_DIR/$arm/job-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$.out.txt"
printf '[capacity] Driver log: %s\n' "$log"
exec >"$log" 2>&1
export PYTHONPATH="$PIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SLURM_OVERLAP="${SLURM_OVERLAP:-1}" SLURM_EXPORT_ENV=ALL
export PIER_CAPACITY_MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)"
export PIER_CAPACITY_MASTER_PORT="${PIER_CAPACITY_MASTER_PORT:-29561}"
snapshot="${PIER_CAPACITY_TOKENIZER:-$PIER_ROOT/local/qwen/models/Qwen2.5-3B/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b}"
exec "$PIER_PYTHON" experiments/qwen/capacity.py \
    --output-dir "$PIER_CAPACITY_DIR" --arm "$arm" --snapshot "$snapshot" \
    --start-layers "${PIER_CAPACITY_START_LAYERS:-36}" \
    --ceiling-layers "${PIER_CAPACITY_MAX_LAYERS:-256}" \
    --budget-seconds "${PIER_CAPACITY_BUDGET_SECONDS:-3300}"

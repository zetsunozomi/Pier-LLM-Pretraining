#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-joint-qwen
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=pier-joint-qwen-%A_%a.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PIER_ROOT/experiments/joint/environment.sh"
source experiments/qwen/log_output.sh
pier_capture_output joint-qwen
scenario="${1:-fixed}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PIER_PYTHON" experiments/joint/qwen.py --scenario "$scenario" \
    --repeat-id "${SLURM_ARRAY_TASK_ID:-1}" \
    --output-dir "$PIER_OUT_DIR/campaign" "$@"

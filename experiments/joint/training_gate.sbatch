#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-joint-training-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=pier-joint-training-gate-%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PIER_ROOT/experiments/joint/environment.sh"
[[ "$PIER_JOINT_NODES" == 1 ]] || { echo 'Training gate requires exactly one four-GPU node.' >&2; exit 2; }
source experiments/qwen/log_output.sh
pier_capture_output joint-training-gate
exec "$PIER_PYTHON" experiments/joint/training_gate.py --output-dir "$PIER_OUT_DIR/gate"

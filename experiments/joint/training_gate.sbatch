#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-joint-training-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=pier-joint-training-gate-%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PIER_ROOT/experiments/joint/environment.sh"
[[ "$PIER_JOINT_NODES" == 1 ]] || { echo 'Training gate requires exactly one four-GPU node.' >&2; exit 2; }
if [[ -n "${PIER_JOINT_TRAINING_RESUME:-}" ]]; then
    gate_dir="$PIER_JOINT_TRAINING_RESUME"
    extra=(--output-dir "$gate_dir" --resume)
    echo "Continuing training gate: $gate_dir"
else
    source experiments/qwen/log_output.sh
    pier_capture_output joint-training-gate
    gate_dir="$PIER_OUT_DIR/gate"
    extra=(--output-dir "$gate_dir")
fi
exec "$PIER_PYTHON" experiments/joint/training_gate.py "${extra[@]}" "$@"

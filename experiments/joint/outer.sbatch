#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-joint-outer
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=pier-joint-outer-%A_%a.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PIER_ROOT/experiments/joint/environment.sh"
stage="${1:-verify}"
if [[ $# -gt 0 ]]; then shift; fi
source experiments/qwen/log_output.sh
pier_capture_output "joint-$stage"
tp="${PIER_JOINT_TP:-2}"
repeat="${SLURM_ARRAY_TASK_ID:-1}"
if [[ "$stage" != verify ]]; then
    "$PIER_PYTHON" experiments/joint/require_gate.py "${PIER_JOINT_GATE:?set the passed CUDA gate JSON}" \
        --world "$((PIER_JOINT_NODES * 4))" --tp "$tp"
fi
case "$stage" in
    verify) command=(experiments/joint/verify.py --tp "$tp") ;;
    pipeline|transitions)
        command=(experiments/joint/benchmark.py --tp "$tp" --scenario "$stage" --repeat-id "$repeat") ;;
    numerics) command=(experiments/joint/numerics.py --tp "$tp") ;;
    *) echo 'Expected verify, pipeline, transitions, or numerics.' >&2; exit 2 ;;
esac
srun --nodes="$PIER_JOINT_NODES" --ntasks="$PIER_JOINT_NODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --gpu-bind=none bash experiments/joint/node.sh \
    "${command[@]}" "$@" --output "$PIER_OUT_DIR/report.json"

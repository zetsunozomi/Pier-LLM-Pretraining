#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-n2-main-orp
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=pier-n2-main-%A_%a.out
#SBATCH --export=ALL

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
[[ -n "${PIER_ROOT:-}" && -f "$PIER_ROOT/experiments/qwen/n2.sbatch" ]] || {
    echo 'Cannot locate Pier; set PIER_ROOT to the repository root.' >&2; exit 2;
}
export PIER_ROOT
# O = naive unsharded/pageable CPU state; OS is the separate sharded ablation.
export PIER_N2_SUITE=baselines PIER_N2_ARMS=O,R,P
# Default: one short calibration window. Main windows use --time=01:00:00.
export PIER_N2_PROFILE="${PIER_N2_PROFILE:-pilot}"
export PIER_N2_REPEATS=1
export PIER_N2_REPEAT_START="${SLURM_ARRAY_TASK_ID:-${PIER_N2_REPEAT_START:-1}}"
export PIER_N2_EXPECTED_GPU='NVIDIA A100-SXM4-40GB'
exec bash "$PIER_ROOT/experiments/qwen/n2.sbatch"

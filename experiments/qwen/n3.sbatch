#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-n3-cohorts
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=pier-n3-%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
# Use the same launcher/training path as N2, including interactive allocations.
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
export PIER_N2_SUITE=cohorts
export PIER_N2_PROFILE="${PIER_N2_PROFILE:-pilot}"
export PIER_N2_EXPECTED_GPU='NVIDIA A100-SXM4-40GB'
export PIER_N3_REFERENCE="${PIER_N3_REFERENCE-$PIER_ROOT/out/n2-58678984-20260921-000738.rUk4TS}"
exec bash "$PIER_ROOT/experiments/qwen/n2.sbatch"

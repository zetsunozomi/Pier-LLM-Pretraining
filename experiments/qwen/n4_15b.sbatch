#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-n4-qwen15b-orp
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=pier-n4-qwen15b-%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

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
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 8 ]] || {
    echo 'N4-1.5B requires an eight-node allocation (32 A100-40GB GPUs); submit with sbatch.' >&2
    exit 2
}
export PIER_ROOT
# Freeze the model-size coverage point, including stale variables from 3B runs.
export PIER_QWEN_MODEL_SIZE=1.5B
export PIER_QWEN_SNAPSHOT="$PIER_ROOT/local/qwen/models/Qwen2.5-1.5B/8faed761d45a263340a0528343f099c05c9a4323"
export PIER_N2_SUITE=baselines PIER_N2_ARMS=O,R,P
export PIER_N2_PROFILE=pilot PIER_N2_REPEATS=1 PIER_N2_REPEAT_START=1
export PIER_N2_COHORT=2 PIER_N2_WORKSPACE_MIB=64
export PIER_N2_EXPECTED_GPU='NVIDIA A100-SXM4-40GB'
export PIER_QWEN_DATA_PREFIX=
# Reuse the tested launcher, logging and measurement path; no new backend.
exec bash "$PIER_ROOT/experiments/qwen/n2.sbatch"

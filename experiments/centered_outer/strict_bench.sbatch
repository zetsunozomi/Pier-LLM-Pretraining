#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --job-name=pier-strict-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=pier-strict-bench-%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
if [[ -z "${PIER_ROOT:-}" ]]; then
    script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    for candidate in "$script_root" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        if [[ -f "$candidate/experiments/centered_outer/benchmark_outer.py" ]]; then
            PIER_ROOT="$candidate"
            break
        fi
    done
fi
cd "${PIER_ROOT:?Cannot find Pier; set PIER_ROOT to the repository root}"
export PIER_ROOT="$PWD"
[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo 'Run with sbatch, or bash inside an existing Slurm GPU allocation.' >&2; exit 2;
}
export PIER_STRICT_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
profile="${PIER_STRICT_PROFILE:-smoke}"
case "$profile:$PIER_STRICT_NODES" in
    smoke:1)
        tp=1; gate_nproc=4; elements=16777217 ;;
    flat32:8)
        tp=2; gate_nproc=2
        # run-1-P/cycles-rank-0.json in n2-58728454-20260922-075100.gEXpVH:
        # padded_master_bytes=6172176384, hence 1543044096 FP32 coordinates.
        elements=1543044096 ;;
    *) echo 'Use profile smoke with 1 node, or flat32 with 8 nodes.' >&2; exit 2 ;;
esac
source "$PIER_ROOT/experiments/qwen/log_output.sh"
pier_capture_output "strict-bench-$profile"
phase=setup
trap 'code=$?; printf "[strict-bench] phase=%s exit=%s artifacts=%s\n" "$phase" "$code" "$PIER_OUT_DIR"' EXIT
export PIER_PYTHON="${PIER_PYTHON:-$(command -v python)}"
export PYTHONPATH="$PIER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 SLURM_EXPORT_ENV=ALL
export SLURM_OVERLAP="${SLURM_OVERLAP:-1}"
export PIER_STRICT_MASTER_PORT="${PIER_STRICT_MASTER_PORT:-29671}"
if [[ "$PIER_STRICT_NODES" == 1 ]]; then
    export PIER_STRICT_MASTER_ADDR=127.0.0.1
else
    export PIER_STRICT_MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)"
fi
launch() {
    srun --nodes="$PIER_STRICT_NODES" --ntasks="$PIER_STRICT_NODES" \
        --ntasks-per-node=1 --kill-on-bad-exit=1 --gpu-bind=none \
        bash "$PIER_ROOT/experiments/centered_outer/strict_bench_node.sh" "$@"
}
printf '[strict-bench] profile=%s nodes=%s benchmark_tp=%s elements=%s\n' \
    "$profile" "$PIER_STRICT_NODES" "$tp" "$elements"
phase=correctness
# flat32: 2 GPUs/node x 8 nodes validate a 16-learner tree before timing
# two corresponding-coordinate groups of 16 learners on all 32 GPUs.
launch "$gate_nproc" experiments/centered_outer/reference/verify_executor.py \
    --device cuda --schedule contiguous --output "$PIER_OUT_DIR/correctness.json"
phase=benchmark
export PIER_STRICT_MASTER_PORT=$((PIER_STRICT_MASTER_PORT + 1))
launch 4 experiments/centered_outer/benchmark_outer.py \
    --device cuda --tp "$tp" --elements "$elements" --cohort 2 \
    --workspace-mib 64 --warmup 3 --samples 10 --repeats 1 \
    --arms reference contiguous recenter --output "$PIER_OUT_DIR/benchmark.json"
phase=summary
"$PIER_PYTHON" experiments/centered_outer/summarize_strict.py "$PIER_OUT_DIR"
phase=complete

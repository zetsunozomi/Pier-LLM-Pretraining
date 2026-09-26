#!/usr/bin/env bash
# One torchrun launcher per Slurm node; invoked by strict_bench.sbatch.
set -euo pipefail
cd "${PIER_ROOT:?}"
nproc="${1:?number of workers per node required}"
shift
exec "${PIER_PYTHON:?}" -m torch.distributed.run \
    --nnodes="${PIER_STRICT_NODES:?}" --nproc_per_node="$nproc" \
    --node_rank="${SLURM_PROCID:-0}" --max_restarts=0 \
    --master_addr="${PIER_STRICT_MASTER_ADDR:?}" \
    --master_port="${PIER_STRICT_MASTER_PORT:?}" "$@"

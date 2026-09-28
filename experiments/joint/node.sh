#!/usr/bin/env bash
set -euo pipefail
cd "${PIER_ROOT:?}"
exec "${PIER_PYTHON:?}" -m torch.distributed.run \
    --nnodes="${PIER_JOINT_NODES:?}" --nproc_per_node=4 \
    --node_rank="${SLURM_PROCID:-0}" --max_restarts=0 \
    --master_addr="${PIER_JOINT_MASTER_ADDR:?}" \
    --master_port="${PIER_JOINT_MASTER_PORT:?}" "$@"

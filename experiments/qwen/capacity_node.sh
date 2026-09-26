#!/usr/bin/env bash
set -euo pipefail
cd "${PIER_ROOT:?}"
exec "$PIER_PYTHON" -m torch.distributed.run \
    --nnodes=8 --node_rank="${SLURM_PROCID:?}" \
    --master_addr="${PIER_CAPACITY_MASTER_ADDR:?}" \
    --master_port="${PIER_CAPACITY_MASTER_PORT:-29561}" \
    --nproc_per_node=4 --max_restarts=0 \
    experiments/qwen/capacity_train.py --trial "$1"

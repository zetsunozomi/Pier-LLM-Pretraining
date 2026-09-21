#!/usr/bin/env bash
# One torchrun agent per allocated node; workers see all four local GPUs.
set -euo pipefail
cd "${PIER_ROOT:?}"
if [[ "${PIER_N2_NODES:-1}" == 1 ]]; then
    exec "$PIER_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc_per_node=4 --max_restarts=0 \
        experiments/qwen/n2_train.py --output-dir "$1" --case "$2"
else
    exec "$PIER_PYTHON" -m torch.distributed.run \
        --nnodes="$PIER_N2_NODES" --node_rank="${SLURM_PROCID:?}" \
        --master_addr="${PIER_N2_MASTER_ADDR:?}" --master_port="$PIER_N2_MASTER_PORT" \
        --nproc_per_node=4 --max_restarts=0 \
        experiments/qwen/n2_train.py --output-dir "$1" --case "$2"
fi

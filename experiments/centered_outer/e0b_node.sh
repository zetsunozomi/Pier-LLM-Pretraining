#!/usr/bin/env bash
set -euo pipefail
cd "$PIER_ROOT"
exec "$PIER_PYTHON" -u -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --node_rank=0 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --max_restarts=0 \
  experiments/centered_outer/e0b_train.py --case "$1" --output-dir "$PIER_E0B_RUN_DIR"

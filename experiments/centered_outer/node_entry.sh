#!/usr/bin/env bash
set -euo pipefail
cd "$PIER_ROOT"
phase=$1
shift
if [[ "$phase" == megatron-dp1 ]]; then
    "$PIER_PYTHON" experiments/centered_outer/manifest.py --node-only --output-dir "$PIER_E0_RUN_DIR"
fi
if [[ "$phase" == protocol ]]; then
    target=experiments/centered_outer/reference/verify_executor.py
    args=(--device cuda --output "$PIER_E0_RUN_DIR/protocol.json")
else
    target=experiments/centered_outer/megatron_gate.py
    args=(--inner-dp-size "${phase##*dp}" --output-dir "$PIER_E0_RUN_DIR/$phase")
fi
exec "$PIER_PYTHON" -u -m torch.distributed.run \
    --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$PIER_GPUS_PER_NODE" \
    --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --max_restarts=0 "$target" "${args[@]}"

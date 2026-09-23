# Main comparison: naive CPU offload (O), GPU resident (R), Pier (P)

2026-09-23 correction: **O does not use state sharding.** The old sharded
offload path is OS, a supporting ablation alongside G/W and the cohort sweep.
All three main rows report complete-cycle throughput, outer time, and GPU memory.

## Run the new O/R/P pilot

After syncing this implementation to the cluster:

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
PIER_N2_PROFILE=pilot PIER_N2_WORKSPACE_MIB=64 PIER_N2_COHORT=2 PIER_N2_REPEAT_START=1 PIER_QWEN_DATA_PREFIX= sbatch --export=ALL experiments/qwen/n2_main.sbatch
```

This requests eight nodes / 32 A100-40GB GPUs, one O/R/P group, 100 steps per
method, and 30 minutes. The new O has no GPU timing yet; 30 minutes is an
initial budget. In an existing GPU allocation, use the same variables with
`bash experiments/qwen/n2_main.sbatch`; the allocation's node count and time
limit apply. The wrapper fixes O/R/P and one repeat even if stale variables
name another suite. The default log interval is one iteration.

Logs and JSON go to a fresh `out/n2-*/` directory, including `out.txt`, the
manifest with frozen commands/source hashes, per-rank receipts, and results.
No downloads or package changes are needed. O must show `backend=cpu_offload`
in `manifest.json`; a historical `backend=gather` O is the old sharded path.

## What O does

`megatron/core/outer_sync/cpu_offload.py` implements
`--outer-arm cpu_offload --outer-cpu-offload`:

1. Keep a full FP32 reference and outer momentum replica in pageable CPU
   memory per learner/TP coordinate. Neither R nor M is state-sharded.
2. For each parameter in canonical order, copy its reference to GPU using a
   blocking copy, then subtract the current FP32 master.
3. AllReduce the displacement across corresponding learners and divide by K.
   Each nonempty parameter has its own collective; there is no bucket fusion.
4. Copy the averaged displacement to CPU and apply the same Nesterov formula
   to CPU R/M. Copy the updated reference back to the GPU master, blocking.
5. The shared runtime performs the normal master-to-model commit.

This is explicitly the **naive unsharded CPU baseline**: no pinned state,
prefetch, overlap, persistent GPU staging pool, or tiling. It performs actual
state transfers and updates, without artificial waits or duplicate updates.
It is an author implementation, not an optimized or third-party-native result.
Forward parameters, masters, inner AdamW, data, and recomputation match R/P.
The shared launcher retains its existing `OMP_NUM_THREADS=1` for every arm;
O records the actual PyTorch CPU update thread count in its allocation receipt.

O uses dynamic whole-parameter temporary buffers. The workspace setting caps
the tiled R/P/G/W/OS executors only; O does not claim that cap. Compare actual
GPU allocated/reserved peaks, and record its buffer policy in the summary.

For the current 3B/TP2 coordinate size, full CPU R+M is approximately 11.50 GiB
per rank, versus 0.72 GiB for the old K=16 sharded offload. Nominal per-boundary
copy arguments total 11.50 GiB H2D plus 5.75 GiB D2H per rank. These are tensor
API bytes, not measured PCIe/NVLink traffic. The collector verifies full-state
sizes, actual CPU placement and pageable storage on every rank. Persistent
state bytes are not process/node host peak memory.

## OS: preserve the sharded offload ablation

OS maps to the unchanged `gather + --outer-cpu-offload`: pinned CPU R/M
shards, tiled GPU reference AllGather, centered ReduceScatter, GPU owner
update, host writeback and updated-parameter AllGather. To include it in a
separate pilot, use `PIER_N2_ARMS=O,OS,P` with **`n2.sbatch`**, since the main
wrapper fixes O/R/P. OS differs in sharding, pinning, tiling and update
placement together; that comparison alone does not isolate each factor.

The existing `out/n2-58779130-20260922-193352.BDRzSL` run recorded the old path
as O. Its manifests and results remain unchanged. Treat it as sharded-offload
evidence in the paper: 81,859 tokens/s and 7.705 s outer, versus that run's P
82,583 tokens/s and 6.337 s outer. The paired +0.885% throughput belongs to
this old implementation, not to the new naive O.

## Later: three separate formal jobs

After the pilot establishes a suitable time limit:

```bash
PIER_N2_PROFILE=main PIER_N2_WORKSPACE_MIB=64 PIER_N2_COHORT=2 PIER_QWEN_DATA_PREFIX= sbatch --export=ALL --array=1-3 --time=01:00:00 experiments/qwen/n2_main.sbatch
```

Each task has one paired O/R/P group, 250 steps per method. Global repeat IDs
1/2/3 select different recorded orders; input/model seed stays fixed. Pair
within each allocation before aggregating across jobs. If O exceeds the
budget, split groups with a common anchor, retaining the same useful work.

## Local validation

Eleven N2 checks pass: real Megatron argument parsing, direct/spooled launch,
repeat ordering, full-pageable-state verification, paired ratios, and exact
read-only recollection of all archived 4/8/32-GPU reports including old O.
The new executor passes a four-process NumPy-oracle comparison over three
updates, including noncontiguous groups and singleton groups. The shared
runtime passes optimizer/skip/commit and exact midcycle checkpoint restore
with O. These checks do not measure CUDA transfer behavior or GPU speed.

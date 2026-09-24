# Main comparison: naive CPU offload (O), GPU resident (R), Pier (P)

2026-09-23 correction: **O does not use state sharding.** The old sharded
offload path is OS, a supporting ablation alongside G/W and the cohort sweep.
All three main rows report complete-cycle throughput, outer time, and GPU memory.

## Adopted core result

The user selected `out/n2-58781491-20260923-012423.b4EudG` as the paper's
core O/R/P comparison. All three completed on the same 32 A100-40GB allocation:

| Arm | Tokens/s | Outer + commit (s) | Allocated / reserved GiB |
|---|---:|---:|---:|
| O | 79,628.74 | 12.256 | 27.22 / 28.78 |
| R | 83,256.57 | 4.994 | 33.40 / 34.37 |
| P, s=2 | 82,697.07 | 6.121 | 30.52 / 31.51 |

P improves throughput over O by 3.85% and reduces outer time by 50.05%.
Against R, it saves 2.87 GiB allocated per GPU with 0.67% lower throughput.
Each arm has one launch, 50 warmup steps and one measured 50-step cycle.
Total training-launch wall time was 18.3 minutes. Additional repeats and
250-step windows are optional extensions, not a prerequisite for this table.

The 32-GPU s=1/2/16 cohort sweep and homogeneous eight-GPU O/R/P point are
complete (`out/n3-58815173-20260924-024052.i2jLI7` and
`out/n2-58821531-20260924-055127.IfV9bj`). The remaining GPU point is
Qwen2.5-1.5B/TP2 on 32 GPUs. Model selection is now connected; use the
dedicated entrypoint below.

## Next experiment: Qwen2.5-1.5B O/R/P

Download the six pinned files first, using `bash experiments/qwen/download_qwen15b.sh`
on a login node, or the standalone copy already provided. The snapshot directory is:

```text
/pscratch/sd/s/syfan/Pier/local/qwen/models/Qwen2.5-1.5B/8faed761d45a263340a0528343f099c05c9a4323/
```

After syncing this change to the cluster:

```bash
cd /pscratch/sd/s/syfan/Pier
git pull --ff-only
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
sbatch experiments/qwen/n4_15b.sbatch
```

This requests eight nodes / 32 A100-40GB GPUs and 30 minutes. It fixes
Qwen2.5-1.5B base, TP2, K=16, O/R/P, P cohort=2, a 64 MiB R/P workspace,
synthetic tokens, full recomputation, and 50 warmup + 50 measured steps per
method. Existing model/profile/data/cohort variables are overridden to preserve
this coverage point. Missing/incomplete snapshot files fail before worker launch;
the shared model loader still checks the pinned checkpoint contents.

The tested `n2.sbatch` launch and measurement path is reused. Full output remains
`out/n2-<job>-<time>.<unique>/out.txt`; the manifest records `model_size=1.5B`
and its revision. The top-level Slurm file is `pier-n4-qwen15b-<job>.out`.
Email is ALL to `sf850@scarletmail.rutgers.edu`. An existing **eight-node**
allocation can run `bash experiments/qwen/n4_15b.sbatch`; a one-node shell
cannot supply this 32-GPU coverage point and is rejected.

For custom runs through `n2.sbatch`, `PIER_QWEN_MODEL_SIZE` selects the pin,
default snapshot and training model together; it defaults to `3B`.
`PIER_QWEN_SNAPSHOT` can override the location, but must contain the selected
model. Historical manifests without `model_size` continue to use 3B.

## Reproduce the adopted O/R/P window

After syncing this implementation to the cluster:

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
PIER_N2_PROFILE=pilot PIER_N2_WORKSPACE_MIB=64 PIER_N2_COHORT=2 PIER_N2_REPEAT_START=1 PIER_QWEN_DATA_PREFIX= sbatch --export=ALL experiments/qwen/n2_main.sbatch
```

This requests eight nodes / 32 A100-40GB GPUs, one O/R/P group, 100 steps per
method, and 30 minutes. The completed window used about 18.3 minutes of
training-launch wall time in total. In an existing GPU allocation, use the same variables with
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

## Optional longer repeated measurements

Only when extra variability evidence is needed; these jobs are not the next
required step or a prerequisite for the adopted core table:

```bash
PIER_N2_PROFILE=main PIER_N2_WORKSPACE_MIB=64 PIER_N2_COHORT=2 PIER_QWEN_DATA_PREFIX= sbatch --export=ALL --array=1-3 --time=01:00:00 experiments/qwen/n2_main.sbatch
```

Each task has one paired O/R/P group, 250 steps per method. Global repeat IDs
1/2/3 select different recorded orders; input/model seed stays fixed. Pair
within each allocation before aggregating across jobs. If O exceeds the
budget, split groups with a common anchor, retaining the same useful work.

## Local validation

2026-09-25 model-size change: five focused tests in `test_n2_model_size.py`
pass, including the actual shell chain for interactive/spooled Slurm launches,
stale-environment overrides, wrong-allocation rejection, selected snapshot
checking, and legacy 3B argument compatibility. All eight archived N2/N3
summaries recollect exactly without modifying their raw data. Shell syntax
checks pass. These are launcher checks, not a completed 1.5B GPU run.
The broader existing `test_n2.py` suite could not import in this local test
environment because `regex` is absent; no cluster packages were changed.

Previously completed validation:

Eleven N2 checks pass: real Megatron argument parsing, direct/spooled launch,
repeat ordering, full-pageable-state verification, paired ratios, and exact
read-only recollection of all archived 4/8/32-GPU reports including old O.
The new executor passes a four-process NumPy-oracle comparison over three
updates, including noncontiguous groups and singleton groups. The shared
runtime passes optimizer/skip/commit and exact midcycle checkpoint restore
with O. The separate GPU window above now supplies CUDA execution and
performance evidence; local tests supply targeted update/restore checks.

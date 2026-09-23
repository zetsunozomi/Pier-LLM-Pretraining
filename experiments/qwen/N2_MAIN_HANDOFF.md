# Main comparison: CPU offload (O), GPU resident (R), Pier (P)

2026-09-23. The primary table now compares O/R/P. Existing G/W measurements
remain supporting execution controls, and the cohort sweep remains an ablation.
All three main rows report both complete-cycle throughput and GPU memory.

## First run: 30-minute pilot

After syncing the changed code to the cluster:

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
PIER_N2_PROFILE=pilot PIER_N2_WORKSPACE_MIB=64 PIER_N2_COHORT=2 \
PIER_N2_REPEAT_START=1 PIER_QWEN_DATA_PREFIX= \
  sbatch --export=ALL experiments/qwen/n2_main.sbatch
```

The wrapper requests eight nodes / 32 A100-40GB GPUs, one O/R/P group,
100 steps per method, and 30 minutes. O has not been timed on this workload;
the time limit is an initial budget, not a completion guarantee. The first run
decides whether the existing offload transfer schedule needs work.
An existing allocation can run `bash experiments/qwen/n2_main.sbatch` with the
same variables; its existing nodes and time limit apply.

The wrapper fixes the main arms and one repeat even if stale environment values
name another suite. The usual recipe/data/workspace controls remain available.
Logs and JSON go into the printed `out/n2-*/out.txt` directory. Downloads and
model tensors remain under ignored scratch paths. No new package is required.

## What O actually does

O is the existing native `gather` executor with `--outer-cpu-offload`:

1. Keep FP32 reference and outer momentum shards in pinned CPU memory.
2. Copy tile shards to the GPU; AllGather reference blocks before centering.
3. Form reference minus local masters; native SUM ReduceScatter and division.
4. Apply the same owner Nesterov update; write R/M shards back to the CPU.
5. AllGather updated parameters and perform the same master-to-model commit.

Forward parameters, FP32 masters and the inner AdamW optimizer stay on the GPU
in all three arms. Current copies are blocking, without asynchronous prefetch
or overlap. This is an author implementation, not a native third-party run or
an optimized-offload claim. Any improvement in transfers receives a separate
identified implementation/configuration, with the initial results retained.

The collector checks actual R/M device, bytes and pinned state on every rank.
It reports maximum per-rank persistent host R/M bytes, **not** process/node host
peak or all pinned memory. The normal GPU allocator peaks remain separate.
Time ratios use O from the same global repeat and allocation; an O failure
leaves ratios empty while retaining other methods' measurements.

## Later: three formal repeats as separate jobs

After the pilot establishes the recipe and reasonable time limit:

```bash
PIER_N2_PROFILE=main PIER_N2_WORKSPACE_MIB=64 PIER_N2_COHORT=2 \
PIER_QWEN_DATA_PREFIX= \
  sbatch --export=ALL --array=1-3 --time=01:00:00 experiments/qwen/n2_main.sbatch
```

Each array task contains one O/R/P comparison, 250 steps per method. Tasks
receive global repeat IDs 1/2/3 and the corresponding predeclared randomized
orders; the model/input seed remains fixed. Each writes a unique output
directory. Pair within an allocation/repeat before summarizing across jobs.
If O needs more time, split method groups with a common comparison anchor
instead of silently shortening training or combining all repeats into a long job.

The earlier G/R/W/P and N3 entrypoints remain usable. The main wrapper is the
new default handoff; it does not rerun the old four-hour job.

## Local validation

Eleven N2 checks passed, including the real Megatron argument parser, O/R/P
interactive/spooled launch, split repeat order, host-placement rejection,
paired O ratios, and read-only recollection of the archived 4/8/32-GPU results.
The four-process CPU/Gloo native-runtime regression also passed with the new
state-storage receipts, optimizer/commit checks and checkpoint restore. These
are local software checks; O's CUDA transfers and GPU performance await the pilot.

# Ordered Pier communication fast path

The opt-in `--outer-pier-schedule contiguous` keeps the existing centered FP32
graph and removes staging copies from the communication tree. `reference`
remains the default and is the implementation used in the existing paper tables.
No GPU speedup has been measured for this change.

## Submit the first GPU tests

`strict_bench.sbatch` combines the CUDA correctness gate, old/new/W benchmark,
and an outer-only summary. Both profiles request 30 minutes and email ALL events
to sf850@scarletmail.rutgers.edu. `smoke` uses one four-GPU node, TP1, K=4,
and 16,777,217 synthetic flat coordinates. `flat32` uses eight four-GPU nodes,
TP2, K=16, s=2, and 1,543,044,096 FP32 coordinates per participant, matching
the padded master-coordinate count in the existing Qwen3B TP2 report.

Both use 64 MiB workspace, three warmup updates, ten timed updates per arm,
and one randomized block containing `reference`, `contiguous`, and `recenter`.
They execute outer updates directly; they do not run inner-training steps or
load Qwen weights. In `flat32`, the correctness phase uses two workers per node
to test the 16-learner tree; timing then uses all four GPUs/node in two TP
coordinate groups. The gate checks the exact executor hash used for timing.

### Local Mac: verify and commit

The current local test environment is `/private/tmp/pier-fastpath-venv/bin/python`.
It is a local verification environment, not the cluster interpreter.

```bash
cd /Users/shuyuanfan/Pier-latest
bash -n experiments/centered_outer/strict_bench.sbatch experiments/centered_outer/strict_bench_node.sh
GLOO_SOCKET_IFNAME=lo0 PYTHONPATH="$PWD/tests/outer_sync:$PWD/tests/qwen:$PWD" \
  /private/tmp/pier-fastpath-venv/bin/python -m unittest \
  test_contiguous test_schedule_config test_strict_bench test_n2 -v
git diff --check
git add \
  megatron/core/outer_sync/executor.py \
  megatron/core/outer_sync/runtime.py \
  megatron/core/outer_sync/checkpoint.py \
  megatron/training/arguments.py \
  experiments/qwen/n2_config.py experiments/qwen/n2_summary.py \
  experiments/centered_outer/reference/verify_executor.py \
  experiments/centered_outer/benchmark_outer.py \
  experiments/centered_outer/strict_bench.sbatch \
  experiments/centered_outer/strict_bench_node.sh \
  experiments/centered_outer/summarize_strict.py \
  experiments/centered_outer/CONTIGUOUS_SCHEDULE.md \
  tests/outer_sync/test_contiguous.py tests/outer_sync/test_schedule_config.py \
  tests/outer_sync/test_strict_bench.py tests/outer_sync/test_runtime.py \
  tests/qwen/test_n2.py
git diff --cached --stat
git commit -m "Add ordered Pier fast path and GPU comparison workflow"
git push origin snapshot-main
```

### Cluster login node: pull and submit

```bash
cd /pscratch/sd/s/syfan/Pier
git pull --ff-only origin snapshot-main
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
smoke_job=$(sbatch --parsable --nodes=1 --export=ALL,PIER_STRICT_PROFILE=smoke \
  experiments/centered_outer/strict_bench.sbatch)
printf 'Smoke job: %s\n' "$smoke_job"
flat_job=$(sbatch --parsable --nodes=8 --dependency="afterok:${smoke_job%%;*}" \
  --export=ALL,PIER_STRICT_PROFILE=flat32 experiments/centered_outer/strict_bench.sbatch)
printf '32-GPU job: %s\n' "$flat_job"
```

Submit the second command after the first returns a valid job ID. The dependency
keeps the 32-GPU job from starting if smoke fails. To stage the jobs manually,
omit the second submission until smoke finishes and inspect its `results.txt`.
For a single-node interactive GPU allocation, the alternative to the first
submission is:

```bash
PIER_STRICT_PROFILE=smoke bash experiments/centered_outer/strict_bench.sbatch
```

### Inspect and return evidence

The short `pier-strict-bench-JOBID.out` file points to the complete log. Each run
gets `out/strict-bench-PROFILE-JOBID-TIMESTAMP.RANDOM/` with `out.txt`,
`correctness.json`, `benchmark.json`, `summary.json`, and `results.txt`.
These top-level text files already pass the repository's existing ignore rules.
Use the actual job IDs if reconnecting with a new shell:

```bash
cat out/strict-bench-smoke-"${smoke_job%%;*}"-*/results.txt
cat out/strict-bench-flat32-"${flat_job%%;*}"-*/results.txt
git add out/strict-bench-smoke-"${smoke_job%%;*}"-* out/strict-bench-flat32-"${flat_job%%;*}"-*
git commit -m "Record strict Pier GPU correctness and outer comparison"
git push origin snapshot-main
```

Then on the Mac:

```bash
cd /Users/shuyuanfan/Pier-latest
git pull --ff-only origin snapshot-main
```

The summary reports the new strict path's latency reduction against the old
strict path and each strict path's remaining overhead against W. Negative
speedups remain in the report. This flat-master run decides whether to proceed
to the existing Qwen integration; its timings and memory peaks do not replace
full-model paper measurements.

Scope fixed by the user: optimize Pier only. Keep O/OS/G/R/W implementations
and their established benchmark configurations unchanged. In particular, do not
move future fused arithmetic or packing into a shared helper used by baseline
executors. The schedule flag is routed only to the Pier arm; selecting it for
a native baseline is rejected. Baseline replays use their recorded settings.

## Implemented

- Pack coordinate-owner subranges in bit-reversed order. The low-bit-first
  learner reduction tree then retains a contiguous half of the bank at every
  level. It sends the other half directly and performs one elementwise add per
  level, with the original left/right operand order.
- Upper AllGather receives directly into the vacated, disjoint bank interval.
  Cohort AllGather uses a low-bit-first schedule, sending and receiving final
  contiguous pack intervals without staging. AllGather performs no arithmetic.
- Persistent R/M ownership, tensor workspace capacity, logical sent bytes,
  leaf subtraction, division and the unfused Nesterov operations stay the same.
  R/M checkpoints retain their original tensor layout. The checkpoint recipe
  records the new schedule and conservatively rejects cross-schedule restart.

For K=16 and s=2, upper-tree additions fall from seven to three tensor launches
per tile. Upper communication drops 21 staging `copy_` calls; cohort gathering
drops two. Reordering reference reads/writes adds 14 row-copy calls, for a net
reduction of nine `copy_` calls per tile. These are source-level operation counts,
not measured GPU kernel counts or a speedup estimate. Small row copies and the
remaining Python packing can still limit performance.

## Correctness and local evidence

CPU/Gloo checks on PyTorch 2.14.0 passed 32 operator configurations and four
toy-training configurations on eight processes, spanning all nested cohorts,
partial tiles, padding, cancellation and signed-zero inputs. The independent
NumPy oracle checks all FP32 R/M/master bits and BF16 commits. Storage guards
reject new exposed tensor allocations inside the executor.

The four-process integration regression additionally checks fragmented parameter
coordinates, noncontiguous process groups, optimizer skips, identical training
trajectories across schedules and same-schedule checkpoint restart. CPU evidence
does not certify CUDA/NCCL execution or performance.

Before timing on a four-GPU allocation, run the CUDA correctness gate:

```bash
torchrun --standalone --nproc_per_node=4 \
  experiments/centered_outer/reference/verify_executor.py \
  --device cuda --schedule contiguous --output /tmp/pier-contiguous-cuda.json
```

Repeat with the actual multi-node launch topology before using main-scale data.

## Isolated outer benchmark

`benchmark_outer.py` times actual outer execution plus BF16 commit without
running 50 inner steps. It compares old/new Pier, GPU reference reconstruction
(G), and pinned, sharded CPU offload (OS). Its offload arm is explicitly OS,
not the paper's naive, full-replica O. It records every rank's raw wall times,
per-update rank maxima, warmup/sample counts, order, workspace, memory peaks
and source hashes. Repeats within one process launch are labeled as such.

For a four-GPU smoke benchmark:

```bash
torchrun --standalone --nproc_per_node=4 \
  experiments/centered_outer/benchmark_outer.py \
  --cohort 2 --workspace-mib 64 --warmup 3 --samples 20 --repeats 3 \
  --output /tmp/pier-outer-smoke.json
```

Use `--elements` for the actual FP32 coordinate count per TP participant and
`--tp 2` to create corresponding-coordinate groups on a larger allocation.
The default size is only a smoke workload. This flat-master benchmark excludes
Qwen parameter fragmentation and inner training; do not present it as measured
token throughput. A four-process CPU/TP2 smoke run exercised all four arms and
the report path; its timings are not GPU performance evidence.

### Strict Pier versus W (raw-weight averaging)

The paper now uses outer-update latency consistently. The existing Qwen3B
co-run medians are 6.197639 s for strict Pier (s=2) and 4.215410 s for W:
47.02% extra outer time. Closing that gap requires reducing strict Pier's
own latency by 31.98%, rather than by 47.02%. The earlier 1.1--1.3% ratio
measured complete training cycles at interval 50; it is not an outer-update
overhead and is not the optimization acceptance metric.

The benchmark accepts `recenter` to replay the unchanged W executor, with
its own existing workspace formula and fully sharded R/M. For a four-GPU
comparison in an existing allocation:

```bash
mkdir -p out
PIER_COMPARE_OUT=$(mktemp -d "$PWD/out/strict-relaxed.XXXXXX")
"${PIER_PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  experiments/centered_outer/benchmark_outer.py \
  --cohort 2 --workspace-mib 64 --warmup 3 --samples 20 --repeats 3 \
  --arms reference contiguous recenter \
  --output "$PIER_COMPARE_OUT/summary.json" > "$PIER_COMPARE_OUT/out.txt" 2>&1
```

This command is a small synthetic smoke comparison, not the 32-GPU paper
workload. Run the correctness gate
above first. Compare old versus new strict Pier at fixed cohort/workspace;
compare the resulting outer latency with W from the same launch. The W/P
comparison still changes both numerical contract and layout/backend. Retain
raw samples and memory peaks, and confirm any measured gain in Qwen before
replacing the paper's original observations. No target percentage is assumed
achieved by the local operation-count reduction.

The three-arm entry point passed a four-process CPU/Gloo smoke run with 65
coordinates and cohort size 2. Report checks confirmed the W executor,
fully sharded R/M, W-specific workspace formula, no reference AllGather,
modeled payload, and maximum-rank sample aggregation. These checks validate
benchmark wiring, not CUDA performance or bitwise equality between W and Pier.

Pier workspace candidates (64/256 MiB) may be profiled separately; baseline
replays retain their established settings. Report any workspace differences
explicitly. Keep original, new Pier, G and OS results, including configurations
where the new schedule is slower.

## Qwen integration

The existing N2 launcher accepts `PIER_N2_PIER_SCHEDULE=contiguous`. It changes
only Pier's outer schedule; all training kernels, batches, precision and
recomputation settings remain those of the shared configuration. For example:

```bash
PIER_N2_PIER_SCHEDULE=contiguous PIER_N2_PROFILE=main \
  PIER_N2_ARMS=O,OS,G,R,P sbatch --nodes=8 --export=ALL experiments/qwen/n2.sbatch
```

This runs the existing longer repeated training protocol and can be expensive;
use the isolated benchmark and correctness gate first. Existing source manifests
record the changed launch arguments. Allocation receipts record the schedule.
Outer+commit latency and complete-cycle throughput remain separate measured
quantities; neither is inferred from the other.

## Next optimizations, after GPU profiling

1. Fuse reference centering and the exact cohort subtree. For s=2, compute
   `(R-W0)+(R-W1)` with explicit FP32 rounding, not `2*R-(W0+W1)`.
2. Fuse Pier's owner division and Nesterov update while preserving each rounding
   point and disabling FMA contraction. Keep this kernel behind the Pier backend;
   do not change G/OS or the other baseline executors.
3. Batch fragmented master packing/writeback, then evaluate double buffering
   with CUDA events. Charge every extra buffer against the workspace cap and
   preserve last-reader/writeback ordering; simply removing waits is incorrect.

Profile before choosing among these. Use complete outer-update plus model-commit
latency as the optimization metric, with bitwise checks and memory peaks beside
it. Do not dilute the strict/relaxed difference by dividing by inner-training
time; synchronization-interval selection remains a separate algorithmic choice.

# N3: one allocation for Pier s=1, s=2 and s=K

**Current status (2026-09-23):** the homogeneous eight-GPU s=1/2/4 run is
complete at `out/n3-58684056-20260921-014323.UmGdL9`. It is included in the
draft's sharding ablation table. The 32-GPU O/R/P core result is also adopted.
The next missing anchor-scale ablation is **32 GPUs, s=1/2/16**.

## Next: 32-GPU cohort ablation

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
PIER_N2_PROFILE=pilot PIER_N2_REPEATS=1 PIER_N2_WORKSPACE_MIB=64 PIER_QWEN_DATA_PREFIX= PIER_N3_REFERENCE= sbatch --export=ALL --nodes=8 --time=00:30:00 experiments/qwen/n3.sbatch
```

The existing launcher chooses K=16 and s=1/2/16 automatically, with 100 steps
per setting (50 warmup, 50 measured). The new allocation includes its own
s=2 anchor. Keep the 64 MiB workspace and full recomputation fixed. Thirty
minutes is an initial budget: s=16 has not been timed at this scale. Retain
completed results if interrupted; split later work with a shared anchor or
use a one-hour request if necessary. Do not require three longer repetitions
before using the current core result. Mail directives are ALL to
`sf850@scarletmail.rutgers.edu`; logs remain under `out/n3-*/out.txt`.

The instructions below describe the already completed eight-GPU run and remain
available for reproduction. Both use the shared N2 training/measurement path;
no new optimizer/backend is introduced.

## Reproduce the completed eight-GPU run

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
PIER_N2_PROFILE=pilot PIER_QWEN_DATA_PREFIX= \
  sbatch --export=ALL experiments/qwen/n3.sbatch
```

The script requests **two nodes, eight A100-40GB GPUs, one hour**. The explicit
`gpu&hbm40g` constraint follows the [NERSC node selection documentation](https://docs.nersc.gov/jobs/policy/).
It exports the environment and checks the GPU model before loading Qwen.
For an already allocated two-node job with four A100-40GB GPUs per node, use
`bash experiments/qwen/n3.sbatch` with the same environment. Bash uses the
existing allocation; it cannot change its GPU model or request extra nodes.

Runs on that allocation:

1. `run-1-P-s1`: full reference per learner.
2. `run-1-P-s2`: node-local cohort candidate and same-allocation reference point.
3. `run-1-P-s4`: fully sharded reference at K=4.

Each starts from the pinned Qwen2.5-3B snapshot. TP2, BF16 model, FP32 optimizer
and outer state, synthetic tokens, sequence 2048, microbatch 1, accumulation 8,
r=50, full activation recompute and a 64 MiB executor workspace remain fixed.
Each uses 50 warmup and 50 measured steps. Iteration logs print every step.
Based on the previous eight-GPU runs, allow roughly 20 minutes for all three;
the one-hour request leaves startup and runtime margin. An OOM/failed case is
retained and later cases continue.

The previous s=2 measurement used a 40GB/80GB mixed allocation. This is why s=2
is repeated once here: the three cohort timings should share actual devices.
No HF parity, full-state checkpoint or tensor-dump phase is added.

## Outputs

The terminal prints `out/n3-<job>-<time>.<unique>/out.txt`. In that directory:

- `results.txt` and `summary.json` update after every case; compare complete-cycle
  tokens/s, outer+commit seconds and maximum-rank allocated/reserved GiB.
- `speedup/P(s=2)` uses **this allocation's** s=2 run. A missing/failed s=2 leaves
  ratios blank. Different cohorts are separate configurations, not repetitions.
- Per-case logs and rank JSON receipts preserve each measurement and failure.
- `historical-n2.txt` / `.json` contain the prior eight-GPU G/P/R/W results as
  historical context. They are never used as speedup denominators for the new
  allocation. The original artifacts remain unchanged.

Historical context defaults to
`out/n2-58678984-20260921-000738.rUk4TS`. Override with `PIER_N3_REFERENCE`, or
set it empty to omit the copy. Missing historical context does not block GPU
work. A normal prior `git pull` brings the text receipts back to the cluster.

All output JSON/log/txt files, including nested cohort rank receipts, are
Git-visible. Model, dataset and cache binaries remain under ignored `local/`.
Return the one printed directory using the existing Git workflow.

## What this run decides

It tests whether an intermediate cohort gives a useful speed/memory choice.
Use the measured curve to select configurations for the 32-GPU main comparison.
These are one-window observations. The current next step is the anchor-scale
cohort curve, with the core O/R/P result already adopted. The recipe also supports the main
window via `PIER_N2_PROFILE=main`; at 32 GPUs the cohorts become 1, 2 and 16.

## Local checks

Nine N2/N3 tests cover the real training parser, shell entrypoints (interactive
and Slurm spool), node launch arguments, continued execution after subprocess
failure, cohort grouping and same-allocation ratios, hardware mismatch
rejection, frozen launch arguments, and read-only recollection of both archived
GPU pilots. These are local software checks, not N3 GPU measurements.

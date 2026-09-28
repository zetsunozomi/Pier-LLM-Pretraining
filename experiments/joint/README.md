# Joint reference-refresh experiments

These entrypoints implement the current paper's missing runtime experiments.
They do not reuse historical timings as concurrent controls. All CUDA results
remain pending until the cluster jobs below run. Local CPU/Gloo evidence checks
the production algorithm, training trajectory, restart, and experiment wiring;
it is not CUDA performance evidence.

## Runtime

Select `--outer-runtime centered --outer-arm pier --outer-pier-schedule joint
--outer-joint-config /absolute/config.json`. The JSON is frozen in the launch
manifest and complete-state checkpoint recipe. The original `reference` and
`contiguous` paths and O/OS/G/R/W executors remain available.

- Reference pages have stable `(canonical shard, offset)` identities. Changing
  execution tile size combines pages without migrating them. Momentum shard j
  stays at corresponding-group rank j for the entire run and across restart.
- Raw learner values reach old reference owners one leaf at a time. A binary
  carry stack retains the original adjacent FP32 reduction tree with O(log s)
  partial tiles rather than s input tiles. Upper reduction preserves the tree.
- Each live slot owns a CUDA stream and communicator. A deterministic wave
  launches independent consumers, then returns completed tiles while later
  slots may still consume inputs. Actual CUDA events guard old-state release,
  momentum writeback, master/model commit, new-reference writes and slot reuse.
  CPU/Gloo uses the same operations synchronously; no CPU overlap is claimed.
- Returned values write model destinations and new reference pages directly.
  Retained pages are reused after their last read; obsolete allocations are
  released before admitting new destinations. There is no whole-model staging
  allocation or reference-migration exchange. Remote M has separately counted
  stage and writeback traffic.
- The planner enumerates `(target cohort, tile capacity, slot count)` against
  every rank's transition peak and final-state budget. Tensor workspace, actual
  other live CUDA allocations and explicit library/allocator headroom all count.
  It fails before mutation if no progress-safe schedule exists. Unsupported
  distributed optimizers and TP/learner changes remain outside this scope.
- `single` is the paged one-slot executor; `pipeline` keeps the layout fixed and
  tunes slots/tiles; `separate` independently selects layout and current-layout
  pipeline, then rebuilds references from committed masters in a separate local
  pass; `joint` constructs the next layout during return and selects jointly.
  Separate composition drops old state before allocating replacements, so it
  is not deliberately penalized with an unnecessary full old+new copy.

The paged variants keep the same canonical momentum homes. In particular,
s=2 may incur remote M traffic even with a fixed layout. Therefore the Qwen
campaign also replays the original **colocated static** executor at s=1/2/16;
the paged single-slot arm alone is not described as the best static baseline.

The budget trace distinguishes `transition_mib` (available at the outer boundary)
from `next_phase_mib` (the next training phase). A contraction may temporarily
need more than its final footprint. Both are announced before executing the
boundary. During measured training the following interval's actual allocated
peak is checked against the previous announcement. Budgets are an explicit
reservation scenario, not a claim that production jobs naturally changed their
memory requirements. The caller must reserve future training/library needs;
an allocation that already exceeds physical memory cannot be recovered here.

## Cluster preparation and gate

Prefer an interactive allocation for one-node/four-GPU correctness work.
Request **30 minutes** for short jobs; estimate larger jobs from returned phase
timings before choosing a longer limit. Leave about five minutes for startup,
Slurm cleanup and returning logs. Backfill is handled by the scheduler; a short
honest walltime helps fit available gaps but does not guarantee a start time.
See the [NERSC scheduling guide](https://docs.nersc.gov/jobs/scheduling/) and
[interactive guide](https://docs.nersc.gov/jobs/interactive/).

First sync the checkout **before** requesting the allocation. From the repository
on Perlmutter, using the existing environment (no package/model download):

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_ROOT="$PWD"
export PIER_OUT_ROOT="$PWD/out"
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
# Request 1 node / 4 A100-40GB / 00:30:00 with your usual salloc account.
# Once the allocation is ready, start immediately:
unset PIER_JOINT_TRAINING_RESUME
bash experiments/joint/training_gate.sbatch --max-phases 2
```

Run the script with `bash` directly inside the allocation: its `#SBATCH` lines
are ignored, and its internal `srun` launches the workers with explicit GPU
resources. Do not wrap it in another `srun` or submit another allocation from
inside the interactive one. If using batch instead, submit from the login node:

```bash
sbatch --time=00:30:00 experiments/joint/training_gate.sbatch --max-phases 2
```

This first chunk runs `reference-tp1` and `single-tp1`. It records each phase's
elapsed walltime in `exit.json`. The driver defaults to a 1,500-second budget,
caps each Slurm step to the remaining budget, and saves `summary.json` after
every phase. It stops between phases if the remaining budget is too small for
another phase based on the timings observed in that invocation. These are
guards, not a claim that an unmeasured phase will finish within 30 minutes.
If you start late in an existing allocation, pass a smaller `--budget-seconds`.
An unfinished phase is a failed gate requiring inspection, never a passed gate.

The complete 13-phase gate uses the actual `pretrain_gpt.py` entrypoint,
with BF16/FP32 AdamW, dropout, an injected global skip, all four variants, TP1,
TP2 and inner-DP2. It checks complete trajectories and checkpoint restore both
at attempt 4 (an outer boundary) and attempt 5 (inside a local cycle), against
the static reference. It needs no pretrained model or downloaded corpus.
Its workers explicitly set `NCCL_ALGO=Ring` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
for Megatron's deterministic correctness mode, with that environment frozen in
the launch receipt. Performance jobs retain their separately recorded settings.

For a step-by-step first run, run only the first chunk and return its evidence
for a quick check before continuing. `status: incomplete` with no errors and
11 pending phases is expected after two successful phases. The launcher prints
the exact fresh `out/joint-training-gate-.../` directory in the terminal or Slurm output.
After the job exits, use `git add out/joint-training-gate-<actual-run>/` on the
cluster to return the complete text evidence, including `gate/summary.json`,
`gate/manifest.json`, all phase/rank JSON files and logs. The repository ignore
rules retain nested JSON/log/txt files while excluding binary checkpoints and
caches. Keep the checkpoint files on the cluster. Return the directory even
when a phase fails or the time limit is reached; logs can diagnose a partial run.

After that quick check, clean partial gates can continue in a new allocation:

```bash
# Use the actual gate directory from the previous chunk; keep checkpoints there.
export PIER_JOINT_TRAINING_RESUME=/absolute/path/to/out/joint-training-gate-RUN/gate
bash experiments/joint/training_gate.sbatch --max-phases 2
```

The phase count can be adjusted after measuring the first chunk. Resume checks
the frozen source, configs and phase plan, validates completed phase evidence,
and skips those phases. It refuses changed sources or damaged/interrupted phases;
return these for diagnosis before retrying. Only all 13 validated phases produce
`status: passed`. Do not run two continuations concurrently in the same directory.
Return the same directory after each chunk; checkpoints stay on the cluster.

| Stage | Nodes / GPUs | Allocation plan |
|---|---|---|
| Real-training/restart gate | 1 / 4 | Interactive, 30 min per chunk; first chunk is two phases |
| Optional small operator gate | 1 / 4 | Interactive, 30 min; does not replace the full topology gate |
| Required TP2/K16 operator gate | 8 / 32 | Batch, 30 min initial limit |
| W numerical diagnostic | 8 / 32 | Batch, 30 min initial limit |
| Pipeline/transition scans and traces | 8 / 32 | Measure a small configuration subset first; size later batches for 30 min when feasible |
| Full Qwen fixed/changing-budget experiments | 8 / 32 | Measure complete case duration first; preserve full cycles and paired comparisons when splitting |

The full operator gate is the next stage after the complete training gate passes:

```bash
sbatch --time=00:30:00 experiments/joint/outer.sbatch verify
```

It defaults to eight four-GPU A100-40GB nodes, TP2, K16. The Slurm output
points to `out/joint-verify-*/report.json`. The CUDA gate checks the production
executor against an independent NumPy FP32 oracle, all directed cohort edges,
partial/padded tiles, slot counts 1/2/4, remote momentum, fragmented master/BF16
commit and serialized state restore.

Set the **actual returned path**, then keep that source checkout unchanged:

```bash
export PIER_JOINT_GATE=/absolute/path/to/out/joint-verify-RUN/report.json
export PIER_JOINT_TRAINING_GATE=/absolute/path/to/out/joint-training-gate-RUN/gate/summary.json
```

The Slurm performance entrypoints verify the operator gate's CUDA status, world/TP topology
and source hashes. A CPU gate or stale hash is rejected. For an inexpensive
four-GPU preliminary gate use `PIER_JOINT_TP=1 bash experiments/joint/outer.sbatch verify`
inside a one-node interactive allocation;
it does not replace the 32-GPU gate.
Qwen campaigns additionally require the passed real-training/restart gate.

## Required experiments and exact entrypoints

Each `--array=1-3%1` creates three independent job launches, serialized to limit
concurrent allocation demand. Samples inside one launch are not independent
repeats. Run each command separately; these jobs are not automatically submitted
by editing this repository.

| Evidence | Command after the gate |
|---|---|
| Fixed-layout pipeline/tile/slot scan | `sbatch --array=1-3%1 experiments/joint/outer.sbatch pipeline` |
| Directed transitions, standalone versus joint conversion, calibration | `sbatch --array=1-3%1 experiments/joint/outer.sbatch transitions` |
| Diagnostic memory ledger and CUDA stage overlap | `sbatch experiments/joint/outer.sbatch transitions --trace --capacities 262144 --slots 2 --samples 1` |
| W numerical differences | `sbatch --time=00:30:00 experiments/joint/outer.sbatch numerics` |
| Full Qwen fixed-budget four-way ablation and strong static/W controls | `sbatch --array=1-3%1 experiments/joint/qwen.sbatch fixed` |
| Full Qwen announced-budget adaptation and cumulative savings | `sbatch --array=1-3%1 experiments/joint/qwen.sbatch budgets` |
| Forced contraction and expansion, including all requested endpoint directions | `sbatch --array=1-3%1 experiments/joint/qwen.sbatch forced --only separate-s1 joint-s1` |
| Optional full-model diagnostic trace | `sbatch experiments/joint/qwen.sbatch budgets --only joint-s1 --trace` |

The flat scan uses the existing 3B/TP2 padded coordinate count, 64 MiB tensor
workspace, page size 65,536 elements, capacities 65,536/262,144/1,048,576,
slots 1/2/4, and s=1/2/16. Every requested configuration is either timed or
explicitly marked infeasible. It writes each completed configuration immediately
so an interrupted job retains partial evidence. Warmup is two updates and each
launch mean uses five timed maximum-rank outer+commit samples. The allocation
time limits are adjustable ceilings, not predictions of runtime.

Qwen uses the existing native 3B/TP2, BF16/FP32, sequence-2048, microbatch-one,
eight-accumulation, full-recomputation recipe and r=50. Fixed budgets use two
warmup and three measured cycles (250 steps); changing budgets use two warmup
and eight measured cycles (500 steps). The phase budgets are 35→32→29.5→32→35
GiB, with a 39 GiB transition ceiling and 1.5 GiB training/library headroom. They are configurable
via `--low-mib`, `--middle-mib`, `--high-mib`; record any change as a new campaign.
Forced runs use s=1→2→16→2→1. Static controls that do not fit are retained and
excluded from the best feasible static comparison. Each method has its own
fresh training process, and the full source/argv/data/initialization receipt.

`--only` permits splitting a long campaign into explicit subsets. Do not pool
different subsets as if they were one completed paired campaign: the summary
rejects changed planned cases across repeats. The fixed/budgets default runs
all four variants, static layouts, original colocated controls and G/OS/R/W.
Control-only subsets retain the common budget trace and are summarized against
the same measured-memory criterion; they do not by themselves complete the
four-way experiment. Paired comparisons reject missing or misaligned cycles.

To inspect the launch plan without GPUs or a model snapshot:

```bash
python experiments/joint/qwen.py --scenario budgets --plan-only \
  --gate /path/to/future-gate.json --output-dir /tmp/pier-joint-plan
```

## Aggregation and planner calibration

Use three actual independent reports from the **same** scan recipe:

```bash
python experiments/joint/summarize.py \
  /path/to/repeat1/report.json /path/to/repeat2/report.json /path/to/repeat3/report.json \
  --output /path/to/transition-costs.json
```

This reports median/min/max/stddev of the three launch means and keeps each
configuration's rejected runs. The transition scan includes steady configurations
for every execution mode, so its `costs` entries can calibrate all four variants.
To use that table, add `--cost-table /path/to/transition-costs.json` to the Qwen
commands. Source, world, TP, coordinate count and workspace must match. Without
a table the planner explicitly reports an **uncalibrated bandwidth/launch model**;
it does not pretend that predicted costs were measured. Flat-master costs remain
estimates for fragmented Qwen parameters, and the Qwen results assess the actual
selected configurations.

For a completed Qwen scenario:

```bash
python experiments/joint/qwen_summary.py \
  /path/to/repeat1/campaign/campaign.json \
  /path/to/repeat2/campaign/campaign.json \
  /path/to/repeat3/campaign/campaign.json \
  --output /path/to/qwen-budget-summary.json
```

The result contains complete-cycle tokens/s, outer+commit time, allocated and
reserved peaks, per-boundary layout/tile/slot choices, budget feasibility,
separate reference/momentum payloads, and cumulative saved outer seconds versus
separate composition, W, and the best feasible static control. This curve shows
whether/when transition costs are recovered. Raw timelines distinguish switching
rounds from steady rounds. No physical-link/NVML measurements or convergence
claims are inferred from logical payload or synthetic tokens.

W is a valid real-arithmetic DiLoCo update; moving centering after the native
reduction changes FP32 rounding. `numerics.py` records master/momentum errors,
BF16 disagreements and a constructed witness. Bitwise disagreement alone does
**not** establish inferior training quality. Report W's speed and memory even if
it wins, alongside the numerical contract; no baseline is silently discarded.

## Local verification

The available local test interpreter is `/private/tmp/pier-fastpath-venv/bin/python`.
On macOS, use explicit loopback rendezvous for the runnable entrypoint smoke tests
because `torchrun --standalone` can select a host name rewritten by local DNS.

```bash
GLOO_SOCKET_IFNAME=lo0 PYTHONPATH="$PWD/tests/outer_sync:$PWD/tests/qwen:$PWD" \
  /private/tmp/pier-fastpath-venv/bin/python -m unittest \
  test_joint test_joint_runtime test_joint_experiments test_joint_reporting \
  test_joint_shell test_schedule_config test_n2 -v
GLOO_SOCKET_IFNAME=lo0 /private/tmp/pier-fastpath-venv/bin/python \
  -m torch.distributed.run --nnodes=1 --master-addr=127.0.0.1 --master-port=29817 \
  --nproc-per-node=4 experiments/joint/verify.py --device cpu --tp 2 \
  --output /tmp/pier-joint-cpu-gate.json
```

Use a fresh output path for every attempt. Local checks cannot certify NCCL
stream progress, CUDA overlap or GPU memory peaks; the supplied CUDA gate and
measurement jobs establish those properties on the target hardware.

The [implementation audit](../centered_outer/JOINT_IMPLEMENTATION.md) maps the
paper's required mechanisms to production code, local tests and cluster
entrypoints. The planner computes the exact stripe-prefix storage envelope
without enumerating every model tile for every candidate; the independent
literal lifetime simulation checks 1,536 combinations. Runtime storage accounting
is incremental, including during diagnostic tracing, so admission bookkeeping
does not repeatedly scan the full reference.

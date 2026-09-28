# Joint reference-refresh experiments

These entrypoints implement the current paper's missing runtime experiments.
They do not reuse historical timings as concurrent controls. CUDA performance results
remain pending until the corresponding measurement jobs run. Local CPU/Gloo evidence checks
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

## Paper-first submission workflow

The objective is to replace the marked draft results with actual measurements.
The [paper progress ledger](PAPER_PROGRESS.md) maps every marked result group to
its data source. No standalone validation allocation is a paper prerequisite.
The completed four-GPU training/restart gate is supporting evidence already in
`out/`; do not rerun it merely because a launcher or reporting script changes.
The optional `verify` and `training_gate` entrypoints remain developer diagnostics.

The next submission directly measures **Table VI, joint runtime, fixed budget**:

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_ROOT="$PWD"
export PIER_OUT_ROOT="$PWD/out"
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
sbatch experiments/joint/paper.sbatch joint-s1 1
```

This requests **8 nodes / 32 A100-40GB GPUs / 30 minutes** and runs exactly one
Qwen2.5-3B/TP2 case: 100 warmup steps followed by 150 measured steps, r=50.
It is independent repeat 1, not a separate pilot or correctness test. Historical
250-step static runs took about 14 minutes per case; the new joint executor's
walltime remains to be measured. One case leaves substantially more margin than
putting two historical 14-minute cases into a 30-minute allocation. If interrupted,
return its logs and partial cycle reports; do not treat an incomplete window as
finished evidence. The launcher retains pinned-input, finite-loss/model, model
commit, complete-cycle, hardware and source checks inside the measurement.
It does not add an oracle suite to the performance timer.

The output is printed as `out/joint-paper-fixed-.../`. Return that whole text
result directory, including `campaign/campaign.json`, per-case `summary.json`,
`manifest.json`, per-rank `cycles-rank-*.json`, initialization/worker receipts and
logs. No checkpoint or model binary needs to be transferred. Review this first
result before submitting another method or repeat; this command starts no array
and submits no dependent jobs.

`paper.sbatch CASE REPEAT` accepts the four variants (`single-s1/s2/s16`,
`pipeline-s1/s2/s16`, `separate-s1`, `joint-s1`) and the original colocated static
controls (`reference-s1/s2/s16`). REPEAT is 1, 2 or 3. Each invocation preserves
the same full measurement window. Do not use a new repeat ID for the samples
inside one launch. A failed case stops its campaign so it does not silently spend
more allocation time on later cases.

For split case jobs, aggregate their returned campaign files with:

```bash
python experiments/joint/qwen_summary.py --split-allocations \
  /path/to/case-repeat-1/campaign/campaign.json \
  /path/to/another-case-repeat-1/campaign/campaign.json \
  --output /path/to/paper-progress.json
```

Keep adding the actual returned files as experiments finish. Missing rows and
insufficient independent repeats stay incomplete. Cross-allocation comparisons
are explicitly **unpaired**, even if they share a repeat number; only cases
actually co-run on the same allocation may be labeled paired. Changed sources,
case configs, input recipes or GPU/software types and duplicate case/repeat
receipts are rejected. The summary exposes measured allocated/reserved peaks
alongside latency. It does not turn the first repeat into a finished Table VI.

Without `--cost-table`, planner selection uses the recorded analytic model.
The selected configurations and observed costs are real measurements, but must
not be described as an empirically calibrated optimum. Calibration scans are
not a mandatory extra submission; decide whether any are needed from the main
results. The paper's fixed-budget and changing-budget results remain separate.

Optional prior receipts can be checked by explicitly passing `--gate` or
`--training-gate` to `qwen.py` / `qwen.sbatch`. They are never picked up implicitly
from old environment variables and are not required by `paper.sbatch`.

## Additional measurement entrypoints (submit only after the preceding quick check)

The commands below are capabilities, not a queue to submit now. Each
`--array=1-3%1` creates three independent job launches; use it only after the
first result has been reviewed. Do not add the diagnostic scans unless they
answer a specific marked paper result. Samples inside one launch are not
independent repeats. The long-form Qwen entrypoint has a six-hour default for
whole campaigns; prefer `paper.sbatch` for one fixed-budget case in 30 minutes.

| Evidence | Available command |
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

`--only` permits splitting a long campaign into explicit subsets. Use
`qwen_summary.py --split-allocations` to combine them with explicit unpaired
comparison labels and case/repeat coverage checks. The fixed/budgets default runs
all four variants, static layouts, original colocated controls and G/OS/R/W.
Control-only subsets retain the common budget trace and are summarized against
the same measured-memory criterion; they do not by themselves complete the
four-way experiment. Paired comparisons reject missing or misaligned cycles.

To inspect the launch plan without GPUs or a model snapshot:

```bash
python experiments/joint/qwen.py --scenario budgets --plan-only \
  --output-dir /tmp/pier-joint-plan
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

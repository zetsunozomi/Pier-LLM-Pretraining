# G/R/W/T baselines and complete-cycle instrumentation

**2026-09-23 update:** the 32-GPU G/R/W/P window (three independent launches
per arm) and eight-GPU cohort pilot are complete. The paper main table now uses
[CPU-offloaded reconstruction / GPU resident / Pier](../qwen/N2_MAIN_HANDOFF.md).
O reuses `gather` with `--outer-cpu-offload`; its current pinned-host copies are
blocking. The new launcher records and verifies actual state placement and
splits formal repeats into short jobs. O GPU timing is still pending. Older
status paragraphs below are implementation history, not the current run order.

Status (2026-09-21): implemented in the shared training adapter and checked on
CPU/Gloo. Four/eight-GPU [N2 pilots](../qwen/N2_HANDOFF.md) now contain actual
Qwen G/P/R/W cycle throughput, outer time and allocator peaks. All four arms
completed the eight-GPU pilot; the four-GPU resident arm hit OOM. The next run
is the [N3 cohort comparison](../qwen/N3_HANDOFF.md), one eight-GPU allocation
for s=1/2/K. E0c/E0d are not performance prerequisites.

## Actual implementation

`megatron/core/outer_sync/collectives.py` provides three tiled native collective
executors. All use FP32 masters/R/M, canonical parameter coordinates, the same
Nesterov update, and the existing master-to-model commit. They run through
`CenteredRuntime`, the existing optimizer skip consensus and complete per-rank
checkpoint code. No separate toy optimizer is used by their training path.

| CLI arm | Paper arm | Outer operation | R per rank | M per rank |
|---|---|---|---|---|
| `gather` | G | Gather R, center local W, native RS, owner update, AG | width | width |
| `gather` + `--outer-cpu-offload` | O | Pinned-host R/M tiles, GPU gather/center/RS/update/AG, host writeback | width (CPU) | width (CPU) |
| `resident` | R | Center against replicated R, native RS, owner update, AG | K × width | width |
| `recenter` | W | Native raw-W RS, divide, recenter, owner update, AG | width | width |
| `pier` / omitted | P | Existing ordered reference-owner executor | K/s × width | width |

Here `width = ceil(N/K)` FP32 elements, N is one corresponding TP coordinate's
master size, and K is its outer-group size. G/R/W assign contiguous reference/
momentum shard j to group rank j. Native collectives operate on that group,
including noncontiguous global-rank groups. All tensors are allocated once;
the final partial tile and padded suffix are handled explicitly.

Native SUM order is opaque. G/R do not claim Pier's fixed-tree bitwise contract;
W additionally moves centering after reduction. The existing `--outer-verify`
oracle is deliberately rejected for these arms, rather than falsely certifying
bitwise equality. N2 records finite loss, successful updates and an untimed
final model-commit check; a short numerical comparison can accompany the paper
table without blocking first timings. Same-arm restart retains the
native tile configuration; switching arms is rejected even when state shapes
are identical. Existing Pier checkpoints without an arm field retain their
meaning. Public Qwen initialization remains weights-only.

## Equal workspace caps

The new `--outer-workspace-mib 64` (or 256) option derives the largest tile T
within that cap, bounded by width. It is mutually exclusive with an explicit
`--outer-tile-elements`. This caps explicit executor tensor workspace, excluding
R/M, model/inner optimizer and library/allocator storage:

| Arm | Workspace bytes for T coordinates per owner |
|---|---|
| G / R | 4T(2K+3) |
| W | 4T(K+4) |
| P | 4T((2s+3)(K/s)+1) |

Allocation receipts report the actual tensor bytes separately from persistent
R/M. Native payload receipts report API buffer sizes and modeled 3b (G) or 2b
(R/W), with b = 4(K−1)width. They set physical wire bytes to null: a buffer size
or logical communication formula is not a measured link counter.

These are initial native-collective implementations, not completed strong-
baseline tuning. GPU collective choices, tile candidates and scheduling must
receive equal tuning budgets. Blocking pinned-host state copies are functional
only; they are not the required optimized O/offload baseline. O, S/server,
asynchronous multi-slot execution and distributed optimizer remain outstanding.
The T/DTensor training backend below is now connected but still needs GPU
numerical checks and the same tuning budget as the other arms.

## T: explicit DTensor layouts in the shared training path

`megatron/core/outer_sync/dtensor.py` implements `--outer-arm dtensor`. It uses
the same FP32 parameter coordinates, successful-step clock, inner AdamW,
master-to-model commit, host/device outer-state option, complete checkpoint
and cycle meter as P/G/R/W. It supports every nested power-of-two cohort,
including endpoints, and tiles all coordinates with explicit zero padding.
It no longer requires the separate full-vector expressibility demonstration.

For K=sg and outer rank a*s+b, the two-dimensional mesh is ordered (b,a).
Its rank map uses the actual corresponding TP/inner-DP group, including
noncontiguous global ranks. Every default-world rank constructs the disjoint
meshes in the same sorted order; overlapping or inconsistent maps fail.
Persistent R holds g owner shards and M holds one, matching P's state layout.

Each tile packs t coordinates from all K owner intervals. DTensor transforms
`(Shard(learner-b), Shard(learner-a))` to coordinate ownership, centers locally
against the corresponding R, then transforms a `(Shard, Partial(sum))` vector
to `(Shard, Shard)`. The owner applies the shared unfused FP32 Nesterov update.
Two explicit redistributions refresh R and return the updated master. The
layout is supplied by the experiment; this does not evaluate automatic planning.
These APIs are documented for the cluster's
[PyTorch 2.6 DTensor](https://docs.pytorch.org/docs/2.6/distributed.tensor.html).

Native reductions and local tensor reduction do not promise P's prescribed
addition tree. `--outer-verify` remains rejected for T, as for G/R/W; future
GPU numerical diagnostics must use its native-SUM contract. Checkpoints keep
the arm, cohort and native tile/cap configuration and reject changed layouts
or tile settings on exact restore. The CPU tests prove same-configuration
restart on that tested backend, not cross-version/CUDA numerical invariance.

### T workspace accounting

Permanent caller buffers hold **4t(K+2g+2)** bytes: packed W, R staging, a
partial-sum bank, M staging and direction. At most **4t(K+g+1)** additional
storage is held by exposed DTensor redistribution outputs. Tile selection
therefore reserves **4t(2K+3g+3)** bytes under `--outer-workspace-mib`; it does
not count only the permanent buffers. Reports separate permanent allocation
from this explicit live-storage allowance and label dynamic outputs.

DTensor's internal redistribution intermediates and collective/library/allocator
storage are **additional**. This bound is not a measured total GPU-memory cap
or proof that two arms with the same option have equal total peaks. All arms
still need measured peak memory and equal tuning opportunities before the
paper comparison. In particular, CPU's AllToAll fallback can allocate different
intermediates and cannot establish the GPU memory or wire behavior. The
payload report keeps logical 2b separately from `physical_wire_bytes: null`.

An argument fragment for later real-data integration, not a ready E1 job:

```text
--outer-runtime centered --local-sgd-inner-average
--outer-arm dtensor --outer-cohort-size 2 --outer-workspace-mib 64
```

CPU/Gloo coverage exercises the actual DTensor collectives for K=1/2/4,
s=1/2/4 when valid, noncontiguous groups, sizes 1/5/7/9/23/65, several tiles,
padding, all R/M owners and BF16 model commit. Exactly representable dyadic
fixtures match bitwise; ordinary FP32 uses explicit local-test tolerances.
The real AdamW/dropout/skip path restores midcycle and reproduces every final
state digest for each cohort. Its CPU cycle records also pass the existing
all-rank collector. No native collective is mocked in these tests.

The first run found PyTorch's mesh coordinate return type differed between
the documented 2.6 list and the local 2.14 tuple; normalizing that representation
fixed the ownership check. This is not a GPU compatibility test. No new Slurm
job is added: E0c remains next, followed by E0d; native-SUM numerical validation
and strong-baseline GPU tuning remain necessary before E1.

After this integration, all 38 outer-sync tests passed (64.232 s) and all 27
Qwen tests passed (48.329 s). Logs: `/private/tmp/pier-outer-with-dtensor-tests.log`
and `/private/tmp/pier-qwen-after-dtensor-tests.log`. These local results use
PyTorch 2.14 CPU/Gloo; the cluster's PyTorch 2.6/CUDA path is still untested.

## Complete-cycle measurements

`--outer-measure-dir PATH --outer-warmup-cycles 2` enables the opt-in recorder in
`cycle_metrics.py`; it is called by the actual training loop before an attempt,
after loss-token accumulation, and around outer execution plus model commit.
The ordinary E0b/E0c commands do not enable it.

- A window begins after the previous device work drains and ranks synchronize;
  it ends after the successful-step interval's outer update and model commit
  have drained. Rank-local wall durations and the maximum across ranks are saved.
- Data fetch, all inner attempts and intervening within-cycle logging are in
  the window. Initialization/loading and boundary measurement barriers/report
  writes are outside; logging/callbacks between independently started cycle
  windows are also excluded and labeled. This is not end-to-end GPU-hour accounting.
- The GPT loss routine already reduces non-padding loss-token counts across DP;
  the meter uses those actual counts after accumulation and checks agreement
  across all ranks. It does not multiply or sum again across TP/DP replicas.
- A skipped optimizer step contributes time and processed tokens, but zero
  successful tokens. Throughput is successful global loss tokens divided by
  the maximum rank cycle duration.
- All warmup cycles remain in the raw report. Only complete post-warmup cycles
  with positive successful-token counts receive throughput samples. A resumed
  midcycle prefix or trailing incomplete cycle is labeled and excluded.
- Optional PyTorch allocated/reserved peaks are per rank and per cycle. Device-
  level/NVML peaks, node host/pinned peaks and physical wire traffic remain null.
- Save/evaluation/profiler/manual-GC modes and the full-state correctness oracle
  are rejected in the initial measurement mode; train_iters must be 1..500.

The atomic `cycles-rank-N.json` reports contain raw samples and remain labeled
`performance_result: false`: the recorder alone cannot validate the actual
workload, paired/tuned comparison, confidence intervals, all memory tiers, or
total GPU-hours. The raw-cycle collector below now validates the timing/token
bookkeeping; a workload/job/tuning evidence gate is still needed before these
data can populate the paper.

## Read-only cycle collection

`cycle_summary.py` reads all `cycles-rank-N.json` files from one or more completed
measurement directories and writes a separate JSON result. It does not modify
input reports and refuses to overwrite its output. Example, once real training
measurement runs exist:

```bash
python experiments/centered_outer/cycle_summary.py \
  /absolute/path/to/run-1 /absolute/path/to/run-2 \
  --output /absolute/path/to/cycle-summary.json
```

The producer now records a shared run UUID, the planned attempted-step budget,
and initial/final successful-step clocks. The reader requires every rank to
finish that declared budget (at most 500 attempted steps), checks contiguous
cycle coverage and agreement across ranks, and recomputes every slowest-rank
duration, token count relation, warmup/partial label and throughput. Missing or
duplicated ranks, a mixed UUID, an unfinished report, inconsistent numeric fields,
or missing tail evidence fails collection. Supplying the same UUID twice also
fails; distinct UUIDs alone do not prove statistically independent runs.

Each completed run contributes **one run sample**:

```text
sum(successful global loss tokens in eligible cycles)
---------------------------------------------------
sum(slowest-rank seconds in those same cycles)
```

This is not the unweighted mean of cycle throughputs. For example, 200 tokens
in 4 seconds and 200 tokens in 16 seconds give 20 tokens/s over the window,
whereas averaging their individual rates would incorrectly give 31.25. Already
global tokens are counted once, without multiplying by world size. Warmup,
resumed prefixes and trailing partials remain visible with exclusion reasons;
none contributes to that run sample. Raw maximum-rank outer/commit durations
and PyTorch allocator peaks remain separately labelled.

The reader defaults to GPU-labelled reports. `--allow-cpu` is only for explicit
fixture review and retains `GPU_executed: false`. It records the exact input
file bytes/hashes, and returns `raw_cycles_validated` with
`performance_result: false`: it cannot prove source/model/data/hardware identity,
successful launcher exit, a passed numerical gate, equal tuning, independent
paired runs or total GPU-hours from these cycle files. It produces no confidence
interval or pooled paired comparison. Physical-wire and device-level memory
fields remain unmeasured; v1 reports claiming values in them are rejected.

The new schema applies to future runs; older unversioned cycle fixtures are
rejected, not rewritten. Accepted E0a/E0b correctness evidence is unaffected.
This adds no Slurm job and does not change the E0c conversion command.

The final outer-sync suite passed 35 tests (43.180 s), including six collector
checks using real four-process CPU/Gloo recorder output and controlled clocks.
The tests detect replica token overcounting, wrong duration/rate arithmetic,
resume-prefix/warmup mistakes, missing or mixed evidence and duplicate runs;
CLI checks confirm input preservation and output overwrite refusal. The
allocator schema case is synthetic, not a CUDA result. Log:
`/private/tmp/pier-outer-with-cycle-summary-tests.log`.

Example **argument fragments for later integration**, not a ready cluster job:

```text
--outer-runtime centered --local-sgd-inner-average
--outer-arm gather --outer-cohort-size 1 --outer-workspace-mib 64
--outer-measure-dir <fresh-run-directory> --outer-warmup-cycles 2
--eval-iters 0
```

R/W use their respective arm names; P uses `pier` and its selected cohort.
All preserve the same inner recipe, data order and successful-step interval.
The next real-data Qwen gate must establish that recipe and full training-state
behavior before the E1 launcher is finalized.

The Qwen adapter now accepts the same explicit activation-recompute settings
for all these arms, including full uniform/block and selective core attention;
see [the Qwen activation contract](../qwen/README.md#activation-recomputation-for-the-later-training-gate).
Checkpoint restoration rejects changing that schedule. The feature has CPU
math/reexecution checks only, not measured GPU memory savings or a tuned policy.

## Initial G/R/W integration evidence and its limits

The outer suite has 26 checks. The first complete run passed 25 and exposed a
misplaced block in a test fixture; after correcting the fixture, all eight
affected runtime/parser checks passed (9.683 s). The final explicit-runtime
argument guard is covered by a targeted parser check. All 14 Qwen tests also
passed after these changes (28.935 s). No CPU timing is treated as a GPU result.

New coverage includes real Gloo AllGather/ReduceScatter, repeated R/M updates,
nondivisible sizes, changing tiles, fixed storage, BF16 model commits, coordinate
maps, noncontiguous groups, K=1, equal caps, and rejection of aliases. A real
FP32 cancellation witness confirms W differs from centered G/R; exactly
representable dyadic fixtures establish state correctness without assuming
native SUM is generally bitwise invariant.

The shared runtime tests exercise each arm's AdamW/dropout/skip and exact
midcycle restart, reject cross-arm restore, and run the meter through that
same path. Clock-controlled four-process tests verify that global tokens are
not counted four times, skipped updates are excluded from the numerator, the
slowest rank sets cycle duration, and warmup/partial fragments are excluded.
Invalid, fractional, nonfinite or rank-disagreeing token counts fail closed.

Logs: `/private/tmp/pier-native-cycle-tests.log`,
`/private/tmp/pier-native-cycle-runtime-final.log`,
`/private/tmp/pier-cycle-args-final.log`,
`/private/tmp/pier-qwen-after-cycle-tests.log`.

Native collective API reference:
[PyTorch 2.6 distributed collectives](https://docs.pytorch.org/docs/2.6/distributed.html#torch.distributed.reduce_scatter_tensor).

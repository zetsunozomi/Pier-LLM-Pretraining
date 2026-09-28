# Joint reference-refresh implementation work

Scope: implement the current paper's dynamic reference placement, bounded
multi-slot pipeline, fixed momentum homes, shared-budget planner, complete
model commit, restart, memory/communication receipts, and runnable experiment
entrypoints. Keep the original static and native baseline implementations as
historical controls. A new runtime requires new concurrent control measurements.

Completion evidence required:

- Distributed numerical checks of every supported layout transition, partial
  tiles, noncontiguous groups, multiple slots, remote momentum and model commit.
- Runtime and complete-state checkpoint integration, including post-transition
  and mid-local-cycle restart and unchanged inner optimizer state.
- GPU correctness gate, isolated pipeline/configuration experiments, full Qwen
  fixed-budget and announced-budget-change experiments, four-way ablation,
  independent launch repeats, and W numerical/performance comparison.
- Per-rank old/new reference lifetime ledger, actual GPU allocated/reserved
  peaks, explicit payload split, source hashes, complete-cycle throughput and
  aggregation that preserves launch independence.
- Local tests and shell/launcher checks. GPU results are pending until the
  supplied cluster entrypoints are actually run; do not present CPU tests as
  GPU measurements or reuse old static timings for the new runtime.

## Implementation and entrypoint audit

The required functions and experiment entrypoints are implemented. GPU results
are still pending: this document records implementation evidence, not completed
paper measurements. No Slurm job was submitted during implementation.

| Required capability | Production implementation | Verification / experiment entrypoint |
|---|---|---|
| Runtime reference contraction and expansion, fixed M homes | `joint.py`: paged reference ownership, reference-owner update, remote M stage/writeback, return-to-destination refresh | `test_joint.py`; `outer.sbatch verify`; `outer.sbatch transitions`; `qwen.sbatch forced` |
| Bounded multi-slot pipeline and completed-consumer reclamation | `joint.py`: per-slot stream/communicator, completion events, deterministic wave admission, direct master/BF16 commit | Four-process numerical/payload checks; CUDA `verify` plus `transitions --trace` |
| Joint cohort/tile/slot choice under transition and final budgets | `joint_plan.py`, `joint_config.py`: per-rank envelope, measured cost table or explicitly uncalibrated estimate, remaining-round switching cost | Independent lifetime simulation of 1,536 combinations; tight-budget rejection, contraction/expansion and measured-cost payback tests; `pipeline`, `transitions`, `qwen.sbatch budgets` |
| Training semantics and full-state restart | `runtime.py`, `checkpoint.py`, `coordinates.py`: successful-step clock, inner state retention, model commit, active layout/M-home recipe, drained checkpoint | `test_joint_runtime.py`: all four variants and restart at attempts 4/5; `training_gate.sbatch`: actual Megatron TP1/TP2/inner-DP2 with dropout and injected skip |
| Fixed-budget four-way ablation and announced budget sequence | `qwen.py`: single, pipeline, separate composition, joint; original colocated static s=1/2/16 plus native controls | `qwen.sbatch fixed`, `budgets`, `forced`; every variant has the same candidate/workspace allowance |
| Correct timing, memory and communication records | `joint.py`, `cycle_metrics.py`, `benchmark.py`, `qwen_summary.py`: full outer+commit, full cycle, old/new reference lifetime, allocated/reserved peaks, phase payload | Operator payload identities; measured static-budget test; optional CUDA timeline and ledger; complete four-worker benchmark CLI smoke |
| Independent repeats and reproducible launch/summary | Source hashes, frozen argv/config, UUIDs, three independent array launches, launch-mean aggregation, stale-gate rejection | `test_joint_experiments.py`, `test_joint_reporting.py`, `test_joint_shell.py`; actual Slurm shells exercised from a spool directory; all 13 training-gate phases parsed by Megatron |

All script names above are under `experiments/joint/`; all test names are under
`tests/outer_sync/`. [README](../joint/README.md) contains the exact submission,
gate, calibration and aggregation commands. The W numerical/performance controls
are also available, without using bitwise disagreement to claim worse quality.

## Local verification and limits

- Final regression: **68 outer-sync tests passed** (65.978 s), followed by
  **12 N2 tests passed** (4.772 s). Logs are
  `/private/tmp/pier-joint-completion-regression.log` and
  `/private/tmp/pier-joint-completion-n2-regression.log`.
- Actual four-worker CPU CLI checks completed for verification (324 records per
  rank), transitions (40 configurations plus successful aggregation), diagnostic
  pipeline tracing (three execution modes), and the W numerical comparison.
  Reports use `/private/tmp/pier-joint-final-{verify,benchmark,trace,numerics}-gloo.json`;
  every report explicitly records `GPU_executed=false`.
- Both Qwen control-only and 13-phase real-training gate plan-only commands ran
  successfully. All submission/node scripts pass Bash syntax checks, Python
  modules compile, and `git diff --check` passes.
- Interpreter: `/private/tmp/pier-fastpath-venv/bin/python`, PyTorch 2.14, macOS
  CPU/Gloo. Distributed tests use four real processes and local loopback sockets.
- Runtime tests exercise all directed layouts, partial/padded coordinates,
  noncontiguous groups, q=1/2/4, remote M, fragmented masters/BF16 model writes,
  budget rejection before mutation, four-variant training trajectories and full
  state restore. The real CLI verifier also passed 324 records per rank.
- The deterministic training gate's complete argument matrix passed the actual
  Megatron parser. Its worker environment is recorded and includes NCCL Ring and
  deterministic cuBLAS workspace settings. This is launcher validation, not a
  claim that the GPU training phases were run locally.
- A local 3B-coordinate planner check selected the same tuple and peak envelope
  after replacing repeated model-tile enumeration with the exact prefix formula;
  observed planning time changed from about 10.84 s to 0.004 s. These are local
  planning observations, not a training speedup or a CUDA benchmark result.
- The existing static executors and archived result files were not changed.
  The new measurements must use their own source identity and concurrent controls.
- Remaining cluster validation is explicit: NCCL stream progress/overlap,
  measured CUDA peaks, full Qwen runs, and statistical performance conclusions.
  Gates run first and are required by the performance launchers. Neither local
  tests nor plan-only manifests satisfy those gates.

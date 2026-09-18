# Validation record — 2026-09-16

Baseline repository: `1489491b9dfb2963c473566bcab75240ed335dd5`.
Host: macOS, Python 3.11, PyTorch 2.14.0, CUDA unavailable.
Work was checked in an isolated local checkout before applying to Pier-latest.

## Executed checks

| Check | Observed result | Scope |
|---|---|---|
| `python -m unittest discover -s tests/outer_sync -v` | 12 passed | CPU storage/copy and update semantics, normalization/warmup, unsupported adapters, report completeness, shell failure propagation |
| `verify_executor.py --world-size 4 --device cpu` | passed; 24 global operator cases, 3 global toy-training cases | Real local Gloo messages; no CUDA, pinned-host transfer or network performance evidence |
| `bash -n` on `e0.sbatch` and `node_entry.sh` | passed | Bash syntax |
| Python AST parsing of changed/new Python files | passed | Syntax only |
| `git diff --check` | passed | Patch whitespace |
| GPU gate invoked on this CPU-only host | expected exit 1 and failed rank report | Explicit refusal to substitute CPU results for GPU validation |
| Launcher test with synthetic Slurm commands and injected worker exit 23 | exit 23 preserved, failed summary and phase log retained | Shell control flow only; no Slurm allocation or GPU run |

The first sandboxed Gloo attempt was denied local socket access. The same
four-process test passed after allowing loopback sockets; no remote service was used.
At the initial handoff, source hashes for the copied executor and verifier
matched the original reference artifact. The compatibility fix below changes them.

The summary's positive schema fixture is synthetic test data held only in a
temporary directory and removed at test exit. It is not a GPU result.

## Cluster validation pending at the initial handoff

- Real MyDDP backward accumulation/collectives and BF16 optimizer construction.
- Master/model commit and next forward on the GPU backend.
- CUDA/NCCL protocol, host staging, GPU-visible buffer reuse and liveness.

At the initial handoff, all GPU conclusions were pending. The first handoff
covered E0a; the full E0 and all performance experiments remained incomplete. See README.md
for the exact scope and the sbatch command.

## E0a process-group compatibility follow-up — 2026-09-16

The supplied log for job `58423372` reached `protocol` after both Megatron phases.
Its summary reported no Megatron evidence errors. Protocol initialization failed
with `KeyError: None` in `get_process_group_ranks(group)`, before operator checks.
Local PyTorch 2.14 accepts `None` here; the cluster implementation in the traceback
requires an explicit process group. No cluster PyTorch version is inferred from
the traceback.

- The executor now resolves an omitted group to `dist.group.WORLD` and uses that
  group consistently for rank lookup and point-to-point communication.
- The verifier's initial NCCL barrier explicitly selects the current CUDA device.
- A new four-process Gloo regression enforces the cluster API's rejection of
  `None`. It reproduced the original exception before the fix, then passed with
  the fix, exercising implicit/explicit WORLD and subgroups `[0, 2]` / `[1, 3]`.
  It checks master, model, reference and momentum values for every cohort size.
- The existing CPU/Gloo matrix passed again: 24 operator configurations and
  3 toy-training configurations. This is CPU evidence, not a CUDA/NCCL pass.

The follow-up requested a complete E0a rerun with a new manifest and phase reports
so source hashes would remain consistent. Its supplied result is recorded below.


## E0a cluster rerun reported passed — job 58425240

Evidence source: the user's pasted Slurm output in this task. The full artifact
archive has not yet been copied back or independently inspected locally.

- The printed summary reports `status: passed`, `stage: E0a`, `world_size: 4`,
  `GPU_executed: true`, and an empty error list.
- Both `megatron-dp1` and `megatron-dp2` completed; the summarizer accepted their
  per-rank normalization, outer-update, model-consumer and local-restore evidence.
  The DP2 case uses ordinary replicated optimizers, not a distributed optimizer.
- The CUDA protocol reports 48 global operator configurations and 3 toy-training
  configurations. Per-rank duplicate records are not additional experiments.
- The 48 operator configurations cover 3 cohort sizes, 4 tensor lengths, 2 tile
  sizes, and 2 state tiers (device/host). This is why the count doubles the CPU
  matrix's 24 configurations.
- Both printed reports explicitly set `performance_result: false`.

Cluster artifact directory:
`/pscratch/sd/s/syfan/Pier/local/centered_outer/e0-58425240`.
Retain the directory and `pier-e0-58425240.out`; inspect the manifest, node metadata
and full per-rank/protocol reports before attributing results to exact hardware,
software versions or source revisions. The pasted output alone does not supply
those identities or timings.

This closes the reported four-rank E0a correctness run. It does not complete the
full E0 or validate Qwen conversion/training, the full `pretrain_gpt` loop,
successful-step scheduling and skip/restore integration, production per-learner
checkpoints, distributed optimizers, concurrent CUDA slots, GPU performance or
physical wire traffic. The next implementation stage is the production coordinate
adapter and consistent successful-step/restore integration, followed by a small
end-to-end GPU gate before scaling performance experiments.


## Full text evidence received and reviewed — commit e3a702bc

The 15 files under `local/centered_outer/e0-58425240/` are now available locally
(309149 bytes). All per-rank evidence, protocol coverage, rank maps, explicit
allocation formulas and logical payload formulas were checked. Environment:
4 A100-SXM4-40GB GPUs on one node, PyTorch 2.6.0+cu124, NCCL 2.21.5.

Source audit: all 358 locally present sources match the run manifest. Twelve
missing legacy/data sources exactly match old main revision e0621757. One
explicitly ignored training_legacy.py remains unavailable. Strict local summary
revalidation therefore reports only the source-set mismatch; report validation
otherwise passes. The original run artifacts were kept unchanged.

E0a component correctness is accepted within its stated scope, with the source
archive limitation recorded separately. Production training integration and
performance are still pending. See [the complete review](E0A_REVIEW_58425240.md).

## E0b production integration — 2026-09-17

Implementation based on snapshot-main `e3a702bc`; changes remain in the worktree
for the user's normal Git commit/push/pull workflow. No E0b GPU job was run locally.

Host: macOS, Python 3.11, PyTorch 2.14.0, CUDA unavailable. CPU environment:
`/Users/shuyuanfan/local-papers/pier/reshape-sep13/.venv/bin/python`;
extra import dependencies regex/PyYAML/einops are in `/private/tmp/pier-e0b-pydeps`.
These temporary host paths are not required by the cluster launcher.

| Executed check | Observed result | Scope |
|---|---|---|
| `python -m unittest discover -s tests/outer_sync -v` | 18 passed, 20.097 s | Existing regressions plus production coordinates/clock/skip/checkpoint, argument and report checks |
| Production runtime, four Gloo processes | passed | Real FP32Optimizer/AdamW/autograd; cohorts 1/2/4 have identical trajectories |
| Checkpoint at attempt 5, resume through attempt 11 | passed | Dropout, Python/NumPy/Torch RNG, model/master, optimizer, scheduler, R/M and full trace match continuous execution |
| Only rank 2 has an Inf gradient | passed | All ranks skip before any model/inner state changes; independent of injected skip vote |
| Production outer group builder | passed | TP1/inner1, TP1/inner2, TP2/inner1 using real Gloo groups and actual executor exchanges |
| Eight real Megatron argument sets and MockGPT token bounds | passed | No unknown options; tokenizer covers longest mock document and EOD; unsupported rerun/dist-optimizer rejected |
| E0b report schema, missing/corrupt evidence, relocated archive | passed | Strict failure on incomplete results, incorrect clock/group/restore/consumer/trajectory; fixtures removed after test |
| E0a and E0b shell launchers with worker exit 23 | passed | Exit 23, phase logs, failed summary preserved; no GPU work claimed |
| Original reference verifier, four Gloo processes | passed | 24 global operator configurations and 3 global toy-training configurations after moving executor/spec into core |
| Shell syntax, Python AST, `git diff --check` | passed | Source/shell syntax and whitespace only |

Local test logs: `/private/tmp/pier-e0b-local-tests.log` and
`/private/tmp/pier-e0b-reference-cpu4.log`; CPU protocol report:
`/private/tmp/pier-e0b-reference-cpu4.json`. They are local diagnostic outputs,
not files to put in the GPU artifact directory or paper results.

Preflight found that the original tiny vocabulary did not cover native MockGPT
tokens; the harness now uses base vocabulary 4096 plus EOD and normal padding.
An unfused core-model launch no longer probes nvcc for unused fused extensions;
it still compiles the dataset helper needed by MockGPTDataset.

Restored 12 `megatron/legacy/data/*.py` files from old main `e0621757`; their hashes
match E0a job 58425240 exactly. `.gitignore` now ignores root `/data/`, avoiding
the accidental exclusion of package source. New manifests include the actual
entry point, Python/extension sources and launchers, and separately record an
optional unused training_legacy.py. Historical E0a results remain unmodified.

Next required evidence: the actual four-GPU E0b job in [E0B_HANDOFF.md](E0B_HANDOFF.md).
This runs real `pretrain_gpt.py` with BF16/CUDA/NCCL, TP2, inner DP2 and pinned-host
state, and compares complete model/loss/inner-state trajectories across cohorts
and process restart. Real training, CUDA transfers and full Megatron startup
have not been validated by the CPU tests above. No throughput, GPU peak memory,
physical wire traffic or Qwen result is claimed.

## E0b first GPU feedback and checkpoint wrapper fix — 2026-09-17

User-provided terminal output from `e0b-58457867-20260917-015035` reports
`GPU_executed: true` and all twelve s2/s4/host-versus-s1 rank comparisons as
bitwise equal. The split phase reaches attempt 5 (40 consumed samples, one
injected skipped step), then fails in checkpoint serialization:
`Float16Module.state_dict() got an unexpected keyword argument 'destination'`.
Resume, TP2 and inner-DP2 have not run. The complete artifact archive has not
yet been received locally; this is a partial result from the pasted output,
not a passed E0b gate or performance evidence.

Root cause: `training.py` imports the **legacy** Float16Module, while DDP's
base `state_dict` forwards `destination`. The similarly named core wrapper
already accepted it. Added the missing keyword and forwarding to the legacy
wrapper, preserving its existing positional prefix/keep_vars arguments.

Added two CPU regressions using the actual legacy BF16 wrapper and DDP base
serialization methods. Both reproduce the exact reported TypeError before
the fix and pass afterward. They cover BF16 parameters and persistent
buffers, integer buffers, strict save/clear/load bitwise restoration,
destination identity, prefix, keep_vars and parent-module traversal.
They do not exercise CUDA gradient buffers or claim GPU checkpoint recovery.

All 20 `tests/outer_sync` checks passed across the initial run and a targeted
rerun: 18 passed directly, while two four-process Gloo checks initially hit
the sandbox's `uv_bind: operation not permitted`; those two passed with local
loopback access (7.995 s). Logs are
`/private/tmp/pier-e0b-wrapper-fix-tests.log` and
`/private/tmp/pier-e0b-wrapper-fix-gloo-tests.log`.
`git diff --check` passed. A fresh complete GPU run is still required after
the user's normal Git update; earlier source manifests/results stay unchanged.

## E0b resume recipe normalization — 2026-09-17

User-provided output from `e0b-58457867-20260917-020105` reaches the resume
phase and fails at `checkpoint recipe differs: ['num_query_groups']`. The
summary contains no split failure and retains all twelve successful
s2/s4/host-versus-s1 comparisons. This supports progress past serialization;
it does not establish successful resume or TP2/inner-DP2 execution.

The real argument parser leaves `num_query_groups=1` when GQA is disabled,
while `core_transformer_config_from_args` builds ordinary attention with
four groups (one per head). Both FLOP and memory estimation overwrite the
argument to four during training. Saving then compares four to the freshly
parsed one on restart. The new recipe records the effective group count,
without dropping the field or disabling mismatch checks. Mismatch errors
now show both saved and current values.

A regression using the actual parser, TransformerConfig builder and both
estimators fails before the fix (`1 != 4`) and passes afterward. Explicit
GQA counts remain distinct. The four-process runtime checkpoint regression
now saves with the post-reporting count four, reloads with the initial CLI
count one, and matches the uninterrupted trajectory; it also rejects a
changed GQA count. Five argument/report/launcher tests passed (10.654 s),
and two runtime tests passed with local Gloo loopback access (3.989 s).
Logs: `/private/tmp/pier-e0b-recipe-tests.log` and
`/private/tmp/pier-e0b-recipe-runtime-tests.log`.

The launcher now labels stage index and START/DONE/FAILED, explains the
intentional split exit, and retains immediate stop on nonzero exit. Its
failure regression confirms no DONE marker or next-stage launch after a
worker exits 23. Shell syntax and `git diff --check` pass. GPU validation
remains pending; rerun with a fresh directory and preserve previous evidence.

## E0b passing cluster summary received — 2026-09-17

The user returned the complete terminal summary for
`e0b-58457867-20260917-021226`, finished UTC 2026-09-17 09:15:21.770642.
All eight cases completed; status passed, GPU_executed true, errors empty,
and all sixteen s2/s4/host/resume trajectory comparisons are bitwise equal.
The final dp2-s2 process exited zero. NCCL reports missing explicit process-group
destruction on exit; it did not prevent this run from completing the gate.

The artifact directory is not present locally yet. This records the received
cluster result, not an independent revalidation of absent files. No original
manifest/summary has been recreated. Full evidence transfer and local review
remain pending; another E0b GPU run is not currently required. See
[the scoped result review](E0B_REVIEW_58457867.md). Qwen and performance evidence
remain absent; no speed, GPU-memory or wire-traffic result is inferred.

## Complete E0b archive review and Qwen training preparation — 2026-09-17

The passing archive is now present locally at
`local/centered_outer/e0b-58457867-20260917-021226/`: 75 files, 773,864 bytes,
32 rank reports, 32 launch records and eight phase logs. Read-only revalidation
reproduced the received summary (excluding the regenerated timestamp), all
sixteen bitwise trajectory comparisons and empty errors. All 390 recorded
source hashes matched the worktree at review time, before the Qwen changes
below. The immutable run source is commit
`025f5139a38a18462057e50e6e63f480c49cdeda`; local review was performed at
`576c52067d8fef45df3e842bd6de4492e3f88fc2`. See
[the scoped review](E0B_REVIEW_58457867.md) and
[the file-level audit](E0B_AUDIT_58457867.json). Original manifests, summaries
and rank reports were not rewritten. Later source changes must not be compared
to this historical run as though they were its source.

The paper now records the scoped four-A100 small-GPT outer-state, next-consumer
and same-layout restart correctness results. Its 42 performance cells remain
TBD. The PDF was rebuilt and the edited pages were visually inspected; no
Qwen, throughput, GPU peak-memory or physical-wire result was inferred from E0b.

Added `pretrain_qwen.py`, the native model training contract and JSONL indexed
data preprocessing with pinned tokenizer probes and source/data hashes. The
model loads public weights before DDP/master/optimizer construction. Qwen
snapshot and data identity are now part of the centered checkpoint recipe;
old non-Qwen runs retain `None` for this optional field. These additions have
not run on CUDA and do not alter the accepted E0b evidence.

Local CPU verification: seven Qwen tests passed (7.400 s) and all 21
outer/checkpoint tests passed (16.538 s). This includes 28 native-vs-HF FP32
rank/configuration records, TP2 BF16 source loading, all seven pinned
architecture/TP parser contracts, and real tokenizer/indexed-dataset checks.
After adding the last unsupported-option guards, the parser test passed
again (0.020 s). The final two runtime checks also passed (4.200 s), including
changed Qwen data-identity rejection and successful full-state restoration
with the original identity. Gloo checks used authorized local loopback access.
Logs are `/private/tmp/pier-qwen-integration-tests.log`,
`/private/tmp/pier-outer-after-qwen-tests.log`,
`/private/tmp/pier-qwen-contract-final.log`, and
`/private/tmp/pier-qwen-checkpoint-identity-tests.log`.

No full-size Qwen checkpoint was downloaded or loaded locally. Cluster
snapshot/tokenizer and real-corpus paths are still to be established. There is
no new GPU submission script in this increment; the next gate must validate
real-checkpoint CUDA conversion and training before performance work. The
existing E0b gate does not need to be rerun merely for this archive review.

## E0c Qwen real-checkpoint gate prepared — 2026-09-17

Added an explicit pinned-snapshot download/check helper and an offline Slurm
gate for Qwen2.5-3B FP32/BF16 conversion on TP1/TP2. The launcher requests one
node, four GPUs and one hour, supports the existing interactive allocation,
and stops on the first failed phase. It produces full HF reference gradients
on scratch and checks all native parameters, logits and gradients in bounded
comparison chunks, using fixed versioned tolerances. The reference and native
models run in separate processes to avoid simultaneous residency. This does
not execute the Qwen training entrypoint or any optimizer/outer update.

All 14 Qwen CPU tests passed (29.978 s), including the new reference/native
I/O integration with a tiny saved HF model, both dtypes, real four-process
Gloo groups and TP1/TP2. Those temporary reports carry GPU_executed=false.
Negative checks reject bad/corrupted tensors, nonfinite/scaled/flipped
gradients, missing rank/parameter coverage, changed source identity and a
failed phase. The shell failure test records the original exit 23, runs the
summary and starts no subsequent phase. Git ignore checks include top-level
JSON/logs and exclude all downloaded weights/reference tensors.
Log: `/private/tmp/pier-qwen-with-e0c-tests.log`.

No full-size weight download or GPU job was performed locally. No E0c result
exists yet. User actions and the exact scope/resource/storage contract are in
[E0C_HANDOFF.md](../qwen/E0C_HANDOFF.md). E0b remains accepted; historical evidence
is unchanged. The performance cells remain TBD and the overall goal is active.

## Native G/R/W arms and complete-cycle recorder — 2026-09-17

Implemented native AllGather/ReduceScatter G/R/W arms in the same parameter-
coordinate, optimizer/skip and checkpoint path as Pier, plus equal explicit
workspace caps. Their numerical contracts are separate from the fixed-tree
oracle; cross-arm restoration is rejected. Added actual training-loop cycle
hooks and GPT loss-mask token counting, rank-duration maxima, skip accounting,
warmup/partial-cycle labels and clearly scoped allocator peaks. No GPU timing,
physical wire bytes, optimized offload/server/DTensor result or tuned baseline
is claimed. Details: [BASELINES_AND_METRICS.md](BASELINES_AND_METRICS.md).

The first 24-test outer run passed before cycle instrumentation. The expanded
26-test run passed 25 and found one test-fixture placement error (NameError),
not a production failure. After moving that block into the intended native-
runtime test and restoring an adjacent existing parser assertion, all eight
affected runtime/parser tests passed in 9.683 s. The final guard rejects use
of new options under the legacy runtime and has a targeted parser regression.
All 14 Qwen tests passed again (28.935 s). Logs are
`/private/tmp/pier-native-outer-tests.log`, `/private/tmp/pier-native-cycle-tests.log`,
`/private/tmp/pier-native-cycle-runtime-final.log`,
`/private/tmp/pier-cycle-args-final.log`, and `/private/tmp/pier-qwen-after-cycle-tests.log`.

No E0c artifact directory has been received locally. E0c remains the next
requested one-node/four-GPU/one-hour job; no additional GPU job is requested
for this increment. The full experiment goal remains incomplete.

## Streamed full-coordinate oracle — 2026-09-17

Added explicit file-backed full R/M oracle storage with bounded collective,
arithmetic and comparison tiles. The default memory oracle retains its 1M
local-parameter cap. Both modes check the same ordered FP32 update, all master
coordinates and owned R/M including padding. Model-commit checks now cast
bounded chunks. Streamed checkpoints store full R/M hash receipts and rebuild
disposable oracle files from restored owners, including verification of every
local R replica. They do not copy full oracle vectors into checkpoints.
Memory-mode zero momentum initialization now explicitly matches positive-zero
executor initialization. Details: [STREAMED_ORACLE.md](STREAMED_ORACLE.md).

The three new tests passed (4.365 s), then the complete outer-sync suite passed
all 29 tests (32.530 s). Real four-process Gloo checks compare memory/streamed
trajectories for s1/s2/s4, restore a midcycle checkpoint, reject changed hashes
and a corrupted non-source R replica, and check all 1,000,017 coordinates of
an over-cap vector while forbidding full-flat CPU copies. Deliberate corruption
of its last coordinate is detected. BF16 commit checking also detects the
last coordinate. File tests check exact bits, bounds and truncation. Four Qwen
parser/data/preflight tests passed (0.058 s); `git diff --check` passed.

Logs: `/private/tmp/pier-streamed-oracle-tests.log`,
`/private/tmp/pier-streamed-full-outer-tests.log`, and
`/private/tmp/pier-streamed-qwen-contract-tests.log`.

The ledger describes explicit oracle scratch and logical file sizes, not
total process RSS or peak GPU memory. OS file cache, model/optimizer storage
and full checkpoint loading remain outside that bound. No CUDA/full-Qwen
training result was produced and no performance cell changed. No additional
Slurm job is requested; E0c remains the next one-node/four-GPU/one-hour gate.

## Native Qwen activation recomputation — 2026-09-17

Connected full uniform/block and selective core-attention recomputation to the
native Qwen builder and training initializer, including existing TP-sharded
saved activations for full recomputation at TP>1. Defaults remain disabled.
Checks reject ignored options, invalid layer counts and uniform groups that
would run beyond this checkout's final layer. Initialization receipts record
the effective settings; centered checkpoint recipes reject a changed schedule
while preserving the historical disabled-mode interpretation.

The new four-process Gloo test produced 152 per-rank mode/layout/dtype records
across TP1/TP2/TP4, FP32/BF16, tied/untied embeddings, uniform/block layer counts,
selective attention and TP-sharded saved inputs. Every logit, loss and parameter
gradient matches the disabled-recompute result byte-for-byte. Hooks confirm
actual backward reexecution and absence of recompute in eval. BF16 cases use
the training initializer; parser checks cover all seven pinned model layouts.
The real outer checkpoint regression rejects a changed recompute recipe before
loading and still resumes with the original disabled recipe.

Final complete suites: 16 Qwen tests passed (41.934 s) and 29 outer-sync tests
passed (38.799 s); `git diff --check` passed. Logs:
`/private/tmp/pier-qwen-after-recompute-tests.log` and
`/private/tmp/pier-outer-after-recompute-tests.log`. Earlier focused tests passed
two recompute checks (6.310 s) and two parser/data checks (0.099 s); the final
suite additionally checks exact tensor bytes, including signed zeros.

All new numerical records use tiny, zero-dropout CPU models with adapted CUDA
device/RNG access. They do not establish CUDA RNG behavior, full-size/2K
training, peak memory or performance. No E0c artifact directory is present
locally and no verified cluster process/job handle is available. This increment
does not request a new job or change the E0c launcher: it remains one node,
four GPUs and one hour, with the real training gate still required afterward.

## Read-only raw-cycle collector — 2026-09-17

Added `cycle_summary.py` and versioned cycle reports with a shared run UUID,
planned attempted-step budget and initial/final successful-step clocks. The
collector requires complete all-rank evidence, rejects mixed/duplicated runs,
early termination, missing windows, inconsistent durations/counts and altered
warmup/partial labels, and computes a single total-tokens/total-seconds sample
per run. Already-global tokens are never multiplied by rank count. It retains
excluded raw cycles and hashes the exact input bytes without rewriting them.

The focused initial four tests passed (12.801 s). The final complete outer
suite passed all 35 tests (43.180 s), including six collector tests built on
actual four-process CPU/Gloo recorder output with controlled clocks. They
cover a real skip, unequal cycle durations, rank maxima, recovery starting
midcycle, warmup, trailing partials, missing/mixed/damaged evidence, strict JSON,
input preservation, duplicate-run rejection and output overwrite refusal.
The allocator-field test is a deliberately synthetic schema fixture, not CUDA
execution or a memory measurement. `git diff --check` passed.

Logs: `/private/tmp/pier-cycle-summary-tests.log` and
`/private/tmp/pier-outer-with-cycle-summary-tests.log`.

The collector returns `raw_cycles_validated` with `performance_result: false`.
It does not infer source/model/data/hardware identity, numerical training
correctness, successful launcher exit, equal tuning, independent paired runs,
confidence intervals or total GPU-hours. Distinct UUIDs are not statistical
independence evidence. Old unversioned cycle fixtures are rejected, not upgraded
or rewritten. No GPU run or new performance result was produced; the E0c
one-node/four-GPU/one-hour conversion job remains the next external action.

## Pinned optional text source and Parquet preprocessing — 2026-09-17

Added a replaceable FineWeb-Edu single-shard default with its full immutable
revision, exact 2,152,819,114-byte size and publisher SHA-256. The source metadata
was checked against the publisher file/pointer/commit pages; no public shard was
downloaded locally. `prepare_corpus.py` only downloads on explicit `--download`,
checks pinned package/tokenizer/source identity, and verifies existing outputs
before reuse. It does not replace wrong/partial data or install packages.
Source/order/budget/token counts enter the indexed-data v2 receipt and the Qwen
training/checkpoint identity. JSONL and historical v1 receipts remain supported.

The same preprocessing routine now reads actual Parquet text batches as well
as JSONL. Tests use a local real Parquet file and a small real fast-tokenizer
fixture; both paths produce identical `.bin/.idx` bytes. Other checks cover
whole-document token-budget prefixes, multilingual/newline text, null/missing
columns, changed corpus identity/bytes, modified indexed files, read-only reuse,
explicit fixed-revision download dispatch and legacy receipt acceptance.
Network downloads are mocked only in the dispatch test, not labelled as actual
corpus preparation. PyArrow 22.0.0 was installed in `/private/tmp/pier-data-pydeps`
for these checks; the user's Python environment was not modified.

The initial focused run passed three tests and found a legacy test fixture
whose three-token dataset was too short for its requested sequence length
three plus next-token target. The fixture was corrected to sequence length two;
the production minimum-length check was retained. The complete Qwen suite then
passed all 20 tests (37.459 s), including all existing model/gradient/E0c checks.
`git diff --check` and Git ignore checks for raw Parquet and indexed binary data
passed. Logs: `/private/tmp/pier-corpus-tests.log` and
`/private/tmp/pier-qwen-with-corpus-tests.log`.

Public-source tokenization with the pinned full Qwen tokenizer, actual corpus
counts, final workload selection, full training recipe and GPU training remain
pending. No performance cell changed and no new Slurm job was added. E0c still
requests one node/four GPUs/one hour and does not require this corpus.

## E0d full-Qwen training/restart gate — 2026-09-17

Added the fixed E0d configuration, real `pretrain_qwen.py` dispatch wrapper,
preflight/evidence collector and one-node/four-GPU/two-hour Slurm launcher.
It requires passed E0c evidence on identical source files, the pinned 3B
snapshot and audited real indexed data. Five phases exercise TP2/inner-DP1,
2K sequences, eight accumulated microbatches per learner, BF16 model/FP32
training state, full uniform recomputation, host/device outer state, a skip
and full-state save/resume. Each complete trajectory has 11 attempts, 10
successful updates and outer boundaries at 4/7/10. Resume does not save extra
checkpoints. Runtime reports now include consumed/omitted sampler counters.

The validator requires all 20 rank reports, source-bound launch and initializer
records, full coordinate coverage, 16 per-rank bitwise trajectory comparisons,
post-outer agreement and restored-file hashes matching the split checkpoint's
publication receipt. It copies/hashes 21 prerequisite E0c JSON files and rejects
the first failed/missing stage. Whole-model oracle files and the checkpoint
stay on scratch; about 429.38 GB additional free space is required, including
32 GiB headroom. This is a file-space allowance, not a total memory/fit claim.

Local checks use the actual Megatron parser and checkpoint recipe, fake Slurm
dispatch for shell fail-fast/exit behavior, mocked entrypoint dispatch/cleanup,
preflight ordering and temporary synthetic acceptance/negative-control JSON.
No synthetic GPU-labelled metadata is retained as experimental evidence.
The first focused run exposed two test fixture issues (missing deterministic
NCCL environment and macOS temporary-path canonicalization); both were fixed
without weakening production checks. Complete Qwen/outer suites then passed
27 tests (49.566 s) and 35 tests (57.209 s). After adding explicit save/restore
receipt binding, all seven E0d tests passed again (5.309 s). Shell syntax,
Git ignore scope and `git diff --check` passed.

Logs: `/private/tmp/pier-qwen-with-e0d-tests.log`,
`/private/tmp/pier-outer-with-e0d-tests.log`, and
`/private/tmp/pier-e0d-final-tests.log`. Instructions are in
`experiments/qwen/E0D_HANDOFF.md`. No corpus download, GPU job, Qwen GPU training
result or performance measurement was produced. E0c (one node/four GPUs/one
hour) is still the next external action; E0d follows after its acceptance.

## T/DTensor shared training backend — 2026-09-17

Connected explicit DTensor reference-owner layouts to the production parameter
coordinates and `CenteredRuntime`, including the full cohort family, host/device
state option, master-to-model commit, same inner optimizer/skip handling,
checkpoint and cycle meter. Ordered global mesh construction supports disjoint
noncontiguous corresponding-TP groups. Native SUM retains its own numerical
contract; the Pier fixed-tree oracle remains rejected for this arm.

The executor tiles complete coordinates, including padding, and reserves its
caller-held redistribution outputs in the explicit workspace tile allowance.
DTensor internal intermediates and library/allocator storage remain additional;
no total GPU peak bound or equal measured peak is claimed. Full checkpoint
recipes retain the native tile and cohort. Changing that recipe cannot silently
restore; same-configuration CPU restart reproduces all final state digests.

The actual four-process Gloo tests cover K=1/2/4, every valid s, noncontiguous
groups, partial tiles, sizes 1/5/7/9/23/65, all master/R/M coordinates, BF16
commit, aliases, dyadic exactness and ordinary FP32 tolerances. Actual AdamW,
dropout, a synchronized skip, scheduler/RNG/sampler counters and midcycle full
checkpoint restore run for s=1/2/4. Their CPU cycle reports pass the all-rank
collector, including useful-token accounting after the skip. No collective is
mocked and no CPU elapsed time is reported as a GPU measurement.

The first focused run identified PyTorch 2.6's documented list versus local
2.14's tuple mesh-coordinate representation. Normalizing it fixed the check;
the three focused tests then passed (8.330 s). The final complete outer suite,
including parser and cycle-collector integration checks, passed 38 tests
(64.232 s); all 27 Qwen tests passed (48.329 s). `git diff --check` passed.

Logs: `/private/tmp/pier-dtensor-tests.log`,
`/private/tmp/pier-outer-with-dtensor-tests.log`,
`/private/tmp/pier-qwen-after-dtensor-tests.log`.
The used public APIs were checked against official PyTorch 2.6 documentation;
actual 2.6/CUDA behavior, native-SUM numerical diagnostics, memory, physical
traffic and equal-budget tuning remain unvalidated. O/S, multislot scheduling,
distributed optimizer and the full E1 comparison are still incomplete.
No new Slurm script, GPU run, paper performance value or binary result artifact
was added. E0c remains one node/four GPUs/one hour and E0d one node/four GPUs/two
hours after the same-source E0c gate passes.

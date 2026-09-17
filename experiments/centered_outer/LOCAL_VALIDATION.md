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

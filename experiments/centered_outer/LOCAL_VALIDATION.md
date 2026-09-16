# Local validation — 2026-09-16

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

All GPU conclusions remain pending. This round is the first handoff, E0a;
the full E0 and all performance experiments remain incomplete. See README.md
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

Rerun the complete E0a job after syncing these changes. Keep the old job directory
for diagnosis; the new job must produce its own manifest and phase reports so
source hashes remain consistent. CUDA/NCCL protocol completion is still pending.

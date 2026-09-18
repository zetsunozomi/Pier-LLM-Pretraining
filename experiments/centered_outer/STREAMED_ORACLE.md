# Full-state verification with bounded oracle tiles

Status: implemented and checked with four-process CPU/Gloo; CUDA and full-size
Qwen training are still pending. This is a correctness tool, not a performance
measurement path. It does not change the accepted E0a/E0b archives.

## Enabling it

For an otherwise valid centered Pier correctness run, add:

```text
--outer-verify
--outer-verify-storage streamed
--outer-verify-tile-elements 65536
--outer-trace-dir /absolute/path/to/a/fresh/run
```

The run must have 1–500 attempted training steps. Use a fresh trace directory
also when restoring a checkpoint; existing oracle files are never overwritten.
The default `memory` oracle retains its one-million-local-parameter cap. The
explicit `streamed` mode removes that oracle allocation limit without sampling
coordinates or weakening the comparison. It applies only to the ordered Pier
arm; native G/R/W reductions have different numerical contracts.

These flags are not a complete Qwen recipe or a new Slurm submission command.
E0c remains the next requested job: one node, four GPUs, one hour, with commands
in [E0C_HANDOFF.md](../qwen/E0C_HANDOFF.md). E0c checks model conversion and does
not use this training-state oracle. A real-data training recipe and measured
large-model memory/storage fit are still required afterward.

## What is checked

At every outer boundary, the oracle gathers each learner's old FP32 masters a
tile at a time, subtracts the previous independent reference, applies the
ordered adjacent-pair FP32 reduction tree, and computes the FP32 Nesterov
update. The complete reference and momentum vectors live in private files.
It then checks every master coordinate and every locally owned R/M coordinate
bitwise, including zero padding. It records exact master-coordinate coverage
and the number of checked boundaries.

Existing runtime checks still cover successful-step scheduling, skipped-step
state, inner moments, and master-to-model commit before the next forward.
The model-commit check now casts and compares bounded chunks, including the
last partial chunk, instead of casting an entire large parameter at once.

## Storage and memory accounting

For `N` local master coordinates, `K` learners in the outer group and
`T = min(oracle_tile_elements, N)`, each rank has:

| Allocation | Explicit size |
|---|---:|
| Oracle reference and momentum files | `8 × N` bytes |
| Persistent device gather/local scratch | `4 × (K + 1) × T` bytes |
| Persistent CPU gather buffer | `4 × K × T` bytes |

For K=4 and T=65,536, device scratch is 1,310,720 bytes (1.25 MiB) per rank.
Additional R/M/arithmetic, file-I/O and comparison temporaries are chunked.
Reports expose this separate `oracle_allocation` ledger. It is **not** total
process RSS or measured peak GPU memory: OS file cache, collective-library
storage, model/optimizer/executor state and checkpoint loading are excluded.
Each rank keeps full local R/M oracle files, so aggregate disk usage includes
all ranks, even when their coordinates describe replicated model partitions.

Files are under `<outer_trace_dir>/.oracle/rank-N/`, are disposable, and are
excluded by `.gitignore` wherever the trace directory is placed. Ordinary
JSON/log evidence can use the existing Git return workflow; the oracle files
do not need to be transferred. File I/O, extra collectives, hashing and
comparisons make this deliberately expensive; do not report its runtime as
training throughput. Disk capacity and I/O time still need a cluster check.

## Checkpoint restoration

The normal full training checkpoint still saves model/master, optimizer and
owned R/M state. Streamed mode adds full-vector R/M SHA-256 receipts and a
checked-boundary count, without duplicating the full oracle vectors in the
checkpoint. On restore, it reconstructs fresh oracle files from restored R/M
owners, checks both full-vector hashes and all local R replicas, and verifies
the boundary count against the successful-step clock.

Midcycle masters are not used to reconstruct R: they may already contain
unsynchronized inner updates. Changing oracle storage mode on restore is
rejected. Default memory-mode checkpoint recipe compatibility is retained.
Checkpoint loading still has its existing full rank-state CPU memory cost;
the streamed oracle does not make that operation bounded by its tile size.

## Executed local checks — 2026-09-17

- Memory and streamed oracles produce identical complete trajectories for
  cohort sizes 1, 2 and 4 with real four-process Gloo communication.
- Midcycle checkpoint restoration matches uninterrupted continuation; changed
  oracle digests and a corrupted non-source R replica are rejected.
- A 1,000,017-coordinate runtime run checks every coordinate while the old
  memory mode correctly rejects its size; a changed final coordinate is
  detected. Streamed execution forbids the full-flat CPU-copy helper in tests.
- BF16 model commit detects a changed final coordinate in a large parameter.
- File-vector checks cover exact signed-zero bits, bounds, exclusive creation,
  dtype rejection and actual file truncation.

All 29 outer-sync tests passed (32.530 s); four Qwen parser/data/preflight tests
also passed (0.058 s). Logs are `/private/tmp/pier-streamed-full-outer-tests.log`
and `/private/tmp/pier-streamed-qwen-contract-tests.log`. These are CPU results;
no new GPU run, full-size Qwen training or paper performance result is claimed.

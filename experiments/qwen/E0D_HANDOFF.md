# E0d: real Qwen full-state training and restart gate

Status: implementation and local CPU checks are available; **no E0d GPU result
exists**. E0b is accepted. Run and review [E0c](E0C_HANDOFF.md) first; its passed
conversion evidence must match the exact source files used by E0d. This gate
does not produce a throughput, memory-peak, wire-traffic or model-quality result.

## Launch after E0c passes

After the normal Git commit/push and cluster pull, use the E0b Python environment
with the pinned Qwen dependencies described in the E0c handoff. Prepare audited
text data on a networked login/transfer node or a permitted CPU preprocessing
allocation, before requesting this GPU job:

```bash
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
"$PIER_PYTHON" experiments/qwen/prepare_corpus.py --model 3B --download
```

The replaceable default is the fixed FineWeb-Edu shard in [CORPUS.md](CORPUS.md):
about 2.15 GB source download and at most 300M tokens/1M documents. It requires
`pyarrow==22.0.0` and the pinned tokenizer packages. No packages are installed
by the command. An existing source/preprocessed dataset is supported; use
`PIER_QWEN_DATA_PREFIX` for a nondefault audited `.bin/.idx/.manifest.json`
prefix. For a custom model snapshot, pass `--snapshot "$PIER_QWEN_SNAPSHOT"`
to corpus preparation and preserve that variable when submitting the job.

Set `PIER_E0C_EVIDENCE` to the **actual passed E0c artifact directory**, replacing
the example path below, then submit from the cluster repository root:

```bash
export PIER_E0C_EVIDENCE=/absolute/path/to/passed/e0c-run
sbatch experiments/qwen/e0d.sbatch
```

The header requests **1 node, 4 GPUs, 2 hours**, `m4431 / regular / gpu`, using
the user's working account convention. Inside an existing one-node/four-GPU
interactive allocation, `bash experiments/qwen/e0d.sbatch` uses that allocation;
it does not request new resources or extend its remaining time. Sequential
`srun` steps use `SLURM_OVERLAP=1`, and torchrun uses `--max_restarts=0`.
Two hours is a requested limit, not a measured execution-time estimate.

Use `bash`, not `source`. As with E0c, the launcher verifies the repository
selected from explicit `PIER_ROOT`, its script location, `SLURM_SUBMIT_DIR`,
or the current directory. Slurm's spool copy and an unrelated interactive
allocation directory are supported. Startup prints repository/snapshot paths.

The preflight checks the full pinned snapshot, E0c raw text reports and source
hashes, tokenizer/data identities and indexed counts, four visible CUDA GPUs,
and free scratch. It requires at least 360,449 indexed tokens to cover this
fixed attempted-step window without an epoch repeat. Sources must remain
unchanged between E0c and E0d; do not mix evidence across code revisions.

## Fixed correctness workload

| Setting | Value |
|---|---|
| Model / topology | Qwen2.5-3B, TP2, inner DP1, two learners, four ranks |
| Sequence / batching | 2,048 tokens, microbatch 1, 8 accumulated microbatches per learner, global batch 16 |
| Precision | BF16 live model; FP32 masters, gradients, AdamW moments and outer R/M |
| Activation policy | Full layer recomputation, uniform groups of one; saved activations not distributed |
| Inner optimizer | AdamW, LR 1e-4 → 1e-5 cosine over 11 attempts, no warmup, betas .9/.95, epsilon 1e-8, weight decay .1, clip 1 |
| Outer update | Pier fixed tree, interval 3 successful steps, momentum .9, learning rate .7 |
| Explicit workspace / oracle | 64 MiB per rank; streamed full-coordinate oracle, 65,536-element tiles |
| Injected skip | Attempt 3 on rank 1, propagated to every rank |

The interval of three is specific to this short correctness gate. It does not
replace the paper's proposed interval-50 performance workload. Initial model
weights come from the pinned snapshot; optimizer and outer state begin fresh
except for the explicit resume phase.

Five sequential processes run, with START/DONE/FAILED labels:

| Phase | Outer state | Attempted steps |
|---|---|---|
| `s2-device` | Cohort 2, R/M on GPU | 1–11, baseline trajectory |
| `s1-host` | Cohort 1, R/M on host | 1–11 |
| `s2-host` | Cohort 2, R/M on host | 1–11 |
| `split` | Cohort 2, R/M on GPU | 1–5, save full per-rank checkpoint, intentional exit |
| `resume` | Same configuration as split | Restore at 5, execute 6–11 |

Each full trajectory must have 11 attempts, 10 successful updates, outer
boundaries at attempts 4/7/10 and a checked consumer forward at 11. The skipped
attempt consumes its data but changes neither model/master nor inner optimizer
state. The framework's `skipped_train_samples` counter tracks sampler omissions,
not the injected optimizer skip; it remains zero, while consumed samples reach
176 (80 at the split).

## Acceptance and retained evidence

Every rank must report complete coordinate/oracle coverage and finite actual
training loss. The validator checks full master/model/inner-state digests,
skip/outer-state invariants, expected rank groups, model/data/recompute identity,
source-bound launch arguments and counters. It requires 16 rank comparisons:
12 complete host/resume trajectories plus four split prefixes must equal the
baseline bitwise. Post-outer model/master parameters must agree across learners
for each corresponding TP rank. Different payload ledgers are excluded from
trajectory equality.

The checkpoint includes model, FP32 masters, optimizer moments, scheduler,
outer R/M, RNG, sampler counters, clock and trace. A portable
`checkpoint-manifest.json` binds each resume rank's verified checkpoint hash and
byte count to the split save. E0c's 21 text evidence files are copied and hashed
under `prerequisite-e0c/`, so returned E0d evidence includes its prerequisite.

First OOM, nonzero phase exit, missing evidence or numerical mismatch fails
the gate and stops remaining phases. Phases are separate training processes;
their planned startups are not an automatic retry loop. Groups are explicitly
destroyed when the wrapper exits. The resume phase does not save another large
checkpoint.

### Scratch requirement and Git return

For this TP2 layout each rank has 1,543,044,096 local parameter coordinates
(TP-replicated norms are included). Preflight requires **429,379,026,944 free
bytes, about 429.38 GB / 399.89 GiB**, beyond existing model/data/E0c files:

- Five retained full oracle pairs across four ranks: 246,887,055,360 bytes.
- Conservative allowance for one complete checkpoint: 148,132,233,216 bytes.
- Extra headroom: 32 GiB.

This is a file-space allowance, not measured storage usage or a memory bound;
filesystem quota, page cache, host/pinned memory, CUDA workspaces, activation
peaks and checkpoint-loading peaks remain separate. The intended first node
is four A100-40GB GPUs; actual fit and elapsed time are not yet validated.

Output is `local/qwen/e0d-JOBID-TIMESTAMP/`. Git allows top-level JSON/logs,
case JSON/logs and prerequisite E0c JSON; model/data, dataset cache, binary
checkpoint and `.oracle/` files remain ignored on scratch. Normal result return
needs **Git only, no LFS**. Preserve cluster binaries for diagnosis. Reviewing
JSON verifies reported identity/coverage/comparisons; recomputing tensor
comparisons requires those retained binaries or a real rerun.

Passing E0d permits the next gate for native baseline numerical behavior and
then equally tuned complete-cycle measurements. It does not fill a performance
TBD or validate distributed optimizer, PP/CP/MoE, changing topology, multislot
execution, larger models, long-run convergence or final quality.

## Local checks

The E0d tests exercise the real Megatron argument parser for every phase,
split/resume recipe compatibility, real shell control flow with a fake local
Slurm dispatcher, entrypoint dispatch and cleanup, preflight rejection ordering,
Git return rules and scratch arithmetic. Temporary synthetic JSON archives
exercise acceptance plus missing-rank, changed-data/state/source/checkpoint,
incomplete-coordinate, CPU-label and failure rejection. These metadata fixtures
are not GPU experiment results. Existing tiny-model Qwen and four-process Gloo
outer-state regressions provide the separate local model/runtime checks.

The complete Qwen suite passed 27 tests (49.566 s), and the outer-sync suite
passed 35 tests (57.209 s). After adding the saved-to-restored checkpoint receipt
binding, all seven E0d tests passed again (5.309 s). Shell syntax and
`git diff --check` passed. Logs are `/private/tmp/pier-qwen-with-e0d-tests.log`,
`/private/tmp/pier-outer-with-e0d-tests.log` and
`/private/tmp/pier-e0d-final-tests.log`. No GPU job was submitted locally.

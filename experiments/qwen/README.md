# Qwen2.5 native-model preparation

**Current next run (2026-09-21): [N2 performance comparison](N2_HANDOFF.md),
`bash experiments/qwen/n2.sbatch`.** G/P/R/W use the actual Qwen training loop
with cycle timing and allocator peaks. E0c/E0d passed reports are not required.
The preparation history below describes the earlier validation work.

Status (2026-09-17): native builder, TP weight mapping and snapshot preflight are
implemented and tested on CPU. No pretrained weights were downloaded, no Qwen
GPU job ran, and no Qwen training/performance result is claimed. The user has
returned a passing [E0b summary](../centered_outer/E0B_REVIEW_58457867.md);
the complete 75-file archive has also passed local review. The native training
entrypoint and JSONL preprocessing/identity contract are now implemented; real
checkpoint CUDA validation and the actual corpus recipe are next. The new
[E0c handoff](E0C_HANDOFF.md) now provides a pinned snapshot preparation command
and a one-node/four-GPU/one-hour conversion gate; it has not run on GPU yet.

## Implemented

- `megatron/core/models/qwen/config.py`: validate dense Qwen2.5 architecture,
  exact vocabulary, GQA, RoPE and tied embeddings; derive source tensor shapes.
- `megatron/core/models/qwen/model.py`: native Megatron GPT using local attention,
  GQA/QKV bias, SwiGLU and an explicit RMSNorm cast order; no HF/FSDP wrapper.
- `megatron/core/models/qwen/weights.py`: stream local safetensors into existing
  model parameters before optimizer construction, with complete key/shape and
  actual TP-coordinate checks. This is weights-only initialization.
- `preflight.py`: check pinned architecture alone, or verify a local checkpoint
  and tokenizer file bytes against the fixed public revision.
- `pins.json` and the three original config files: revision/config identities,
  exact unique parameter counts, published file sizes and LFS/Git identities.

The loader packs Q/K/V **within each query group**, then takes a rank's groups;
concatenating whole Q/K/V matrices gives a different layout. SwiGLU takes each
rank's gate and up slice separately before concatenation. Row-parallel output
and down projections split input columns. Norms are replicated. Tied output
weights must be absent or bitwise equal to the embedding; an extra optimizer
parameter is not created for the tied output.

The native builder preserves the checkpoint's exact vocabulary. Generic
tokenizer padding can add output classes and change the softmax denominator;
the training entry point preserves this invariant before tokenizer creation. For example,
Qwen2.5-3B has 151,936 embedding rows, divisible by TP2 but not by 128 × TP2.

Supported architectural layouts in this preparation stage:

| Model | Unique parameters | Validated TP splits | Published safetensors bytes |
|---|---:|---|---:|
| 1.5B | 1,543,714,304 | 1 / 2 | 3,087,467,144 |
| 3B | 3,085,938,688 | 1 / 2 | 6,171,926,992 |
| 7B | 7,615,616,512 | 1 / 2 / 4 | 15,231,271,888 |

These bytes are checkpoint file sizes, not network measurements or measured
runtime memory. TP4 for 1.5B/3B is rejected because these models have two KV groups.
PP/CP/EP, sequence parallelism, sliding windows, RoPE scaling and performance
kernel selection are outside this builder's current scope.

## Read-only preflight

Architecture only, from the repository root:

```bash
python experiments/qwen/preflight.py --model 3B --tp 2
```

To inspect an already downloaded local snapshot:

```bash
python experiments/qwen/preflight.py --model 3B --tp 2 \
  --checkpoint /absolute/path/to/the/pinned/Qwen2.5-3B/snapshot \
  --output /tmp/qwen-3b-snapshot-check.json
```

The second command reads and hashes full weight files; it may take time on shared
storage. It does not download, convert to another on-disk checkpoint, modify the
source, construct a GPU model or start training. It requires all selected model
and tokenizer files from the fixed revision. `architecture_checked` alone is not
checkpoint validation; `local_snapshot_checked` alone is not model or tokenizer
behavior validation. Both explicitly report `ready_for_training: false`.

The source reader requires safetensors 0.7.0, as pinned in the repository's
requirements. Model parity tests use Transformers 4.57.3, also already pinned
there. The normal E0b launcher does not import these new Qwen modules.

## Actual local evidence

Environment: macOS, Python 3.11, Torch 2.14.0, Transformers 4.57.3, safetensors 0.7.0.
Additional packages were installed only in `/private/tmp/pier-qwen-pydeps`.
No installed cluster package or existing local Python environment was upgraded.

The initial builder/mapping stage passed five tests in `tests/qwen`
(the training/data additions below bring the current total to seven):

1. Four real Gloo processes, seven architecture/TP configurations and 28 rank
   records: native Megatron vs HF full logits, loss and every parameter gradient.
   Small random models preserve the official 12:2, 16:2 and 28:4 head:KV ratios,
   tied/untied choices, and nonzero QKV biases. This does not use real pretrained
   weights. Maximum absolute errors: logits **4.172325e-7**, gradients
   **2.086163e-7**, loss **4.768372e-7**. FP32 checks use atol 2e-6, rtol 2e-5.
2. Sharded safetensors slices equal the in-memory source; missing bias tensors,
   conflicting tied weights and invalid TP layouts are rejected.
3. BF16 RMSNorm output, input gradient and gain gradient are bitwise equal to HF
   on the deterministic local fixture. This is not whole-model BF16 equivalence.
4. All three pinned parameter counts and the seven planned TP layouts match.
5. Same-sized but changed file contents fail both LFS SHA256 and Git-blob checks.

CPU parity uses the actual Megatron model and tensor-parallel math. The test
selects CPU for Megatron's hard-coded current-device calls and substitutes a
no-op CUDA RNG context only with dropout disabled; it supplies an explicit CPU
causal mask. These test adaptations are not present in the production builder,
and they do not establish CUDA correctness or performance.

Commands used:

```bash
# Requires transformers and safetensors plus local loopback socket access.
python -m unittest discover -s tests/qwen -v
python experiments/qwen/preflight.py --model 3B --tp 2
```

Local diagnostic log: `/private/tmp/pier-qwen-tests.log`; architecture-only report:
`/private/tmp/pier-qwen-3b-tp2-preflight.json`. No test fixture is kept as GPU evidence.

## Remaining integration

1. E0b archive review is complete; retain its immutable run/source identities.
2. Exercise the new training entrypoint on the cluster after model conversion
   validation; CPU tests do not establish real BF16 Qwen training behavior.
3. Verify BF16 CUDA logits/gradients and captured real checkpoint state, including
   TP2 (and TP4 for 7B), then validate the centered runtime on those models;
   CPU mapping checks do not complete E0.
4. Pin tokenizer behavior, real-data subset/order and optimizer hyperparameters;
   label public-weight initialization as weights-only warm start.
5. Build/tune strong baselines and collect complete-cycle throughput, actual
   memory and link traffic before filling the paper's performance cells.

The unfused model is an explicit correctness recipe. Performance experiments
still need appropriate inner kernels and equal tuning for every arm.

## Primary sources

The configs were copied byte-for-byte from the paper's existing hardware-plan
archive. Their original sources are the fixed official revisions:
[1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/8faed761d45a263340a0528343f099c05c9a4323/config.json),
[3B](https://huggingface.co/Qwen/Qwen2.5-3B/blob/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/config.json),
[7B](https://huggingface.co/Qwen/Qwen2.5-7B/blob/d149729398750b98c0af14eb82c78cfe92750796/config.json).
Published file identities were retrieved without authentication through the HF
model-info API for these exact revisions; `pins.json` records the endpoint URLs.
HF layout and normalization were checked against
[Transformers v4.57.3 Qwen2](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen2/modeling_qwen2.py).


## Training/data integration added after E0b

- `pretrain_qwen.py` uses the real Megatron pretrain loop, dataset provider and
  forward/loss path. It verifies the pinned snapshot on rank zero, broadcasts
  validation failure to all ranks, and returns a loaded native model before
  DDP/master/optimizer/R/M construction. Public weights remain explicitly
  **weights-only initialization**; an optional centered checkpoint restores
  the complete later training state.
- `megatron/core/models/qwen/training.py` installs architecture defaults before
  argument validation/tokenizer creation, fixes the exact output vocabulary,
  and rejects incompatible overrides and unsupported topology/kernel options.
  This initial unfused path is for correctness, not a tuned performance arm.
- `experiments/qwen/prepare_data.py` converts local UTF-8 JSONL or Parquet text into
  Megatron `.bin/.idx` using the pinned tokenizer with no implicit special tokens
  and one explicit EOS per document. It keeps a source-order prefix of whole
  documents, refuses overwrites, and records source/tokenizer/data hashes,
  tokenization probes, counts and budgets in `.manifest.json`.
- The training entrypoint requires that receipt and checks data file bytes,
  tokenizer identity/behavior and token bounds; it does not accept arbitrary
  pretokenized data or mock data under a real-data label. `qwen_recipe` is part
  of the centered checkpoint recipe, so changed source/data identities cannot
  silently restore old optimizer/outer state.

Preprocessing, after the snapshot/tokenizer and local JSONL are available:

```bash
python experiments/qwen/prepare_data.py --model 3B \
  --snapshot /absolute/path/to/pinned/Qwen2.5-3B \
  --input-jsonl /absolute/path/to/source.jsonl \
  --output-prefix /absolute/path/to/prepared/qwen3b \
  --max-documents 100000 --max-tokens 20000000
```

These are path placeholders for a user-provided corpus, not cluster locations
we have verified. This command consumes CPU/storage only and performs no download.
A [pinned FineWeb-Edu default and download/preprocessing helper](CORPUS.md) now
provides an alternative without supplying a corpus path. It has not downloaded
or prepared the public shard locally; final workload selection, counts and
hyperparameters still need to be fixed before an experiment.
The E0c launcher collects Qwen entrypoint/data sources in its manifest and runs
the real-checkpoint conversion gate before training output can be used as evidence.
The default in-memory `--outer-verify` oracle retains its 1M-local-parameter
limit. The explicit `--outer-verify-storage streamed` path checks every
coordinate using file-backed R/M and bounded tiles; CPU tests cover a vector
above that limit and full checkpoint restoration. See the
[streamed oracle contract](../centered_outer/STREAMED_ORACLE.md) for flags,
disk/scratch accounting and remaining limits. CUDA correctness and full-size
Qwen training memory/storage fit are still pending; this is not a performance
path or a completed 3B training validation.

Current local checks: 7 Qwen tests passed (7.400 s), including the existing
28 CPU FP32 forward/gradient records plus TP2 BF16 initialization through the
new training adapter; 21 outer/checkpoint tests passed (16.538 s). New parser
checks cover all seven pinned architecture/TP layouts and reject vocabulary,
RoPE, GQA and step-budget mismatches. The real tokenizer/indexed-data test
checks EOS/order/budgets and rejects modified bytes or a different revision.
Logs: `/private/tmp/pier-qwen-integration-tests.log` and
`/private/tmp/pier-outer-after-qwen-tests.log`. The final checkpoint regression
also rejects changed Qwen data identity before restoration, then restores the
original identity and matches the uninterrupted trajectory (two runtime tests,
4.200 s; `/private/tmp/pier-qwen-checkpoint-identity-tests.log`). No full-size model was loaded
locally and no Qwen GPU or performance result has been produced.

## E0c submission now available

See [E0C_HANDOFF.md](E0C_HANDOFF.md) for the exact commands, fixed numerical
thresholds, memory/storage estimates and artifact return policy. The launcher
supports `sbatch`, or `bash` inside an existing one-node/four-GPU allocation.
`prepare_snapshot.py --download` explicitly fetches missing pinned public files
on a networked login node; the GPU job is offline. FP32/BF16 and TP1/TP2 each
check complete weights, logits, loss and per-parameter gradients against HF.
It does not run the Qwen training loop or claim performance. Large reference
tensors stay on scratch; Git includes only top-level JSON/log evidence.

### Complete launcher output

`e0c.sbatch`, `e0c_fp64.sbatch`, and `e0d.sbatch` automatically create a fresh
`out/<stage>-<job>-<timestamp>.<unique>/out.txt` under the repository. Once the
repository is located, all stdout and stderr (including preflight errors and
the final summary) go into that file. The terminal prints its absolute path;
use `tail -f /absolute/path/to/out.txt` to follow progress. Each attempt gets a
new directory, including repeated runs within one interactive allocation.
`PIER_OUT_ROOT` optionally overrides the parent `out/` directory.

This works with both `bash` and `sbatch`. The Slurm `pier-*.out` file contains
the pointer to `out.txt`; Slurm opens that bootstrap file before the script can
create directories. Per-phase logs and JSON evidence still reside in the
reported `local/qwen/` run directory. Git allows `.txt`, `.log`, and `.json`
files directly inside each `out/` run directory; binaries remain ignored.

## E0d full-state training gate

[E0D_HANDOFF.md](E0D_HANDOFF.md) provides the next launcher after E0c passes on
the same source files: **one node, four GPUs, two hours**. It fixes real 3B/TP2,
2K sequences, eight accumulated microbatches per learner, BF16/FP32 mixed
training, uniform full-layer recomputation, interval-three outer updates and
an injected optimizer skip. Five phases compare GPU/host state placement and
full-state split/resume against one bitwise trajectory. This is a correctness
recipe; the later interval-50 performance workload remains separate.

The preflight checks E0c, tokenizer/data identity and actual indexed counts,
and about **429 GB of additional free scratch** for complete oracle/checkpoint
evidence. All numerical coordinates are checked through the streamed oracle.
Git returns JSON/logs, including the saved-checkpoint publication receipt and
archived E0c text evidence; large binaries stay on scratch. No Qwen GPU training
or performance result is claimed. The immediate external action remains E0c.

## Activation recomputation for the later training gate

The native builder and `pretrain_qwen.py` now pass explicit activation
recomputation settings into the existing Megatron implementation. The default
remains disabled. This permits the experiment plan's common activation policy
and later equal-budget tuning across Pier and the shared native baseline arms;
it does not select a final training recipe or establish that 3B/2K fits on GPU.

Supported argument fragments for an otherwise valid training command:

```text
# Recompute each complete transformer layer during backward.
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1

# Alternatively, recompute the first N complete layers.
--recompute-granularity full --recompute-method block --recompute-num-layers N

# Alternatively, recompute only core attention.
--recompute-activations
```

`N` is an integer from 1 to the model's layer count. For uniform groups, the
group size must divide that count: this checkout's forward loop does not clamp
the last group, so nondivisible sizes are rejected before model allocation.
Selective/core-attention recomputation takes no method or layer count. With
full recomputation and TP>1, `--distribute-saved-activations` additionally uses
Megatron's existing TP split/gather of saved inputs. PP/CP/sequence-parallel
support is unchanged and remains outside this Qwen adapter's current contract.

Initialization reports record the effective settings. Centered checkpoints
record granularity, method, layer count and saved-activation distribution and
reject a changed configuration on restore; historical disabled-mode recipes
retain their meaning. Recompute settings must be held fixed across paired arms
unless the experiment explicitly gives every arm the same retuning budget.

Local checks run the actual checkpoint autograd function and native model on
tiny Qwen configurations using real four-process Gloo TP1/TP2/TP4. FP32 and BF16
tests check all logits, loss and parameter gradients byte-for-byte against
recomputation disabled, count actual backward reexecution, and verify eval
does not recompute. BF16 cases go through the training initializer. CUDA
device/RNG access is adapted for these zero-dropout CPU tests; CUDA RNG,
full-size/2K training, peak memory and throughput remain unvalidated.

The final run passed all 16 Qwen tests (41.934 s), including 152 per-rank
recompute records, and all 29 outer-sync tests (38.799 s). Logs:
`/private/tmp/pier-qwen-after-recompute-tests.log` and
`/private/tmp/pier-outer-after-recompute-tests.log`.

No new Slurm job is requested for this change. E0c still runs conversion in
eval mode without activation recomputation; passing it will not certify this
training feature. The later full-state Qwen training gate must exercise the
chosen activation policy on the GPU.

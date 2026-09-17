# Qwen2.5 native-model preparation

Status (2026-09-17): native builder, TP weight mapping and snapshot preflight are
implemented and tested on CPU. No pretrained weights were downloaded, no Qwen
GPU job ran, and no Qwen training/performance result is claimed. The next cluster
submission remains [E0b](../centered_outer/E0B_HANDOFF.md).

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
the future training entry point must retain this invariant. For example,
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

Five tests passed in `tests/qwen`:

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

1. Review real E0b training/skip/restore evidence.
2. Wire the native Qwen builder and loader into the training entry point before
   FP32 master/optimizer/R/M construction, and preserve all architecture flags,
   exact vocabulary and source hashes in checkpoint recipes.
3. Verify BF16 CUDA logits/gradients and captured real checkpoint state, including
   TP2/TP4, then connect the centered runtime; CPU mapping checks do not complete E0.
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

# E0c: real Qwen2.5-3B CUDA conversion gate

Status: two real 3B GPU runs completed their HF FP32 reference and stopped
at native FP32/TP1 under the original numerical contract. **E0c is not accepted.**
E0b is already accepted and does not need to be rerun for this gate.

The user-returned log for `e0c-58511509-20260918-012718` reports the same failing
parameter on all four TP1 ranks: `decoder.layers.2.mlp.linear_fc2.weight`
(HF layer 2 `mlp.down_proj.weight`, zero-based layer numbering). Its returned
rank-0 statistics show 2 violations among 22,544,384 elements, relative L2
error 5.76894e-6 and maximum absolute error 4.95017e-5. Logits and loss pass.
This is a small overall discrepancy; the cause is not yet established, and
neither TP2 nor BF16 ran. The full cluster JSON/log artifacts should be retained
and returned through Git; the local analysis so far uses the pasted evidence.
The second run, `e0c-58511509-20260918-014121`, reproduced the same aggregate
gradient statistics and identified the same two coordinates on all four ranks:

| Weight coordinate | Native | HF | Absolute error / permitted error |
|---|---|---|---|
| `[325, 3189]` | 0.00942956004 | 0.00940536521 | 1.10574235 |
| `[1843, 3189]` | -0.02219339833 | -0.02216718532 | 1.07283331 |

Both use input channel 3189. This does not establish the cause: local gradient
summation and different forward/backward operands must be separated. A fresh
E0c run now records those operands and the decomposition described below.
Tolerances and both failed runs' outcomes remain unchanged.

## Run after the normal Git update

From the cluster repository root, use the same `diloco` Python that ran E0b:

```bash
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
sbatch experiments/qwen/e0c.sbatch
```

Place the eight pinned files listed below in the snapshot directory before
launching. Downloads with curl or the Hugging Face CLI are supported; no
separate preparation command is required for an existing snapshot. E0c
preflight checks all file identities inside the GPU allocation.

If files are not yet available, `prepare_snapshot.py --model 3B --download`
is an optional downloader for a login/transfer node with network access
and a working Hugging Face Python environment. The download is about
**6.18 GB**; no package is installed or upgraded by this command.
Required model packages are already pinned in the
repo: `transformers==4.57.3`, `safetensors==0.7.0`, `huggingface-hub==0.36.0`.
The `diloco` environment still needs to actually contain them; preflight fails
clearly if the two model-library versions differ. Do not replace its CUDA Torch
installation with a CPU wheel.

Default snapshot directory:
`local/qwen/models/Qwen2.5-3B/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/`.
The directory must directly contain:

```text
config.json
merges.txt
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
vocab.json
```

For another location, export `PIER_QWEN_SNAPSHOT=/absolute/path/to/snapshot`
before launching. The existing files must match the fixed revision; a matching
directory name is insufficient. Preflight does not download or overwrite files.

The Slurm header requests **1 node, 4 GPUs, 1 hour**, `m4431 / regular / gpu`.
Inside an already allocated **one-node, four-GPU** interactive session, use
`bash experiments/qwen/e0c.sbatch` instead of the last `sbatch` command. The
script verifies the allocation and four visible GPUs. Direct `bash` does not
request GPUs or extend the interactive allocation's remaining time.
Internal sequential `srun` steps default to `SLURM_OVERLAP=1` so they can share
the interactive shell's allocation, as in the accepted E0b invocation.

Use `bash`, not `source`. The launcher resolves the repository from an explicit
`PIER_ROOT`, the actual script location, `SLURM_SUBMIT_DIR`, or the current
working directory, verifying repository markers before using a candidate.
This supports Slurm's spool copy and interactive sessions allocated from a
different directory. Startup prints both the repository and snapshot paths;
a missing-directory error includes the actual attempted snapshot path.

## What runs

Six phases (two single-process HF references and four four-process native
launches), with explicit START/DONE/FAILED markers and no automatic retry:

1. HF FP32 reference: pinned BF16 source weights cast to FP32, exact source
   comparison, full logits and all parameter gradients saved to scratch.
2. Native FP32 / TP1: four ranks compare every mapped weight, output and gradient.
3. Native FP32 / TP2: two TP groups, all four ranks checked.
4. HF BF16 reference, same checkpoint and inputs.
5. Native BF16 / TP1.
6. Native BF16 / TP2.

HF and native models run sequentially, in separate processes, so their full
models/gradients do not coexist on one GPU. FP32 reference/native TP1 each need
about 24.69 GB just for weights and gradients; activations and CUDA workspaces
are additional. This is a storage estimate, not a measured GPU peak. The
intended first machine is the user's four A100-40GB node; actual fit remains a
GPU validation item. OOM stops the gate and is retained as a failure.

Input is two authored English/Chinese conversion probes, each 64 tokens, with
positions 0..63, no implicit special tokens, no dropout, no optimizer step.
The exact IDs and tokenizer/model/source identities are stored in the manifest.
This probe does not select or validate a real training corpus. The model uses
unfused eager attention, TF32 disabled, deterministic algorithms and no reduced
precision GEMM reduction. Loss is explicitly FP32 mean next-token cross entropy.

## Fixed numerical acceptance

Source-to-HF and source-to-native weights must be bitwise equal after the
declared dtype conversion. The whole output vocabulary and every parameter
gradient are checked, including tied embeddings; no sampled-coordinate pass.
Raw max error, squared norms, dot products, changed-element counts and coverage
are retained per tensor. `e0c_metrics.py` defines the versioned thresholds:

| Quantity | FP32 | BF16 |
|---|---|---|
| Logits | every element: atol 2e-4, rtol 2e-4 | relative L2 ≤ 0.02 and cosine ≥ 0.999 |
| Each parameter gradient | every element: atol 2e-5, rtol 2e-4 | relative L2 ≤ 0.15 and cosine ≥ 0.99 |
| Mean CE loss | atol 2e-5, rtol 2e-5 | absolute error ≤ 0.05 |

These are preregistered engineering conversion checks, not a proof of exact
BF16 trajectories or model quality. A zero reference gradient requires an
exact zero native gradient; any nonfinite value fails. BF16 gradients are
checked **per parameter**, so a large embedding cannot hide a wrong smaller
projection. Thresholds cannot be overridden through the launcher. A failed
comparison must be investigated before changing any numerical contract.

Elementwise comparisons also retain the logical tensor `shape`, `tolerance`,
`worst_element` (largest absolute-error/tolerance ratio), and the first eight
`outside_tolerance_samples` in logical flattening order. Each point contains
its flat index, tensor coordinates, actual/reference values, absolute error,
allowed tolerance and ratio. The full violation count is retained even when
there are more than eight samples; nonfinite elements still fail and are
counted separately. BF16 gradient/logit acceptance remains aggregate-based.
Failed gradient statistics are printed to the phase log as well as rank JSON.
These are diagnostic fields only; they do not change contract version 1,
acceptance, model arithmetic or the source-bound evidence requirements.

### Targeted FP32 gradient diagnostic

HF FP32 and native FP32/TP1 capture the input activation and incoming output
gradient of layer 2's down projection (zero-based). Hooks copy the operands to
CPU in the same `[batch, sequence, channel]` order and return no replacements.
TP2 and BF16 arithmetic and acceptance are unchanged. For tiny CPU fixtures,
the probe selects the last layer when layer 2 does not exist.

For each sampled violation (or the worst element when the tensor passes),
`e0c_linear_probe.py` recomputes the weight-gradient dot product using FP64
products and `math.fsum` over the captured FP32 operands. The report separates:

- Each implementation's actual gradient minus its recomputed dot product.
- The difference between the two recomputed dot products, further decomposed
  as `sum((native_x - hf_x) * hf_g) + sum(native_x * (native_g - hf_g))`.
- Cancellation within each sum, operand-vector errors and the four token
  positions with the largest product differences.

These are diagnostics of **FP32 operands**, not a whole-model FP64 reference
and not an alternate pass criterion. They locate whether the observed delta
arises locally or in the operands; upstream differences still need their own
investigation. A failing gradient keeps E0c failed and stops subsequent phases.

Every rank writes `linear-probe-fp32-tp1-rankN.json` and prints its points with
the prefix `[E0c linear probe]`. The JSON binds the input/manifest, HF/native
reports and both raw operand files by SHA-256. Return these small JSON files
and logs via Git after running the same launcher; no new packages are needed.

Final acceptance requires both HF references and all **16 native rank/layout/
dtype reports**, correct complete parameter coverage, unchanged source/input
identities, no worker failure record and a zero launcher exit. NCCL groups are
explicitly destroyed by each native process. The script stops on the first
failed phase; later absent results are not interpreted as passes.

## Artifacts and return path

Output: `local/qwen/e0c-JOBID-TIMESTAMP/`. Use a fresh directory for every run.
Top-level `*.json` and `*.log` are allowed by `.gitignore`, so use the same
Git commit/push and local pull workflow as E0b. **No Git LFS is needed for the
normal report return.**

Downloaded weights, `reference-fp32/`, `reference-bf16/` and `linear-traces/`
remain ignored. The targeted probe adds about 6.68 MB per trace (one HF plus
four native traces, about 33.42 MB total), retained only on cluster scratch.
Together the full gradient references need **18,515,632,128 bytes (~18.52 GB)**
plus outputs/metadata; the preflight requires another 2 GiB of free scratch.
Keep them on the cluster for mismatch diagnosis. Neither weights nor reference
tensors are copied into the Git report tree. Their hashes are recorded in the
HF reference reports. Local JSON review can check coverage/acceptance without
the binary references; recomputing the tensor comparison requires the retained
cluster tensors or another real run, not the JSON alone.

No new performance cell can be filled from this gate. Passing E0c only permits
the next stage: real-data Qwen training with full optimizer/outer-state checks,
followed by equally tuned baselines and complete-cycle measurements.

## Local verification

For the targeted linear diagnostic, all 15 E0c tests passed in 25.872 s on CPU
(`test_e0c`, `test_e0c_linear_probe`, `test_e0c_metrics_diagnostics`). They cover
unchanged linear outputs/input gradients/weight gradients with capture enabled,
canonical sequence/batch layout, controlled cancellation and operand changes,
and actual tiny HF/native FP32/BF16 TP1/TP2 runs. The integration also verifies
diagnostic hashes and that JSON is included but trace tensors remain ignored.
Log: `/private/tmp/pier-e0c-linear-probe-tests.log`. These tests do not establish
the cause of the real 3B GPU mismatch.

The initial E0c implementation passed all 14 Qwen tests in 29.978 s on CPU. Its integration regression
executes these exact reference/native routines using a tiny saved HF checkpoint,
both dtypes and real four-process Gloo TP1/TP2 groups, with test-only CPU device
adapters. Its temporary reports explicitly set `GPU_executed: false`, so they
cannot pass the production CUDA evidence check. Other checks reject corrupted
references, missing rank/parameter reports, changed source identity, nonfinite
values, sign/scaling errors, and a failed launcher phase; Git ignore checks
confirm text reports are included and model/reference tensors are excluded.
The test uses no full-size pretrained model and establishes no GPU fit or
numerical result. Log: `/private/tmp/pier-qwen-with-e0c-tests.log`.

Primary API references:
[HF fixed-revision downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download),
[Transformers 4.57.3 Qwen2 model](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen2/modeling_qwen2.py),
[PyTorch 2.6 distributed shutdown](https://docs.pytorch.org/docs/2.6/distributed.html#shutdown).

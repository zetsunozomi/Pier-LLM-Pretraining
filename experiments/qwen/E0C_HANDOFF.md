# E0c: real Qwen2.5-3B CUDA conversion gate

Status: launcher and local CPU regressions are ready; **no E0c GPU result exists**.
E0b is already accepted and does not need to be rerun for this gate.

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

Downloaded weights and `reference-fp32/`, `reference-bf16/` remain ignored.
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

All 14 Qwen tests passed in 29.978 s on CPU. The new integration regression
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

# A100 40GB: maximum trainable Qwen depth

This experiment produces one maximum feasible layer count and logical parameter
count for each outer-state method. It uses the existing R/G/OS/P implementations.
It does not generate an additional throughput sweep or load pretrained weights.

## Submit on Perlmutter

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
bash experiments/qwen/capacity_submit.sh
```

The default array scans R, G, OS, P2, P16, one arm at a time, with eight
A100-SXM4-40GB nodes / 32 GPUs per job and a one-hour wall limit. Mail is ALL to
sf850@scarletmail.rutgers.edu. An allocation saves each completed probe; if it runs
out of time, repeat **the same submit command**. Completed arms are omitted and
unfinished arms resume their saved search. Do not submit the same arm twice while
its previous job is pending or running. A per-arm lock rejects concurrent scans.

Half-hour allocations and individual arms are also supported:

```bash
PIER_CAPACITY_MINUTES=30 bash experiments/qwen/capacity_submit.sh R G
bash experiments/qwen/capacity_submit.sh OS P2 P16
```

O (the existing naive CPU baseline) and W can be requested explicitly using the
same command. O retains its existing uncapped whole-parameter scratch; it must
not be described as using the tiled workspace cap of the other methods.

For an existing **eight-node** exclusive allocation:

```bash
bash experiments/qwen/capacity.sbatch R
```

The default `PIER_CAPACITY_DIR` is `out/capacity40-depth-v1`. Set a new directory
for a fresh campaign. All driver logs, per-trial `out.txt` files, rank receipts,
and summaries live there. Only JSON and text are tracked. No checkpoints are
saved. The four tokenizer files from the existing pinned Qwen2.5-3B snapshot are
used; no new model download is needed. Set `PIER_CAPACITY_TOKENIZER` only if that
snapshot is stored somewhere else.

## Fixed recipe

- Qwen2.5-3B architecture: hidden 2048, FFN 11008, Q/KV heads 16/2,
  vocabulary 151936, tied embeddings. **Only depth changes.**
- TP2, PP1, inner DP1, K16; sequence 2048, microbatch 1, accumulation 8,
  global batch 128; BF16, FP32 masters/gradients/state; full uniform layer-wise
  activation recomputation; AdamW inner / Nesterov outer every 50 successful steps.
- 64 MiB managed workspace for R/G/OS/P2/P16 (and optional W), as in N2.
- Random initialization, seed 1234, synthetic tokens, identical recipe for every
  arm at a given depth. Existing runtime checks identical initial masters across
  learners. Logical parameter counts are checked against actual model tensors,
  counting tied embeddings and TP-replicated parameters once.

## Search and acceptance

1. Probe 36 layers (3.085938688B). If successful, double depth until CUDA OOM.
   If 36 fails, halve depth until a successful lower bound is found.
2. Binary search between successful and failed depths to a one-layer gap.
3. Each probe is a fresh distributed process and executes **51 successful steps**:
   first optimizer-state allocation, one actual outer update and one subsequent
   training step are included.
4. Confirm the candidate maximum in a fresh process with **151 successful steps**
   (three complete outer cycles plus one step). If it OOMs, reduce the bound and
   continue. `complete` requires that depth to pass and the next layer to OOM.
5. Require every rank to finish with finite model/loss and committed master/model
   weights. Only explicit CUDA OutOfMemoryError establishes an OOM bound. Generic
   Slurm/NCCL errors, CPU OOM, skipped steps, incomplete evidence and timeouts are
   errors or interruptions, never capacity bounds.

The search assumes feasibility decreases with depth under this fixed recipe.
The default ceiling is 256 layers. Reaching it successfully is reported as
`search_ceiling_reached`, **not** as a measured maximum. Change the ceiling only
in a fresh campaign. Job time estimates are scheduling aids, not performance
results. `needs_resume` means the saved search must continue in another allocation.

## Read the five limits

```bash
"$PIER_PYTHON" experiments/qwen/capacity.py \
  --output-dir out/capacity40-depth-v1 --summary
```

Each `<arm>/summary.json` records `max_confirmed_layers`,
`max_confirmed_parameters`, `smallest_oom_layers`, and `exact_layer_boundary`.
Only `status=complete` / `exact_layer_boundary=true` is a resolved maximum.
The statement supported is the largest depth that completes this fixed training
window on this hardware and recipe, not long-run convergence or model quality.

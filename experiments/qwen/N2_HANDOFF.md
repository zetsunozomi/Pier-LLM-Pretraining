# N2: first Qwen performance comparison

**Current handoff (2026-09-23):** use [O/R/P main comparison](N2_MAIN_HANDOFF.md)
and `experiments/qwen/n2_main.sbatch`. The 32-GPU G/R/W/P three-repeat window
and eight-GPU cohort pilot have completed. G/W now support the mechanism study;
the next main measurement uses naive unsharded CPU offload (O). The returned
sharded offload pilot is retained as OS ablation evidence. The instructions
below retain the earlier G/R/W/P workflow for reference.

**Historical progress (2026-09-21):** the four-GPU and eight-GPU pilots have returned.
The eight-GPU pilot completed G/P/R/W; P reached 21,125.86 tokens/s versus
20,853.64 for G and 21,112.62 for R, with 31.60 GiB allocated peak versus
R's 34.47 GiB. This was one measured cycle on mixed 40GB/80GB nodes.
The next run is the [N3 cohort comparison](N3_HANDOFF.md) on one uniform
eight-GPU allocation. The original launch instructions below remain available.

Current entrypoint: **`bash experiments/qwen/n2.sbatch`**. This runs actual
Qwen2.5-3B weights through `pretrain_qwen.py` and the shared G/P/R/W training
backends. E0c/E0d passed reports are **not prerequisites**. Existing numerical
reports are untouched; returned pilot evidence lives under `out/n2-*`.

## First run: interactive one node, four GPUs

After syncing the code to the cluster, inside the existing four-GPU allocation:

```bash
cd /pscratch/sd/s/syfan/Pier
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
PIER_N2_PROFILE=pilot PIER_QWEN_DATA_PREFIX= bash experiments/qwen/n2.sbatch
```

The existing eight-file Qwen snapshot is used automatically at
`local/qwen/models/Qwen2.5-3B/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.
Override `PIER_QWEN_SNAPSHOT` only if it lives elsewhere. No model download or
dependency installation is performed. Both direct `bash` and `sbatch` work.

The first comparison uses **synthetic tokens**, the pretrained Qwen model,
TP2/PP1/inner-DP1, BF16 model and FP32 optimizer/outer states. It is a systems
performance measurement, with no claim about corpus learning quality. If an
existing Qwen-tokenized indexed corpus and its preprocessing receipt are ready,
set `PIER_QWEN_DATA_PREFIX=/absolute/prefix/train` instead; `.bin`, `.idx` and
`.manifest.json` must exist. All arms use the same input and initialization.

| Setting | Pilot (default) | Main measurement window |
|---|---:|---:|
| Outer interval | 50 successful updates | 50 successful updates |
| Sequence / microbatch / accumulation | 2048 / 1 / 8 | 2048 / 1 / 8 |
| Warmup / measured cycles | 1 / 1 | 2 / 3 |
| Steps per arm | 100 | 250 |
| Independent launches per arm | 1 | 3 |
| Arm order | G, P, R, W | shuffled with recorded seeds per repeat |
| Explicit workspace cap | 64 MiB each | 64 MiB each, then equal 256 MiB candidate |
| Pier cohort | s=2 | s=2 |

The pilot first produces G/P, then adds R/W. It retains r=50 and the main
per-learner workload; only scale and sample count are smaller. Every arm starts
fresh. A failed/OOM arm is recorded and the other arms still run. Failed runs
are never speedup denominators. The existing single-slot Pier executor is being
measured; speedup is not guaranteed.

## Where to look

The terminal prints one path such as `out/n2-<job>-<time>.<unique>/out.txt`.
Worker output is captured automatically; nothing needs to be copied from terminal
scrollback. Under that same directory:

- `results.txt`: readable table, refreshed after **each arm**.
- `summary.json`: tokens/s, outer+commit mean seconds, peak allocated/reserved
  GiB, paired speedup against G, final loss and per-rank allocation metadata.
- `manifest.json`: complete recipe, arm order, source hashes and Git revision.
- `run-1-G.log`, etc.: full per-arm logs.
- `run-1-G/*.json`, etc.: raw per-rank cycles, pretrained initialization,
  worker/hardware receipts, launcher exit and any errors.

`tail -f <printed-directory>/out.txt` shows progress from a second shell.
Throughput is successful loss tokens divided by **sum of slowest-rank complete
cycle durations**. Warmup is excluded. Outer includes master-to-model commit;
memory is maximum-rank PyTorch allocated/reserved, not NVML/device-wide memory.
Startup, loading and untimed final commit checks are outside the cycle metric.
All runs keep finite-loss checks; a single final master→model check is untimed.
No full oracle, reference tensor dump, evaluation or checkpoint save runs here.

All JSON/log/txt evidence is Git-visible, including nested rank receipts.
Weights and dataset caches stay under ignored `local/` paths. To return one run:

```bash
git add -- out/n2-<job>-<time>.<unique>
git diff --cached --stat
git commit -m "Record N2 Qwen pilot measurements"
git push
```

## Next sizes, using the same entrypoint

For a two-node/eight-GPU pilot:

```bash
PIER_N2_PROFILE=pilot sbatch --nodes=2 experiments/qwen/n2.sbatch
```

For the 32-GPU main window, after the pilot shows what needs changing:

```bash
PIER_N2_PROFILE=main sbatch --nodes=8 --time=04:00:00 experiments/qwen/n2.sbatch
```

The allocation needs four GPUs per node. Each node launches one torchrun agent
and four GPU workers; static rendezvous uses the allocation's first hostname.
The time request is a starting budget, not a measured runtime guarantee.
The main window still does not by itself finish the paper baseline set: the
independent comparison S, equal 64/256 MiB tuning and a short native-reduction
numerical comparison remain. They do not block this first timing result.

Optional controls: `PIER_N2_ARMS=G,P` for just the first comparison,
`PIER_N2_WORKSPACE_MIB=256`, `PIER_N2_COHORT=1` or a power of two dividing K,
and `PIER_N2_REPEATS=1..3`. These are recorded; compare matching configurations.
To regenerate a summary after syncing artifacts:

```bash
python experiments/qwen/n2_summary.py out/n2-<job>-<time>.<unique>
```

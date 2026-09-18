# Pinned text input for the later Qwen training gate

The proposed default is one fixed FineWeb-Edu shard. It provides a reproducible
text source while the user's existing corpus path is unknown; local JSONL or
Parquet remains supported. E0c uses its authored conversion probes and does
not require this corpus. No public shard has been downloaded/preprocessed
locally, and no real-data Qwen training result is claimed.

## Source and selection

[`corpus_pin.json`](corpus_pin.json) fixes the source to:

- Dataset: `HuggingFaceFW/fineweb-edu`, `sample-10BT`, train split, English.
- Revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- File: `sample/10BT/000_00000.parquet`.
- Source size: **2,152,819,114 bytes (about 2.15 GB)**.
- SHA-256: `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871`.

The publisher's [file page](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/000_00000.parquet),
[LFS pointer metadata](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/raw/main/sample/10BT/000_00000.parquet)
and [commit](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/commit/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9)
were checked on 2026-09-17. Downloads use the full fixed revision and verify
content bytes, never a moving `main` reference. The
[dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) lists
ODC-By; the receipt preserves this attribution and source identity.

Selection is a source-order prefix of complete, nonempty `text` documents in
that shard. No extra quality filtering, ranking or random subsampling is
performed. It is a bounded systems-workload input, not a representative claim
about the whole corpus or a model-quality evaluation set.

## One-time preparation

Use the same Python environment as E0c after the pinned Qwen snapshot/tokenizer
is available. Download on a networked login/transfer node; perform tokenization
where the cluster permits CPU preprocessing:

```bash
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
"$PIER_PYTHON" experiments/qwen/prepare_corpus.py --model 3B --download
```

This is **CPU/storage work with no GPU allocation**. It does not install or
upgrade packages. The repository-pinned versions are checked before download:
`pyarrow==22.0.0`, `transformers==4.57.3`, `tokenizers==0.22.2`, and, when
downloading, `huggingface-hub==0.36.0`. A custom existing model snapshot can be
passed with `--snapshot "$PIER_QWEN_SNAPSHOT"`; only its config/tokenizer files
are needed for this step. Model weights are not fetched by the corpus helper.

The defaults cap preprocessing at **1,000,000 documents and 300,000,000 Qwen
tokens**, including one appended EOS per document. These are upper limits,
not measured corpus counts. A document that would exceed the token budget
ends the prefix; the helper does not skip it and cherry-pick later documents.
Actual counts and the source row cursor are in the receipt. The 300M-token
budget corresponds to at most 1.2 GB of int32 `.bin` data, plus index/receipt,
download/cache storage and preprocessing buffers.

For a smaller correctness-only input, use an explicit budget, for example:

```bash
"$PIER_PYTHON" experiments/qwen/prepare_corpus.py --model 3B --download \
  --max-tokens 20000000 --max-documents 100000
```

Changed budgets get distinct default output directories. Source data is reused
after verification. Repeating an identical command verifies existing indexed
files and the entire preprocessing recipe, then reuses them without rewriting.
Wrong source bytes, partial outputs, different recipes or modified indexed
files stop the command rather than silently replacing them.

Default source root:

```text
local/qwen/data/fineweb-edu/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/
```

The printed `output_prefix` points to
`indexed/<model-revision>/docs-<limit>-tokens-<limit>/train` below that root.
It is the prefix to use with `pretrain_qwen.py --data-path` or the E0d launcher's
`PIER_QWEN_DATA_PREFIX`. The helper does not launch training or establish
that the resulting dataset covers a requested run without epoch reuse; that
must be checked against measured token counts and the final global batch.

If source data already exists in the expected layout, omit `--download` and
optionally pass `--source-dir`; missing data then fails without network access.
The default `local/qwen/data/` tree is ignored by Git. Return initialization,
manifest and run JSON/log evidence, not the corpus or `.bin/.idx` files.

## Existing local data

The original `prepare_data.py --input-jsonl ...` interface remains available.
An alternative `--input-parquet ...` reads a local file's `text` column using
row batches; the two input flags are mutually exclusive. Both paths use the
same tokenizer, ordering, document/token limits and indexed-data writer.
Generic local files receive their real content hashes without a fabricated
public-dataset provenance label.

The new `pier-qwen-indexed-v2` receipt records source format and parser versions,
source and output hashes, tokenizer files/probes, budgets, actual counts and
optional fixed-corpus provenance. Training accepts this receipt and historical
`pier-qwen-jsonl-v1` receipts. The receipt hash and public provenance enter the
Qwen training/checkpoint identity, so changing the corpus cannot silently resume
an old trajectory. Arrow's decoding buffers are additional to the row batches;
the helper does not claim a bounded total RSS or measured preprocessing time.

[E0d](E0D_HANDOFF.md) now fixes the short correctness optimizer, activation,
batching/data-sampler and outer-state recipe and checks the actual indexed
counts before launch. Its GPU validation remains pending. This data helper
adds no Slurm job and does not change E0c's one-node/four-GPU/one-hour request.

Local verification: all 20 Qwen tests passed (37.459 s), including real local
Parquet versus JSONL byte-identical tokenization, whole-document limits,
source/output corruption rejection, verified reuse and v1/v2 training receipt
acceptance. Test data/tokenizers are small fixtures; public-shard download is
mocked only in the download-dispatch test. No public-source or GPU training
result is inferred. Log: `/private/tmp/pier-qwen-with-corpus-tests.log`.

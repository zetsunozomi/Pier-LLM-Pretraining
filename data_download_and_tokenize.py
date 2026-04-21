import argparse
import multiprocessing
import os
import sys
import time
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from megatron.training.tokenizer import build_tokenizer
from megatron.core.datasets import indexed_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def download_tokenizer_files(save_dir):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_vocabulary(save_dir)
    print(f"Tokenizer files saved to {save_dir}")


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        ids = {}
        lens = {}
        doc_ids = Encoder.tokenizer.tokenize(text)
        if len(doc_ids) > 0:
            if self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
        ids['text'] = doc_ids
        lens['text'] = [len(doc_ids)]
        return ids, lens


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='tokenizer download')
    parser.add_argument('--tokenizer-save-dir', type=str, required=True,
                        help='Directory to save downloaded GPT-2 tokenizer files')

    group = parser.add_argument_group(title='input data')
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--dataset-split', type=str, default='train')
    parser.add_argument('--cache-dir', type=str, default=None)

    group = parser.add_argument_group(title='tokenizer')
    parser.add_argument('--tokenizer-type', type=str, default='GPT2BPETokenizer')
    parser.add_argument('--tokenizer-model', type=str, default=None)
    parser.add_argument('--vocab-file', type=str, default=None)
    parser.add_argument('--merge-file', type=str, default=None)
    parser.add_argument('--append-eod', action='store_true')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128)
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1)
    parser.add_argument('--vocab-extra-ids', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)

    group = parser.add_argument_group(title='output data')
    parser.add_argument('--output-prefix', type=str, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--dataset-fraction', type=float, default=1.0)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Step 1: download tokenizer files
    os.makedirs(args.tokenizer_save_dir, exist_ok=True)
    download_tokenizer_files(args.tokenizer_save_dir)

    # wire vocab/merge paths if not explicitly provided
    if args.vocab_file is None:
        args.vocab_file = os.path.join(args.tokenizer_save_dir, 'vocab.json')
    if args.merge_file is None:
        args.merge_file = os.path.join(args.tokenizer_save_dir, 'merges.txt')

    # Step 2: load dataset
    dataset_kwargs = {
        "path": args.dataset_path,
        "name": args.dataset_name,
        "split": args.dataset_split,
        "cache_dir": args.cache_dir,
    }
    print(f"Loading dataset {args.dataset_path}...")
    ds = load_dataset(**dataset_kwargs)

    if args.dataset_fraction < 1.0:
        total_len = len(ds)
        target_len = int(total_len * args.dataset_fraction)
        print(f"Slicing dataset: using {target_len} out of {total_len} samples ({args.dataset_fraction*100}%)")
        ds = ds.select(range(target_len))

    print(f"Final dataset size to process: {len(ds)}")

    # Step 3: tokenize and write indexed dataset
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)

    output_bin_file = "{}_{}_{}.bin".format(args.output_prefix, 'text', 'document')
    output_idx_file = "{}_{}_{}.idx".format(args.output_prefix, 'text', 'document')

    builder = indexed_dataset.IndexedDatasetBuilder(
        output_bin_file,
        dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
    )

    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    total_docs = len(ds)
    print("Starting tokenization...")
    start_time = time.time()

    def text_generator():
        for i in range(total_docs):
            yield ds[i]['text']

    encoded_docs = pool.imap(encoder.encode, text_generator(), chunksize=100)

    for i, (doc, lens) in enumerate(encoded_docs):
        if i % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i}/{total_docs} docs ({i/elapsed:.2f} docs/s)", flush=True)
        builder.add_document(doc['text'], lens['text'])

    builder.finalize(output_idx_file)
    pool.close()
    pool.join()
    print("Done!")


if __name__ == "__main__":
    main()

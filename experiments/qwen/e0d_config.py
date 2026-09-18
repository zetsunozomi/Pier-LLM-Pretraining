"""Fixed four-GPU Qwen2.5-3B training-state gate; no performance claim."""

from pathlib import Path
import json

CASES = {
    's2-device': {'cohort': 2, 'host': False},
    's1-host': {'cohort': 1, 'host': True},
    's2-host': {'cohort': 2, 'host': True},
    'split': {'cohort': 2, 'host': False, 'stop': 5},
    'resume': {'cohort': 2, 'host': False, 'resume': True},
}
ATTEMPTS, SEQUENCE, GLOBAL_BATCH = 11, 2048, 16
RECOMPUTE = {'recompute_granularity': 'full', 'recompute_method': 'uniform',
             'recompute_num_layers': 1, 'distribute_saved_activations': False}
NOT_VALIDATED = ['tuned native G/R/W/O/S/T baseline performance', 'multi-slot CUDA pipeline',
                 'distributed optimizer, PP/CP/MoE or topology-changing recovery',
                 'other model sizes/topologies or activation policies',
                 'steady-state throughput, total peak memory or physical wire traffic',
                 'long-run convergence or model quality']


def default_data_prefix():
    root = Path(__file__).resolve().parents[2]
    corpus = json.loads((root / 'experiments/qwen/corpus_pin.json').read_text())
    model = json.loads((root / 'experiments/qwen/pins.json').read_text())['models']['3B']
    return (root / 'local/qwen/data/fineweb-edu' / corpus['revision'] / 'indexed'
            / model['revision'] / 'docs-1000000-tokens-300000000/train')


def training_args(case, output, snapshot, data_prefix):
    config, output = CASES[case], Path(output)
    directory = output / f'case-{case}'
    args = [
        '--qwen-model-size', '3B', '--qwen-snapshot', str(snapshot),
        '--qwen-trace-dir', str(directory), '--tokenizer-model', str(snapshot),
        '--data-path', str(data_prefix), '--split', '100,0,0',
        '--seq-length', str(SEQUENCE), '--num-workers', '0', '--dataloader-type', 'single',
        '--data-cache-path', str(output / 'dataset-cache'),
        '--tensor-model-parallel-size', '2', '--pipeline-model-parallel-size', '1',
        '--num-subgroup', '2', '--micro-batch-size', '1', '--global-batch-size', str(GLOBAL_BATCH),
        '--train-iters', str(ATTEMPTS), '--optimizer', 'adam',
        '--lr', '0.0001', '--min-lr', '0.00001', '--lr-decay-iters', str(ATTEMPTS),
        '--lr-decay-style', 'cosine', '--lr-warmup-iters', '0', '--weight-decay', '0.1',
        '--adam-beta1', '0.9', '--adam-beta2', '0.95', '--adam-eps', '1e-8', '--clip-grad', '1.0',
        '--bf16', '--accumulate-allreduce-grads-in-fp32', '--local-sgd-inner-average',
        '--deterministic-mode', '--recompute-granularity', 'full',
        '--recompute-method', 'uniform', '--recompute-num-layers', '1',
        '--outer-runtime', 'centered', '--outer-arm', 'pier', '--outer-cohort-size', str(config['cohort']),
        '--outer-workspace-mib', '64', '--outer-sync-interval', '3',
        '--outer-momentum', '0.9', '--outer-learning-rate', '0.7', '--momentum-warmup-steps', '0',
        '--outer-verify', '--outer-verify-storage', 'streamed', '--outer-verify-tile-elements', '65536',
        '--outer-trace-dir', str(directory), '--outer-inject-skip-at', '3', '--outer-inject-skip-rank', '1',
        '--eval-iters', '0', '--eval-interval', '1000', '--log-interval', '1', '--seed', '1234',
        '--ckpt-format', 'torch', '--save-interval', '5',
    ]
    if config['host']:
        args.append('--outer-cpu-offload')
    if case == 'split':
        args += ['--save', str(output / 'checkpoints'), '--exit-interval', '5']
    if config.get('resume'):
        # Reuse the split checkpoint, but do not write redundant large checkpoints.
        args += ['--load', str(output / 'checkpoints')]
    return args


if __name__ == '__main__':
    print(default_data_prefix())

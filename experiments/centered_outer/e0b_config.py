"""One-node/four-GPU real pretrain_gpt correctness matrix (no model downloads)."""

CASES = {
    'tp1-s1': dict(tp=1, inner=1, cohort=1),
    'tp1-s2': dict(tp=1, inner=1, cohort=2),
    'tp1-s4': dict(tp=1, inner=1, cohort=4),
    'tp1-host': dict(tp=1, inner=1, cohort=2, host=True),
    'split': dict(tp=1, inner=1, cohort=2, stop=5),
    'resume': dict(tp=1, inner=1, cohort=2, resume=True),
    'tp2-s2': dict(tp=2, inner=1, cohort=2),
    'dp2-s2': dict(tp=1, inner=2, cohort=2),
}


def training_args(case, output):
    config = CASES[case]
    dp = 4 // config['tp']
    args = [
        '--num-layers', '2', '--hidden-size', '64', '--ffn-hidden-size', '128',
        '--num-attention-heads', '4', '--seq-length', '64', '--max-position-embeddings', '64',
        '--tokenizer-type', 'NullTokenizer', '--vocab-size', '4096', '--make-vocab-size-divisible-by', '128',
        '--mock-data', '--split', '100,0,0', '--num-workers', '0', '--dataloader-type', 'single',
        '--tensor-model-parallel-size', str(config['tp']), '--pipeline-model-parallel-size', '1',
        '--num-subgroup', str(dp // config['inner']), '--micro-batch-size', '1',
        '--global-batch-size', str(2 * dp), '--train-iters', '11',
        '--lr', '0.001', '--min-lr', '0.0001', '--lr-decay-iters', '11', '--lr-decay-style', 'cosine',
        '--lr-warmup-iters', '0', '--weight-decay', '0.1', '--adam-beta1', '0.9', '--adam-beta2', '0.95',
        '--clip-grad', '1.0', '--hidden-dropout', '0.1', '--attention-dropout', '0.1',
        '--bf16', '--accumulate-allreduce-grads-in-fp32', '--local-sgd-inner-average',
        '--transformer-impl', 'local', '--no-masked-softmax-fusion', '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion', '--no-rope-fusion', '--no-persist-layer-norm',
        '--no-gradient-accumulation-fusion', '--deterministic-mode',
        '--outer-runtime', 'centered', '--outer-cohort-size', str(config['cohort']),
        '--outer-tile-elements', '257', '--outer-sync-interval', '3',
        '--outer-momentum', '0.9', '--outer-learning-rate', '0.7', '--momentum-warmup-steps', '0',
        '--outer-verify', '--outer-trace-dir', str(output / f'case-{case}'),
        '--outer-inject-skip-at', '3', '--outer-inject-skip-rank', '1',
        '--eval-iters', '0', '--eval-interval', '1000', '--log-interval', '1',
        '--seed', '1234', '--ckpt-format', 'torch', '--save-interval', '5',
    ]
    if config.get('host'):
        args.append('--outer-cpu-offload')
    if case in ('split', 'resume'):
        args += ['--save', str(output / 'checkpoints')]
    if config.get('stop'):
        args += ['--exit-interval', str(config['stop'])]
    if config.get('resume'):
        args += ['--load', str(output / 'checkpoints')]
    return args

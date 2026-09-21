"""Small, inspectable Qwen N2 recipes; no E0c/E0d prerequisite."""

import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
REVISION = '3aab1f1954e9cc14eb9509a215f9e5ca08227a9b'
ARMS = {'G': 'gather', 'R': 'resident', 'W': 'recenter', 'P': 'pier'}


def configuration(env=None):
    env = os.environ if env is None else env
    profile = env.get('PIER_N2_PROFILE', 'pilot')
    if profile not in ('pilot', 'main'):
        raise ValueError('PIER_N2_PROFILE must be pilot or main')
    suite = env.get('PIER_N2_SUITE', 'baselines')
    if suite not in ('baselines', 'cohorts'):
        raise ValueError('PIER_N2_SUITE must be baselines or cohorts')
    nodes = int(env.get('SLURM_JOB_NUM_NODES', env.get('SLURM_NNODES', '1')))
    if nodes not in (1, 2, 4, 8):
        raise ValueError('N2 supports 1, 2, 4 or 8 nodes, four GPUs each')
    arms = ('P' if suite == 'cohorts' else env.get('PIER_N2_ARMS', 'G,P,R,W')).replace(' ', ',').split(',')
    arms = [arm for arm in arms if arm]
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('PIER_N2_ARMS must contain distinct G,P,R,W labels')
    cohort = int(env.get('PIER_N2_COHORT', '2'))
    learners = nodes * 2
    if cohort < 1 or cohort & (cohort - 1) or learners % cohort:
        raise ValueError('PIER_N2_COHORT must be a power of two dividing the learner count')
    workspace = int(env.get('PIER_N2_WORKSPACE_MIB', '64'))
    if workspace not in (64, 256):
        raise ValueError('PIER_N2_WORKSPACE_MIB must be 64 or 256')
    repeats = int(env.get('PIER_N2_REPEATS', '1' if profile == 'pilot' else '3'))
    if not 1 <= repeats <= 3:
        raise ValueError('PIER_N2_REPEATS must be 1..3')
    prefix = env.get('PIER_QWEN_DATA_PREFIX') or None
    root = Path(env.get('PIER_ROOT', ROOT)).resolve()
    return dict(repository=str(root), profile=profile, suite=suite,
                stage='N3' if suite == 'cohorts' else 'N2',
                cohorts=sorted(set((1, 2, learners))) if suite == 'cohorts' else [cohort],
                log_interval=1, expected_gpu=env.get('PIER_N2_EXPECTED_GPU') or None,
                nodes=nodes, world_size=nodes * 4, tp=2, learners=learners,
                arms=arms, cohort=cohort, workspace_mib=workspace, repeats=repeats,
                interval=50, warmup_cycles=1 if profile == 'pilot' else 2,
                measured_cycles=1 if profile == 'pilot' else 3,
                attempts=100 if profile == 'pilot' else 250,
                sequence=2048, microbatch=1, accumulation=8, global_batch=learners * 8,
                snapshot=str(Path(env.get('PIER_QWEN_SNAPSHOT',
                    root / 'local/qwen/models/Qwen2.5-3B' / REVISION)).resolve()),
                data_prefix=str(Path(prefix).resolve()) if prefix else None,
                data_kind='indexed_text' if prefix else 'synthetic_tokens',
                seed=1234, order_seed=421, model='Qwen/Qwen2.5-3B', revision=REVISION)


def cases(config):
    result = []
    for repeat in range(config['repeats']):
        sweep = config.get('suite') == 'cohorts'
        order = [(arm, s) for arm in config['arms']
                 for s in (config['cohorts'] if sweep and arm == 'P'
                           else [config['cohort'] if arm == 'P' else 1])]
        # The first pilot gets the direct G/P comparison as soon as possible.
        # Main runs use a recorded order fixed before any measurements exist.
        if config['profile'] == 'main':
            random.Random(config['order_seed'] + repeat).shuffle(order)
        for arm, cohort in order:
            suffix = f'-s{cohort}' if sweep else ''
            result.append({'id': f'run-{repeat + 1}-{arm}{suffix}', 'repeat': repeat + 1,
                           'arm': arm, 'backend': ARMS[arm],
                           'cohort': cohort})
    return result


def training_args(config, case, directory):
    directory = Path(directory)
    options = {
        '--qwen-model-size': '3B', '--qwen-snapshot': config['snapshot'],
        '--qwen-trace-dir': str(directory), '--tokenizer-model': config['snapshot'],
        '--split': '100,0,0', '--seq-length': config['sequence'], '--num-workers': 0,
        '--dataloader-type': 'single',
        '--data-cache-path': str(Path(config['repository']) / 'local/qwen/n2-dataset-cache'),
        '--tensor-model-parallel-size': 2, '--pipeline-model-parallel-size': 1,
        '--num-subgroup': config['learners'], '--micro-batch-size': 1,
        '--global-batch-size': config['global_batch'], '--train-iters': config['attempts'],
        '--optimizer': 'adam', '--lr': '.0001', '--min-lr': '.00001',
        '--lr-decay-iters': 250, '--lr-decay-style': 'cosine', '--lr-warmup-iters': 0,
        '--weight-decay': '.1', '--adam-beta1': '.9', '--adam-beta2': '.95',
        '--adam-eps': '1e-8', '--clip-grad': 1,
        '--recompute-granularity': 'full', '--recompute-method': 'uniform',
        '--recompute-num-layers': 1,
        '--outer-runtime': 'centered', '--outer-arm': case['backend'],
        '--outer-cohort-size': case['cohort'], '--outer-workspace-mib': config['workspace_mib'],
        '--outer-sync-interval': config['interval'], '--outer-momentum': '.9',
        '--outer-learning-rate': '.7', '--momentum-warmup-steps': 0,
        '--outer-measure-dir': str(directory), '--outer-warmup-cycles': config['warmup_cycles'],
        '--eval-iters': 0, '--log-interval': config.get('log_interval', 1), '--seed': config['seed'],
    }
    argv = [item for pair in options.items() for item in (pair[0], str(pair[1]))]
    argv += ['--bf16', '--accumulate-allreduce-grads-in-fp32', '--local-sgd-inner-average']
    if config['data_prefix']:
        argv += ['--data-path', config['data_prefix']]
    else:
        argv += ['--mock-data', '--qwen-synthetic-benchmark']
    return argv


def launch_command(config, case, output, *, slurm=True):
    node = ['bash', str(ROOT / 'experiments/qwen/n2_node.sh'), str(output), case['id']]
    if slurm:
        return ['srun', f"--nodes={config['nodes']}", f"--ntasks={config['nodes']}",
                '--ntasks-per-node=1', '--kill-on-bad-exit=1', '--gpu-bind=none', *node]
    if config['nodes'] != 1:
        raise ValueError('multiple nodes require a Slurm allocation')
    return node

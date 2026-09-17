"""Synchronous complete per-rank Local-SGD checkpoints with atomic publication.

Each rank keeps its own model, masters, inner moments, R/M, scheduler and RNG.
The single deterministic Megatron sampler resumes from consumed_train_samples.
Topology changes, asynchronous saves and weights-only imports are separate work.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .runtime import digest


RECIPE_FIELDS = (
    'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups',
    'kv_channels', 'seq_length', 'max_position_embeddings', 'position_embedding_type',
    'rotary_percent', 'rotary_base', 'normalization', 'norm_epsilon',
    'hidden_dropout', 'attention_dropout', 'untie_embeddings_and_output_weights',
    'tensor_model_parallel_size', 'num_subgroup', 'micro_batch_size', 'global_batch_size',
    'seed', 'data_path', 'data_args_path', 'split', 'mock_data', 'tokenizer_type', 'vocab_size',
    'padded_vocab_size', 'train_iters', 'lr', 'min_lr', 'lr_decay_iters', 'lr_decay_style',
    'lr_warmup_iters', 'lr_warmup_fraction', 'adam_beta1', 'adam_beta2', 'adam_eps',
    'weight_decay', 'clip_grad', 'optimizer', 'bf16', 'local_sgd_inner_average',
    'outer_sync_interval', 'outer_momentum', 'outer_learning_rate', 'outer_cohort_size',
    'outer_cpu_offload', 'outer_verify', 'outer_inject_skip_at', 'outer_inject_skip_rank',
)


def recipe(args):
    return {key: getattr(args, key, None) for key in RECIPE_FIELDS}


def rng_state():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
        state['tracker'] = get_cuda_rng_tracker().get_states()
    return state


def restore_rng(state):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'].cpu())
        get_cuda_rng_tracker().set_states({k: v.cpu() for k, v in state['tracker'].items()})


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def save(runtime, iteration, scheduler, flops):
    args = runtime.args
    if iteration != runtime.clock.attempted:
        raise ValueError('checkpoint is not at a completed attempted-step boundary')
    root = Path(args.save)
    directory = root / f'iter_{iteration:07d}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        raise FileExistsError(f'completed checkpoint already exists: {directory}')
    if runtime.device.type == 'cuda':
        torch.cuda.synchronize(runtime.device)
    state = {'format': 'pier-centered-v1', 'rank': runtime.rank, 'world': runtime.world,
             'iteration': iteration, 'flops': flops, 'recipe': recipe(args),
             'coordinate_schema': runtime.coordinates.schema,
             'coordinate_fingerprint': runtime.coordinates.fingerprint,
             'outer_ranks': runtime.peers, 'inner_ranks': runtime.inner_ranks,
             'model': [module.state_dict() for module in runtime.model],
             'optimizer': runtime.optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'reference': runtime.executor.reference, 'momentum': runtime.executor.momentum,
             'clock': asdict(runtime.clock), 'rng': rng_state(),
             'consumed_train_samples': args.consumed_train_samples,
             'consumed_valid_samples': args.consumed_valid_samples,
             'skipped_train_samples': args.skipped_train_samples,
             'consumer_checks': runtime.consumer_checks,
             'pending_consumer': runtime.pending_consumer,
             'events': runtime.events,
             'oracle_reference': runtime.oracle_reference, 'oracle_momentum': runtime.oracle_momentum}
    if runtime.verify:
        state['optimizer_digest'] = digest(state['optimizer'])
        state['model_digest'] = digest(state['model'])
    temporary = directory / f'rank-{runtime.rank}.pt.tmp'
    destination = directory / f'rank-{runtime.rank}.pt'
    torch.save(state, temporary)
    os.replace(temporary, destination)
    files = [None] * runtime.world
    dist.all_gather_object(files, {'rank': runtime.rank, 'name': destination.name,
                                  'bytes': destination.stat().st_size, 'sha256': file_hash(destination)})
    if runtime.rank == 0:
        complete = {'format': 'pier-centered-v1', 'iteration': iteration,
                    'world': runtime.world, 'files': files}
        atomic_json(directory / 'complete.json', complete)
        atomic_json(root / 'latest_centered.json', {'iteration': iteration, 'directory': directory.name})
    dist.barrier()


def load(runtime, load_dir, scheduler):
    root = Path(load_dir)
    if (root / 'complete.json').is_file():
        directory = root
    else:
        latest = json.loads((root / 'latest_centered.json').read_text())
        directory = root / latest['directory']
        if directory.parent != root or not directory.name.startswith('iter_'):
            raise ValueError('invalid centered checkpoint directory')
    complete = json.loads((directory / 'complete.json').read_text())
    if complete['format'] != 'pier-centered-v1' or complete['world'] != runtime.world:
        raise ValueError('checkpoint format or world size differs')
    expected_ranks = list(range(runtime.world))
    if sorted(record['rank'] for record in complete['files']) != expected_ranks:
        raise ValueError('incomplete per-rank checkpoint')
    record = next(row for row in complete['files'] if row['rank'] == runtime.rank)
    if record['name'] != f'rank-{runtime.rank}.pt':
        raise ValueError('checkpoint rank filename mismatch')
    path = directory / record['name']
    if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
        raise ValueError('checkpoint rank file size/hash mismatch')
    # This is the private format written above, not a third-party model importer.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['format'] != 'pier-centered-v1' or state['world'] != runtime.world
            or state['rank'] != runtime.rank or state['iteration'] != complete['iteration']
            or state['outer_ranks'] != runtime.peers or state['inner_ranks'] != runtime.inner_ranks
            or state['coordinate_fingerprint'] != runtime.coordinates.fingerprint
            or state['coordinate_schema'] != runtime.coordinates.schema):
        raise ValueError('checkpoint ownership or parameter coordinates differ')
    expected_recipe = recipe(runtime.args)
    differences = [key for key in expected_recipe if expected_recipe[key] != state['recipe'].get(key)]
    if differences:
        raise ValueError(f'checkpoint recipe differs: {differences}')
    for key in ('reference', 'momentum'):
        expected = getattr(runtime.executor, key)
        if state[key].dtype != expected.dtype or state[key].shape != expected.shape:
            raise ValueError(f'checkpoint {key} dtype/shape differs')
    runtime.clock.load_state_dict(state['clock'])
    if runtime.clock.attempted != state['iteration']:
        raise ValueError('checkpoint iteration and clock differ')
    for module, model_state in zip(runtime.model, state['model'], strict=True):
        module.load_state_dict(model_state, strict=True)
    runtime.optimizer.load_state_dict(state['optimizer'])
    runtime.optimizer.commit_outer_update()
    scheduler.load_state_dict(state['scheduler'])
    with torch.no_grad():
        runtime.executor.reference.copy_(state['reference'])
        runtime.executor.momentum.copy_(state['momentum'])
    if runtime.verify:
        runtime.coordinates.assert_model_committed()
        if digest(runtime.optimizer.state_dict()) != state['optimizer_digest']:
            raise AssertionError('restored inner optimizer or FP32 master differs')
        if digest([module.state_dict() for module in runtime.model]) != state['model_digest']:
            raise AssertionError('restored model differs')
    for name in ('consumed_train_samples', 'consumed_valid_samples', 'skipped_train_samples'):
        setattr(runtime.args, name, state[name])
    runtime.oracle_reference = state['oracle_reference']
    runtime.oracle_momentum = state['oracle_momentum']
    runtime.events = state['events']
    runtime.consumer_checks = state['consumer_checks']
    runtime.pending_consumer = True
    runtime.pending_rng = state['rng']
    runtime.restored = True
    runtime.restore_evidence = {'iteration': state['iteration'], 'rank_file_sha256': record['sha256'],
                                'rank_file_bytes': record['bytes'],
                                'consumed_train_samples': state['consumed_train_samples'],
                                'clock': state['clock']}
    dist.barrier()
    return state['iteration'], state['flops']

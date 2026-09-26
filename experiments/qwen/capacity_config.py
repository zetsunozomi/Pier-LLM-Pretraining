"""Fixed A100-40GB capacity recipe and resumable integer-depth search (stdlib only)."""

import hashlib
import json
from pathlib import Path

from experiments.qwen.n2_config import ROOT, training_args as n2_training_args

ARMS = {'R': ('R', 1), 'G': ('G', 1), 'OS': ('OS', 1),
        'P2': ('P', 2), 'P16': ('P', 16), 'O': ('O', 1), 'W': ('W', 1)}
DEFAULT_ARMS = ('R', 'G', 'OS', 'P2', 'P16')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')


def architecture(layers):
    if type(layers) is not int or layers < 1:
        raise ValueError('layers must be a positive integer')
    cfg = json.loads((ROOT / 'experiments/qwen/qwen2.5-3B-config.json').read_text())
    cfg['num_hidden_layers'] = cfg['max_window_layers'] = layers
    return cfg


def parameters(layers):
    """Logical parameters, counting tied embeddings and TP-replicated norms once."""
    c = architecture(layers)
    h, f, v = c['hidden_size'], c['intermediate_size'], c['vocab_size']
    kv = c['num_key_value_heads'] * (h // c['num_attention_heads'])
    per_layer = 2*h*h + 2*h*kv + 3*h*f + 2*h + h + 2*kv
    return v*h*(1 if c['tie_word_embeddings'] else 2) + h + layers*per_layer


def recipe(snapshot):
    return dict(repository=str(ROOT), snapshot=str(Path(snapshot).resolve()),
                nodes=8, world_size=32, tp=2, learners=16,
                sequence=2048, microbatch=1, accumulation=8, global_batch=128,
                interval=50, workspace_mib=64, seed=1234, log_interval=1,
                warmup_cycles=0, data_prefix=None, model_size='3B',
                initialization='random', data_kind='synthetic_tokens',
                architecture_family='Qwen2.5-3B depth variants', base_architecture=architecture(36),
                expected_gpu='NVIDIA A100-SXM4-40GB',
                probe_steps=51, confirm_steps=151)


def training_args(config, arm, layers, steps, directory):
    from experiments.qwen.n2_config import ARMS as BACKENDS
    label, cohort = ARMS[arm]
    case = dict(arm=label, backend=BACKENDS[label], cohort=cohort)
    cfg = dict(config, attempts=steps)
    argv = n2_training_args(cfg, case, directory)
    for option in ('--qwen-model-size', '--qwen-snapshot'):
        index = argv.index(option)
        del argv[index:index+2]
    argv += ['--capacity-layers', str(layers)]
    return argv


def tokenizer_identity(snapshot):
    """Check only the four small tokenizer files; no pretrained weights needed."""
    pins = json.loads((ROOT / 'experiments/qwen/pins.json').read_text())['models']['3B']['files']
    result = {}
    for name in TOKENIZER_FILES:
        content = (Path(snapshot) / name).read_bytes()
        pin = pins[name]
        sha = hashlib.sha256(content).hexdigest()
        expected = pin.get('sha256') or pin.get('lfs_sha256')
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if len(content) != pin['bytes'] or (sha != expected if expected else blob != pin['git_blob_id']):
            raise ValueError(f'tokenizer file differs from the 3B pin: {name}')
        result[name] = sha
    return result


def next_trial(trials, *, start=36, ceiling=256):
    """Exponential bracketing, binary search, then a fresh 151-step confirmation.

    GPU OOMs bound the search. Timeouts/errors never tighten a bound. A later
    confirmation OOM supersedes a shorter successful probe at the same depth.
    Returns (layers, phase), or (None, terminal status).
    """
    if not 1 <= start <= ceiling:
        raise ValueError('require 1 <= start <= ceiling')
    oom = [t['layers'] for t in trials if t['status'] == 'oom']
    high = min(oom, default=ceiling + 1)
    passed = [t for t in trials if t['status'] == 'passed' and t['layers'] < high]
    low = max((t['layers'] for t in passed), default=0)
    if not oom and not passed:
        return start, 'probe'
    if high == 1:
        return None, 'no_feasible_model'
    if low == ceiling or (low and high == low + 1):
        if any(t['layers'] == low and t['phase'] == 'confirm' for t in passed):
            return None, 'complete' if oom else 'search_ceiling_reached'
        return low, 'confirm'
    if not low:
        return max(1, high // 2), 'probe'
    if not oom:
        return min(ceiling, low * 2), 'probe'
    return (low + high) // 2, 'probe'

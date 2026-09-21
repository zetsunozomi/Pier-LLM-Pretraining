#!/usr/bin/env python3
"""Preflight and evidence validation for E0c; never synthesize GPU results."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.centered_outer.manifest import command, source_hashes as megatron_sources
from experiments.qwen.e0c import configure_gpu, write_json
from experiments.qwen.e0c_metrics import CONTRACT, accepts
from experiments.qwen.preflight import pinned_architecture, preflight
from megatron.core.models.qwen.weights import parameter_mappings, sha256_file

PROBES = (
    'A distributed learner stores model weights, computes gradients, and applies an update. '
    'The reference and the native implementation receive exactly the same sequence of tokens. '
    'We compare every output class and every parameter gradient, including shared embeddings. ',
    '分布式训练需要明确参数、梯度和数据的对应关系。这里使用固定的短文本检查模型转换。'
    '每个进程读取相同的输入，使用相同的位置编号，并保存完整比较结果。数值检查不是性能测量。',
)
NOT_VALIDATED = ['real-data training recipe', 'Qwen optimizer/outer-state trajectory and restart',
                 'strong baseline performance', 'GPU peak memory or physical wire traffic',
                 'other model sizes, PP/CP/MoE, distributed optimizer or multi-slot pipeline']


def source_hashes():
    result = megatron_sources()
    paths = [ROOT / 'pretrain_qwen.py', ROOT / 'requirements.txt', ROOT / '.gitignore']
    paths += [p for p in (ROOT / 'experiments/qwen').iterdir()
              if p.is_file() and p.suffix in ('.py', '.json', '.sh', '.sbatch')]
    result.update({str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)})
    return result


def make_manifest(directory, snapshot, *, collect_numerical_mismatches=False):
    import torch
    import transformers
    from transformers import AutoTokenizer
    environment = configure_gpu()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('refusing to reuse an E0c evidence directory')
    directory.mkdir(parents=True, exist_ok=True)
    report = preflight('3B', 2, snapshot)
    # Reference gradients are FP32 + BF16, plus outputs and filesystem overhead.
    required = report['unique_parameters'] * 6 + (2 << 30)
    if shutil.disk_usage(directory).free < required:
        raise RuntimeError(f'E0c needs at least {required} free scratch bytes for full reference gradients')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokens = [tokenizer.encode(text * 8, add_special_tokens=False)[:64] for text in PROBES]
    if any(len(row) != 64 for row in tokens):
        raise ValueError('tokenizer probe is shorter than the fixed 64-token fixture')
    if any(not 0 <= t < report['architecture']['vocab'] for row in tokens for t in row):
        raise ValueError('probe token exceeds the exact checkpoint vocabulary')
    inputs = {'purpose': 'authored conversion probes; not a real-data training corpus',
              'texts': list(PROBES), 'tokens': tokens, 'positions': [list(range(64))] * 2,
              'add_special_tokens': False, 'batch': 2, 'sequence_length': 64}
    write_json(directory / 'inputs.json', inputs)
    manifest = {'stage': 'E0c', 'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'purpose': 'real Qwen2.5-3B numerical conversion check; no training/performance result',
                'snapshot_path': str(snapshot.resolve()), 'snapshot': report, 'contract': CONTRACT,
                'world_size': 4, 'layouts': [1, 2], 'dtypes': ['fp32', 'bf16'],
                'collect_numerical_mismatches': collect_numerical_mismatches,
                'inputs_sha256': sha256_file(directory / 'inputs.json'),
                'source_sha256': source_hashes(), 'git_head': command(['git', 'rev-parse', 'HEAD']),
                'git_status': command(['git', 'status', '--porcelain']), 'python': sys.executable,
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'environment': environment,
                'reference_gradient_storage_bytes': report['unique_parameters'] * 6,
                'not_validated': NOT_VALIDATED, 'performance_result': False}
    write_json(directory / 'manifest.json', manifest)


def validate_native(record, manifest, dtype, tp, rank, manifest_sha, input_sha, reference_sha):
    _, arch = pinned_architecture('3B')
    errors = []
    expected = {'phase': 'native', 'status': 'passed', 'dtype': dtype, 'tp': tp, 'rank': rank,
                'world_size': 4, 'tp_rank': rank % tp, 'performance_result': False,
                'optimizer_steps': 0, 'manifest_sha256': manifest_sha,
                'inputs_sha256': input_sha, 'reference_report_sha256': reference_sha}
    for key, value in expected.items():
        if record.get(key) != value:
            errors.append(f'{key} differs')
    if record.get('collect_numerical_mismatches', False) != manifest.get('collect_numerical_mismatches', False):
        errors.append('numerical collection mode differs')
    if not record.get('environment', {}).get('GPU_executed'):
        errors.append('missing actual GPU execution')
    mapping = parameter_mappings(arch, tp, rank % tp)
    shape_counts = {p.target: math.prod(p.shape) for p in mapping}
    if record.get('gradient_elements') != sum(shape_counts.values()):
        errors.append('incomplete gradient element coverage')
    for kind in ('weights', 'gradients'):
        values = record.get(kind, {})
        if set(values) != set(shape_counts):
            errors.append(f'{kind} parameter coverage differs')
            continue
        for name, count in shape_counts.items():
            value = values[name]
            if value.get('elements') != count:
                errors.append(f'{kind} shape/count differs: {name}')
            if kind == 'weights':
                valid = value.get('bitwise_equal') is True
            else:
                valid = accepts(value, dtype, 'gradient')
            if not valid:
                errors.append(f'{kind} mismatch: {name}')
    for kind, count in (('logits', 2 * 64 * arch.vocab), ('loss', 1)):
        value = record.get(kind, {})
        if value.get('elements') != count or not accepts(value, dtype, kind):
            errors.append(f'{kind} mismatch/incomplete coverage')
    return errors


def summarize(directory, launcher_exit=0, *, check_current_source=True):
    errors, comparisons = [], []
    collect_mismatches = False
    manifest_path, inputs_path = directory / 'manifest.json', directory / 'inputs.json'
    try:
        manifest = json.loads(manifest_path.read_text())
        collect_mismatches = manifest.get('collect_numerical_mismatches', False)
        manifest_sha, input_sha = sha256_file(manifest_path), sha256_file(inputs_path)
        pin, arch = pinned_architecture('3B')
        if (manifest['stage'] != 'E0c' or manifest['contract'] != CONTRACT
                or manifest['inputs_sha256'] != input_sha
                or manifest['snapshot']['revision'] != pin['revision']
                or manifest['snapshot']['architecture'] != asdict(arch)
                or manifest['snapshot']['config_sha256'] != pin['config_sha256']
                or not manifest['snapshot']['checkpoint_verified']
                or not manifest['snapshot']['tokenizer_files_verified']
                or manifest['snapshot']['unique_parameters'] != arch.unique_parameters()):
            errors.append('manifest contract/snapshot/input identity differs')
        files = manifest['snapshot']['files']
        if (set(files) != set(pin['files'])
                or any(files[name]['bytes'] != value['bytes']
                       or (value.get('lfs_sha256') and files[name]['sha256'] != value['lfs_sha256'])
                       for name, value in pin['files'].items())):
            errors.append('snapshot file identities differ from pinned checkpoint')
        if check_current_source and manifest['source_sha256'] != source_hashes():
            errors.append('current source differs from the run manifest; inspect at the recorded revision')
        for dtype in ('fp32', 'bf16'):
            ref_path = directory / f'hf-{dtype}.json'
            reference = json.loads(ref_path.read_text())
            expected_shapes = {name: list(shape) for name, shape in arch.hf_shapes().items()}
            expected_counts = {name: math.prod(shape) for name, shape in expected_shapes.items()}
            if (reference['status'] != 'reference_written' or reference['dtype'] != dtype
                    or reference['manifest_sha256'] != manifest_sha or reference['inputs_sha256'] != input_sha
                    or not reference['environment']['GPU_executed']
                    or reference['performance_result'] is not False or reference['optimizer_steps'] != 0
                    or not math.isfinite(reference['loss'])
                    or any(reference['loading_info'].get(k) for k in
                           ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
                    or reference['gradient_elements'] != arch.unique_parameters()
                    or {n: x['shape'] for n, x in reference['gradients'].items()} != expected_shapes
                    or {n: x['elements'] for n, x in reference['gradients'].items()} != expected_counts
                    or {n: x['elements'] for n, x in reference['weights'].items()} != expected_counts
                    or not all(x['bitwise_equal'] for x in reference['weights'].values())):
                errors.append(f'HF {dtype} reference contract/coverage failed')
            for tp in (1, 2):
                for rank in range(4):
                    path = directory / f'native-{dtype}-tp{tp}-rank{rank}.json'
                    try:
                        record = json.loads(path.read_text())
                        problems = validate_native(record, manifest, dtype, tp, rank,
                                                   manifest_sha, input_sha, sha256_file(ref_path))
                        errors += [f'{path.name}: {problem}' for problem in problems]
                        comparisons.append({'dtype': dtype, 'tp': tp, 'rank': rank, 'passed': not problems})
                    except (OSError, KeyError, ValueError, TypeError) as exc:
                        errors.append(f'{path.name}: {exc}')
    except (OSError, KeyError, ValueError, TypeError) as exc:
        errors.append(f'missing/invalid evidence: {exc}')
    if launcher_exit:
        errors.append(f'launcher exited {launcher_exit}')
    failures = sorted(p.name for p in directory.glob('failure-*.json'))
    if failures:
        errors.append(f'worker failures: {failures}')
    # JSON evidence is portable; large reference arrays intentionally stay on scratch.
    result = {'status': 'passed' if not errors and len(comparisons) == 16 else 'failed',
              'stage': 'E0c', 'errors': errors, 'comparisons': comparisons,
              'collect_numerical_mismatches': collect_mismatches,
              'native_reports_collected': len(comparisons),
              'performance_result': False, 'training_validated': False,
              'not_validated': NOT_VALIDATED,
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    result['GPU_conversion_validated'] = result['status'] == 'passed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--launcher-exit', type=int, default=0)
    parser.add_argument('--collect-numerical-mismatches', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        report = summarize(args.output_dir, args.launcher_exit)
        write_json(args.output_dir / 'summary.json', report)
        print(json.dumps(report, indent=2))
        raise SystemExit(report['status'] != 'passed')
    if args.snapshot is None:
        parser.error('--snapshot is required for preflight')
    make_manifest(args.output_dir, args.snapshot,
                  collect_numerical_mismatches=args.collect_numerical_mismatches)


if __name__ == '__main__':
    main()

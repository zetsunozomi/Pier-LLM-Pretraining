#!/usr/bin/env python3
"""Preflight and portable evidence review for full-state Qwen training E0d."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qwen.e0c import configure_gpu, write_json
from experiments.qwen.e0c_evidence import source_hashes, summarize as summarize_e0c
from experiments.qwen.e0d_config import ATTEMPTS, CASES, GLOBAL_BATCH, NOT_VALIDATED, RECOMPUTE, SEQUENCE, training_args
from experiments.qwen.preflight import pinned_architecture, preflight
from megatron.core.models.qwen.weights import parameter_mappings, sha256_file
from pretrain_qwen import data_identity, snapshot_identity


def read(path):
    return json.loads(Path(path).read_text())


def file_record(path):
    path = Path(path)
    return {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def prerequisite(directory, *, check_current_source=True):
    saved = read(directory / 'summary.json')
    checked = summarize_e0c(directory, check_current_source=check_current_source)
    if (saved.get('status') != 'passed' or saved.get('GPU_conversion_validated') is not True
            or saved.get('errors') != [] or checked['status'] != 'passed'):
        raise ValueError(f'E0d requires accepted E0c evidence: {checked["errors"]}')
    names = ['manifest.json', 'inputs.json', 'summary.json', 'hf-fp32.json', 'hf-bf16.json']
    names += [f'native-{dtype}-tp{tp}-rank{rank}.json'
              for dtype in ('fp32', 'bf16') for tp in (1, 2) for rank in range(4)]
    return {name: file_record(directory / name) for name in names}


def scratch_estimate(local_elements):
    # Five retained pairs of full local oracle vectors on each of four ranks.
    elements = sum(local_elements[rank % 2] for rank in range(4))
    return {'oracle_files_bytes': len(CASES) * 8 * elements,
            'checkpoint_conservative_bytes': 24 * elements,
            'headroom_bytes': 32 << 30,
            'required_free_bytes': (len(CASES) * 8 + 24) * elements + (32 << 30),
            'scope': 'new E0d files only; excludes existing weights/corpus/E0c references and filesystem quota',
            'measured_peak_memory': False}


def make_manifest(directory, snapshot, data_prefix, e0c_directory):
    from transformers import AutoTokenizer
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    environment = configure_gpu('E0d')
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('E0d requires a fresh evidence directory')
    prerequisite_files = prerequisite(e0c_directory)
    report = preflight('3B', 2, snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    args = SimpleNamespace(mock_data=False, data_path=[str(data_prefix)], data_args_path=None,
                           per_split_data_args_path=None, seq_length=SEQUENCE)
    data = data_identity(args, report, tokenizer=tokenizer)
    dataset = IndexedDataset(str(data_prefix))
    if (len(dataset) != data['documents'] or int(dataset.sequence_lengths.sum()) != data['tokens']
            or data['tokens'] < ATTEMPTS * GLOBAL_BATCH * SEQUENCE + 1):
        raise ValueError('indexed data count differs or cannot cover this fixed training window without epoch reuse')
    del dataset
    estimate = scratch_estimate(report['local_parameter_elements'])
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < estimate['required_free_bytes']:
        raise RuntimeError(f'E0d needs {estimate["required_free_bytes"]} free scratch bytes for full oracle/checkpoint evidence')
    archived = directory / 'prerequisite-e0c'
    archived.mkdir()
    for name in prerequisite_files:
        shutil.copyfile(e0c_directory / name, archived / name)
        if file_record(archived / name) != prerequisite_files[name]:
            raise ValueError('E0c evidence changed while archiving its text records')
    manifest = {'stage': 'E0d', 'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'purpose': 'real Qwen optimizer/outer-state/restore correctness, not performance',
                'snapshot_path': str(snapshot.resolve()), 'snapshot': report,
                'data_prefix': str(data_prefix.resolve()), 'data': data,
                'run_output': str(directory.resolve()), 'source_sha256': source_hashes(),
                'e0c_files': prerequisite_files, 'e0c_source_directory': str(e0c_directory.resolve()),
                'world_size': 4, 'tp_size': 2, 'inner_dp': 1, 'outer_learners': 2,
                'cases': CASES, 'attempts': ATTEMPTS, 'sequence_length': SEQUENCE,
                'global_batch': GLOBAL_BATCH, 'micro_batch': 1, 'microbatches_per_learner': 8,
                'recompute': RECOMPUTE, 'scratch_estimate': estimate, 'environment': environment,
                'qwen_recipe': {'snapshot_sha256': snapshot_identity(report), 'architecture': report['architecture'],
                                'data': data, 'eod': tokenizer.eos_token_id, 'initialization': 'weights-only warm start'},
                'not_validated': NOT_VALIDATED, 'performance_result': False}
    write_json(directory / 'manifest.json', manifest)
    return manifest


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_case(case, rank, record, launch, initialization, manifest, manifest_sha):
    errors, config = [], CASES[case]
    end = config.get('stop', ATTEMPTS)
    successful, boundaries = end - 1, (end - 1) // 3
    peers = list(range(rank % 2, 4, 2))
    n = manifest['snapshot']['local_parameter_elements'][rank % 2]
    expected = {'status': 'partial' if case == 'split' else 'passed', 'rank': rank, 'world_size': 4,
                'GPU_executed': True, 'backend': 'nccl', 'arm': 'pier', 'cohort': config['cohort'],
                'state_tier': 'host' if config['host'] else 'device', 'oracle_storage': 'streamed',
                'coordinate_numel': n, 'outer_ranks': peers, 'inner_ranks': [rank],
                'workspace_budget_mib': 64., 'restored': bool(config.get('resume')),
                'pending_consumer': False, 'consumed_train_samples': end * GLOBAL_BATCH,
                'consumed_valid_samples': 0, 'skipped_train_samples': 0,
                'clock': {'interval': 3, 'attempted': end,
                                                     'successful': successful, 'boundaries': boundaries},
                'performance_result': False, 'error': None}
    for key, value in expected.items():
        if record.get(key) != value:
            errors.append(f'{key} differs')
    if record.get('consumer_checks', 0) < boundaries or not valid_digest(record.get('coordinate_fingerprint')):
        errors.append('missing coordinate/next-forward evidence')
    allocation = record.get('oracle_allocation') or {}
    oracle_expected = {'kind': 'streamed-full-coordinate-fixed-tree', 'coordinates_per_vector': n,
                       'last_master_coordinates_checked': n, 'boundaries_checked': boundaries,
                       'file_logical_bytes': 8 * n, 'tile_elements': 65536,
                       'persistent_device_scratch_bytes': 3 * 65536 * 4, 'performance_result': False}
    if any(allocation.get(key) != value for key, value in oracle_expected.items()):
        errors.append('incomplete full-coordinate oracle coverage')
    argv = training_args(case, Path(manifest['run_output']), manifest['snapshot_path'], manifest['data_prefix'])
    if (launch.get('case') != case or launch.get('rank') != rank or launch.get('entrypoint') != 'pretrain_qwen.py'
            or launch.get('argv') != argv or launch.get('manifest_sha256') != manifest_sha
            or launch.get('GPU_executed') is not True
            or launch.get('environment', {}).get('GPU_executed') is not True):
        errors.append('missing manifest-bound real training launch')
    _, arch = pinned_architecture('3B')
    init_expected = {'tp_size': 2, 'tp_rank': rank % 2, 'local_parameter_elements': n,
                     'loaded_parameters': len(parameter_mappings(arch, 2, rank % 2)),
                     'unique_model_elements': arch.unique_parameters(), 'optimizer_state_restored': False,
                     'loaded_before_optimizer_construction': True, 'exact_output_vocabulary': arch.vocab,
                     'architecture': manifest['snapshot']['architecture'], 'activation_recompute': RECOMPUTE,
                     'qwen_recipe': manifest['qwen_recipe'], 'initialization': 'weights-only warm start'}
    if any(initialization.get(key) != value for key, value in init_expected.items()):
        errors.append('initial Qwen model/data/recompute identity or coverage differs')
    if config.get('resume'):
        restored = record.get('restore_evidence') or {}
        if (restored.get('iteration') != 5 or restored.get('consumed_train_samples') != 5 * GLOBAL_BATCH
                or restored.get('clock') != {'interval': 3, 'attempted': 5, 'successful': 4, 'boundaries': 1}
                or not valid_digest(restored.get('rank_file_sha256')) or restored.get('rank_file_bytes', 0) <= 0):
            errors.append('missing verified full checkpoint restoration')
    events = record.get('events', [])
    if len(events) != end:
        errors.append('missing attempted-step trajectory')
        return errors
    for attempted, event in enumerate(events, 1):
        skip = attempted == 3
        success = attempted - int(attempted >= 3)
        boundary = not skip and success % 3 == 0
        flags = {'attempted': attempted, 'successful': success, 'boundaries': success // 3,
                 'skipped': skip, 'outer_boundary': boundary,
                 'oracle_states_bitwise': True if boundary else None,
                 'outer_retained_inner_state': True if boundary else None,
                 'skip_retained_optimizer': True if skip else None}
        if any(event.get(key) != value for key, value in flags.items()):
            errors.append(f'attempt {attempted}: scheduling/state checks differ')
        for key in ('master_sha256', 'model_sha256', 'inner_sha256'):
            if not valid_digest(event.get(key)):
                errors.append(f'attempt {attempted}: missing full {key}')
            if skip and event.get(key) != events[attempted - 2].get(key):
                errors.append(f'attempt {attempted}: skipped state changed')
        loss = event.get('loss', {}).get('lm loss')
        if type(loss) not in (int, float) or not math.isfinite(loss):
            errors.append(f'attempt {attempted}: missing/invalid actual training loss')
    if len({event.get('master_sha256') for event in events}) < 2:
        errors.append('no parameter trajectory change across successful updates')
    return errors


def comparable(events):
    return [{key: value for key, value in event.items() if key != 'payload'} for event in events]


def summarize(directory, launcher_exit=0, *, check_current_source=True):
    errors, comparisons, reports = [], [], {}
    try:
        manifest = read(directory / 'manifest.json')
        manifest_sha = sha256_file(directory / 'manifest.json')
        pin, arch = pinned_architecture('3B')
        if (manifest['stage'] != 'E0d' or manifest['world_size'] != 4 or manifest['tp_size'] != 2
                or manifest['inner_dp'] != 1 or manifest['outer_learners'] != 2 or manifest['cases'] != CASES
                or manifest['attempts'] != ATTEMPTS or manifest['sequence_length'] != SEQUENCE
                or manifest['global_batch'] != GLOBAL_BATCH or manifest['micro_batch'] != 1
                or manifest['microbatches_per_learner'] != 8 or manifest['recompute'] != RECOMPUTE
                or manifest['performance_result'] is not False or not manifest['environment']['GPU_executed']):
            errors.append('E0d workload manifest differs')
        node = read(directory / 'node-0.json')
        if node.get('cuda_available') is not True or node.get('visible_devices') != 4:
            errors.append('missing four-GPU node preflight')
        if check_current_source and manifest['source_sha256'] != source_hashes():
            errors.append('current sources differ from E0d manifest; review at its recorded source')
        archived = directory / 'prerequisite-e0c'
        if prerequisite(archived, check_current_source=False) != manifest['e0c_files']:
            errors.append('archived E0c evidence bytes changed')
        e0c_manifest = read(archived / 'manifest.json')
        # Snapshot preflight reports also contain layout fields; the content
        # identity deliberately excludes those and is checked separately below.
        if (e0c_manifest['source_sha256'] != manifest['source_sha256']
                or snapshot_identity(e0c_manifest['snapshot']) != snapshot_identity(manifest['snapshot'])):
            errors.append('E0c conversion and E0d source/model identity differ')
        expected_counts = [sum(math.prod(item.shape) for item in parameter_mappings(arch, 2, rank)) for rank in range(2)]
        if (manifest['snapshot']['local_parameter_elements'] != expected_counts
                or manifest['snapshot']['tp_size'] != 2
                or manifest['snapshot']['architecture'] != asdict(arch)):
            errors.append('TP2 parameter coverage differs')
        data = manifest['data']
        eos = read(ROOT / 'experiments/qwen' / pin['config_filename'])['eos_token_id']
        expected_recipe = {'snapshot_sha256': snapshot_identity(manifest['snapshot']),
                           'architecture': asdict(arch), 'data': data, 'eod': eos,
                           'initialization': 'weights-only warm start'}
        if (manifest['qwen_recipe'] != expected_recipe
                or not valid_digest(data['manifest_sha256']) or data['documents'] < 1
                or data['tokens'] < ATTEMPTS * GLOBAL_BATCH * SEQUENCE + 1
                or data['tokenization']['append_eod'] != eos
                or set(data['files']) != {'.bin', '.idx'}
                or data['files']['.bin']['bytes'] != data['tokens'] * 4
                or any(not valid_digest(item['sha256']) or item['bytes'] <= 0
                       for item in [*data['files'].values(), data['input']])):
            errors.append('indexed data identity or Qwen initialization recipe differs')
        if manifest['scratch_estimate'] != scratch_estimate(expected_counts):
            errors.append('full-state scratch estimate differs')
        checkpoint = read(directory / 'checkpoint-manifest.json')
        checkpoint_files = checkpoint['files']
        if (checkpoint['format'] != 'pier-centered-v1' or checkpoint['iteration'] != 5
                or checkpoint['world'] != 4 or len(checkpoint_files) != 4
                or {item['rank'] for item in checkpoint_files} != set(range(4))
                or any(item['name'] != f'rank-{item["rank"]}.pt' or item['bytes'] <= 0
                       or not valid_digest(item['sha256']) for item in checkpoint_files)):
            errors.append('complete per-rank checkpoint publication receipt differs')
        saved_ranks = {item['rank']: item for item in checkpoint_files}
        for case in CASES:
            folder = directory / f'case-{case}'
            reports[case] = []
            if list(folder.glob('failure-rank-*.json')):
                errors.append(f'{case}: worker failure present')
            for rank in range(4):
                try:
                    report = read(folder / f'rank-{rank}.json')
                    launch = read(folder / f'launch-rank-{rank}.json')
                    initialization = read(folder / f'initialization-rank-{rank}.json')
                    errors += [f'{case} rank {rank}: {message}' for message in
                               validate_case(case, rank, report, launch, initialization, manifest, manifest_sha)]
                    if case == 'resume':
                        restored = report.get('restore_evidence') or {}
                        if (restored.get('rank_file_sha256') != saved_ranks[rank]['sha256']
                                or restored.get('rank_file_bytes') != saved_ranks[rank]['bytes']):
                            errors.append(f'resume rank {rank}: restored bytes differ from split checkpoint')
                    reports[case].append(report)
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    errors.append(f'{case} rank {rank}: {exc}')
            if len(reports[case]) == 4:
                for a, b in ((0, 2), (1, 3)):
                    for left, right in zip(reports[case][a]['events'], reports[case][b]['events']):
                        if left['outer_boundary'] and any(left[key] != right[key] for key in ('master_sha256', 'model_sha256')):
                            errors.append(f'{case}: learner parameters differ after outer commit')
        if len(reports.get('s2-device', [])) == 4:
            for case in ('s1-host', 's2-host', 'split', 'resume'):
                if len(reports.get(case, [])) != 4:
                    continue
                for rank in range(4):
                    actual = comparable(reports[case][rank]['events'])
                    baseline = comparable(reports['s2-device'][rank]['events'])[:len(actual)]
                    equal = (actual == baseline and reports[case][rank]['coordinate_fingerprint']
                             == reports['s2-device'][rank]['coordinate_fingerprint'])
                    comparisons.append({'case': case, 'rank': rank, 'bitwise_trajectory_equal': equal,
                                        'prefix_only': case == 'split'})
                    if not equal:
                        errors.append(f'{case} rank {rank}: training trajectory differs from s2-device')
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        errors.append(f'missing/invalid E0d evidence: {exc}')
    if launcher_exit:
        errors.append(f'launcher exited {launcher_exit}')
    passed = not errors and len(comparisons) == 16
    return {'status': 'passed' if passed else 'failed', 'stage': 'E0d', 'GPU_executed': passed,
            'entrypoint': 'pretrain_qwen.py', 'world_size': 4, 'cases': list(CASES),
            'errors': errors, 'comparisons': comparisons, 'performance_result': False,
            'not_validated': NOT_VALIDATED, 'finished_utc': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--data-prefix', type=Path)
    parser.add_argument('--e0c-dir', type=Path)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--launcher-exit', type=int, default=0)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.output_dir, args.launcher_exit)
        write_json(args.output_dir / 'summary.json', result)
        print(json.dumps(result, indent=2))
        return int(result['status'] != 'passed')
    if any(value is None for value in (args.snapshot, args.data_prefix, args.e0c_dir)):
        parser.error('preflight requires snapshot, data-prefix and e0c-dir')
    result = make_manifest(args.output_dir, args.snapshot, args.data_prefix, args.e0c_dir)
    print(json.dumps({'stage': 'E0d', 'status': 'preflight_passed', 'scratch_estimate': result['scratch_estimate'],
                      'performance_result': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

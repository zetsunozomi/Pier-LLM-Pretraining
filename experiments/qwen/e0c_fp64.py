#!/usr/bin/env python3
"""Whole-model FP64 diagnostic for E0c; never overrides conversion acceptance."""

import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from experiments.qwen.e0c import configure_gpu, exact_tensor, file_record, write_json
from experiments.qwen.e0c_evidence import CONTRACT, source_hashes
from experiments.qwen.fp64_math import double_softmax, loss64, promote
from experiments.qwen.preflight import preflight
from megatron.core.models.qwen.config import QwenArchitecture
from megatron.core.models.qwen.weights import SafeTensorSource, load_weights, parameter_mappings, sha256_file

# Separate, preregistered diagnostic comparison; NOT an alternative E0c gate.
FP64_LIMITS = {'atol': 1e-9, 'rtol': 1e-8}
FLAGS = {'diagnostic_only': True, 'acceptance_override': False,
         'GPU_conversion_validated': False, 'training_validated': False, 'performance_result': False}


def read_json(path):
    return json.loads(Path(path).read_text())


def compare64(actual, reference):
    if actual.shape != reference.shape or actual.dtype != torch.float64 or reference.dtype != torch.float64:
        raise ValueError('FP64 comparison requires matching FP64 tensors')
    a, b = actual.detach().reshape(-1), reference.detach().reshape(-1)
    result = {'elements': a.numel(), 'shape': list(actual.shape), 'nonfinite': 0,
              'outside_tolerance': 0, 'max_abs': 0., 'error_sq': 0., 'reference_sq': 0.}
    for start in range(0, a.numel(), 1 << 20):
        x, y = a[start:start + (1 << 20)].cpu(), b[start:start + (1 << 20)].cpu()
        finite = torch.isfinite(x) & torch.isfinite(y)
        result['nonfinite'] += int((~finite).sum())
        delta, y = (x - y)[finite], y[finite]
        if delta.numel():
            result['max_abs'] = max(result['max_abs'], float(delta.abs().max()))
            result['error_sq'] += float(delta.dot(delta))
            result['reference_sq'] += float(y.dot(y))
            result['outside_tolerance'] += int((delta.abs() > FP64_LIMITS['atol'] + FP64_LIMITS['rtol'] * y.abs()).sum())
    result['relative_l2'] = math.sqrt(result['error_sq'] / result['reference_sq']) if result['reference_sq'] else None
    result['agrees'] = bool(result['elements'] and not result['nonfinite'] and not result['outside_tolerance'])
    result['limits'] = FP64_LIMITS
    return result


def prepare(directory, previous):
    environment = configure_gpu('E0c FP64 diagnostic')
    old = read_json(previous / 'manifest.json')
    hf_path, native_path = previous / 'hf-fp32.json', previous / 'native-fp32-tp1-rank0.json'
    hf, native = read_json(hf_path), read_json(native_path)
    old_sha, inputs_sha = sha256_file(previous / 'manifest.json'), sha256_file(previous / 'inputs.json')
    if old.get('stage') != 'E0c' or old.get('contract') != CONTRACT or old['inputs_sha256'] != inputs_sha:
        raise ValueError('FP32 run identity/contract differs')
    for report in (hf, native):
        if (report.get('manifest_sha256') != old_sha or report.get('inputs_sha256') != inputs_sha
                or report.get('environment', {}).get('GPU_executed') is not True):
            raise ValueError('FP32 report lacks matching GPU/source/input identity')
    if (hf['status'] != 'reference_written' or hf['dtype'] != 'fp32' or native['dtype'] != 'fp32' or native['tp'] != 1
            or native['rank'] != 0 or native['reference_report_sha256'] != sha256_file(hf_path)):
        raise ValueError('requires the completed HF FP32 and native FP32/TP1 rank0 phases')
    snapshot = Path(old['snapshot_path'])
    checked = preflight('3B', 1, snapshot)
    for key in ('revision', 'architecture', 'files'):
        if checked[key] != old['snapshot'][key]:
            raise ValueError(f'snapshot changed since FP32 run: {key}')
    parameter = 'decoder.layers.2.mlp.linear_fc2.weight'
    comparison = native['gradients'][parameter]
    arch = QwenArchitecture(**checked['architecture'])
    if comparison['shape'] != [arch.hidden, arch.intermediate] or arch.layers < 3:
        raise ValueError('FP32 gradient shape or layer coverage differs')
    points = list(comparison['outside_tolerance_samples'])
    if comparison['worst_element'] is not None and all(p['index'] != comparison['worst_element']['index'] for p in points):
        points.append(comparison['worst_element'])
    if not points or not all(math.isfinite(p[key]) for p in points for key in ('actual', 'reference')):
        raise ValueError('missing finite FP32 diagnostic points')
    if any(len(p['index']) != 2 or not (0 <= p['index'][0] < arch.hidden
                                       and 0 <= p['index'][1] < arch.intermediate) for p in points):
        raise ValueError('FP32 diagnostic coordinate is outside the parameter')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('refusing to reuse FP64 output directory')
    if shutil.disk_usage(directory).free < 2 << 30:
        raise ValueError('FP64 diagnostic requires at least 2 GiB free scratch')
    shutil.copyfile(previous / 'inputs.json', directory / 'inputs.json')
    write_json(directory / 'manifest.json', dict(FLAGS, stage='E0c-FP64-diagnostic', layer=2,
        snapshot_path=str(snapshot), snapshot=checked, inputs_sha256=inputs_sha,
        fp32_run=str(previous.resolve()), fp32_manifest_sha256=old_sha,
        fp32_hf_report_sha256=sha256_file(hf_path), fp32_native_report_sha256=sha256_file(native_path),
        fp32_points=points, fp64_comparison_limits=FP64_LIMITS, source_sha256=source_hashes(),
        environment=environment, gradient_scope=parameter,
        not_validated=['all-parameter gradients', 'TP2', 'BF16', 'E0c acceptance', 'training or performance']))


def run_model(directory, backend):
    from transformers import Qwen2ForCausalLM
    from megatron.core import parallel_state as ps
    from megatron.core.models.qwen.model import build_model
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    environment = configure_gpu('E0c FP64 diagnostic')
    manifest = read_json(directory / 'manifest.json')
    if source_hashes() != manifest['source_sha256'] or sha256_file(directory / 'inputs.json') != manifest['inputs_sha256']:
        raise ValueError('diagnostic source or input bytes changed')
    arch = QwenArchitecture(**manifest['snapshot']['architecture'])
    inputs = read_json(directory / 'inputs.json')
    tokens = torch.tensor(inputs['tokens'], dtype=torch.long, device='cuda')
    positions = torch.tensor(inputs['positions'], dtype=torch.long, device='cuda')
    if not torch.equal(positions, torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)):
        raise ValueError('diagnostic requires the original contiguous E0c positions')
    source = SafeTensorSource(manifest['snapshot_path'])
    try:
        print(f'[E0c FP64] {backend}: loading weights; one gradient tensor enabled', flush=True)
        if backend == 'hf':
            model, loading = Qwen2ForCausalLM.from_pretrained(manifest['snapshot_path'],
                dtype=torch.float32, attn_implementation='eager', local_files_only=True,
                trust_remote_code=False, output_loading_info=True)
            if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError(f'HF load coverage differs: {loading}')
            model = model.cuda().train()
            model.config.use_cache = False
            if {n: tuple(p.shape) for n, p in model.named_parameters()} != arch.hf_shapes():
                raise ValueError('HF parameter coverage differs')
            if not all(exact_tensor(p, source.read(n))['bitwise_equal'] for n, p in model.named_parameters()):
                raise ValueError('HF weights differ from source')
        else:
            dist.init_process_group('nccl', timeout=timedelta(minutes=10))
            if dist.get_world_size() != 1:
                raise ValueError('FP64 diagnostic uses one native TP1 rank')
            ps.initialize_model_parallel(tensor_model_parallel_size=1, num_subgroups=1)
            model_parallel_cuda_manual_seed(421)
            model = build_model(arch, dtype=torch.float32, parallel_output=False).train()
            load_weights(model, source, arch, 1, 0)
            params = dict(model.named_parameters())
            if not all(exact_tensor(params[m.target], m.materialize(source))['bitwise_equal']
                       for m in parameter_mappings(arch)):
                raise ValueError('native weights differ from source')
        name, parameter, precision = promote(model, backend, arch, manifest['layer'])
        print(f'[E0c FP64] {backend}: whole-model forward/backward in FP64', flush=True)
        with double_softmax() as receipt:
            if backend == 'hf':
                logits = model(tokens, position_ids=positions, use_cache=False).logits
            else:
                mask = torch.ones(1, 1, tokens.shape[1], tokens.shape[1], dtype=torch.bool, device='cuda').triu(1)
                logits = model(tokens, positions, mask)
            loss = loss64(logits, tokens)
            loss.backward()
        precision.update(receipt)
        if receipt['fp64_softmax_calls'] != arch.layers:
            raise ValueError('FP64 softmax coverage differs')
        if [n for n, p in model.named_parameters() if p.grad is not None] != [name]:
            raise ValueError('FP64 gradient scope differs')
        tensors = {'logits': logits.detach().cpu().contiguous(), 'loss': loss.detach().cpu().reshape(1),
                   'gradient': parameter.grad.detach().cpu().contiguous()}
        if any(t.dtype != torch.float64 or not bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise ValueError('nonfinite or lower-precision FP64 output')
        raw = directory / 'reference-fp64'
        raw.mkdir(exist_ok=True)
        tensor_path = raw / f'{backend}.safetensors'
        save_file(tensors, str(tensor_path))
        report = dict(FLAGS, status='diagnostic_written', backend=backend, dtype='float64', tp=1,
            environment=environment, precision=precision, parameter=name,
            manifest_sha256=sha256_file(directory / 'manifest.json'),
            inputs_sha256=manifest['inputs_sha256'], tensors=file_record(tensor_path), loss=float(loss.detach()))
        if backend == 'native':
            reference_path = raw / 'hf.safetensors'
            reference_report = read_json(directory / 'hf-fp64.json')
            if (file_record(reference_path) != reference_report['tensors']
                    or reference_report['manifest_sha256'] != report['manifest_sha256']):
                raise ValueError('FP64 reference identity changed')
            reference = load_file(str(reference_path))
            report['comparisons'] = {key: compare64(tensors[key], reference[key]) for key in tensors}
            report['hf_report_sha256'] = sha256_file(directory / 'hf-fp64.json')
            report['fp64_implementations_agree'] = all(s['agrees'] for s in report['comparisons'].values())
            report['points'] = []
            for p in manifest['fp32_points']:
                index = tuple(p['index'])
                hf64, native64 = float(reference['gradient'][index]), float(tensors['gradient'][index])
                point = dict(index=list(index), hf_fp32=p['reference'], native_fp32=p['actual'],
                    hf_fp64=hf64, native_fp64=native64, fp64_native_minus_hf=native64-hf64,
                    hf_fp32_minus_hf_fp64=p['reference']-hf64,
                    native_fp32_minus_hf_fp64=p['actual']-hf64,
                    original_fp32_tolerance=p['tolerance'], original_fp32_ratio=p['ratio'])
                report['points'].append(point)
                print('[E0c FP64 point] ' + json.dumps(point, allow_nan=False), flush=True)
        write_json(directory / f'{backend}-fp64.json', report)
        print(f'[E0c FP64] {backend}: diagnostic written', flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if backend == 'native':
            ps.destroy_model_parallel()


def summarize(directory, launcher_exit):
    errors, native = [], {}
    try:
        manifest = read_json(directory / 'manifest.json')
        manifest_sha = sha256_file(directory / 'manifest.json')
        if sha256_file(directory / 'inputs.json') != manifest['inputs_sha256']:
            errors.append('diagnostic inputs changed')
        for backend in ('hf', 'native'):
            report = read_json(directory / f'{backend}-fp64.json')
            if report['manifest_sha256'] != manifest_sha or report['environment'].get('GPU_executed') is not True:
                errors.append(f'{backend}: missing matching GPU diagnostic')
            if (report['status'] != 'diagnostic_written' or report['dtype'] != 'float64'
                    or report['backend'] != backend or report['tp'] != 1
                    or report['inputs_sha256'] != manifest['inputs_sha256']
                    or any(report.get(key) != value for key, value in FLAGS.items())
                    or report['tensors'] != file_record(directory / 'reference-fp64' / f'{backend}.safetensors')):
                errors.append(f'{backend}: diagnostic metadata or tensor identity differs')
        native = report
        if native['hf_report_sha256'] != sha256_file(directory / 'hf-fp64.json'):
            errors.append('HF report identity changed')
        if not native['fp64_implementations_agree']:
            errors.append('FP64 implementations differ beyond the diagnostic limits')
    except Exception as exc:
        errors.append(str(exc))
    if launcher_exit:
        errors.append(f'launcher exited {launcher_exit}')
    if list(directory.glob('failure-*.json')):
        errors.append('worker failure evidence present')
    result = dict(FLAGS, stage='E0c-FP64-diagnostic', status='diagnostic_complete' if not errors else 'diagnostic_failed',
        errors=errors, fp64_comparisons=native.get('comparisons'), points=native.get('points'),
        note='FP64 agreement is diagnostic evidence only; the original FP32 E0c run remains failed.')
    write_json(directory / 'summary.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return bool(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'hf', 'native', 'summary'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fp32-run', type=Path)
    parser.add_argument('--launcher-exit', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'summary':
        raise SystemExit(summarize(args.output_dir, args.launcher_exit))
    try:
        if args.phase == 'prepare':
            if args.fp32_run is None:
                raise ValueError('--fp32-run is required')
            prepare(args.output_dir, args.fp32_run)
        else:
            run_model(args.output_dir, args.phase)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / f'failure-{args.phase}.json', dict(FLAGS, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

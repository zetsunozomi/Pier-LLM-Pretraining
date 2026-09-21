#!/usr/bin/env python3
"""E0c real Qwen3B CUDA conversion gate; no optimizer or training claim.

The HF reference runs alone, saves all gradients on scratch, then exits. Native
TP ranks run in separate processes so two full models never share GPU memory.
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from experiments.qwen.e0c_metrics import compare_tensor, exact_tensor
from experiments.qwen.e0c_linear_probe import LinearTrace, analyze_linear, probe_layer
from megatron.core.models.qwen.config import QwenArchitecture
from megatron.core.models.qwen.weights import SafeTensorSource, load_weights, parameter_mappings, sha256_file


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def file_record(path):
    return {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def check_files(directory, records):
    for name, expected in records.items():
        if Path(name).name != name or file_record(directory / name) != expected:
            raise ValueError(f'reference bytes changed or invalid filename: {name}')


def configure_gpu(stage='E0c'):
    import safetensors
    import transformers
    if transformers.__version__ != '4.57.3' or safetensors.__version__ != '0.7.0':
        raise RuntimeError(f'{stage} requires transformers==4.57.3 and safetensors==0.7.0; use the prepared environment')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError(f'{stage} requires exactly four visible CUDA GPUs')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('native BF16 CUDA support is required')
    torch.set_num_threads(1)
    torch.manual_seed(421)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    return {'GPU_executed': True, 'device': torch.cuda.get_device_name(local_rank),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'transformers': transformers.__version__, 'safetensors': safetensors.__version__,
            'tf32': False, 'reduced_precision_reduction': False, 'deterministic_algorithms': True}


def cross_entropy(logits, tokens):
    # Match both implementations explicitly: unscaled FP32 mean next-token CE.
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                              tokens[:, 1:].reshape(-1))


def finish_native_comparison(passed, comparisons, collect_mismatches=False):
    """Only a completed, finite numerical mismatch may continue collection.

    All rank reports have already been written. The strict summary still
    rejects their failed comparisons; execution errors never reach this path.
    """
    success = torch.tensor(int(passed), device='cuda')
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    dist.barrier()
    if int(success):
        return
    if collect_mismatches:
        fields = ('error_sq', 'reference_sq', 'actual_sq', 'dot', 'max_abs')
        finite = bool(comparisons) and all(s['elements'] > 0 and s['nonfinite'] == 0
                     and all(math.isfinite(s[key]) for key in fields) for s in comparisons)
        eligible = torch.tensor(int(finite), device='cuda')
        dist.all_reduce(eligible, op=dist.ReduceOp.MIN)
        if int(eligible):
            if dist.get_rank() == 0:
                print('[E0c] NUMERICAL MISMATCH: complete finite rank reports saved; '
                      'collecting remaining phases. E0c acceptance remains failed.', flush=True)
            return
    raise ValueError('numerical conversion check failed; inspect per-parameter JSON; later phases stopped')


def reference(args, manifest):
    from transformers import Qwen2ForCausalLM
    environment = configure_gpu()
    dtype = getattr(torch, {'fp32': 'float32', 'bf16': 'bfloat16'}[args.dtype])
    directory = args.output_dir / f'reference-{args.dtype}'
    directory.mkdir()
    inputs = json.loads((args.output_dir / 'inputs.json').read_text())
    tokens = torch.tensor(inputs['tokens'], dtype=torch.long, device='cuda')
    positions = torch.tensor(inputs['positions'], dtype=torch.long, device='cuda')
    arch = QwenArchitecture(**manifest['snapshot']['architecture'])
    print(f'[E0c] HF {args.dtype}: loading pinned 3B checkpoint', flush=True)
    model, loading = Qwen2ForCausalLM.from_pretrained(
        manifest['snapshot_path'], dtype=dtype, attn_implementation='eager',
        local_files_only=True, trust_remote_code=False, output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'HF checkpoint loading differs: {loading}')
    model = model.cuda().train()
    model.config.use_cache = False
    parameters = dict(model.named_parameters())
    expected_shapes = arch.hf_shapes()
    if {n: tuple(p.shape) for n, p in parameters.items()} != expected_shapes:
        raise ValueError('HF parameter coverage differs from the pinned architecture')
    source = SafeTensorSource(manifest['snapshot_path'])
    weights = {}
    for name, parameter in parameters.items():
        weights[name] = exact_tensor(parameter, source.read(name))
    if not all(record['bitwise_equal'] for record in weights.values()):
        raise ValueError('HF parameters differ from pinned source weights')
    print(f'[E0c] HF {args.dtype}: full logits and backward', flush=True)
    layer = probe_layer(arch)
    trace = LinearTrace(model.model.layers[layer].mlp.down_proj) if args.dtype == 'fp32' else None
    with trace if trace is not None else nullcontext():
        logits = model(tokens, position_ids=positions, use_cache=False).logits
        loss = cross_entropy(logits, tokens)
        loss.backward()
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
        raise ValueError('HF reference contains nonfinite logits/loss')
    save_file({'logits': logits.detach().cpu().contiguous(), 'loss': loss.detach().cpu().reshape(1)},
              str(directory / 'outputs.safetensors'))
    weight_map, gradients, files = {}, {}, {}
    if trace is not None:
        save_file(trace.complete(), str(directory / 'linear-trace.safetensors'))
        files['linear-trace.safetensors'] = file_record(directory / 'linear-trace.safetensors')
    # One tensor per file bounds CPU serialization memory by the largest tensor.
    for index, (name, parameter) in enumerate(parameters.items()):
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError(f'HF missing/nonfinite gradient: {name}')
        filename = f'gradient-{index:04d}.safetensors'
        save_file({name: parameter.grad.detach().cpu().contiguous()}, str(directory / filename))
        files[filename] = file_record(directory / filename)
        weight_map[name] = filename
        gradients[name] = {'shape': list(parameter.shape), 'elements': parameter.numel()}
    write_json(directory / 'model.safetensors.index.json', {'weight_map': weight_map})
    for filename in ('model.safetensors.index.json', 'outputs.safetensors'):
        files[filename] = file_record(directory / filename)
    result = {'status': 'reference_written', 'phase': 'reference', 'dtype': args.dtype,
              'manifest_sha256': sha256_file(args.output_dir / 'manifest.json'),
              'inputs_sha256': sha256_file(args.output_dir / 'inputs.json'),
              'environment': environment, 'weights': weights, 'gradients': gradients,
              'gradient_elements': sum(p.numel() for p in parameters.values()),
              'files': files, 'loading_info': loading, 'loss': float(loss.detach()),
              'performance_result': False, 'optimizer_steps': 0}
    if trace is not None:
        result['linear_probe'] = {'layer': layer, 'file': 'linear-trace.safetensors',
                                  'tensor_order': 'batch,sequence,channel', 'diagnostic_only': True}
    write_json(args.output_dir / f'hf-{args.dtype}.json', result)
    print(f'[E0c] HF {args.dtype}: reference written ({len(gradients)} gradients)', flush=True)


def native(args, manifest):
    from megatron.core import parallel_state as ps
    from megatron.core.models.qwen.model import build_model
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    environment = configure_gpu()
    collect_mismatches = bool(getattr(args, 'collect_numerical_mismatches', False))
    if collect_mismatches != manifest.get('collect_numerical_mismatches', False):
        raise ValueError('numerical collection mode differs from the run manifest')
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        if dist.get_world_size() != 4 or args.tp not in (1, 2):
            raise ValueError('E0c native phase requires world4, TP1/TP2')
        rank = dist.get_rank()
        ref_path = args.output_dir / f'hf-{args.dtype}.json'
        reference_report = json.loads(ref_path.read_text())
        directory = args.output_dir / f'reference-{args.dtype}'
        checks = [None]
        if rank == 0:
            try:
                check_files(directory, reference_report['files'])
                if (reference_report['manifest_sha256'] != sha256_file(args.output_dir / 'manifest.json')
                        or reference_report['inputs_sha256'] != sha256_file(args.output_dir / 'inputs.json')):
                    raise ValueError('HF reference belongs to another source/input manifest')
            except Exception as exc:
                checks[0] = f'{type(exc).__name__}: {exc}'
        dist.broadcast_object_list(checks, src=0)
        if checks[0]:
            raise ValueError(checks[0])
        ps.initialize_model_parallel(tensor_model_parallel_size=args.tp, num_subgroups=4 // args.tp)
        model_parallel_cuda_manual_seed(421)
        arch = QwenArchitecture(**manifest['snapshot']['architecture'])
        dtype = getattr(torch, {'fp32': 'float32', 'bf16': 'bfloat16'}[args.dtype])
        model = build_model(arch, dtype=dtype, parallel_output=False).train()
        tp_rank = ps.get_tensor_model_parallel_rank()
        source = SafeTensorSource(manifest['snapshot_path'])
        load_receipt = load_weights(model, source, arch, args.tp, tp_rank)
        mapping = parameter_mappings(arch, args.tp, tp_rank)
        parameters = dict(model.named_parameters())
        weights = {}
        for entry in mapping:
            weights[entry.target] = exact_tensor(parameters[entry.target], entry.materialize(source))
        if not all(record['bitwise_equal'] for record in weights.values()):
            raise ValueError('native mapped parameters differ from the source')
        inputs = json.loads((args.output_dir / 'inputs.json').read_text())
        tokens = torch.tensor(inputs['tokens'], dtype=torch.long, device='cuda')
        positions = torch.tensor(inputs['positions'], dtype=torch.long, device='cuda')
        length = tokens.shape[1]
        causal = torch.ones(1, 1, length, length, dtype=torch.bool, device='cuda').triu(1)
        print(f'[E0c] rank{rank} native {args.dtype}/TP{args.tp}: forward/backward', flush=True)
        layer = probe_layer(arch)
        trace = (LinearTrace(model.decoder.layers[layer].mlp.linear_fc2, sequence_first=True)
                 if args.dtype == 'fp32' and args.tp == 1 else None)
        with trace if trace is not None else nullcontext():
            logits = model(tokens, positions, causal)
            loss = cross_entropy(logits, tokens)
            loss.backward()
        outputs = load_file(str(directory / 'outputs.safetensors'))
        logits_check = compare_tensor(logits, outputs['logits'], args.dtype, 'logits')
        loss_check = compare_tensor(loss.reshape(1), outputs['loss'], args.dtype, 'loss')
        reference_gradients = SafeTensorSource(directory)
        gradients = {}
        for entry in mapping:
            gradient = parameters[entry.target].grad
            if gradient is None:
                raise ValueError(f'missing native gradient: {entry.target}')
            gradients[entry.target] = compare_tensor(
                gradient, entry.materialize(reference_gradients), args.dtype, 'gradient')
            if not gradients[entry.target]['passed']:
                print(f'[E0c] rank{rank} gradient mismatch {entry.target}: '
                      + json.dumps(gradients[entry.target], sort_keys=True, allow_nan=False), flush=True)
        passed = logits_check['passed'] and loss_check['passed'] and all(x['passed'] for x in gradients.values())
        report = {'status': 'passed' if passed else 'failed', 'phase': 'native',
                  'dtype': args.dtype, 'tp': args.tp, 'tp_rank': tp_rank, 'rank': rank, 'world_size': 4,
                  'manifest_sha256': sha256_file(args.output_dir / 'manifest.json'),
                  'inputs_sha256': sha256_file(args.output_dir / 'inputs.json'),
                  'reference_report_sha256': sha256_file(ref_path),
                  'environment': environment, 'load_receipt': load_receipt, 'weights': weights,
                  'logits': logits_check, 'loss': loss_check, 'gradients': gradients,
                  'gradient_elements': sum(p.numel() for p in parameters.values()),
                  'performance_result': False, 'optimizer_steps': 0}
        report['collect_numerical_mismatches'] = collect_mismatches
        report_path = args.output_dir / f'native-{args.dtype}-tp{args.tp}-rank{rank}.json'
        write_json(report_path, report)
        if trace is not None:
            # Keep raw operands on scratch and hash-bind the small JSON diagnostic.
            trace_directory = args.output_dir / 'linear-traces'
            trace_directory.mkdir(exist_ok=True)
            trace_path = trace_directory / f'fp32-tp1-rank{rank}.safetensors'
            save_file(trace.complete(), str(trace_path))
            reference_trace = directory / 'linear-trace.safetensors'
            if (reference_report.get('linear_probe', {}).get('layer') != layer
                    or file_record(reference_trace) != reference_report['files'].get(reference_trace.name)):
                raise ValueError('linear probe reference identity differs')
            parameter = f'decoder.layers.{layer}.mlp.linear_fc2.weight'
            diagnostic = analyze_linear(load_file(str(reference_trace)), trace.complete(), gradients[parameter])
            diagnostic.update(layer=layer, parameter=parameter, rank=rank, dtype=args.dtype, tp=args.tp,
                              environment=environment, manifest_sha256=report['manifest_sha256'],
                              inputs_sha256=report['inputs_sha256'],
                              reference_report_sha256=report['reference_report_sha256'],
                              native_report_sha256=sha256_file(report_path),
                              trace_files={str(p.relative_to(args.output_dir)): file_record(p)
                                           for p in (reference_trace, trace_path)})
            write_json(args.output_dir / f'linear-probe-fp32-tp1-rank{rank}.json', diagnostic)
            for point in diagnostic['points']:
                print(f'[E0c linear probe] rank{rank} {parameter}: '
                      + json.dumps(point, sort_keys=True, allow_nan=False), flush=True)
        finish_native_comparison(passed, [logits_check, loss_check, *gradients.values()], collect_mismatches)
    finally:
        # Destroy distributed groups explicitly on each rank, after communication.
        if dist.is_initialized():
            dist.destroy_process_group()
        ps.destroy_model_parallel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('reference', 'native'), required=True)
    parser.add_argument('--dtype', choices=('fp32', 'bf16'), required=True)
    parser.add_argument('--tp', type=int, choices=(1, 2), default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--collect-numerical-mismatches', action='store_true',
                        help='native only: retain finite numerical failures and collect later phases; summary still fails')
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / 'manifest.json').read_text())
    try:
        (reference if args.phase == 'reference' else native)(args, manifest)
    except Exception as exc:
        rank = os.environ.get('RANK', '0')
        write_json(args.output_dir / f'failure-{args.phase}-{args.dtype}-tp{args.tp}-rank{rank}.json',
                   {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc(), 'performance_result': False})
        raise


if __name__ == '__main__':
    main()

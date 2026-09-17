"""Centered Local-SGD integration, successful-step clock and validation evidence.

The production path stores only owned R/M and the executor's bounded workspace.
Full CPU/GPU oracle copies and per-step hashes are enabled only by --outer-verify.
"""

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .coordinates import optimizer_coordinates
from .executor import CenteredExecutor, power2


def enabled(args):
    return getattr(args, 'outer_runtime', 'legacy') == 'centered'


def digest(value):
    """Stable value hash, including dtype/shape and exact tensor bits."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach()
            result.update(str((str(data.dtype), tuple(data.shape))).encode())
            flat = data.reshape(-1)
            for start in range(0, flat.numel(), 262144):
                block = flat[start:start + 262144].cpu().contiguous()
                result.update(block.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part)
        else:
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


@dataclass
class StepClock:
    interval: int
    attempted: int = 0
    successful: int = 0
    boundaries: int = 0

    def advance(self, success):
        if self.interval < 1:
            raise ValueError('positive outer interval required')
        self.attempted += 1
        self.successful += int(success)
        boundary = success and self.successful % self.interval == 0
        self.boundaries += int(boundary)
        return boundary

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, state):
        if (state['interval'] != self.interval or not 0 <= state['successful'] <= state['attempted']
                or state['boundaries'] != state['successful'] // self.interval):
            raise ValueError('inconsistent successful-step checkpoint')
        self.attempted, self.successful, self.boundaries = (
            state['attempted'], state['successful'], state['boundaries'])


def validate_args(args):
    """Reject unsupported ownership/restore semantics before model allocation."""
    if not enabled(args):
        return
    if not args.local_sgd_inner_average or args.outer_sync_interval < 1:
        raise ValueError('centered runtime requires inner averaging and positive outer interval')
    if args.momentum_warmup_steps != 0:
        raise ValueError('centered runtime currently uses a fixed recipe without legacy lazy start')
    for name in ('use_distributed_optimizer', 'use_precision_aware_optimizer', 'fp16', 'fp8',
                 'use_custom_fsdp', 'use_torch_fsdp2', 'overlap_param_gather',
                 'overlap_param_gather_with_optimizer_step', 'async_save', 'no_save_optim',
                 'no_save_rng', 'no_load_optim', 'no_load_rng', 'finetune', 'exit_on_missing_checkpoint',
                 'use_checkpoint_args', 'use_legacy_models', 'rampup_batch_size', 'data_parallel_random_init',
                 'calculate_per_token_loss', 'external_cuda_graph', 'moe_use_upcycling',
                 'non_persistent_ckpt_type', 'pretrained_checkpoint', 'ckpt_convert_format',
                 'retro_project_dir', 'vision_pretraining'):
        if getattr(args, name, None):
            raise ValueError(f'{name} has no centered-runtime contract yet')
    if (args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1
            or args.expert_model_parallel_size != 1 or args.num_experts
            or args.virtual_pipeline_model_parallel_size is not None):
        raise ValueError('centered runtime currently requires dense PP1/CP1/EP1')
    if args.dataloader_type != 'single' or args.num_workers != 0:
        raise ValueError('exact cursor restore currently requires single dataloader and zero workers')
    if args.iterations_to_skip:
        raise ValueError('use synchronized optimizer skip handling, not iterations-to-skip')
    if getattr(args, 'rerun_mode', 'disabled') != 'disabled':
        raise ValueError('rerun state-machine recovery has no centered-runtime contract yet')
    if not power2(args.outer_cohort_size) or args.outer_tile_elements < 1:
        raise ValueError('positive tile and power-of-two cohort required')
    if not 0 <= args.outer_momentum < 1 or args.outer_learning_rate <= 0:
        raise ValueError('outer momentum must be in [0,1), with positive learning rate')
    if args.outer_verify and (args.outer_trace_dir is None or args.train_iters is None
                              or not 1 <= args.train_iters <= 500):
        raise ValueError('verification requires an output directory and at most 500 attempted steps')
    if args.outer_inject_skip_at and not args.outer_verify:
        raise ValueError('skip injection is a labeled correctness experiment requiring --outer-verify')


def make_outer_group():
    """Group corresponding TP and inner-DP coordinates across logical learners."""
    from megatron.core import parallel_state as ps
    dp = dist.get_process_group_ranks(ps.get_data_parallel_group())
    inner = dist.get_process_group_ranks(ps.get_data_parallel_sub_group())
    index = inner.index(dist.get_rank())
    if len(dp) % len(inner):
        raise ValueError('nonuniform inner DP groups')
    local = {'dp': dp, 'inner': inner, 'index': index}
    all_maps = [None] * dist.get_world_size()
    dist.all_gather_object(all_maps, local)
    groups = set()
    for mapping in all_maps:
        ranks, size = mapping['dp'], len(mapping['inner'])
        if mapping['inner'] not in [ranks[i:i + size] for i in range(0, len(ranks), size)]:
            raise ValueError('inner DP groups must partition each DP group in rank order')
        groups.add(tuple(ranks[mapping['index']::size]))
    own = None
    for peers in sorted(groups):
        group = dist.new_group(list(peers), timeout=timedelta(seconds=180))
        if dist.get_rank() in peers:
            own = group
    if own is None:
        raise ValueError('rank has no corresponding outer-coordinate group')
    return own, dp, inner


class CenteredRuntime:
    def __init__(self, args, model, optimizer, group=None, inner_ranks=None):
        self.args, self.model, self.optimizer = args, model, optimizer
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        if group is None:
            self.group, self.dp_ranks, self.inner_ranks = make_outer_group()
        else:
            self.group = group
            self.dp_ranks = dist.get_process_group_ranks(group)
            self.inner_ranks = inner_ranks or [self.rank]
        self.peers = dist.get_process_group_ranks(self.group)
        self.k, self.outer_rank = len(self.peers), dist.get_rank(self.group)
        self.s = args.outer_cohort_size
        if not power2(self.k) or self.s > self.k or self.k % self.s:
            raise ValueError('outer learner count and cohort must be nested powers of two')
        optimizer.validate_outer_update_support()
        children = getattr(optimizer, 'chained_optimizers', [optimizer])
        if any(getattr(child, 'grad_scaler', None) is not None for child in children):
            raise ValueError('global skip consensus currently supports FP32 or unscaled BF16')
        self.coordinates = optimizer_coordinates(model, optimizer)
        self.device = self.coordinates.device
        fingerprints = [None] * self.k
        dist.all_gather_object(fingerprints, self.coordinates.fingerprint, group=self.group)
        if len(set(fingerprints)) != 1:
            raise ValueError('different parameter coordinates within an outer group')
        self.clock = StepClock(args.outer_sync_interval)
        self.verify = args.outer_verify
        if self.verify and self.coordinates.numel > 1_000_000:
            raise ValueError('full-state correctness oracle is limited to 1M parameters per TP coordinate')
        width = (self.coordinates.numel + self.k - 1) // self.k
        g, b = self.k // self.s, self.outer_rank % self.s
        state_device = torch.device('cpu') if args.outer_cpu_offload else self.device
        pinned = state_device.type == 'cpu' and self.device.type == 'cuda'
        reference = torch.zeros(g * width, dtype=torch.float32, device=state_device, pin_memory=pinned)
        momentum = torch.zeros(width, dtype=torch.float32, device=state_device, pin_memory=pinned)
        # Initialize only this owner's reference interval; no full flat master.
        start = b * g * width
        valid = max(0, min(g * width, self.coordinates.numel - start))
        if valid:
            self.coordinates.read_into(start, reference[:valid])
        initial = digest([p for _, p, _ in self.coordinates.pairs])
        dist.all_gather_object(fingerprints, initial, group=self.group)
        if len(set(fingerprints)) != 1:
            raise ValueError('initial FP32 reference differs across learners')
        self.executor = CenteredExecutor(None, reference, momentum, cohort=self.s,
                                         tile_elements=args.outer_tile_elements,
                                         coordinates=self.coordinates, group=self.group)
        self.nonfinite = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.unity = torch.ones(1, device=self.device, dtype=torch.float32)
        self.pending_consumer = False
        self.consumer_checks = 0
        self.restored = False
        self.restore_evidence = None
        self.pending_rng = None
        self.events = []
        self.last_optimizer_before = None
        self.oracle_reference = self.coordinates.cpu_flat().numpy().copy() if self.verify else None
        self.oracle_momentum = (self.oracle_reference * 0) if self.verify else None
        self.report_path = None
        if args.outer_trace_dir:
            directory = Path(args.outer_trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.report_path = directory / f'rank-{self.rank}.json'
        if self.verify:
            for module in model:
                module.register_forward_pre_hook(self.before_forward)
        optimizer.centered_runtime = self

    def before_forward(self, module, inputs):
        if self.pending_consumer:
            self.coordinates.assert_model_committed()
            self.consumer_checks += 1
            self.pending_consumer = False

    @torch.no_grad()
    def skip_consensus(self, found_inf):
        # Called after model gradients are copied to FP32 masters and BEFORE any
        # learner changes model, inner moments or its learning-rate schedule.
        self.nonfinite.fill_(float(bool(found_inf)))
        grads = [p.grad for _, p, _ in self.coordinates.pairs if p.grad is not None]
        if grads:
            torch._amp_foreach_non_finite_check_and_unscale_(grads, self.nonfinite, self.unity)
        if (self.clock.attempted + 1 in self.args.outer_inject_skip_at
                and self.rank == self.args.outer_inject_skip_rank):
            self.nonfinite.fill_(1)
        dist.all_reduce(self.nonfinite, op=dist.ReduceOp.MAX)
        skip = bool(self.nonfinite.item())
        if self.verify:
            self.last_optimizer_before = digest(self.optimizer.state_dict()) if skip else None
        return skip

    def _oracle(self):
        # Full reference and gather are validation-only; never enter a perf run.
        import numpy as np
        from .spec import resident_reference
        local = self.coordinates.cpu_flat().to(self.device)
        leaves = [torch.empty_like(local) for _ in self.peers]
        dist.all_gather(leaves, local, group=self.group)
        weights = np.stack([leaf.cpu().numpy() for leaf in leaves])
        return resident_reference(self.oracle_reference, self.oracle_momentum, weights,
                                  self.args.outer_momentum, self.args.outer_learning_rate)

    @torch.no_grad()
    def after_attempt(self, success, attempted, loss_dict):
        if attempted != self.clock.attempted + 1:
            raise ValueError('training iteration and attempted-step clock differ')
        boundary = self.clock.advance(success)
        if self.verify and not success:
            if self.last_optimizer_before is None or digest(self.optimizer.state_dict()) != self.last_optimizer_before:
                raise AssertionError('skipped step changed optimizer state or FP32 masters')
        oracle = self._oracle() if self.verify and boundary else None
        if boundary:
            moments = self._moment_digest() if self.verify else None
            payload = self.executor.step(mu=self.args.outer_momentum, eta=self.args.outer_learning_rate)
            self.optimizer.commit_outer_update()
            self.pending_consumer = True
            if self.verify:
                self._check_oracle(oracle)
                if moments != self._moment_digest():
                    raise AssertionError('outer update modified inner optimizer moments')
                self.oracle_reference = oracle['reference'].copy()
                self.oracle_momentum = oracle['momentum'].copy()
        else:
            payload = None
        if self.verify:
            self.coordinates.assert_model_committed()
            self.events.append({'attempted': self.clock.attempted, 'successful': self.clock.successful,
                                'boundaries': self.clock.boundaries, 'skipped': not success,
                                'outer_boundary': boundary, 'payload': payload,
                                'oracle_states_bitwise': True if boundary else None,
                                'outer_retained_inner_state': True if boundary else None,
                                'skip_retained_optimizer': True if not success else None,
                                'master_sha256': digest([p for _, p, _ in self.coordinates.pairs]),
                                'model_sha256': digest([p for _, _, p in self.coordinates.pairs]),
                                'inner_sha256': self._moment_digest(),
                                'loss': {key: float(value) for key, value in loss_dict.items()}})
            self.write_report('running')

    def _moment_digest(self):
        children = getattr(self.optimizer, 'chained_optimizers', [self.optimizer])
        return digest([child.optimizer.state_dict() for child in children])

    def _check_oracle(self, expected):
        import numpy as np
        from .verification import same
        same(self.coordinates.cpu_flat(), torch.from_numpy(expected['reference']), 'production master vs oracle')
        e = self.executor
        reference = np.pad(expected['reference'], (0, e.width * e.k - e.n))
        momentum = np.pad(expected['momentum'], (0, e.width * e.k - e.n))
        start = e.b * e.g * e.width
        same(e.reference.view(-1), torch.from_numpy(reference[start:start + e.g * e.width]), 'production R')
        start = (e.b * e.g + e.a) * e.width
        same(e.momentum, torch.from_numpy(momentum[start:start + e.width]), 'production M')

    def write_report(self, status, error=None):
        if self.report_path is None:
            return
        report = {'status': status, 'rank': self.rank, 'world_size': self.world,
                  'GPU_executed': self.device.type == 'cuda', 'backend': dist.get_backend(self.group),
                  'torch_version': torch.__version__, 'clock': self.clock.state_dict(),
                  'outer_ranks': self.peers, 'inner_ranks': self.inner_ranks,
                  'coordinate_fingerprint': self.coordinates.fingerprint,
                  'coordinate_numel': self.coordinates.numel, 'cohort': self.s,
                  'state_tier': 'host' if self.args.outer_cpu_offload else 'device',
                  'allocation': self.executor.allocation_bytes(),
                  'restored': self.restored, 'consumer_checks': self.consumer_checks,
                  'restore_evidence': self.restore_evidence,
                  'pending_consumer': self.pending_consumer, 'events': self.events,
                  'performance_result': False, 'error': error}
        self.report_path.write_text(json.dumps(report, indent=2) + '\n')

    def finish(self, iteration):
        if iteration != self.clock.attempted:
            raise ValueError('final iteration differs from attempted steps')
        if self.verify and iteration == self.args.train_iters and self.pending_consumer:
            raise AssertionError('verification window must include the consumer after the last outer boundary')
        self.write_report('passed' if iteration == self.args.train_iters else 'partial')


def build_runtime(args, model, optimizer):
    if enabled(args):
        validate_args(args)
        return CenteredRuntime(args, model, optimizer)
    return None

"""Drained training-cycle measurements, separate from correctness oracles.

Every rank receives the same already-DP-reduced GPT loss-mask token count.
It is checked across ranks, never summed again across TP/DP replicas. Failed
optimizer attempts consume time/tokens but contribute no successful tokens.
Raw measurements do not certify a workload, tuned baseline, or wire counter.
"""

from pathlib import Path
import json
import time
import uuid

import torch
import torch.distributed as dist

FORMAT = 'pier-training-cycles-v1'
TOKEN_SEMANTICS = 'global DP-reduced non-padding loss tokens on successful updates; TP counted once'
TIMING_SCOPE = 'data/inner attempts/intermediate logging/outer/master-model commit, drained at boundaries'
EXCLUDED_FROM_CYCLE = ['initial model/data/checkpoint loading', 'boundary barriers and report I/O',
                       'logging/callbacks between independent cycle windows']


class CycleMeter:
    def __init__(self, device, directory, *, warmup_cycles=2, metadata=None,
                 planned_attempts=None, now=time.perf_counter):
        if warmup_cycles < 0 or not dist.is_initialized():
            raise ValueError('nonnegative warmup and initialized distributed world required')
        self.device = torch.device(device)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f'cycles-rank-{self.rank}.json'
        if self.path.exists():
            raise FileExistsError('cycle reports require a fresh directory for each run')
        self.warmup, self.metadata, self.now = warmup_cycles, metadata or {}, now
        if planned_attempts is not None and (type(planned_attempts) is not int or not 1 <= planned_attempts <= 500):
            raise ValueError('planned attempted-step budget must be an integer from 1 to 500')
        self.planned_attempts = planned_attempts
        identity = [str(uuid.uuid4()) if self.rank == 0 else None]
        dist.broadcast_object_list(identity, src=0)
        self.run_id = identity[0]
        self.initial_clock = self.final_clock = None
        self.tokens = torch.zeros(3, dtype=torch.int64, device=self.device)
        self.started = None
        self.outer_started = None
        self.pending_tokens = None
        self.attempt_started = False
        self.complete_count = 0
        self.records = []
        self.start_attempted = self.start_successful = 0
        self.starts_midcycle = False

    def drain(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def before_attempt(self, clock):
        if self.attempt_started:
            raise RuntimeError('previous measured attempt was not completed')
        if self.initial_clock is None:
            self.initial_clock = clock.state_dict()
        if self.started is None:
            self.drain()
            dist.barrier()
            self.drain()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            self.tokens.zero_()
            self.start_attempted, self.start_successful = clock.attempted, clock.successful
            self.starts_midcycle = clock.successful % clock.interval != 0
            self.started = self.now()
        self.pending_tokens = None
        self.attempt_started = True

    def record_tokens(self, dp_reduced_loss_tokens):
        if not self.attempt_started or self.pending_tokens is not None:
            raise RuntimeError('record exactly one accumulated GPT token count per attempt')
        value = dp_reduced_loss_tokens.detach().reshape(())
        # Validation stays on-device until the cycle boundary; no per-step item().
        valid = torch.isfinite(value) & (value >= 0) & (value == value.round())
        self.tokens[2].add_((~valid).to(dtype=torch.int64))
        self.pending_tokens = value.to(device=self.device, dtype=torch.int64)

    def before_outer(self):
        if not self.attempt_started:
            raise RuntimeError('outer measurement must be inside an attempted step')
        self.drain()
        self.outer_started = self.now()

    def after_attempt(self, success, boundary, clock, payload=None):
        if not self.attempt_started or self.pending_tokens is None:
            raise RuntimeError('measured training must supply its actual DP-reduced loss-mask token count')
        self.tokens[0].add_(self.pending_tokens)
        if success:
            self.tokens[1].add_(self.pending_tokens)
        self.attempt_started = False
        self.pending_tokens = None
        self.final_clock = clock.state_dict()
        if boundary:
            if self.outer_started is None:
                raise RuntimeError('outer boundary lacks its start timestamp')
            self._close(clock, boundary=True, payload=payload)

    def _close(self, clock, *, boundary, payload=None):
        self.drain()  # Includes the final master -> model commit, not just the collective.
        ended = self.now()
        elapsed = ended - self.started
        outer = ended - self.outer_started if self.outer_started is not None else 0.
        if elapsed <= 0 or not 0 <= outer <= elapsed:
            raise RuntimeError('invalid monotonic cycle timestamps')
        # DP tokens are already global; MAX/MIN checks replica agreement, not a SUM.
        reduced = torch.empty(7, dtype=torch.float64, device=self.device)
        reduced[0], reduced[1] = elapsed, outer
        reduced[2:4].copy_(self.tokens[:2])
        reduced[4:6].copy_(-self.tokens[:2])
        reduced[6].copy_(self.tokens[2])
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
        values = reduced.cpu().tolist()
        if values[6] or values[2] != -values[4] or values[3] != -values[5]:
            raise ValueError('invalid or rank-inconsistent DP-reduced token counts')
        successful_steps = clock.successful - self.start_successful
        complete = boundary and not self.starts_midcycle and successful_steps == clock.interval
        if complete:
            self.complete_count += 1
        warmup = complete and self.complete_count <= self.warmup
        eligible = complete and not warmup and values[3] > 0
        record = {'outer_boundary_index': clock.boundaries, 'complete_cycle': complete,
                  'warmup': warmup, 'eligible': eligible,
                  'attempted_start': self.start_attempted, 'attempted_end': clock.attempted,
                  'successful_start': self.start_successful, 'successful_end': clock.successful,
                  'attempts': clock.attempted - self.start_attempted, 'successful_steps': successful_steps,
                  'starts_midcycle': self.starts_midcycle, 'ends_at_outer_boundary': boundary,
                  'local_cycle_seconds': elapsed, 'max_rank_cycle_seconds': values[0],
                  'local_outer_and_commit_seconds': outer, 'max_rank_outer_and_commit_seconds': values[1],
                  'processed_loss_tokens_global': int(values[2]),
                  'successful_loss_tokens_global': int(values[3]),
                  'skipped_loss_tokens_global': int(values[2] - values[3]),
                  'useful_tokens_per_second': values[3] / values[0] if eligible else None,
                  'payload': payload, 'torch_peak_allocated_bytes': None, 'torch_peak_reserved_bytes': None,
                  'device_level_peak_bytes': None, 'physical_wire_bytes': None}
        if self.device.type == 'cuda':
            record['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(self.device)
            record['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(self.device)
        self.records.append(record)
        self.started = self.outer_started = None
        self.write_report('running')

    def write_report(self, status):
        report = {'format': FORMAT, 'run_id': self.run_id,
                  'status': status, 'rank': self.rank, 'world_size': self.world,
                  'GPU_executed': self.device.type == 'cuda', 'metadata': self.metadata,
                  'planned_attempts': self.planned_attempts,
                  'initial_clock': self.initial_clock, 'final_clock': self.final_clock,
                  'warmup_cycles': self.warmup, 'complete_cycles': self.complete_count,
                  'eligible_cycles': sum(r['eligible'] for r in self.records), 'cycles': self.records,
                  'token_semantics': TOKEN_SEMANTICS, 'timing_scope': TIMING_SCOPE,
                  'excluded_from_cycle': EXCLUDED_FROM_CYCLE,
                  'performance_result': False,
                  'not_validated': ['workload identity and tuned comparison', 'device-level peak memory',
                                    'node host/pinned peaks', 'physical link traffic', 'end-to-end allocated GPU-hours']}
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.path)

    def finish(self, clock):
        if self.attempt_started:
            raise RuntimeError('cannot finalize an unfinished measured training attempt')
        if self.initial_clock is None:
            self.initial_clock = clock.state_dict()
        self.final_clock = clock.state_dict()
        if self.started is not None:
            self._close(clock, boundary=False)
        self.write_report('complete')

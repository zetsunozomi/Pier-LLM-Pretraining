"""Complete optimizer/skip/trajectory/restart contract for the joint runtime."""

import copy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.outer_sync.checkpoint import load, restore_rng, save
from megatron.core.outer_sync.joint_config import JointController, validate_config
from megatron.core.outer_sync.runtime import digest
from test_runtime import advance, fixture


CONFIG = dict(version=1, variant='joint', page_elements=2, capacities=[2, 4], slot_counts=[1, 2],
              trace=True, budgets=[dict(round=1, transition_mib=1, next_phase_mib=1, target=2),
                                   dict(round=2, transition_mib=1, next_phase_mib=1, target=4),
                                   dict(round=3, transition_mib=1, next_phase_mib=1, target=1)])


def worker(rank, rendezvous, directory):
    import torch
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    try:
        base, model, optimizer, scheduler = fixture(rank, 1, Path(directory) / 'reference')
        advance(base, model, optimizer, scheduler, 11)
        expected = digest([model.state_dict(), optimizer.state_dict(), scheduler.state_dict()])
        for variant in ('single', 'pipeline', 'separate', 'joint'):
            config = {**CONFIG, 'variant': variant}
            path = Path(directory) / variant
            runtime, model, optimizer, scheduler = fixture(rank, 1, path, schedule='joint', joint_config=config)
            advance(runtime, model, optimizer, scheduler, 11)
            assert digest([model.state_dict(), optimizer.state_dict(), scheduler.state_dict()]) == expected
            assert runtime.consumer_checks == 3 and not runtime.pending_consumer
            expected_state = digest(runtime.executor.state_dict())
            runtime.executor.close()
            for interruption in (4, 5):
                resume = path / f'restart-{interruption}'
                runtime, model, optimizer, scheduler = fixture(rank, 1, resume, schedule='joint', joint_config=config)
                advance(runtime, model, optimizer, scheduler, interruption)
                assert runtime.consumer_checks == (0 if interruption == 4 else 1)
                assert runtime.pending_consumer == (interruption == 4)
                save(runtime, interruption, scheduler, 0.)
                runtime.executor.close()
                runtime, model, optimizer, scheduler = fixture(rank, 1, resume, schedule='joint', joint_config=config)
                with patch('megatron.core.optimizer.optimizer.parallel_state.get_pipeline_model_parallel_world_size', return_value=1):
                    load(runtime, str(resume), scheduler)
                assert runtime.executor.round == runtime.clock.boundaries
                restore_rng(runtime.pending_rng)
                runtime.pending_rng = None
                advance(runtime, model, optimizer, scheduler, 11)
                assert digest([model.state_dict(), optimizer.state_dict(), scheduler.state_dict()]) == expected
                assert digest(runtime.executor.state_dict()) == expected_state
                assert runtime.consumer_checks == (3 if interruption == 4 else 4)
                assert not runtime.pending_consumer
                runtime.executor.close()
        # Exercise the production cycle meter with changing layouts and actual
        # payload traces, independently of full-state oracle measurements.
        path = Path(directory) / 'meter'
        runtime, model, optimizer, scheduler = fixture(
            rank, 1, path, arm='pier', schedule='joint', joint_config=CONFIG, measure=True)
        advance(runtime, model, optimizer, scheduler, 11)
        runtime.finish(11)
        assert runtime.meter.records[1]['payload']['plan']['cohort'] == 4
        runtime.executor.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


class JointRuntimeTests(unittest.TestCase):
    def test_config_validation(self):
        self.assertEqual(validate_config(CONFIG), CONFIG)
        for broken in ({**CONFIG, 'variant': 'typo'}, {**CONFIG, 'slot_counts': [0]},
                       {**CONFIG, 'capacities': [3]}, {**CONFIG, 'unknown': True}):
            with self.assertRaises(ValueError):
                validate_config(broken)
        copied = copy.deepcopy(CONFIG)
        copied['budgets'][1]['round'] = 1
        with self.assertRaises(ValueError):
            validate_config(copied)
        self.assertEqual(JointController(CONFIG).announcement(4)['target'], 1)

    def test_production_training_and_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='pier-joint-runtime-') as directory:
            mp.spawn(worker, args=(f'file://{directory}/rendezvous', directory), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()

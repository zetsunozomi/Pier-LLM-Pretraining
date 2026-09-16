"""Check that partial or stale cluster outputs cannot be reported as a pass."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

EXPERIMENTS = Path(__file__).resolve().parents[2] / 'experiments/centered_outer'
sys.path.insert(0, str(EXPERIMENTS))
from manifest import source_hashes
from summarize import summarize


class ReportTests(unittest.TestCase):
    def test_complete_schema_then_duplicate_case(self):
        # Synthetic schema fixture only, deleted at test exit. It is never a
        # hardware result and is not included in the cluster handoff artifacts.
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            hashes = source_hashes()
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write(output / 'manifest.json', {'nodes': 1, 'gpus_per_node': 4, 'source_sha256': hashes})
            write(output / 'node-0.json', {'cuda_available': True, 'visible_devices': 4})
            for inner in (1, 2):
                for rank in range(4):
                    write(output / f'megatron-dp{inner}/rank-{rank}.json', {
                        'status': 'passed', 'rank': rank, 'GPU_executed': True,
                        'world_size': 4, 'inner_dp_size': inner,
                        'normalization': [{'inner_average': a, 'collective_average': b,
                                           'inner_and_warmup_gradient_bits_match': True}
                                          for a in (False, True) for b in (False, True)],
                        'outer_update': [{'outer_shard': a, 'outer_cpu_offload': b,
                                          'missing_copy_negative_control_observed': True,
                                          'master_model_next_forward_bits_match': True,
                                          'inner_moments_retained': True, 'local_optimizer_restore': True}
                                         for a in (False, True) for b in (False, True)]})
            operators = [{'rank': rank, 's': s, 'N': n, 'tile_elements_per_momentum_owner': tile,
                          'state_tier': tier, 'bitwise_states_and_model': True,
                          'fixed_workspace_storages': True}
                         for rank in range(4) for s in (1, 2, 4) for n in (1, 5, 17, 64)
                         for tile in (1, 7) for tier in ('device', 'host')]
            training = [{'rank': rank, 's': s, 'next_forward_and_inner_moments_bitwise': True,
                         'drained_boundary_executor_restore': True}
                        for rank in range(4) for s in (1, 2, 4)]
            protocol = {'status': 'passed', 'ranks': 4, 'GPU_executed': True, 'backend': 'nccl',
                        'records': {'operator': operators, 'training': training},
                        'source_sha256': {p: hashes['experiments/centered_outer/reference/' + p]
                                          for p in ('executor.py', 'verify_executor.py')}}
            write(output / 'protocol.json', protocol)
            self.assertEqual(summarize(output, 0)['status'], 'passed')
            operators[0] = operators[1].copy()
            write(output / 'protocol.json', protocol)
            self.assertEqual(summarize(output, 0)['status'], 'failed')

    def test_missing_evidence_fails(self):
        with tempfile.TemporaryDirectory() as name:
            result = summarize(Path(name), 0)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['GPU_executed'])

    def test_cpu_protocol_cannot_supply_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / 'manifest.json').write_text(json.dumps({
                'nodes': 1, 'gpus_per_node': 4, 'source_sha256': source_hashes()}))
            (output / 'protocol.json').write_text(json.dumps({
                'status': 'passed', 'ranks': 4, 'GPU_executed': False, 'backend': 'gloo'}))
            result = summarize(output, 0)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('ordered protocol: no complete CUDA/NCCL evidence', result['errors'])

    def test_launcher_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as name:
            result = summarize(Path(name), 17)
            self.assertTrue(any('exit code 17' in e for e in result['errors']))


if __name__ == '__main__':
    unittest.main()

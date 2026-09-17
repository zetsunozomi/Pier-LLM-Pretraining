"""Pinned model counts, supported TP layouts and byte provenance checks."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/qwen'))
from preflight import preflight, verify_file


class PreflightTests(unittest.TestCase):
    def test_all_pinned_model_counts_and_planned_layouts(self):
        for model, count, sizes in (('1.5B', 1543714304, (1, 2)),
                                    ('3B', 3085938688, (1, 2)),
                                    ('7B', 7615616512, (1, 2, 4))):
            for tp in sizes:
                report = preflight(model, tp)
                self.assertEqual(report['unique_parameters'], count)
                self.assertFalse(report['checkpoint_verified'])
                self.assertFalse(report['GPU_executed'])
                self.assertFalse(report['ready_for_training'])
        with self.assertRaises(ValueError):
            preflight('3B', 4)

    def test_file_identity_must_match_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            data = b'local synthetic fixture, not checkpoint evidence'
            path.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            for lfs in (None, sha):
                record = dict(bytes=len(data), lfs_sha256=lfs, git_blob_id=blob)
                self.assertEqual(verify_file(path, record)['sha256'], sha)
                path.write_bytes(data[:-1] + b'X')
                with self.assertRaises(ValueError):
                    verify_file(path, record)
                path.write_bytes(data)


if __name__ == '__main__':
    unittest.main()

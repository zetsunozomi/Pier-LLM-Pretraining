"""Compatibility import for the production centered executor.

The implementation now lives in megatron.core.outer_sync.executor and is shared
by pretrain_gpt and this independent correctness verifier.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from megatron.core.outer_sync.executor import CenteredExecutor, overlaps, power2

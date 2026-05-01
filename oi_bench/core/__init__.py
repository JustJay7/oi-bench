"""
OI-Bench core abstractions.

The three files in this package define the complete contract between
models and tasks. Nothing else in the codebase should bypass these interfaces.

  types.py    — Stimulus, ModelState, TrialResult dataclasses
  adapter.py  — OIModel ABC (implement this to register a model)
  protocol.py — BenchmarkTask ABC (implement this to add a task)
"""

from .types import Stimulus, ModelState, TrialResult
from .adapter import OIModel
from .protocol import BenchmarkTask

__all__ = [
    "Stimulus",
    "ModelState",
    "TrialResult",
    "OIModel",
    "BenchmarkTask",
]

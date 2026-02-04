"""
Inference Strategies for LeRobot Inference Node

This module provides pluggable strategies for observation acquisition:
- MultiProcessStrategy (PRODUCTION DEFAULT): Uses shared memory + worker processes for 30+ Hz
- SingleProcessStrategy (LEGACY): Uses callbacks + threading for debugging (~6 Hz due to GIL)
"""

from .base import InferenceStrategy
from .multi_process import MultiProcessStrategy
from .single_process import SingleProcessStrategy

__all__ = [
    "InferenceStrategy",
    "MultiProcessStrategy",
    "SingleProcessStrategy",
]

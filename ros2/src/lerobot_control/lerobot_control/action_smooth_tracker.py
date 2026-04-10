"""Action smoothness tracker for published robot actions.

Sliding-window tracker that computes per-step and windowed statistics:
- L2 delta between consecutive actions
- Jerk (second derivative of delta)
- Clamped joint percentage (safety limiter activations)
"""

import threading
from collections import deque

import numpy as np


class ActionSmoothTracker:
    """Thread-safe sliding-window tracker for action output smoothness metrics."""

    def __init__(self, action_dim: int, window: int = 100):
        self._lock = threading.Lock()
        self._action_dim = action_dim
        self._history: deque[np.ndarray] = deque(maxlen=window + 2)
        self._deltas: deque[float] = deque(maxlen=window)
        self._jerks: deque[float] = deque(maxlen=window)
        self._clamped_count = 0
        self._step_count = 0

    def record(self, action: np.ndarray, n_clamped: int = 0) -> None:
        """Record one published action and optional clamping info."""
        with self._lock:
            self._history.append(action.copy())
            self._step_count += 1
            if n_clamped > 0:
                self._clamped_count += 1
            if len(self._history) >= 2:
                delta = float(np.linalg.norm(self._history[-1] - self._history[-2]))
                self._deltas.append(delta)
            if len(self._deltas) >= 2:
                jerk = abs(self._deltas[-1] - self._deltas[-2])
                self._jerks.append(jerk)

    def get_stats(self) -> dict | None:
        """Return smoothness statistics for current window. Thread-safe."""
        with self._lock:
            if len(self._deltas) < 2:
                return None
            deltas = np.array(self._deltas)
            return {
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas)),
                "delta_max": float(np.max(deltas)),
                "jerk_mean": float(np.mean(self._jerks)) if self._jerks else 0.0,
                "jerk_max": float(np.max(self._jerks)) if self._jerks else 0.0,
                "clamped_pct": self._clamped_count / max(self._step_count, 1) * 100,
                "step_count": self._step_count,
            }

    def reset(self):
        with self._lock:
            self._history.clear()
            self._deltas.clear()
            self._jerks.clear()
            self._clamped_count = 0
            self._step_count = 0

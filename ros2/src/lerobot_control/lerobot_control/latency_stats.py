import numpy as np
from lerobot.policies.rtc.latency_tracker import LatencyTracker


class LatencyStats(LatencyTracker):
    """LatencyTracker extended with mean/std for logging."""

    def mean(self) -> float:
        vals = list(self._values)
        return float(np.mean(vals)) if vals else 0.0

    def std(self) -> float:
        vals = list(self._values)
        return float(np.std(vals)) if vals else 0.0

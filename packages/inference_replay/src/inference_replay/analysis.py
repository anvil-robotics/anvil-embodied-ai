"""Stall, clamp and event detection over an inference trace.

Ported from `scripts/build_trace_replay.py` with its thresholds unchanged, so this module's
output can be diffed against that script's console summary on the same CSV. That equality is
the cheapest correctness check available for this feature -- do not tune these constants
without re-baselining against it.

The finding that motivated the original script: large |commanded - observed| values come from
per-joint STALLS, not from fast motion. A stall is a joint that has stopped moving while it is
still being commanded meaningfully away from where it sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trace import ARM_CHANNELS, N_CHANNELS, NAMES, Trace

VEL_EPS = 0.002  # rad/step; measured joint motion below this counts as frozen
GAP_EPS = 0.15  # rad; commanded-observed error above this counts as meaningful
MIN_LEN = 5  # steps; shorter runs are noise, not a stall


def intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges where a boolean mask is True."""
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


@dataclass(frozen=True)
class StallEvent:
    """One joint stalled over one contiguous span of steps."""

    channel: int
    name: str
    start: int
    end: int
    steps: int
    t_sec: float
    duration_sec: float
    max_gap: float  # worst commanded-observed error during the stall, rad
    obs_range: float  # how far the joint actually moved during it, rad


@dataclass(frozen=True)
class StallWindow:
    """A span where at least one joint was stalled, and which joints were involved."""

    start: int
    end: int
    t_sec: float
    duration_sec: float
    joints: list[str]


@dataclass
class Analysis:
    gap: np.ndarray  # (n, 16) raw_output - obs_state
    stalls: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    clamps: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    events: list[StallEvent] = field(default_factory=list)
    windows: list[StallWindow] = field(default_factory=list)

    def stall_mask(self, channel: int, n: int) -> np.ndarray:
        return _spans_to_mask(self.stalls.get(channel, []), n)

    def clamp_mask(self, channel: int, n: int) -> np.ndarray:
        return _spans_to_mask(self.clamps.get(channel, []), n)


def _spans_to_mask(spans: list[tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in spans:
        mask[start:end] = True
    return mask


def analyse(trace: Trace) -> Analysis:
    """Detect stalls and limiter clamps across a trace."""
    gap = trace.raw - trace.obs
    # Step-to-step motion. `prepend` keeps the array length so indices line up with `rel`.
    velocity = np.abs(np.diff(trace.obs, axis=0, prepend=trace.obs[:1]))

    stalls: dict[int, list[tuple[int, int]]] = {}
    events: list[StallEvent] = []
    for channel in ARM_CHANNELS:
        frozen_and_commanded = (velocity[:, channel] < VEL_EPS) & (np.abs(gap[:, channel]) > GAP_EPS)
        spans = [span for span in intervals(frozen_and_commanded) if span[1] - span[0] >= MIN_LEN]
        stalls[channel] = spans
        for start, end in spans:
            window = trace.obs[start:end, channel]
            events.append(
                StallEvent(
                    channel=channel,
                    name=NAMES[channel],
                    start=start,
                    end=end,
                    steps=end - start,
                    t_sec=float(trace.rel[start]),
                    duration_sec=float(trace.rel[end - 1] - trace.rel[start]),
                    max_gap=float(np.abs(gap[start:end, channel]).max()),
                    obs_range=float(window.max() - window.min()),
                )
            )
    events.sort(key=lambda e: -e.steps)

    # Where the limiter was saturating: |cmd - obs| sitting exactly at the inferred cap.
    clamps = {
        channel: intervals(np.isclose(np.abs(trace.cmd[:, channel] - trace.obs[:, channel]), trace.cap, atol=1e-6))
        for channel in range(N_CHANNELS)
    }

    any_stalled = np.zeros(trace.n, dtype=bool)
    for spans in stalls.values():
        for start, end in spans:
            any_stalled[start:end] = True

    windows = [
        StallWindow(
            start=start,
            end=end,
            t_sec=float(trace.rel[start]),
            duration_sec=float(trace.rel[end - 1] - trace.rel[start]),
            joints=sorted({e.name for e in events if e.start < end and e.end > start}),
        )
        for start, end in intervals(any_stalled)
    ]

    return Analysis(gap=gap, stalls=stalls, clamps=clamps, events=events, windows=windows)

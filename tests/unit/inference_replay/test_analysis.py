"""Stall/clamp detection. Thresholds are pinned to build_trace_replay.py -- see analysis.py."""

from __future__ import annotations

import numpy as np
import pytest

from inference_replay.analysis import GAP_EPS, MIN_LEN, VEL_EPS, analyse, intervals
from inference_replay.trace import GRIPPER_CHANNELS, N_CHANNELS, NAMES, load_trace

from .conftest import HZ, make_trace, plant_stall, slow_ramp, smooth_motion, write_trace


class TestIntervals:
    def test_empty_mask_has_no_intervals(self):
        assert intervals(np.zeros(10, dtype=bool)) == []

    def test_single_run(self):
        mask = np.zeros(10, dtype=bool)
        mask[3:7] = True
        assert intervals(mask) == [(3, 7)]

    def test_two_runs(self):
        mask = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
        assert intervals(mask) == [(0, 2), (4, 7)]

    def test_run_touching_the_end_is_closed(self):
        mask = np.array([0, 0, 1, 1], dtype=bool)
        assert intervals(mask) == [(2, 4)]

    def test_intervals_are_half_open(self):
        mask = np.ones(5, dtype=bool)
        start, end = intervals(mask)[0]
        assert (start, end) == (0, 5)


def _trace_with_stall(tmp_path, channel: int, steps: int, gap: float = 0.3):
    """A trace where one channel freezes for `steps` while commanded `gap` rad away."""
    n = 200
    stall_from, stall_to = 50, 50 + steps
    obs = smooth_motion(n=n)
    slow_ramp(n, channel, obs)
    raw = obs + 0.001
    plant_stall(obs, raw, channel, stall_from, steps, gap=gap)
    path = tmp_path / "inference_data.csv"
    write_trace(path, obs, raw)
    return load_trace(path), stall_from, stall_to


class TestStallDetection:
    def test_finds_a_planted_stall_on_the_right_channel(self, tmp_path):
        channel = 3
        trace, stall_from, stall_to = _trace_with_stall(tmp_path, channel, steps=20)
        analysis = analyse(trace)
        assert [e.channel for e in analysis.events] == [channel]
        event = analysis.events[0]
        assert event.name == NAMES[channel]
        # The frozen span is detected; its first step is where velocity first read zero.
        assert stall_from <= event.start <= stall_from + 1
        assert event.end == pytest.approx(stall_to, abs=1)
        assert event.max_gap > GAP_EPS
        assert event.obs_range == pytest.approx(0.0, abs=1e-9)

    def test_clean_trace_has_no_stalls(self, good_trace_path):
        assert analyse(load_trace(good_trace_path)).events == []

    # Velocity is a backward difference, so the first sample of a frozen span still carries the
    # motion that arrived at it: planting N frozen samples yields a detected span of N-1.
    def test_span_below_min_len_is_ignored(self, tmp_path):
        trace, _, _ = _trace_with_stall(tmp_path, 3, steps=MIN_LEN)
        assert analyse(trace).events == []

    def test_span_at_min_len_is_reported(self, tmp_path):
        trace, _, _ = _trace_with_stall(tmp_path, 3, steps=MIN_LEN + 1)
        events = analyse(trace).events
        assert len(events) == 1
        assert events[0].steps == MIN_LEN

    def test_frozen_but_barely_commanded_is_not_a_stall(self, tmp_path):
        """A joint held still while commanded to stay still is correct behaviour, not a fault."""
        trace, _, _ = _trace_with_stall(tmp_path, 3, steps=30, gap=GAP_EPS / 2)
        assert analyse(trace).events == []

    @pytest.mark.parametrize("channel", GRIPPER_CHANNELS)
    def test_grippers_are_exempt(self, channel):
        """A gripper legitimately stops against an object while commanded further closed.

        Built directly rather than through a CSV: a gripper commanded 0.3 rad away is exactly
        what the loader's gripper check rejects, so this shape cannot reach analysis via a file.
        """
        n = 200
        obs = smooth_motion(n=n)
        slow_ramp(n, channel, obs)
        raw = obs + 0.001
        plant_stall(obs, raw, channel, start=50, steps=40)
        assert analyse(make_trace(obs, raw)).events == []

    def test_events_are_ranked_by_duration(self, tmp_path):
        n = 400
        obs = smooth_motion(n=n)
        for channel in (2, 4, 6):
            slow_ramp(n, channel, obs)
        # raw is built only after obs is final, so each gap is confined to its planted window.
        raw = obs + 0.001
        for channel, (start, length) in {2: (50, 10), 4: (150, 40), 6: (250, 25)}.items():
            plant_stall(obs, raw, channel, start, length)
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, raw)
        events = analyse(load_trace(path)).events
        assert [e.steps for e in events] == sorted((e.steps for e in events), reverse=True)
        assert events[0].channel == 4

    def test_velocity_threshold_is_respected(self, tmp_path):
        """Creeping faster than VEL_EPS per step is motion, not a stall."""
        n = 200
        channel = 5
        obs = smooth_motion(n=n)
        obs[:, channel] = np.arange(n) * VEL_EPS * 2
        raw = obs + 0.001
        raw[50:100, channel] = obs[50:100, channel] + 0.3
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, raw)
        assert analyse(load_trace(path)).events == []


class TestWindows:
    def test_overlapping_stalls_merge_into_one_window(self, tmp_path):
        n = 200
        obs = smooth_motion(n=n)
        for channel in (2, 4):
            slow_ramp(n, channel, obs)
        raw = obs + 0.001
        for channel in (2, 4):
            plant_stall(obs, raw, channel, start=60, steps=40)
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, raw)
        analysis = analyse(load_trace(path))
        assert len(analysis.events) == 2
        assert len(analysis.windows) == 1
        assert analysis.windows[0].joints == sorted([NAMES[2], NAMES[4]])

    def test_separated_stalls_stay_separate(self, tmp_path):
        n = 300
        obs = smooth_motion(n=n)
        for channel in (2, 4):
            slow_ramp(n, channel, obs)
        raw = obs + 0.001
        for channel, start in ((2, 40), (4, 200)):
            plant_stall(obs, raw, channel, start, steps=30)
        path = tmp_path / "inference_data.csv"
        write_trace(path, obs, raw)
        assert len(analyse(load_trace(path)).windows) == 2

    def test_window_duration_is_in_seconds(self, tmp_path):
        trace, _, _ = _trace_with_stall(tmp_path, 3, steps=30)
        window = analyse(trace).windows[0]
        assert window.duration_sec == pytest.approx(29 / HZ, rel=0.05)


class TestMasks:
    def test_stall_mask_matches_the_reported_spans(self, tmp_path):
        channel = 3
        trace, _, _ = _trace_with_stall(tmp_path, channel, steps=20)
        analysis = analyse(trace)
        mask = analysis.stall_mask(channel, trace.n)
        event = analysis.events[0]
        assert mask.sum() == event.steps
        assert mask[event.start] and mask[event.end - 1]
        assert not mask[event.end]

    def test_untouched_channel_has_an_empty_mask(self, tmp_path):
        trace, _, _ = _trace_with_stall(tmp_path, 3, steps=20)
        analysis = analyse(trace)
        assert not analysis.stall_mask(7, trace.n).any()

    def test_gap_is_raw_minus_observed(self, tmp_path):
        trace, _, _ = _trace_with_stall(tmp_path, 3, steps=20)
        analysis = analyse(trace)
        np.testing.assert_allclose(analysis.gap, trace.raw - trace.obs)
        assert analysis.gap.shape == (trace.n, N_CHANNELS)

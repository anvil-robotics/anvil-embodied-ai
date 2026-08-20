"""Synthetic inference_data.csv fixtures.

Building traces by hand rather than checking in a real CSV keeps the assertions about
*behaviour* (this permutation, this stall shape) legible, and lets a deliberately broken trace
be constructed to prove the alignment gate actually fires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference_replay.analysis import VEL_EPS
from inference_replay.trace import MODEL_TO_CTRL, N_CHANNELS, Trace

HZ = 30.0


def to_csv_order(model_order: np.ndarray) -> np.ndarray:
    """Invert the loader's permutation, so a test can specify channels in model order.

    The loader reads `obs_csv[:, MODEL_TO_CTRL]`, so writing a file that yields a desired
    model-order array means scattering it back through the same index set.
    """
    csv_order = np.zeros_like(model_order)
    csv_order[:, MODEL_TO_CTRL] = model_order
    return csv_order


def write_trace(
    path,
    obs: np.ndarray,
    raw: np.ndarray,
    *,
    action_type: str = "absolute",
    cmd: np.ndarray | None = None,
    hz: float = HZ,
) -> None:
    """Write a monitor-format CSV. `obs`/`raw`/`cmd` are all in model order.

    When `cmd` is omitted it is derived the way the real action limiter derives it, which is
    also what makes the loader's clamp-identity check pass.
    """
    n = len(obs)
    if cmd is None:
        cmd = raw.copy()
    ts = 1000.0 + np.arange(n) / hz

    header = (
        ["timestamp"]
        + [f"obs_state_{i}" for i in range(N_CHANNELS)]
        + [f"raw_output_{i}" for i in range(N_CHANNELS)]
        + [f"control_cmd_{i}" for i in range(N_CHANNELS)]
        + [f"delta_cmd_{i}" for i in range(N_CHANNELS)]
    )
    obs_csv = to_csv_order(obs)
    cmd_csv = to_csv_order(cmd)
    lines = [f"# action_type: {action_type}", "# joint_names: ", ",".join(header)]
    for i in range(n):
        row = (
            [f"{ts[i]:.6f}"]
            + [repr(float(v)) for v in obs_csv[i]]
            + [repr(float(v)) for v in raw[i]]
            + [repr(float(v)) for v in cmd_csv[i]]
            + [repr(float(v)) for v in (cmd[i] - obs[i])]
        )
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


def smooth_motion(n: int = 120, amplitude: float = 0.2) -> np.ndarray:
    """A benign trace: every channel sweeps slowly, grippers kept small."""
    t = np.arange(n) / HZ
    obs = np.zeros((n, N_CHANNELS))
    for channel in range(N_CHANNELS):
        obs[:, channel] = amplitude * np.sin(0.5 * t + channel)
    # Channels 0 and 8 are grippers, in metres; joint-scale values there trip the loader's
    # gripper check, which exists precisely to catch a mis-split arm/gripper layout.
    obs[:, 0] = 0.02 + 0.001 * np.sin(t)
    obs[:, 8] = 0.02 + 0.001 * np.cos(t)
    return obs


def make_trace(obs: np.ndarray, raw: np.ndarray, hz: float = HZ) -> Trace:
    """Build a Trace directly, bypassing the CSV and its alignment gate.

    Needed to exercise the analysis layer on shapes the loader legitimately refuses -- a
    gripper commanded a joint-scale distance, for instance, which is exactly the input the
    gripper-exemption rule is about.
    """
    n = len(obs)
    ts = 1000.0 + np.arange(n) / hz
    cmd = raw.copy()
    cap = float(np.abs(cmd - obs).max())
    return Trace(
        path=Path("<synthetic>"),
        action_type="absolute",
        ts=ts,
        rel=ts - ts[0],
        raw=raw,
        obs=obs,
        cmd=cmd,
        cap=cap,
        residual=0.0,
    )


def plant_stall(
    obs: np.ndarray, raw: np.ndarray, channel: int, start: int, steps: int, gap: float = 0.3
) -> None:
    """Freeze `channel` for `steps` while commanding it `gap` rad away, in place.

    `raw` must already be sized and populated: the gap is measured against the *final* obs, so
    building raw before editing obs silently produces a gap everywhere instead of just here.
    """
    obs[start : start + steps, channel] = obs[start, channel]
    raw[start : start + steps, channel] = obs[start, channel] + gap


def slow_ramp(n: int, channel: int, obs: np.ndarray) -> None:
    """Give one channel a definitely-moving baseline, in place."""
    obs[:, channel] = np.linspace(0.0, 0.1 + n * VEL_EPS * 3, n)


@pytest.fixture
def good_trace_path(tmp_path):
    """A valid, stall-free trace: commands lead observations slightly and are tracked."""
    obs = smooth_motion()
    raw = obs + 0.01  # a small, uniformly-tracked lead
    path = tmp_path / "inference_data.csv"
    write_trace(path, obs, raw)
    return path

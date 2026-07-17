"""GT-replay: model-free validation of the EE Delta(n-(n-1)) transform pipeline.

Permanent, reusable CLI — the FIRST GATE any delta-flow dataset must pass
before training compute is spent on it (see claude_docs/ee-delta-flow-plan.md,
Item 3). Extends the scratchpad prototype
(``ee_diag/task4_roundtrip.py``) into a real tool with two validation modes:

``--encoding absolute`` (pure transform-math round-trip)
    Works on any EE dataset with an absolute ``action`` column (the default,
    pre-existing EE encoding). For each episode, ``t=1..T-1``:
        delta      = ee_delta_forward(action[t], state[t-1])
        recon      = ee_delta_inverse(delta, state[t-1])
    Asserts ``recon`` recovers ``action[t]`` to ~machine precision. This
    validates the transform math itself, independent of mcap_converter.

``--encoding delta`` (baked on-disk column correctness)
    For a dataset converted with ``action_encoding="delta"``, the on-disk
    ``action`` column IS ALREADY the baked delta — there is no absolute
    action column to compare against directly. But because mcap_converter
    builds ``action_abs[t]`` from the SAME raw pose sample as
    ``observation.state[t]`` (position/quat re-encoded to rot6d, nothing
    else), ``action_abs[t]`` is recoverable from ``observation.state[t]``
    alone via ``ee_obs_abs_forward``. So for ``t=1..T-1``:
        action_abs[t] = ee_obs_abs_forward(state[t])
        expected       = ee_delta_forward(action_abs[t], state[t-1])
    is compared against the ACTUAL on-disk ``action[t]`` — this is "is the
    baked column correct", not a round-trip of the math alone. The first
    frame of each episode (``t=0``, self-anchor convention) is checked
    separately against the identity-rot6d / zero-xyz invariant.

``--encoding auto`` (default) reads ``conversion_config.yaml`` in the dataset
root to determine which mode applies; specify explicitly to override.

Bar is strict and does NOT relax for small/smoke-scale datasets — this tests
transform math correctness, independent of dataset size or model quality.
Exit code 0 = PASS (gate clears), 1 = FAIL (do not proceed to training).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Per-arm slice layout (10 action/delta dims, 8 state dims) — bimanual = 2 arms.
_POS_SLICE = (0, 3)
_ROT_SLICE = (3, 9)
_ACTION_DIM_PER_ARM = 10
_STATE_DIM_PER_ARM = 8

# Strict bar — near-machine-precision, does not relax for smoke-scale datasets.
_DEFAULT_ATOL_POS_M = 1e-6
_DEFAULT_ATOL_ROT_DEG = 1e-4


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _pos_slices(action_dim: int) -> list[tuple[int, int]]:
    n_arms = action_dim // _ACTION_DIM_PER_ARM
    return [(a * _ACTION_DIM_PER_ARM + _POS_SLICE[0], a * _ACTION_DIM_PER_ARM + _POS_SLICE[1]) for a in range(n_arms)]


def _rot_slices(action_dim: int) -> list[tuple[int, int]]:
    n_arms = action_dim // _ACTION_DIM_PER_ARM
    return [(a * _ACTION_DIM_PER_ARM + _ROT_SLICE[0], a * _ACTION_DIM_PER_ARM + _ROT_SLICE[1]) for a in range(n_arms)]


def _rot6d_angle_diff_deg(r6d_a: np.ndarray, r6d_b: np.ndarray) -> np.ndarray:
    """Angular difference (deg) between two rot6d-encoded rotations, elementwise batched."""
    from anvil_shared.rotation import matrices_to_quats, rot6ds_to_matrices

    Ra = rot6ds_to_matrices(r6d_a)
    Rb = rot6ds_to_matrices(r6d_b)
    Rdiff = np.einsum("...ij,...kj->...ik", Ra, Rb)  # Ra @ Rb.T
    q = matrices_to_quats(Rdiff)
    ang = 2.0 * np.arccos(np.clip(np.abs(q[..., 3]), 0.0, 1.0))
    return np.degrees(ang)


def _max_pos_rot_err(recon: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    action_dim = recon.shape[-1]
    pos_errs = [
        np.linalg.norm(recon[..., p0:p1] - expected[..., p0:p1], axis=-1)
        for p0, p1 in _pos_slices(action_dim)
    ]
    rot_errs = [
        _rot6d_angle_diff_deg(recon[..., r0:r1], expected[..., r0:r1])
        for r0, r1 in _rot_slices(action_dim)
    ]
    max_pos = float(np.concatenate(pos_errs).max()) if pos_errs else 0.0
    max_rot = float(np.concatenate(rot_errs).max()) if rot_errs else 0.0
    return max_pos, max_rot


def _detect_encoding(dataset_path: Path) -> str:
    """Read conversion_config.yaml's action_encoding, default 'absolute' if absent.

    Goes through ConfigLoader (lenient — this is a frozen historical record, not a config
    being actively authored; strict unknown-key rejection would make old datasets
    unreadable rather than more robust) instead of a raw single-key yaml.safe_load, so a
    pre-rename dataset (schema v1.0, still saying ee_action_encoding) is transparently
    migrated to the current field name rather than silently misread as "absolute" under a
    name it never had.
    """
    from mcap_converter.config.loader import ConfigLoader

    cfg_path = dataset_path / "conversion_config.yaml"
    if not cfg_path.exists():
        log.warning(
            "[gt-replay] %s not found — assuming encoding='absolute' "
            "(pass --encoding explicitly to override)",
            cfg_path,
        )
        return "absolute"

    config = ConfigLoader.from_yaml(str(cfg_path), strict=False)
    log.info(
        "[gt-replay] Detected action_encoding=%r from %s", config.action_encoding, cfg_path
    )
    return config.action_encoding


def _load_episode_arrays(dataset_root: Path, episode_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (action, observation.state) arrays for one episode, sorted by frame_index.

    Reads parquet directly (not through LeRobotDataset) to keep this tool
    independent of any training-time transform — we want the RAW on-disk
    values, exactly as mcap_converter wrote them.
    """
    import pandas as pd

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        log.error("[gt-replay] No parquet files found under %s", data_dir)
        return None

    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["episode_index", "frame_index", "action", "observation.state"])
        sub = df[df["episode_index"] == episode_idx]
        if len(sub):
            frames.append(sub)
    if not frames:
        return None

    sub = pd.concat(frames).sort_values("frame_index")
    actions = np.stack(sub["action"].to_numpy()).astype(np.float64)
    states = np.stack(sub["observation.state"].to_numpy()).astype(np.float64)
    return actions, states


def _replay_episode_absolute(actions: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    """Pure transform-math round-trip: ee_delta_forward -> ee_delta_inverse."""
    from anvil_shared.ee_transform import ee_delta_forward, ee_delta_inverse

    T = len(actions)
    if T < 2:
        return {"skipped": True, "reason": f"episode has only {T} frame(s), need >=2"}

    anchors = states[:-1]
    gt = actions[1:]
    delta = ee_delta_forward(gt, anchors)
    recon = ee_delta_inverse(delta, anchors)
    max_pos, max_rot = _max_pos_rot_err(recon, gt)
    return {"skipped": False, "max_pos_err_m": max_pos, "max_rot_err_deg": max_rot, "n_frames_checked": T - 1}


def _replay_episode_delta(actions: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    """Baked on-disk column correctness: recompute expected delta from state, compare to actual."""
    from anvil_shared.ee_transform import ee_delta_forward, ee_obs_abs_forward

    T = len(actions)
    if T < 1:
        return {"skipped": True, "reason": "episode has 0 frames"}

    action_dim = actions.shape[-1]

    # First-frame self-anchor convention: action[0] should already equal the
    # identity-rot6d / zero-xyz delta (gripper passthrough), per Item 1's
    # documented first-frame choice.
    identity_rot6d = np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], action_dim // _ACTION_DIM_PER_ARM)
    first_pos = np.concatenate([actions[0, p0:p1] for p0, p1 in _pos_slices(action_dim)])
    first_rot = np.concatenate([actions[0, r0:r1] for r0, r1 in _rot_slices(action_dim)])
    first_frame_pos_err = float(np.max(np.abs(first_pos - 0.0))) if len(first_pos) else 0.0
    first_frame_rot_err = float(np.max(np.abs(first_rot - identity_rot6d))) if len(first_rot) else 0.0

    if T < 2:
        return {
            "skipped": False,
            "n_frames_checked": 0,
            "first_frame_pos_err_m": first_frame_pos_err,
            "first_frame_rot_l2_err": first_frame_rot_err,
            "max_pos_err_m": 0.0,
            "max_rot_err_deg": 0.0,
        }

    action_abs = ee_obs_abs_forward(states)  # (T, action_dim) reconstructed from state alone
    anchors = states[:-1]
    expected = ee_delta_forward(action_abs[1:], anchors)
    actual = actions[1:]
    max_pos, max_rot = _max_pos_rot_err(actual, expected)
    return {
        "skipped": False,
        "n_frames_checked": T - 1,
        "first_frame_pos_err_m": first_frame_pos_err,
        "first_frame_rot_l2_err": first_frame_rot_err,
        "max_pos_err_m": max_pos,
        "max_rot_err_deg": max_rot,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GT-replay: model-free validation of the EE Delta(n-(n-1)) transform "
            "pipeline. The first gate a delta-flow dataset must pass before training."
        )
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to the LeRobot v3.0 dataset directory")
    parser.add_argument(
        "--episodes", type=str, default=None,
        help="Comma-separated episode indices to check (default: all episodes)",
    )
    parser.add_argument(
        "--encoding", choices=["auto", "absolute", "delta"], default="auto",
        help="'absolute': pure transform-math round-trip. 'delta': baked on-disk "
             "column correctness. 'auto' (default): read conversion_config.yaml.",
    )
    parser.add_argument("--atol-pos", type=float, default=_DEFAULT_ATOL_POS_M,
                         help=f"Position tolerance in meters (default: {_DEFAULT_ATOL_POS_M:.0e})")
    parser.add_argument("--atol-rot-deg", type=float, default=_DEFAULT_ATOL_ROT_DEG,
                         help=f"Rotation tolerance in degrees (default: {_DEFAULT_ATOL_ROT_DEG:.0e})")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        log.error("[gt-replay] Dataset path does not exist: %s", dataset_root)
        sys.exit(1)

    encoding = args.encoding if args.encoding != "auto" else _detect_encoding(dataset_root)
    if encoding not in ("absolute", "delta"):
        log.error("[gt-replay] Unknown encoding %r (must be 'absolute' or 'delta')", encoding)
        sys.exit(1)

    import json

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        log.error("[gt-replay] %s not found — is this a valid LeRobot v3.0 dataset?", info_path)
        sys.exit(1)
    info = json.loads(info_path.read_text())
    total_episodes = info.get("total_episodes", 0)

    if args.episodes:
        episode_indices = [int(e.strip()) for e in args.episodes.split(",") if e.strip()]
    else:
        episode_indices = list(range(total_episodes))

    log.info(
        "[gt-replay] dataset=%s encoding=%s episodes=%d atol_pos=%.1e m atol_rot=%.1e deg",
        dataset_root, encoding, len(episode_indices), args.atol_pos, args.atol_rot_deg,
    )

    replay_fn = _replay_episode_absolute if encoding == "absolute" else _replay_episode_delta

    results: list[dict[str, Any]] = []
    any_fail = False
    for ep in episode_indices:
        arrays = _load_episode_arrays(dataset_root, ep)
        if arrays is None:
            log.warning("[gt-replay] ep %d: no frames found, skipping", ep)
            continue
        actions, states = arrays
        result = replay_fn(actions, states)
        result["episode"] = ep

        if result.get("skipped"):
            log.info("[gt-replay] ep %d: SKIPPED (%s)", ep, result.get("reason"))
            results.append(result)
            continue

        pos_ok = result["max_pos_err_m"] <= args.atol_pos
        rot_ok = result["max_rot_err_deg"] <= args.atol_rot_deg
        first_frame_ok = True
        if encoding == "delta":
            first_frame_ok = (
                result["first_frame_pos_err_m"] <= args.atol_pos
                and result["first_frame_rot_l2_err"] <= args.atol_rot_deg * 1e-2
                # rot6d L2 tolerance is unit-less; use a tight but not identical
                # threshold to the angular-degree tolerance above.
            )
        passed = pos_ok and rot_ok and first_frame_ok
        result["passed"] = passed
        any_fail = any_fail or not passed

        status = "PASS" if passed else "FAIL"
        extra = ""
        if encoding == "delta":
            extra = (
                f", first_frame_pos_err={result['first_frame_pos_err_m']:.3e} m"
                f", first_frame_rot_l2_err={result['first_frame_rot_l2_err']:.3e}"
            )
        log.info(
            "[gt-replay] ep %d (%d frames checked): %s — max_pos_err=%.3e m, max_rot_err=%.3e deg%s",
            ep, result["n_frames_checked"], status,
            result["max_pos_err_m"], result["max_rot_err_deg"], extra,
        )
        results.append(result)

    checked = [r for r in results if not r.get("skipped")]
    n_pass = sum(1 for r in checked if r.get("passed"))
    n_total = len(checked)
    log.info("[gt-replay] SUMMARY: %d/%d episodes passed (encoding=%s)", n_pass, n_total, encoding)

    if n_total == 0:
        log.error("[gt-replay] No episodes were actually checked — treating as FAIL.")
        sys.exit(1)

    if any_fail:
        log.error("[gt-replay] GATE FAILED — do not proceed to training.")
        sys.exit(1)

    log.info("[gt-replay] GATE PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()

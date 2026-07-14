"""Real-data math identities for the libero_ee study's ``validate-math`` stage.

Each validator loads a dataset group's stored actions/states and asserts the
convert-time construction round-trips to the native command (or the achieved
trajectory) to float precision — the data-level form of the closed-loop
GT-replay oracle. If an identity fails, the dataset construction is wrong and
no amount of training will help, so the bench runner runs these BEFORE any
training. Moved out of the harness (``bench_runner``) so the runner stays
study-agnostic; :data:`MATH_VALIDATORS` is consumed via
``spec.study.math_validators``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anvil_sim.bench_spec import BenchSpec

_MATH_TOLERANCE = 1e-4  # max abs error allowed by the identities


def _native_dataset_root(spec: BenchSpec) -> Path:
    return spec.study.dataset_root(spec.study.baseline_group, spec.task_index)


def _load_local_episode(root: Path, episode: int = 0) -> dict[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local", root=str(root))
    hf = ds.hf_dataset.select_columns(["episode_index", "action", "observation.state"]).with_format(None)
    actions, states = [], []
    for ep, act, sta in zip(hf["episode_index"], hf["action"], hf["observation.state"], strict=True):
        if int(ep) == episode:
            actions.append(np.asarray(act, dtype=np.float64))
            states.append(np.asarray(sta, dtype=np.float64))
    return {"action": np.stack(actions), "state": np.stack(states)}


def _validate_goalabs(spec: BenchSpec) -> dict:
    """goalabs family identity: recovering a native-delta from the stored
    formal goal against the SAME state must reproduce the native dataset's
    own command — pos/rot to float precision, gripper via native_cmd
    passthrough (the dimension bug #4 hid in; validated explicitly now)."""
    from anvil_sim.studies.libero_ee.libero_processor import recovered_delta_native_action

    goal = _load_local_episode(spec.dataset_root)
    native = _load_local_episode(_native_dataset_root(spec))
    n = min(len(goal["action"]), len(native["action"]))
    max_err = 0.0
    for t in range(n):
        act10, state8 = goal["action"][t], goal["state"][t]
        recovered = recovered_delta_native_action(
            reconstructed_pos=act10[:3],
            reconstructed_rot6d=act10[3:9],
            reconstructed_gripper=float(act10[9]),
            current_state=state8.astype(np.float32),
            current_gripper=float(state8[7]),
            gripper_mode="native_cmd",
        )
        expected = np.clip(native["action"][t], -1.0, 1.0)
        max_err = max(max_err, float(np.abs(recovered - expected).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"goalabs identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {"identity": "goalabs->native command", "frames": n, "max_err": max_err}


def _validate_native_abs(spec: BenchSpec) -> dict:
    """native_abs identity (NATIVE-family axis-angle absolute goal): decoding
    the stored 7-dim axis-angle goal back to rot6d and recovering a native
    delta against the frame's OWN state (converted native->quat, since the obs
    column is now native 8-dim axis-angle) must reproduce the native dataset's
    own command to float precision. Same round-trip guarantee as
    _validate_goalabs, but for the native-family observation."""
    from anvil_sim.studies.libero_ee.libero_convert import raw_state_to_anvil
    from anvil_sim.studies.libero_ee.libero_processor import (
        axis_angle_action_to_rot6d,
        recovered_delta_native_action,
    )

    goal = _load_local_episode(spec.dataset_root)
    native = _load_local_episode(_native_dataset_root(spec))
    n = min(len(goal["action"]), len(native["action"]))
    max_err = 0.0
    for t in range(n):
        anvil8 = raw_state_to_anvil(goal["state"][t].astype(np.float32))
        rot6d10 = axis_angle_action_to_rot6d(goal["action"][t])
        recovered = recovered_delta_native_action(
            reconstructed_pos=rot6d10[:3],
            reconstructed_rot6d=rot6d10[3:9],
            reconstructed_gripper=float(rot6d10[9]),
            current_state=anvil8,
            current_gripper=float(anvil8[7]),
            gripper_mode="native_cmd",
        )
        expected = np.clip(native["action"][t], -1.0, 1.0)
        max_err = max(max_err, float(np.abs(recovered - expected).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"native_abs identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {"identity": "native_abs goal -> native command", "frames": n, "max_err": max_err}


def _validate_native_ctrlgoal(spec: BenchSpec) -> dict:
    """native_ctrlgoal identity: decoding the stored 7-dim axis-angle
    controller-goal back to rot6d and inverting the SCALED construction
    (subtract the frame's own state, then divide by robosuite's own
    ``output_max``) must reproduce the native dataset's own command to float
    precision. Unlike ``_validate_native_abs`` (which recovers via
    ``recovered_delta_native_action``'s UNSCALED subtraction, correct for the
    formal goal family), this identity divides by ``OSC_OUTPUT_MAX_POS``/
    ``OSC_OUTPUT_MAX_ROT`` since ``native_delta_to_ctrlgoal``'s goal is
    genuinely scaled, not a formal composition. Tautological w.r.t.
    ``state[t]``, like the other validators here — proves convert math, not
    rollout robustness (see ``research/libero_ee/stage1-closeout.md``)."""
    from anvil_shared.rotation import matrix_to_axis_angle, quat_to_matrix, rot6d_to_matrix

    from anvil_sim.studies.libero_ee.libero_convert import (
        OSC_OUTPUT_MAX_POS,
        OSC_OUTPUT_MAX_ROT,
        raw_state_to_anvil,
    )
    from anvil_sim.studies.libero_ee.libero_processor import axis_angle_action_to_rot6d

    goal = _load_local_episode(spec.dataset_root)
    native = _load_local_episode(_native_dataset_root(spec))
    n = min(len(goal["action"]), len(native["action"]))
    max_err = 0.0
    for t in range(n):
        anvil8 = raw_state_to_anvil(goal["state"][t].astype(np.float32))
        rot6d10 = axis_angle_action_to_rot6d(goal["action"][t].astype(np.float32))
        recovered_pos_delta = (rot6d10[:3] - anvil8[:3]) / OSC_OUTPUT_MAX_POS
        R_state = quat_to_matrix(anvil8[3:7].astype(np.float64))
        R_goal = rot6d_to_matrix(rot6d10[3:9].astype(np.float64))
        recovered_rot_delta = matrix_to_axis_angle(R_goal @ R_state.T) / OSC_OUTPUT_MAX_ROT
        recovered_gripper = float(rot6d10[9])
        recovered = np.concatenate([recovered_pos_delta, recovered_rot_delta, [recovered_gripper]])
        expected = np.clip(native["action"][t], -1.0, 1.0)
        max_err = max(max_err, float(np.abs(recovered - expected).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"native_ctrlgoal identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {
        "identity": "native_ctrlgoal goal -> native command (scaled inverse)",
        "frames": n,
        "max_err": max_err,
    }


def _validate_native_n0(spec: BenchSpec) -> dict:
    """native_n0 identity (NATIVE-family n-0 relativized goal): un-relativizing
    the stored 7-dim axis-angle action against the frame's OWN (native->quat)
    state via ee_rel_world_inverse, then recovering a native delta against that
    same state, must reproduce the native dataset's own command to float
    precision — the data-level form of the n-0 GT-replay oracle at
    n_action_steps=1 (anchor == current state)."""
    from anvil_shared.ee_transform import ee_rel_world_inverse

    from anvil_sim.studies.libero_ee.libero_convert import raw_state_to_anvil
    from anvil_sim.studies.libero_ee.libero_processor import (
        axis_angle_action_to_rot6d,
        recovered_delta_native_action,
    )

    goal = _load_local_episode(spec.dataset_root)
    native = _load_local_episode(_native_dataset_root(spec))
    n = min(len(goal["action"]), len(native["action"]))
    max_err = 0.0
    for t in range(n):
        anvil8 = raw_state_to_anvil(goal["state"][t].astype(np.float32))
        rel10 = axis_angle_action_to_rot6d(goal["action"][t])
        abs10 = ee_rel_world_inverse(rel10.reshape(1, 10), anvil8.reshape(1, 8))[0]
        recovered = recovered_delta_native_action(
            reconstructed_pos=abs10[:3],
            reconstructed_rot6d=abs10[3:9],
            reconstructed_gripper=float(abs10[9]),
            current_state=anvil8,
            current_gripper=float(anvil8[7]),
            gripper_mode="native_cmd",
        )
        expected = np.clip(native["action"][t], -1.0, 1.0)
        max_err = max(max_err, float(np.abs(recovered - expected).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"native_n0 identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {"identity": "native_n0 relativized goal -> native command", "frames": n, "max_err": max_err}


def _validate_native_hand(spec: BenchSpec) -> dict:
    """native_hand identity (world->hand->world round-trip, at the data
    level): rotating each stored hand-frame command back to the world frame
    (via hand_action_to_native, using that frame's OWN EE axis-angle from
    observation.state[3:6]) must reproduce the native dataset's own command
    to float precision — because the eval rotate-back is the exact inverse of
    the convert-time rotation with the same per-step EE orientation. This is
    the data-level form of the closed-loop GT-replay oracle; if it fails, the
    frame convention is wrong and no amount of training will help."""
    from anvil_sim.studies.libero_ee.libero_processor import hand_action_to_native

    hand = _load_local_episode(spec.dataset_root)
    native = _load_local_episode(_native_dataset_root(spec))
    n = min(len(hand["action"]), len(native["action"]))
    max_err = 0.0
    for t in range(n):
        recovered = hand_action_to_native(hand["action"][t], hand["state"][t][3:6])
        max_err = max(max_err, float(np.abs(recovered - native["action"][t]).max()))
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"native_hand identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {"identity": "hand->world command == native", "frames": n, "max_err": max_err}


def _validate_afo_abs(spec: BenchSpec) -> dict:
    """afo_abs identity: unlike native_abs/native_n0/goalabs (formal
    ``state + native_delta`` compositions, validated against the NATIVE
    dataset's own recorded command), afo_abs's target is the REAL OBSERVED
    future pose -- so its identity is decoding the stored 7-dim axis-angle
    action back to rot6d and confirming the POSITION/ROTATION channels
    reproduce the native dataset's OWN state trajectory ``horizon`` frames
    ahead (read from the native dataset, which has the full untruncated
    episode -- afo_abs's own state/action columns are `horizon` frames
    shorter per episode, see convert_episode_afo_abs_actions). The GRIPPER
    channel is checked separately against the native dataset's RECORDED
    command (the UMI decomposition -- gripper is never obs-derived)."""
    from anvil_sim.studies.libero_ee.libero_convert import (
        anvil_state_to_abs_action,
        raw_state_to_anvil,
    )
    from anvil_sim.studies.libero_ee.libero_processor import axis_angle_action_to_rot6d

    horizon = int(spec.dataset_group.rsplit("_h", 1)[1])
    afo = _load_local_episode(spec.dataset_root)
    native = _load_local_episode(_native_dataset_root(spec))
    n = len(afo["action"])
    if len(native["action"]) < n + horizon:
        raise RuntimeError(
            f"afo_abs (h={horizon}) identity: native episode too short "
            f"({len(native['action'])} frames) to check {n} afo frames at horizon {horizon}"
        )
    max_err = 0.0
    for t in range(n):
        rot6d10 = axis_angle_action_to_rot6d(afo["action"][t].astype(np.float32))
        future_anvil8 = raw_state_to_anvil(native["state"][t + horizon].astype(np.float32))
        expected10 = anvil_state_to_abs_action(future_anvil8)
        pose_err = float(np.abs(rot6d10[:9] - expected10[:9]).max())
        gripper_err = abs(float(afo["action"][t][6]) - float(native["action"][t][6]))
        max_err = max(max_err, pose_err, gripper_err)
    if max_err > _MATH_TOLERANCE:
        raise RuntimeError(f"afo_abs (h={horizon}) identity failed: max_err={max_err:.2e} > {_MATH_TOLERANCE}")
    return {
        "identity": f"afo_abs h={horizon} -> observed future pose + native gripper",
        "frames": n,
        "max_err": max_err,
    }


MATH_VALIDATORS = {
    "goalabs": _validate_goalabs,
    "native_abs": _validate_native_abs,
    "native_n0": _validate_native_n0,
    "native_hand": _validate_native_hand,
    "native_ctrlgoal": _validate_native_ctrlgoal,
    "afo_abs_h1": _validate_afo_abs,
    "afo_abs_h5": _validate_afo_abs,
    "afo_abs_h10": _validate_afo_abs,
}

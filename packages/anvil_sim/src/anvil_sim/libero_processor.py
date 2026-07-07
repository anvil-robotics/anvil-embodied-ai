"""ProcessorSteps letting Anvil ee_abs/ee_rel policies run against LIBERO's
native environment via ``lerobot.scripts.lerobot_eval.rollout()``'s
``env_preprocessor``/``env_postprocessor`` hooks.

No custom gym env is needed: arms B (ee_abs) and C (ee_rel) share the exact
same native ``LiberoEnv`` (``control_mode="relative"``) as arm A (native).
These two steps only translate the *format* crossing the policy/env
boundary — the sim's own OSC_POSE controller still does all the actual
low-level control.

Also home to :class:`NativeRot6dActionProcessorStep`, used only by the
experimental 5th arm ``native_rot6d`` (see ``anvil_sim.libero_convert``
module docstring) — a zero-calibration isolation of rot6d vs axis-angle
rotation encoding that needs no paired obs processor at all.

And :class:`ZeroCalActionProcessorStep`, used by the zero-calibration
re-run of the ee_abs/ee_rel/ee_delta ablations via
``env.control_mode="absolute"`` (see :func:`absolute_native_action_from_target`)
instead of the ``NATIVE_POS_SCALE``/``NATIVE_ROT_SCALE`` reconstruction the
other processors above use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from anvil_shared.ee_transform import (
    ee_obs_abs_forward,
    ee_obs_rel_forward,
    ee_rel_inverse,
    ee_rel_world_inverse,
)
from anvil_shared.rotation import (
    matrix_to_axis_angle,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
    rot6d_to_matrix,
)
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.pipeline import ActionProcessorStep, ObservationProcessorStep
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE

# =============================================================================
# Empirically-calibrated robosuite OSC_POSE native-delta convention
# =============================================================================
# Derived from lerobot/libero task_index=10 ("put the bowl on the plate"):
# least-squares fit of native_action = scale * real_delta across ~4500
# consecutive frame pairs (see project notes / TASK-006 for the calibration
# script). Position fits cleanly in the WORLD frame (R^2=0.92 vs 0.04 for
# body frame) -- this is the key empirical difference from Anvil's ee_rel
# (UMI-style), which explicitly rotates translation into the EE's own body
# frame (`body_delta = R_state.T @ world_delta`). Rotation is only weakly
# constrained by this specific task's data (its orientation barely changes:
# the 99th percentile rotation delta is ~0.019 rad) but is still better
# explained by the world frame (R^2=0.49 over all frames, 0.69 restricted to
# the larger-delta subset) than the body frame (R^2=0.11).
NATIVE_POS_SCALE = 82.7763  # native_action_xyz = NATIVE_POS_SCALE * world_delta_xyz (metres)
NATIVE_ROT_SCALE = 7.1523  # native_action_rot = NATIVE_ROT_SCALE * world_delta_axis_angle (radians)

# Gripper action is a saturating open/close command, not a scaled delta --
# confirmed via the env's own no-op action [0,0,0,0,0,0,-1] (hold pose, open
# gripper): -1.0 = open, +1.0 = close.
GRIPPER_OPEN_CMD = -1.0
GRIPPER_CLOSE_CMD = 1.0


def native_action_from_targets(
    target_pos: np.ndarray,
    target_rot6d: np.ndarray,
    target_gripper: float,
    current_pos: np.ndarray,
    current_quat_xyzw: np.ndarray,
    current_gripper: float,
) -> np.ndarray:
    """Convert an absolute Anvil EE target into LIBERO's native 7-dim delta action.

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    world_delta_xyz = target_pos - current_pos
    native_pos = world_delta_xyz * NATIVE_POS_SCALE

    R_current = quat_to_matrix(current_quat_xyzw)
    R_target = rot6d_to_matrix(target_rot6d)
    R_delta_world = R_target @ R_current.T
    world_delta_aa = matrix_to_axis_angle(R_delta_world)
    native_rot = world_delta_aa * NATIVE_ROT_SCALE

    # Push toward the target opening at max rate; robustly sign-agnostic
    # since a wider gripper opening means larger |gripper_qpos| regardless
    # of which finger's coordinate sign we track (see raw_state_to_anvil).
    native_gripper = GRIPPER_CLOSE_CMD if abs(target_gripper) < abs(current_gripper) else GRIPPER_OPEN_CMD

    return np.concatenate([native_pos, native_rot, [native_gripper]]).astype(np.float32)


def native_action_from_world_delta(
    delta_pos: np.ndarray,
    delta_rot6d: np.ndarray,
    target_gripper: float,
    current_gripper: float,
) -> np.ndarray:
    """Convert an already-world-frame-delta Anvil action into LIBERO's native
    7-dim delta action. Used only by the experimental ``ee_delta`` arm (see
    ``anvil_sim.libero_convert.anvil_state_to_delta_action``) — unlike
    :func:`native_action_from_targets`, there is no absolute target to
    reconstruct, so no ``current_pos``/``current_quat_xyzw`` composition is
    needed for position/rotation; ``current_gripper`` is only used for the
    bang-bang gripper comparison.

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    native_pos = delta_pos * NATIVE_POS_SCALE

    R_delta_world = rot6d_to_matrix(delta_rot6d)
    world_delta_aa = matrix_to_axis_angle(R_delta_world)
    native_rot = world_delta_aa * NATIVE_ROT_SCALE

    native_gripper = GRIPPER_CLOSE_CMD if abs(target_gripper) < abs(current_gripper) else GRIPPER_OPEN_CMD

    return np.concatenate([native_pos, native_rot, [native_gripper]]).astype(np.float32)


def absolute_native_action_from_target(
    target_pos: np.ndarray,
    target_rot6d: np.ndarray,
    target_gripper: float,
    current_gripper: float,
    gripper_mode: str = "target_qpos",
) -> np.ndarray:
    """Convert an absolute Anvil EE target directly into LIBERO's native
    7-dim action for ``env.control_mode="absolute"`` — ZERO calibration.

    Confirmed by reading robosuite's OSC controller source
    (``robosuite/controllers/osc.py::set_goal``): when ``use_delta=False``
    (``control_mode="absolute"``), position is used as-is (metres) and
    orientation is interpreted directly as an axis-angle vector (radians) —
    "No scaling of values since these are absolute values" per the source
    comment. This replaces :func:`native_action_from_targets`/
    :func:`native_action_from_world_delta`'s ``NATIVE_POS_SCALE``/
    ``NATIVE_ROT_SCALE`` reconstruction (whose rotation fit was only
    R²=0.49) for the zero-calibration re-run of the ee_abs/ee_rel/ee_delta
    ablations — see ``ZeroCalActionProcessorStep``.

    ``gripper_mode``: see :func:`recovered_delta_native_action` — the same
    qpos-target vs native-command semantics split applies here (the
    ``goalabs`` family stores +/-1 native commands, everything else stores
    qpos-scale targets).

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    native_pos = target_pos  # metres, absolute mode takes it directly, no scale
    native_rot = matrix_to_axis_angle(rot6d_to_matrix(target_rot6d))  # radians, no scale
    if gripper_mode == "native_cmd":
        native_gripper = float(np.clip(target_gripper, -1.0, 1.0))
    elif gripper_mode == "target_qpos":
        native_gripper = (
            GRIPPER_CLOSE_CMD if abs(target_gripper) < abs(current_gripper) else GRIPPER_OPEN_CMD
        )
    else:
        raise ValueError(f"gripper_mode must be 'target_qpos' or 'native_cmd', got {gripper_mode!r}")
    return np.concatenate([native_pos, native_rot, [native_gripper]]).astype(np.float32)


def recovered_delta_native_action(
    reconstructed_pos: np.ndarray,
    reconstructed_rot6d: np.ndarray,
    reconstructed_gripper: float,
    current_state: np.ndarray,
    current_gripper: float,
    gripper_mode: str = "target_qpos",
) -> np.ndarray:
    """Recover a native-command-scale delta from a FORMAL "state + native
    delta" reconstruction (see ``anvil_sim.libero_convert.native_delta_to_goal``),
    for delivery via ``env.control_mode="relative"`` — the fix for
    Experiment 7's v1 catastrophic failure (all 5 conditions, 0%
    pc_success).

    v1 reconstructed an absolute target by SCALING the formal quantity
    ourselves (``* [0.05]*3+[0.5]*3``, robosuite's OSC_POSE ``output_max``)
    and fed it to ``env.control_mode="absolute"``. That assumed the scaled
    quantity was close to what the arm would actually reach — false: real
    per-step displacement is only ~22% of that (impedance-controller lag),
    so the "goal" was ~4.5x too large and destabilized every rollout.

    The fix: don't scale here at all. Subtract the CURRENT REAL state
    (fresh, closed-loop — NOT the stale anchor used to reconstruct
    ``reconstructed_*``) to recover a delta that lives in the SAME formal
    numeric space as LIBERO's own native action column, clip it to
    ``[-1, 1]`` (matching robosuite's own ``scale_action`` clip), and let
    the environment's OWN ``scale_action`` apply whatever the true physical
    scale is when this is fed through with ``env.control_mode="relative"``.
    This never needs to know or guess that scale.

    For the ``abs`` condition (no anchor, obs/target share the same time
    index) this reduces to exactly ``recovered_delta = reconstructed -
    current_state`` — matching how LIBERO's own native delta is defined.
    For n-0/n-(n-1) conditions, ``reconstructed_*`` was already un-relativized
    against a real anchor (via ``ee_rel_world_inverse``/``ee_rel_inverse``,
    or a running accumulator) before being passed here; subtracting the
    CURRENT real state (which may differ from that anchor once execution has
    advanced past the first step of a chunk) is exactly the self-correction
    that makes n-0 anchoring closed-loop.

    ``gripper_mode`` selects how ``reconstructed_gripper`` is interpreted —
    a REAL BUG (the 4th of this benchmark, found by the GT-replay tool on
    its first run) hid here: the ``goalabs`` dataset family stores the
    gripper as LIBERO's native +/-1 COMMAND (``native_delta_to_goal``
    passes ``native_delta7[6]`` through), while ``abs``/``rel``/``delta``
    store a qpos-scale TARGET (~0.002-0.04). The original bang-bang
    comparator ``abs(target) < abs(current_qpos)`` is only meaningful for
    the qpos-scale convention; fed a +/-1 command it is ALWAYS False, so
    the gripper never closes and every goalabs rollout fails at 0%
    regardless of policy quality.

    - ``"target_qpos"`` (default, original behavior): bang-bang against the
      current gripper qpos — for abs/rel/delta-family targets.
    - ``"native_cmd"``: the value already IS the native command; clip to
      [-1, 1] and pass through — for goalabs-family targets.

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    current_pos = current_state[:3]
    current_R = quat_to_matrix(current_state[3:7])
    reconstructed_R = rot6d_to_matrix(reconstructed_rot6d)

    native_delta_pos = np.clip(reconstructed_pos - current_pos, -1.0, 1.0)
    native_delta_rot = np.clip(
        matrix_to_axis_angle(reconstructed_R @ current_R.T), -1.0, 1.0
    )
    if gripper_mode == "native_cmd":
        native_gripper = float(np.clip(reconstructed_gripper, -1.0, 1.0))
    elif gripper_mode == "target_qpos":
        native_gripper = (
            GRIPPER_CLOSE_CMD if abs(reconstructed_gripper) < abs(current_gripper) else GRIPPER_OPEN_CMD
        )
    else:
        raise ValueError(f"gripper_mode must be 'target_qpos' or 'native_cmd', got {gripper_mode!r}")
    return np.concatenate([native_delta_pos, native_delta_rot, [native_gripper]]).astype(np.float32)


def rot6d_action_to_native(rot6d_action10: np.ndarray) -> np.ndarray:
    """Exact inverse of ``anvil_sim.libero_convert.native_action_to_rot6d``
    — used by the experimental 5th arm ``native_rot6d``, which isolates
    rot6d rotation encoding with ZERO calibration/approximation error
    (unlike :func:`native_action_from_targets`/:func:`native_action_from_world_delta`,
    which both apply ``NATIVE_POS_SCALE``/``NATIVE_ROT_SCALE``). Position and
    gripper pass through unchanged; rotation is decoded via
    ``rot6d_to_matrix``/``matrix_to_axis_angle`` — exact, invertible math,
    with no notion of "current state" or "target" needed at all.

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    pos = rot6d_action10[:3]
    native_rot = matrix_to_axis_angle(rot6d_to_matrix(rot6d_action10[3:9]))
    gripper = rot6d_action10[9]
    return np.concatenate([pos, native_rot, [gripper]]).astype(np.float32)


def _extract_robot_state(observation: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull (eef_pos, eef_quat_xyzw, gripper_qpos) out of LIBERO's nested robot_state obs.

    Mirrors ``lerobot.processor.env_processor.LiberoProcessorStep``'s access
    pattern for the same raw observation structure.
    """
    robot_state = observation[OBS_PREFIX + "robot_state"]
    return robot_state["eef"]["pos"], robot_state["eef"]["quat"], robot_state["gripper"]["qpos"]


@dataclass
class AnvilEEObsProcessorStep(ObservationProcessorStep):
    """``env_preprocessor``: LIBERO's native obs -> the state format
    ``anvil_trainer`` actually trained the policy on.

    Mirrors ``LiberoProcessorStep`` for image handling (the same 180-degree
    flip — a documented quirk of this env/dataset family) but replaces the
    state encoding with Anvil's, AND applies the same
    ``anvil_trainer.transforms.EEAbsTransform``/``EERelTransform``
    observation re-encoding used at training time — the policy was NOT
    trained on the raw 8-dim quat state, it was trained on:

    - ``ee_abs`` / ``ee_delta``: ``ee_obs_abs_forward(state8)`` -> 10-dim
      rot6d, absolute. Both use the same observation encoding — they only
      differ in what the ACTION represents (see
      :class:`AnvilEEActionProcessorStep`). ``ee_delta`` is an experimental
      4th arm (see ``anvil_sim.libero_convert``), not part of Anvil's real
      contract.
    - ``ee_rel``: ``ee_obs_rel_forward(state8, anchor=state8)`` -> 10-dim
      rot6d, self-anchored. Since the anchor IS the current observation, this
      is always the fixed vector ``[0,0,0, 1,0,0,0,1,0, gripper]`` (zero
      translation, identity rotation) — only the gripper term carries
      information. That is intentional and matches training exactly
      (``EERelTransform`` anchors every single-step observation to itself).

    ``last_anvil_state`` (the raw absolute 8-dim quat state, NOT the encoded
    10-dim one above) is cached after every call so a paired
    :class:`AnvilEEActionProcessorStep` can use it as the reference state for
    ``ee_rel_inverse`` and for computing the native-delta action. Assumes
    batch size 1 (this benchmark doesn't vectorize envs).
    """

    action_type: str
    last_anvil_state: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.action_type not in ("ee_abs", "ee_rel", "ee_delta"):
            raise ValueError(
                f"action_type must be 'ee_abs', 'ee_rel', or 'ee_delta', got {self.action_type!r}"
            )

    def observation(self, observation: dict) -> dict:
        processed = observation.copy()
        # Rename LIBERO's native camera keys to the names used when we wrote
        # the training dataset (libero_convert.py: "agentview"/"wrist").
        rename = {
            f"{OBS_IMAGES}.image": f"{OBS_IMAGES}.agentview",
            f"{OBS_IMAGES}.image2": f"{OBS_IMAGES}.wrist",
        }
        for old_key, new_key in rename.items():
            if old_key in processed:
                processed[new_key] = processed.pop(old_key)

        for key in list(processed.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                processed[key] = torch.flip(processed[key], dims=[2, 3])

        eef_pos, eef_quat, gripper_qpos = _extract_robot_state(processed)
        processed.pop(OBS_PREFIX + "robot_state", None)

        pos = (eef_pos[0] if eef_pos.dim() > 1 else eef_pos).numpy()
        quat = (eef_quat[0] if eef_quat.dim() > 1 else eef_quat).numpy()
        gripper_vec = gripper_qpos[0] if gripper_qpos.dim() > 1 else gripper_qpos
        gripper = float(gripper_vec[0])

        state8 = np.concatenate([pos, quat, [gripper]]).astype(np.float64)
        self.last_anvil_state = state8.astype(np.float32)

        if self.action_type == "ee_rel":
            state10 = ee_obs_rel_forward(state8.reshape(1, 8), state8.reshape(1, 8))[0]
        else:  # ee_abs and ee_delta share the same absolute obs encoding
            state10 = ee_obs_abs_forward(state8.reshape(1, 8))[0]

        processed[OBS_STATE] = torch.from_numpy(state10.astype(np.float32)).unsqueeze(0)
        return processed

    def transform_features(self, features):
        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != FeatureType.STATE}
        new_features[FeatureType.STATE] = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))}
        return new_features


@dataclass
class AnvilEEActionProcessorStep(ActionProcessorStep):
    """``env_postprocessor``: policy's 10-dim rot6d action -> LIBERO's native 7-dim action.

    ``action_type`` is ``"ee_abs"``, ``"ee_rel"``, or the experimental
    ``"ee_delta"`` (not part of Anvil's real contract — see
    ``anvil_sim.libero_convert`` module docstring). For ``ee_rel`` the
    policy output is first restored to absolute via
    ``anvil_shared.ee_transform.ee_rel_inverse``. For ``ee_delta`` the policy
    output already IS a world-frame delta (no absolute target to
    reconstruct, no chunk anchor needed) — see
    :func:`native_action_from_world_delta`.

    IMPORTANT — the ``ee_rel_inverse`` reference state must be the
    observation the model actually conditioned on when it predicted this
    action's *chunk*, not the observation at the moment this individual
    action is executed. ``ACTPolicy.select_action`` only calls
    ``predict_action_chunk`` when its internal ``_action_queue`` is empty
    (every ``n_action_steps`` calls); the other ``n_action_steps - 1`` calls
    just pop a cached action without looking at the current observation at
    all. Since training relativizes a whole action chunk against the single
    observation at the chunk's start (``anvil_trainer.transforms.EERelTransform``),
    using the fresh per-step observation as the reference here silently
    drifts more and more within each chunk — this exactly matches the real
    ``ee_runtime.ee_rel_restore_chunk(chunk, obs_t)`` contract, which also
    takes one reference observation for the whole chunk, not one per action.
    We can't see the policy's internal queue from here, so we track the
    chunk boundary ourselves via a call counter synced to ``n_action_steps``.
    """

    action_type: str
    obs_step: AnvilEEObsProcessorStep
    n_action_steps: int = 1

    _call_count: int = field(default=0, init=False, repr=False)
    _chunk_anchor: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.action_type not in ("ee_abs", "ee_rel", "ee_delta"):
            raise ValueError(
                f"action_type must be 'ee_abs', 'ee_rel', or 'ee_delta', got {self.action_type!r}"
            )

    def reset_episode_state(self) -> None:
        """Re-align chunk tracking with the policy's own replan schedule —
        MUST be called at every episode start (real bug #5): the policy
        replans at episode-local step 0 after ``policy.reset()``, but this
        step's call counter used to run on ACROSS episodes, so unless an
        episode's length happened to be a multiple of ``n_action_steps``,
        every episode after the first reconstructed targets against a chunk
        anchor captured at the WRONG time (initially even one from the
        previous episode's scene). See ``eval_libero_ee``'s rollout wrapper.
        """
        self._call_count = 0
        self._chunk_anchor = None

    def action(self, action: torch.Tensor) -> torch.Tensor:
        if self.obs_step.last_anvil_state is None:
            raise RuntimeError(
                "AnvilEEActionProcessorStep needs an observation processed by its "
                "paired AnvilEEObsProcessorStep before the first action."
            )
        current_state = self.obs_step.last_anvil_state

        if self._call_count % self.n_action_steps == 0:
            self._chunk_anchor = current_state.copy()
        self._call_count += 1

        act10 = (action[0] if action.dim() > 1 else action).detach().cpu().numpy()

        if self.action_type == "ee_rel":
            act10 = ee_rel_inverse(act10.reshape(1, 10), self._chunk_anchor.reshape(1, 8))[0]
            native = native_action_from_targets(
                target_pos=act10[:3],
                target_rot6d=act10[3:9],
                target_gripper=float(act10[9]),
                current_pos=current_state[:3],
                current_quat_xyzw=current_state[3:7],
                current_gripper=float(current_state[7]),
            )
        elif self.action_type == "ee_delta":
            # Already a world-frame delta -- no absolute target to
            # reconstruct, no chunk anchor involved.
            native = native_action_from_world_delta(
                delta_pos=act10[:3],
                delta_rot6d=act10[3:9],
                target_gripper=float(act10[9]),
                current_gripper=float(current_state[7]),
            )
        else:  # ee_abs
            # Converting the absolute target into a native step-delta DOES
            # use the fresh current physical state, not the chunk anchor —
            # this is "how far do I need to move from here-now", unrelated
            # to how the model's own output was originally referenced.
            native = native_action_from_targets(
                target_pos=act10[:3],
                target_rot6d=act10[3:9],
                target_gripper=float(act10[9]),
                current_pos=current_state[:3],
                current_quat_xyzw=current_state[3:7],
                current_gripper=float(current_state[7]),
            )
        return torch.from_numpy(native).unsqueeze(0)

    def transform_features(self, features):
        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != FeatureType.ACTION}
        new_features[FeatureType.ACTION] = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        return new_features


@dataclass
class NativeRot6dActionProcessorStep(ActionProcessorStep):
    """``env_postprocessor`` for the experimental 5th arm ``native_rot6d``:
    policy's 10-dim action (native's own position/gripper unchanged, rotation
    packed as rot6d) -> LIBERO's native 7-dim action, via
    :func:`rot6d_action_to_native`.

    Unlike :class:`AnvilEEActionProcessorStep`, this needs no paired obs
    processor, no ``current_state``, and no chunk-anchor tracking — the
    decoding is a pure, self-contained format conversion with zero
    calibration. The observation side for this arm reuses lerobot's own
    stock ``LiberoProcessorStep`` unchanged (see ``eval_native_rot6d.py``),
    since this arm's ``observation.state`` is byte-identical to native's.
    """

    def action(self, action: torch.Tensor) -> torch.Tensor:
        act10 = (action[0] if action.dim() > 1 else action).detach().cpu().numpy()
        native = rot6d_action_to_native(act10)
        return torch.from_numpy(native).unsqueeze(0)

    def transform_features(self, features):
        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != FeatureType.ACTION}
        new_features[FeatureType.ACTION] = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        return new_features


_ZERO_CAL_MODES = ("abs", "rel_world", "rel_hand", "rel_world_seq", "rel_hand_seq")


@dataclass
class ZeroCalActionProcessorStep(ActionProcessorStep):
    """``env_postprocessor`` for the zero-calibration re-run of the
    ee_abs/ee_rel/ee_delta ablations, using ``env.control_mode="absolute"``
    (see :func:`absolute_native_action_from_target`) instead of
    :func:`native_action_from_targets`/:func:`native_action_from_world_delta`'s
    ``NATIVE_POS_SCALE``/``NATIVE_ROT_SCALE`` reconstruction. Pair with
    :class:`AnvilEEObsProcessorStep` using ``action_type="ee_abs"`` (for
    ``mode="abs"``/``"rel_world_seq"``) or ``action_type="ee_rel"`` (for
    ``mode="rel_world"``/``"rel_hand"`` — the self-anchored single-step obs
    encoding is frame-invariant, see ``test_obs_rel_world_matches_body_frame_when_self_anchored``,
    so ``ee_rel``'s existing obs encoding is reused unchanged for both).

    Five modes. ``"abs"``/``"rel_world"``/``"rel_hand"``/``"rel_world_seq"``
    were the 4 control-variable conditions of the original (zero-cal,
    act-from-obs) 6th-round experiment; ``"rel_hand_seq"`` was added for the
    7th-round "goal" target family (see ``libero_convert.py``'s module
    docstring) to complete the 2x2 (world/hand x n-0/n-(n-1)) grid:

    - ``"abs"``: policy output IS the absolute target directly.
    - ``"rel_world"``: SE(3)-relative, WORLD frame, anchored to chunk start
      (n-0) — reconstruct via ``ee_rel_world_inverse``
      (``--action-type=ee_rel_world``).
    - ``"rel_hand"``: SE(3)-relative, HAND frame, anchored to chunk start
      (n-0) — reconstruct via ``ee_rel_inverse``
      (``--action-type=ee_rel``).
    - ``"rel_world_seq"``: per-step CONSECUTIVE world-frame delta (n-(n-1)).
      No fixed anchor reconstruction is possible (each step's target
      depends on the previous step's reconstructed target, not the
      original chunk-start anchor), so this mode maintains a running
      absolute target, accumulated one step at a time and reset at chunk
      boundaries: ``new_pos = running_pos + delta_pos``,
      ``new_R = R_delta @ R_running`` (see
      ``anvil_sim.libero_convert.anvil_state_to_delta_action``'s
      ``R_delta = R_next @ R_current.T`` convention).
    - ``"rel_hand_seq"``: per-step CONSECUTIVE HAND-frame delta (n-(n-1)) —
      same running-target accumulation as ``"rel_world_seq"``, but the
      delta is expressed in the PREVIOUS running target's own frame:
      ``new_pos = running_pos + R_running @ delta_pos``,
      ``new_R = R_running @ R_delta`` (see
      ``anvil_sim.libero_convert.convert_episode_goal_handseq_actions``'s
      ``R_delta = R_prev.T @ R_next`` convention).

    ``deliver`` controls the FINAL step, after ``target_pos``/``target_rot6d``/
    ``target_gripper`` have been reconstructed by the branches above:

    - ``"absolute"`` (default): :func:`absolute_native_action_from_target`
      — feed the reconstructed target directly to
      ``env.control_mode="absolute"``, zero scaling. This is the ORIGINAL
      6th-round zero-cal behavior, unchanged, still used by the
      ``zerocal_abs``/``zerocal_rel_world``/``zerocal_rel_hand``/
      ``zerocal_rel_world_seq`` action types (numbers preserved).
    - ``"relative"``: :func:`recovered_delta_native_action` — subtract the
      CURRENT real state from the reconstructed target to recover a
      native-command-scale delta, clip to ``[-1, 1]``, and feed to
      ``env.control_mode="relative"``, letting robosuite's own
      ``scale_action`` apply the true physical scale. Used by the 7th-round
      ``zerocal_goal_*`` "goal" family (see ``libero_convert.py``'s module
      docstring) — reconstructing an ABSOLUTE target and feeding it
      directly (the ``"absolute"`` path) caused catastrophic failure there
      because the reconstructed target is a formal ``state + native_delta``
      composition, not a physically achievable pose by itself.
    """

    mode: str
    obs_step: AnvilEEObsProcessorStep
    n_action_steps: int = 1
    deliver: str = "absolute"
    gripper_mode: str = "target_qpos"  # "native_cmd" for the goalabs family — see recovered_delta_native_action

    _call_count: int = field(default=0, init=False, repr=False)
    _chunk_anchor: np.ndarray | None = field(default=None, init=False, repr=False)
    _running_target: np.ndarray | None = field(
        default=None, init=False, repr=False
    )  # rel_world_seq / rel_hand_seq only

    def __post_init__(self) -> None:
        if self.mode not in _ZERO_CAL_MODES:
            raise ValueError(f"mode must be one of {_ZERO_CAL_MODES}, got {self.mode!r}")
        if self.deliver not in ("absolute", "relative"):
            raise ValueError(f"deliver must be 'absolute' or 'relative', got {self.deliver!r}")
        if self.gripper_mode not in ("target_qpos", "native_cmd"):
            raise ValueError(
                f"gripper_mode must be 'target_qpos' or 'native_cmd', got {self.gripper_mode!r}"
            )

    def reset_episode_state(self) -> None:
        """Re-align chunk tracking with the policy's replan schedule at every
        episode start — see AnvilEEActionProcessorStep.reset_episode_state
        (real bug #5). Also clears the seq modes' running-target accumulator,
        which otherwise carried the previous episode's pose into the next."""
        self._call_count = 0
        self._chunk_anchor = None
        self._running_target = None

    def action(self, action: torch.Tensor) -> torch.Tensor:
        if self.obs_step.last_anvil_state is None:
            raise RuntimeError(
                "ZeroCalActionProcessorStep needs an observation processed by its "
                "paired AnvilEEObsProcessorStep before the first action."
            )
        current_state = self.obs_step.last_anvil_state

        chunk_start = self._call_count % self.n_action_steps == 0
        if chunk_start:
            self._chunk_anchor = current_state.copy()
        self._call_count += 1

        act10 = (action[0] if action.dim() > 1 else action).detach().cpu().numpy()

        if self.mode == "abs":
            target_pos, target_rot6d, target_gripper = act10[:3], act10[3:9], float(act10[9])
        elif self.mode == "rel_world":
            abs10 = ee_rel_world_inverse(act10.reshape(1, 10), self._chunk_anchor.reshape(1, 8))[0]
            target_pos, target_rot6d, target_gripper = abs10[:3], abs10[3:9], float(abs10[9])
        elif self.mode == "rel_hand":
            abs10 = ee_rel_inverse(act10.reshape(1, 10), self._chunk_anchor.reshape(1, 8))[0]
            target_pos, target_rot6d, target_gripper = abs10[:3], abs10[3:9], float(abs10[9])
        else:  # rel_world_seq / rel_hand_seq — accumulate consecutive deltas from the anchor
            if chunk_start:
                self._running_target = self._chunk_anchor.copy()
            R_running = quat_to_matrix(self._running_target[3:7])
            R_delta = rot6d_to_matrix(act10[3:9])
            if self.mode == "rel_world_seq":
                # R_delta = R_next @ R_current.T (see anvil_state_to_delta_action)
                new_pos = self._running_target[:3] + act10[:3]
                new_R = R_delta @ R_running  # R_next = R_delta @ R_current
            else:  # rel_hand_seq
                # R_delta = R_prev.T @ R_next (see convert_episode_goal_handseq_actions)
                new_pos = self._running_target[:3] + R_running @ act10[:3]
                new_R = R_running @ R_delta  # R_next = R_prev @ R_delta
            new_gripper = float(act10[9])
            self._running_target = np.concatenate(
                [new_pos, matrix_to_quat(new_R), [new_gripper]]
            ).astype(np.float32)
            target_pos, target_rot6d, target_gripper = new_pos, matrix_to_rot6d(new_R), new_gripper

        if self.deliver == "absolute":
            native = absolute_native_action_from_target(
                target_pos=target_pos,
                target_rot6d=target_rot6d,
                target_gripper=target_gripper,
                current_gripper=float(current_state[7]),
                gripper_mode=self.gripper_mode,
            )
        else:  # "relative"
            native = recovered_delta_native_action(
                reconstructed_pos=target_pos,
                reconstructed_rot6d=target_rot6d,
                reconstructed_gripper=target_gripper,
                current_state=current_state,
                current_gripper=float(current_state[7]),
                gripper_mode=self.gripper_mode,
            )
        return torch.from_numpy(native).unsqueeze(0)

    def transform_features(self, features):
        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != FeatureType.ACTION}
        new_features[FeatureType.ACTION] = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        return new_features

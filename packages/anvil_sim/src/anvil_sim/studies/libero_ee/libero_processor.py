"""ProcessorSteps letting the LIBERO-EE study's policies run against LIBERO's
native environment via ``lerobot.scripts.lerobot_eval.rollout()``'s
``env_preprocessor``/``env_postprocessor`` hooks. No custom gym env is
needed — every arm shares the exact same native ``LiberoEnv``; these steps
only translate the *format* crossing the policy/env boundary, the sim's own
OSC_POSE controller still does all the actual low-level control.

:class:`AnvilEEObsProcessorStep` is the Anvil EE observation encoding (used
by the ``goalabs`` family's ``zerocal_goal_*`` eval, and as a standalone
diagnostics state probe elsewhere).

:class:`NativeRot6dActionProcessorStep` (``native_rot6d`` arm) and
:class:`NativeHandObsProcessorStep`/:class:`NativeHandActionProcessorStep`
(``native_hand`` arm) are self-contained format converters over native's own
observation, no obs/action calibration involved.

:class:`ZeroCalActionProcessorStep` is the zero-calibration re-run
(``env.control_mode="absolute"`` or ``"relative"``, see
:func:`absolute_native_action_from_target` / :func:`recovered_delta_native_action`)
used by the ``goalabs``/``native_abs``/``native_n0`` "goal" target family.
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
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
    rot6d_to_matrix,
)
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.pipeline import ActionProcessorStep, ObservationProcessorStep
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE

# Gripper action is a saturating open/close command, not a scaled delta --
# confirmed via the env's own no-op action [0,0,0,0,0,0,-1] (hold pose, open
# gripper): -1.0 = open, +1.0 = close.
GRIPPER_OPEN_CMD = -1.0
GRIPPER_CLOSE_CMD = 1.0


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
    comment — see ``ZeroCalActionProcessorStep``.

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
    — used by the ``native_rot6d`` arm, which isolates rot6d rotation
    encoding with ZERO calibration/approximation error. Position and gripper
    pass through unchanged; rotation is decoded via
    ``rot6d_to_matrix``/``matrix_to_axis_angle`` — exact, invertible math,
    with no notion of "current state" or "target" needed at all.

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    pos = rot6d_action10[:3]
    native_rot = matrix_to_axis_angle(rot6d_to_matrix(rot6d_action10[3:9]))
    gripper = rot6d_action10[9]
    return np.concatenate([pos, native_rot, [gripper]]).astype(np.float32)


def axis_angle_action_to_rot6d(aa_action7: np.ndarray) -> np.ndarray:
    """Decode a 7-dim ``[pos(3), axis-angle(3), gripper]`` action into the
    10-dim ``[pos(3), rot6d(6), gripper]`` rot6d layout — the EXACT inverse of
    :func:`rot6d_action_to_native`, and identical math to
    ``anvil_sim.libero_convert.native_action_to_rot6d``.

    Used by ``native_abs``/``native_n0`` (the axis-angle counterparts of the
    rot6d ``goalabs`` conditions ``zerocal_goal_abs``/``zerocal_goal_world_n0``):
    those store the formal
    goal ``G[t] = state[t] + native_delta[t]`` with its rotation as AXIS-ANGLE
    instead of rot6d, so the eval action step decodes it back to rot6d here
    and then runs the IDENTICAL abs / rel_world reconstruction and
    recovered-delta relative delivery as the rot6d ``goalabs`` family. Because
    ``axis_angle_to_matrix``/``matrix_to_rot6d`` are exact invertible math,
    this round-trips losslessly (proven zero-error by the same
    ``native_action_to_rot6d``/``rot6d_action_to_native`` pair the
    ``native_rot6d`` arm relies on).

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    pos = aa_action7[:3]
    rot6d = matrix_to_rot6d(axis_angle_to_matrix(aa_action7[3:6]))
    gripper = aa_action7[6]
    return np.concatenate([pos, rot6d, [gripper]]).astype(np.float32)


def hand_action_to_native(hand_action7: np.ndarray, ee_axis_angle3: np.ndarray) -> np.ndarray:
    """Exact inverse of
    :func:`anvil_sim.libero_convert.native_action_to_hand`: rotate a
    HAND-frame native command back to the WORLD frame using the CURRENT obs
    EE orientation, for ``env.control_mode="relative"`` delivery exactly like
    ``native``.

    Used only by the ``native_hand`` arm — the missing "hand-frame +
    n-(n-1) + relative-delivery" cell that isolates the FRAME factor (world
    vs hand) with the NATIVE command representation. Because the rotation
    here (``R_ee @``) is the exact inverse of the convert-time rotation
    (``R_ee.T @``) with the SAME per-step EE orientation, replaying the
    dataset's own hand-frame actions through this reconstructs native's
    world-frame command exactly (the benchmark's GT-replay oracle). Position
    and rotation are the linear/angular parts of one spatial command,
    rotated by ``R_ee = axis_angle_to_matrix(ee_axis_angle3)``; gripper is a
    frame-invariant open/close command, passed through unchanged (already a
    native +/-1 command — no bang-bang, unlike the goalabs family).

    Pure function (no torch/env dependency) so it can be unit-tested directly.
    """
    R_ee = axis_angle_to_matrix(ee_axis_angle3)
    pos_world = R_ee @ hand_action7[:3]
    rot_world = R_ee @ hand_action7[3:6]
    return np.concatenate([pos_world, rot_world, [hand_action7[6]]]).astype(np.float32)


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

    - ``ee_abs``: ``ee_obs_abs_forward(state8)`` -> 10-dim rot6d, absolute.
    - ``ee_rel``: ``ee_obs_rel_forward(state8, anchor=state8)`` -> 10-dim
      rot6d, self-anchored. Since the anchor IS the current observation, this
      is always the fixed vector ``[0,0,0, 1,0,0,0,1,0, gripper]`` (zero
      translation, identity rotation) — only the gripper term carries
      information. That is intentional and matches training exactly
      (``EERelTransform`` anchors every single-step observation to itself).

    ``last_anvil_state`` (the raw absolute 8-dim quat state, NOT the encoded
    10-dim one above) is cached after every call — used by
    :class:`ZeroCalActionProcessorStep` as the reference state for the goal
    reconstruction, and as a standalone diagnostics state probe elsewhere.
    Assumes batch size 1 (this benchmark doesn't vectorize envs).
    """

    action_type: str
    last_anvil_state: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.action_type not in ("ee_abs", "ee_rel"):
            raise ValueError(f"action_type must be 'ee_abs' or 'ee_rel', got {self.action_type!r}")

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
        else:  # ee_abs
            state10 = ee_obs_abs_forward(state8.reshape(1, 8))[0]

        processed[OBS_STATE] = torch.from_numpy(state10.astype(np.float32)).unsqueeze(0)
        return processed

    def transform_features(self, features):
        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != FeatureType.STATE}
        new_features[FeatureType.STATE] = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))}
        return new_features


@dataclass
class NativeRot6dActionProcessorStep(ActionProcessorStep):
    """``env_postprocessor`` for the ``native_rot6d`` arm: policy's 10-dim
    action (native's own position/gripper unchanged, rotation packed as
    rot6d) -> LIBERO's native 7-dim action, via :func:`rot6d_action_to_native`.

    This needs no paired obs processor, no ``current_state``, and no
    chunk-anchor tracking — the decoding is a pure, self-contained format
    conversion with zero calibration. The observation side for this arm
    reuses lerobot's own stock ``LiberoProcessorStep`` unchanged, since this
    arm's ``observation.state`` is byte-identical to native's.
    """

    def action(self, action: torch.Tensor) -> torch.Tensor:
        act10 = (action[0] if action.dim() > 1 else action).detach().cpu().numpy()
        native = rot6d_action_to_native(act10)
        return torch.from_numpy(native).unsqueeze(0)

    def transform_features(self, features):
        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != FeatureType.ACTION}
        new_features[FeatureType.ACTION] = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        return new_features


@dataclass
class NativeHandObsProcessorStep(ObservationProcessorStep):
    """``env_preprocessor`` for the ``native_hand`` arm: produces the SAME
    8-dim ``observation.state`` (``[eef_pos(3), eef_axis-angle(3),
    gripper_qpos(2)]``) and original ``image``/``image2`` camera keys that
    ``native`` trains and evals on — by COMPOSING lerobot's own stock
    :class:`~lerobot.processor.env_processor.LiberoProcessorStep` (so the obs
    encoding is byte-identical to ``native``'s, not re-derived) — and
    additionally caches the EE world-frame orientation (axis-angle,
    ``state[3:6]``) so the paired :class:`NativeHandActionProcessorStep` can
    rotate the policy's hand-frame command back to the world frame.

    Unlike :class:`AnvilEEObsProcessorStep`, it does NOT rename cameras to
    ``agentview``/``wrist`` nor re-encode the state to 10-dim rot6d — the
    ``native_hand`` policy is a plain ``lerobot-train`` policy over native's
    own observation schema (only the ACTION column was rotated at
    dataset-write time).

    ``last_anvil_state`` (Anvil 8-dim ``[pos, quat, gripper]``) is also
    cached so the optional ``--trace-dir`` writer works unchanged.
    """

    last_ee_axis_angle: np.ndarray | None = field(default=None, init=False, repr=False)
    last_anvil_state: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from lerobot.processor.env_processor import LiberoProcessorStep

        self._inner = LiberoProcessorStep()

    def observation(self, observation: dict) -> dict:
        processed = self._inner.observation(observation)
        state = processed[OBS_STATE]
        state_np = (state[0] if state.dim() > 1 else state).detach().cpu().numpy().astype(np.float64)
        axis_angle = state_np[3:6]
        self.last_ee_axis_angle = axis_angle.astype(np.float32)
        quat = matrix_to_quat(axis_angle_to_matrix(axis_angle))
        self.last_anvil_state = np.concatenate(
            [state_np[:3], quat, [state_np[6]]]
        ).astype(np.float32)
        return processed

    def transform_features(self, features):
        return self._inner.transform_features(features)


@dataclass
class NativeHandActionProcessorStep(ActionProcessorStep):
    """``env_postprocessor`` for the ``native_hand`` arm: rotate the policy's
    predicted 7-dim HAND-frame native command back to the WORLD frame (via
    :func:`hand_action_to_native`, using the current obs EE orientation
    cached by the paired :class:`NativeHandObsProcessorStep`), then deliver
    the 7-dim world-frame native command via ``env.control_mode="relative"``
    exactly like ``native``.

    Per-step and stateless: the native command is a per-step delta, so there
    is no chunk anchor / running target (hence no ``reset_episode_state``
    needed), matching how ``native`` is replayed "direct".
    """

    obs_step: NativeHandObsProcessorStep

    def action(self, action: torch.Tensor) -> torch.Tensor:
        if self.obs_step.last_ee_axis_angle is None:
            raise RuntimeError(
                "NativeHandActionProcessorStep needs an observation processed by its "
                "paired NativeHandObsProcessorStep before the first action."
            )
        act7 = (action[0] if action.dim() > 1 else action).detach().cpu().numpy()
        native = hand_action_to_native(act7, self.obs_step.last_ee_axis_angle)
        return torch.from_numpy(native).unsqueeze(0)

    def transform_features(self, features):
        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != FeatureType.ACTION}
        new_features[FeatureType.ACTION] = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        return new_features


_ZERO_CAL_MODES = ("abs", "rel_world", "rel_hand")


@dataclass
class ZeroCalActionProcessorStep(ActionProcessorStep):
    """``env_postprocessor`` for the "goal" target family (``goalabs``/
    ``native_abs``/``native_n0``), using ``env.control_mode="absolute"`` (see
    :func:`absolute_native_action_from_target`) or ``"relative"`` (see
    :func:`recovered_delta_native_action`) delivery — zero calibration either
    way. Pair with :class:`AnvilEEObsProcessorStep` using ``action_type="ee_abs"``
    (for ``mode="abs"``) or ``action_type="ee_rel"`` (for ``mode="rel_world"``/
    ``"rel_hand"`` — the self-anchored single-step obs encoding is
    frame-invariant, see ``test_obs_rel_world_matches_body_frame_when_self_anchored``,
    so ``ee_rel``'s existing obs encoding is reused unchanged for both).

    Three modes:

    - ``"abs"``: policy output IS the absolute target directly.
    - ``"rel_world"``: SE(3)-relative, WORLD frame, anchored to chunk start
      (n-0) — reconstruct via ``ee_rel_world_inverse``
      (``--action-type=ee_rel_world``).
    - ``"rel_hand"``: SE(3)-relative, HAND frame, anchored to chunk start
      (n-0) — reconstruct via ``ee_rel_inverse``
      (``--action-type=ee_rel``).

    ``deliver`` controls the FINAL step, after ``target_pos``/``target_rot6d``/
    ``target_gripper`` have been reconstructed by the branches above:

    - ``"absolute"`` (default): :func:`absolute_native_action_from_target`
      — feed the reconstructed target directly to
      ``env.control_mode="absolute"``, zero scaling.
    - ``"relative"``: :func:`recovered_delta_native_action` — subtract the
      CURRENT real state from the reconstructed target to recover a
      native-command-scale delta, clip to ``[-1, 1]``, and feed to
      ``env.control_mode="relative"``, letting robosuite's own
      ``scale_action`` apply the true physical scale. Used by the "goal"
      family (``zerocal_goal_*``, ``native_abs``, ``native_n0``) —
      reconstructing an ABSOLUTE target and feeding it directly (the
      ``"absolute"`` path) caused catastrophic failure there because the
      reconstructed target is a formal ``state + native_delta`` composition,
      not a physically achievable pose by itself.
    """

    mode: str
    obs_step: AnvilEEObsProcessorStep
    n_action_steps: int = 1
    deliver: str = "absolute"
    gripper_mode: str = "target_qpos"  # "native_cmd" for the goalabs family — see recovered_delta_native_action
    # "rot6d" (default): policy output is the 10-dim rot6d layout the goalabs
    # dataset stores. "axis_angle": policy output is the 7-dim
    # [pos, axis-angle, gripper] layout native_abs/native_n0 store — decoded
    # to rot6d up front via axis_angle_action_to_rot6d, after which every
    # branch below is identical.
    action_encoding: str = "rot6d"
    # rel_world / rel_hand reconstruction anchor. False (default): anchor to the
    # CHUNK-START pose (n-0), correct for the rot6d goalabs world-n0/hand-n0
    # family, whose absolute-goal column is relativized against the chunk start
    # at anvil-trainer LOAD time (EERelWorldTransform). True: anchor to the
    # PER-FRAME current pose. Required for the axis-angle native_n0 condition,
    # whose column bakes the relativization against each frame's OWN observed
    # pose at convert time (libero_convert.goal_state_to_n0_axis_angle_action) —
    # a static lerobot-train column cannot carry a chunk-start anchor (chunks
    # exist only at inference). Chunk-start inverse of that per-frame column only
    # matches at n_action_steps=1 (frame == chunk start); at n>1 it diverges over
    # the chunk and collapses to 0%. Per-frame inverse recovers the goal at all n.
    per_frame_anchor: bool = False

    _call_count: int = field(default=0, init=False, repr=False)
    _chunk_anchor: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in _ZERO_CAL_MODES:
            raise ValueError(f"mode must be one of {_ZERO_CAL_MODES}, got {self.mode!r}")
        if self.deliver not in ("absolute", "relative"):
            raise ValueError(f"deliver must be 'absolute' or 'relative', got {self.deliver!r}")
        if self.gripper_mode not in ("target_qpos", "native_cmd"):
            raise ValueError(
                f"gripper_mode must be 'target_qpos' or 'native_cmd', got {self.gripper_mode!r}"
            )
        if self.action_encoding not in ("rot6d", "axis_angle"):
            raise ValueError(
                f"action_encoding must be 'rot6d' or 'axis_angle', got {self.action_encoding!r}"
            )

    def reset_episode_state(self) -> None:
        """Re-align chunk tracking with the policy's replan schedule at every
        episode start (real bug #5): the policy replans at episode-local step
        0 after ``policy.reset()``, but the call counter used to run on
        ACROSS episodes, so unless an episode's length happened to be a
        multiple of ``n_action_steps``, every episode after the first
        reconstructed targets against a chunk anchor captured at the WRONG
        time. See ``eval_libero_ee``'s rollout wrapper."""
        self._call_count = 0
        self._chunk_anchor = None

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
        if self.action_encoding == "axis_angle":
            # native_abs / native_n0 family: 7-dim [pos, axis-angle, gripper]
            # -> 10-dim rot6d, lossless (see axis_angle_action_to_rot6d).
            # Every branch below then operates on rot6d exactly as for the
            # goalabs family.
            act10 = axis_angle_action_to_rot6d(act10)

        # Chunk-start anchor (rot6d goalabs n-0) or per-frame anchor (axis-angle
        # native_n0, whose column is baked per-frame) — see per_frame_anchor.
        anchor = current_state if self.per_frame_anchor else self._chunk_anchor
        if self.mode == "abs":
            target_pos, target_rot6d, target_gripper = act10[:3], act10[3:9], float(act10[9])
        elif self.mode == "rel_world":
            abs10 = ee_rel_world_inverse(act10.reshape(1, 10), anchor.reshape(1, 8))[0]
            target_pos, target_rot6d, target_gripper = abs10[:3], abs10[3:9], float(abs10[9])
        else:  # rel_hand
            abs10 = ee_rel_inverse(act10.reshape(1, 10), anchor.reshape(1, 8))[0]
            target_pos, target_rot6d, target_gripper = abs10[:3], abs10[3:9], float(abs10[9])

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

"""Derive native/ee_abs/ee_rel datasets from a single `lerobot/libero` task.

Arm A (native) is written here too, as a plain LOCAL copy of the source's own
8-dim/7-dim schema (no Anvil transform at all — see :func:`_make_native_writer`).
It used to train directly against the Hub dataset filtered via
``LeRobotDataset("lerobot/libero", episodes=...)``, but that combination hits
a real ``lerobot`` upstream bug: ``lerobot_train.py``'s ``EpisodeAwareSampler``
(used whenever a policy has ``drop_n_last_frames``, e.g. Diffusion Policy)
indexes by the GLOBAL absolute frame index from dataset metadata, but
``LeRobotDataset.__getitem__`` expects a index *relative* to the
episode-filtered ``hf_dataset`` — confirmed via two isolated repros with
plain ``lerobot-train`` (filtered+diffusion crashes with
``IndexError: ... out of bounds``; unfiltered+diffusion works fine). ACT has
no ``drop_n_last_frames`` attribute so it never hits this path, which is why
``native`` + ACT worked before while ``native`` + Diffusion did not. Writing
a local subset dataset (this module already does the same for B/C) sidesteps
it entirely: no Hub-side episode filtering happens at load time, so metadata
and the loaded rows are the same size and global == relative.

Arms B (ee_abs) and C (ee_rel) are written here as new local LeRobot v3.0
datasets, reusing ``mcap_converter.core.writer.LeRobotWriter``'s existing EE
feature schema (8-dim quat state, 10-dim rot6d action per arm) so this
exercises the exact same writer code path real MCAP-derived EE datasets use.

Action encoding for B/C is act-from-obs (matches mcap_converter's convention
for real robot data): ``action_abs[t] = encode(state[t+1])``, NOT LIBERO's
own native delta column — see the module docstring in
``anvil_shared.ee_transform`` for why the two "relative" conventions aren't
interchangeable (translation frame differs; see :mod:`anvil_sim.libero_processor`
for the empirically-calibrated native delta convention used at eval time).

A 4th, EXPERIMENTAL arm ``ee_delta`` is also written here (see
:func:`convert_episode_delta_actions`) — NOT part of Anvil's real
``joint_abs``/``ee_abs``/``ee_rel`` contract. It exists only to isolate
whether rot6d rotation encoding itself (vs the act-from-obs absolute-pose
formulation ee_abs/ee_rel use) explains their lower closed-loop success rate
vs native: same world-frame delta as native, just rot6d instead of
axis-angle. It's trained by reusing ``anvil-trainer --action-type=ee_abs``
(whose ``EEAbsTransform`` never touches ``action``, only
``observation.state``, so whatever's stored in this dataset's action column
reaches the policy unchanged) — do not mistake its checkpoint's
``anvil_config.json`` (which will say ``action_type: "ee_abs"``) for a real
Anvil ee_abs run.

``ee_delta`` turned out NOT to cleanly isolate rot6d: it also introduced (a)
approximation error from ``NATIVE_POS_SCALE``/``NATIVE_ROT_SCALE`` (rotation
fit only R²=0.49) and (b) blind open-loop chunk execution with no
closed-loop correction (unlike ee_abs/ee_rel, which recompute
``target - current`` fresh every step). A 5th, EXPERIMENTAL arm
``native_rot6d`` (see :func:`native_action_to_rot6d`) fixes this: it keeps
native's observation AND action values completely unchanged (same 8-dim
state, same controller-scale numbers, no calibration at all) and only
re-encodes the action's rotation component as rot6d via a lossless,
invertible axis-angle-shaped embedding — no physical-unit conversion, no
approximation error, no confound besides the encoding itself. It trains via
plain ``lerobot-train`` (same path as native, no anvil-trainer at all).

A 7th, EXPERIMENTAL "goal" family (2 more datasets: ``goalabs``,
``delta-hand``, see :func:`native_delta_to_goal`) fixes a target-definition
mismatch in ee_abs/ee_rel: those use act-from-obs (``action[t] =
encode(state[t+1])``), but ``state[t+1]`` is the physically ACHIEVED next
state (goal + impedance-controller tracking error), not the commanded goal
itself. ``G[t] = state[t] + native_delta[t]`` (UNSCALED — see
:func:`native_delta_to_goal`'s docstring for why an earlier version that
scaled this by robosuite's ``output_max`` caused catastrophic closed-loop
failure) is a purely FORMAL composition, analogous to
:func:`native_action_to_rot6d`'s treatment of rotation commands: it carries
no physical meaning by itself. ``goalabs`` is the shared dataset for 3 of
the 5 Experiment 7 conditions (``abs``, plus ``world-n0``/``hand-n0``,
relativized at LOAD TIME by ``anvil_trainer``'s existing
``EERelWorldTransform``/``EERelTransform``). The other 2 conditions
(n-(n-1), consecutive) are built from REAL achieved states, NOT this formal
goal construction (see :func:`convert_episode_delta_hand_actions`'s
docstring for why): condition #4 (``world-n(n-1)``) REUSES the existing
``delta`` dataset/checkpoint unchanged; condition #5 (``hand-n(n-1)``, the
one genuinely new condition) is the new ``delta-hand`` dataset. All 5
conditions are eval'd by RECOVERING a native-delta-shaped quantity relative
to the REAL current state at eval time (see
``anvil_sim.libero_processor.recovered_delta_native_action``) and feeding
it to ``env.control_mode="relative"`` — letting robosuite's own
``scale_action`` apply the true physical scale, so nothing here needs to
know or guess what that scale actually is.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from anvil_shared.ee_transform import ee_rel_world_forward
from anvil_shared.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
    rot6d_to_matrix,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap_converter.config.schema import DataConfig
from mcap_converter.core.writer import LeRobotWriter

log = logging.getLogger(__name__)

SOURCE_REPO_ID = "lerobot/libero"
TASK_INDEX = 10
# Task text for the DEFAULT task_index above. convert() resolves the actual
# text for whatever --task-index it is given from the source dataset's task
# metadata (see task_text_for_index) — this constant is documentation only.
TASK_TEXT = "put the bowl on the plate"
# lerobot/libero's global task_index is NOT the same numbering LIBERO's own
# benchmark/env uses internally — confirmed by querying
# `libero.libero.benchmark.get_benchmark_dict()["libero_goal"]`, whose
# task_id=8 has this exact task text (not task_id=0; no simple offset
# formula, this requires the actual per-suite lookup). Needed by
# `create_libero_envs(task=LIBERO_ENV_SUITE, gym_kwargs={"task_ids": [LIBERO_ENV_TASK_ID]})`
# for closed-loop eval (see eval_libero_ee.py).
LIBERO_ENV_SUITE = "libero_goal"
LIBERO_ENV_TASK_ID = 8
ARM_ID = "panda"
CAMERA_NAMES = ["agentview", "wrist"]
SOURCE_CAMERA_KEYS = {"agentview": "observation.images.image", "wrist": "observation.images.image2"}

DEFAULT_OUTPUT_ROOT = Path("data/datasets/ee-space")


def task_episode_indices(ds: LeRobotDataset, task_index: int) -> list[int]:
    """Return the sorted list of global episode indices for ``task_index``.

    Reads the raw (non-video) columns only — no image decoding. Note
    ``LeRobotDataset(..., episodes=...)`` filters by these ORIGINAL global
    episode indices (they are preserved, not remapped, when filtered).
    """
    hf = ds.hf_dataset.select_columns(["episode_index", "task_index"]).with_format(None)
    pairs = zip(hf["episode_index"], hf["task_index"], strict=True)
    return sorted({int(e) for e, t in pairs if int(t) == task_index})


def task_text_for_index(ds: LeRobotDataset, task_index: int) -> str:
    """Resolve the language-instruction string for ``task_index`` from the
    source dataset's task metadata — the derived datasets' per-frame
    ``task`` field must carry the RIGHT instruction for whatever task is
    being converted (the old module-level ``TASK_TEXT`` constant silently
    wrote task 10's text for every task)."""
    for text, row in ds.meta.tasks.iterrows():
        if int(row["task_index"]) == task_index:
            return str(text)
    raise ValueError(f"task_index {task_index} not found in {SOURCE_REPO_ID} task metadata")


def raw_state_to_anvil(state8: np.ndarray) -> np.ndarray:
    """LIBERO native ``[pos(3), axis-angle(3), gripper_qpos(2)]`` -> Anvil ``[x,y,z,qx,qy,qz,qw,gripper]``.

    Gripper: LIBERO's ``gripper_qpos`` is the two (mirrored, opposite-sign)
    finger positions; we take finger 0 as the single representative scalar
    (arbitrary but consistent choice — the two fingers are rigidly coupled
    by the parallel gripper mechanism, so no information is lost for our
    purposes).
    """
    pos = state8[:3]
    quat = matrix_to_quat(axis_angle_to_matrix(state8[3:6]))
    gripper = state8[6]
    return np.concatenate([pos, quat, [gripper]]).astype(np.float32)


def anvil_state_to_abs_action(anvil_state8: np.ndarray) -> np.ndarray:
    """Encode an absolute Anvil EE state as the matching 10-dim rot6d action."""
    pos = anvil_state8[:3]
    quat = anvil_state8[3:7]
    gripper = anvil_state8[7]
    rot6d = matrix_to_rot6d(quat_to_matrix(quat))
    return np.concatenate([pos, rot6d, [gripper]]).astype(np.float32)


def anvil_state_to_delta_action(anvil_state8: np.ndarray, next_anvil_state8: np.ndarray) -> np.ndarray:
    """World-frame delta from one Anvil state to the next, in Anvil's 10-dim
    rot6d action layout — but representing a DELTA, not an absolute pose.

    Used only for the experimental ``ee_delta`` arm (see
    :func:`convert_episode_delta_actions`), NOT part of Anvil's real ee_abs/
    ee_rel contract. Position is stored in physical units (metres, world
    frame) and rotation as rot6d of the world-frame delta rotation — the
    same quantities :func:`anvil_sim.libero_processor.native_action_from_targets`
    computes internally (``target_pos - current_pos``, ``R_target @ R_current.T``),
    just persisted directly instead of derived from an absolute target at
    eval time. Scaling to LIBERO's native controller units happens once, at
    eval time, in ``native_action_from_world_delta`` — this stays in
    physical units to match the ``ee_abs``/``ee_rel`` convention of storing
    real metres/radians, not pre-scaled controller values.
    """
    pos_delta = next_anvil_state8[:3] - anvil_state8[:3]
    R_current = quat_to_matrix(anvil_state8[3:7])
    R_next = quat_to_matrix(next_anvil_state8[3:7])
    rot6d_delta = matrix_to_rot6d(R_next @ R_current.T)
    gripper_abs = next_anvil_state8[7]
    return np.concatenate([pos_delta, rot6d_delta, [gripper_abs]]).astype(np.float32)


def convert_episode_delta_actions(anvil_states: np.ndarray) -> np.ndarray:
    """Derive the WORLD-FRAME delta EE action for one episode — a 4th,
    experimental arm (``ee_delta``) used only to isolate whether rot6d
    encoding itself (vs the act-from-obs absolute-pose formulation) explains
    ee_abs/ee_rel's lower closed-loop success rate vs native. Same
    act-from-obs pairing (state[t] -> state[t+1]) as
    :func:`convert_episode_actions`, but stores the delta directly instead
    of the absolute target — structurally identical to LIBERO's own native
    delta, just rot6d instead of axis-angle, and NOT pre-scaled for the
    native controller (see :func:`anvil_state_to_delta_action`).
    """
    n = anvil_states.shape[0]
    targets = np.vstack([anvil_states[1:], anvil_states[-1:]])  # shift by one, repeat last
    action_delta = np.stack(
        [anvil_state_to_delta_action(anvil_states[t], targets[t]) for t in range(n)]
    )
    return action_delta.astype(np.float32)


def native_delta_to_goal(anvil_state8: np.ndarray, native_delta7: np.ndarray) -> np.ndarray:
    """Formally compose the current state with the native command — ``goal =
    state + native_delta`` — WITHOUT any scaling.

    v1 of this function multiplied ``native_delta7`` by robosuite's
    ``OSC_POSE`` ``output_max`` (``[0.05]*3+[0.5]*3``) to reconstruct what it
    assumed was the controller's exact internal target. That was WRONG: an
    empirical check against real ``lerobot/libero`` episodes showed the true
    per-step displacement is only ~22% of that (fitted scale ≈0.011 vs the
    assumed 0.05) — the impedance controller never fully reaches its
    internal aim point within one control step (that's normal OSC/PD
    behavior, not a bug), so ``state + delta*0.05`` is not close to
    anything physically achievable and fed directly into
    ``env.control_mode="absolute"`` it caused catastrophic closed-loop
    failure (all 5 Experiment 7 conditions: 0% pc_success).

    The fix: don't scale at all here, and don't feed this "goal" to the
    controller directly. Treat ``native_delta7``'s pos/rot components
    PURELY FORMALLY (same trick as :func:`native_action_to_rot6d` — they are
    NOT real metres/radians, just numbers in LIBERO's own command range),
    composed onto the real state via ordinary vector-add (position) /
    rotation composition (orientation, treating the raw numbers as if they
    were an axis-angle vector — exact, invertible math regardless of what
    they represent). The RESULT is only meaningful once it is later
    RECOVERED relative to a real state at eval time (see
    ``anvil_sim.libero_processor.recovered_delta_native_action``) and fed to
    the environment via ``env.control_mode="relative"`` — letting
    robosuite's own ``scale_action`` apply whatever the true physical scale
    actually is, so we never need to know or guess it.

    Args:
        anvil_state8: Anvil ``[x,y,z,qx,qy,qz,qw,gripper]`` state at time t.
        native_delta7: LIBERO's own recorded ``[Δxyz,Δaxis-angle,gripper]``
            command at time t (``NATIVE_ACTION_DIM``).

    Returns:
        Anvil ``[x,y,z,qx,qy,qz,qw,gripper]``-shaped array (same layout as
        ``anvil_state8``) — a FORMAL composition, not a physically
        meaningful absolute pose by itself. Gripper taken directly from
        ``native_delta7[6]`` (LIBERO's gripper command is already an
        absolute open/close value, not a delta — see ``NATIVE_ACTION_DIM``
        docstring).
    """
    delta = np.clip(native_delta7[:6], -1.0, 1.0)
    goal_pos = anvil_state8[:3] + delta[:3]
    R_state = quat_to_matrix(anvil_state8[3:7])
    R_goal = axis_angle_to_matrix(delta[3:6]) @ R_state
    goal_gripper = native_delta7[6]
    return np.concatenate([goal_pos, matrix_to_quat(R_goal), [goal_gripper]]).astype(np.float32)


def convert_episode_goal_states(anvil_states: np.ndarray, native_actions: np.ndarray) -> np.ndarray:
    """Derive the per-step absolute goal trajectory ``G[t]`` (Anvil 8-dim
    state layout) for one episode — see :func:`native_delta_to_goal`. Shared
    building block for the ``goalabs`` dataset (Experiment 7 conditions
    #1/#2/#3: ``abs``/``world-n0``/``hand-n0``).
    """
    n = anvil_states.shape[0]
    return np.stack(
        [native_delta_to_goal(anvil_states[t], native_actions[t]) for t in range(n)]
    ).astype(np.float32)


def convert_episode_goal_abs_actions(goal_states: np.ndarray) -> np.ndarray:
    """Encode the absolute goal trajectory as the 10-dim rot6d action —
    Experiment 7 condition #1 (``abs``), and the shared dataset for
    conditions #2/#3 (``world-n0``/``hand-n0``, relativized at load time by
    ``EERelWorldTransform``/``EERelTransform``).
    """
    return np.stack([anvil_state_to_abs_action(g) for g in goal_states]).astype(np.float32)


def goal_state_to_axis_angle_action(goal_state8: np.ndarray) -> np.ndarray:
    """Encode an absolute Anvil EE goal state as a 7-dim AXIS-ANGLE action
    ``[pos(3), axis-angle(3), gripper]`` — the axis-angle analogue of
    :func:`anvil_state_to_abs_action` (which uses 10-dim rot6d).

    Rotation is stored as axis-angle via
    ``matrix_to_axis_angle(quat_to_matrix(quat))``; this is EXACTLY the layout
    ``native_action_to_rot6d`` inverts, so
    ``anvil_sim.libero_processor.axis_angle_action_to_rot6d`` decodes it back
    to rot6d losslessly at eval. Shared encoder for the ``goalabs_aa`` dataset
    group — the axis-angle counterpart of ``goalabs`` — serving the
    ``native_abs`` (absolute goal) and ``native_n0`` (chunk-start / n-0
    relativized) conditions, which differ from the rot6d ``goalabs`` family
    (``abs``/``world-n0``) ONLY in this rotation encoding. Gripper is the
    goal's native +/-1 command (``native_delta_to_goal`` passthrough), carried
    through unchanged.
    """
    pos = goal_state8[:3]
    aa = matrix_to_axis_angle(quat_to_matrix(goal_state8[3:7]))
    gripper = goal_state8[7]
    return np.concatenate([pos, aa, [gripper]]).astype(np.float32)


def convert_episode_goal_aa_actions(goal_states: np.ndarray) -> np.ndarray:
    """Encode the absolute goal trajectory as the 7-dim AXIS-ANGLE action —
    the ``goalabs_aa`` group's action column, shared by ``native_abs`` and
    ``native_n0`` (see :func:`goal_state_to_axis_angle_action`). Axis-angle
    counterpart of :func:`convert_episode_goal_abs_actions`.
    """
    return np.stack([goal_state_to_axis_angle_action(g) for g in goal_states]).astype(np.float32)


def goal_state_to_n0_axis_angle_action(
    goal_state8: np.ndarray, anchor_state8: np.ndarray
) -> np.ndarray:
    """Express the absolute goal RELATIVE to a per-frame WORLD-frame anchor
    (n-0) and encode as a 7-dim AXIS-ANGLE action ``[pos(3), axis-angle(3),
    gripper]`` — the ``native_n0`` dataset column.

    This is the NATIVE-family (``lerobot-train`` raw, 8-dim native
    observation) counterpart of the rot6d ``goalabs`` world-n0 condition
    (which relativizes at anvil-trainer LOAD time via ``EERelWorldTransform``).
    Because ``native_n0`` trains on raw ``lerobot-train`` with NO load-time
    transform, the relativization is BAKED into the stored column here:
    ``rel = ee_rel_world_forward(goal, anchor)``, where ``anchor`` is that
    frame's own observed EE pose (``anvil_state[t]``, quat layout). Rotation is
    stored as axis-angle (via ``rot6d_to_matrix``/``matrix_to_axis_angle``) so
    the eval action step decodes it back to rot6d losslessly
    (``axis_angle_action_to_rot6d``) and runs the IDENTICAL
    ``ee_rel_world_inverse`` chunk-start reconstruction as the rot6d goalabs
    world-n0 condition. Gripper is the goal's native +/-1 command, carried
    through unchanged (``ee_rel_world_forward`` passes gripper through).
    """
    goal10 = anvil_state_to_abs_action(goal_state8)  # [pos, rot6d, gripper]
    rel10 = ee_rel_world_forward(goal10.reshape(1, 10), anchor_state8.reshape(1, 8))[0]
    pos = rel10[:3]
    aa = matrix_to_axis_angle(rot6d_to_matrix(rel10[3:9]))
    gripper = rel10[9]
    return np.concatenate([pos, aa, [gripper]]).astype(np.float32)


def convert_episode_goal_n0_aa_actions(
    goal_states: np.ndarray, anchor_states: np.ndarray
) -> np.ndarray:
    """Relativize every per-frame goal against that frame's own observed EE
    pose (n-0, world frame) and encode as 7-dim axis-angle — the ``native_n0``
    group's action column (see :func:`goal_state_to_n0_axis_angle_action`).
    ``anchor_states`` is the episode's Anvil observation trajectory
    (``anvil_states``), the same states the policy conditions on.
    """
    n = goal_states.shape[0]
    return np.stack(
        [goal_state_to_n0_axis_angle_action(goal_states[t], anchor_states[t]) for t in range(n)]
    ).astype(np.float32)


def anvil_state_to_hand_delta_action(anvil_state8: np.ndarray, next_anvil_state8: np.ndarray) -> np.ndarray:
    """HAND-frame (UMI-style body-frame) delta from one Anvil state to the
    next, in Anvil's 10-dim rot6d action layout — the hand-frame analogue of
    :func:`anvil_state_to_delta_action`, used only by
    :func:`convert_episode_delta_hand_actions` (Experiment 7 condition #5,
    ``hand-n(n-1)``).

    Unlike the world-frame delta (plain subtraction / left-multiply), the
    translation is projected into the PREVIOUS (``anvil_state8``) frame and
    rotation composes on the right: ``pos_delta = R_prev.T @ (pos_next -
    pos_prev)``, ``R_delta = R_prev.T @ R_next`` (so ``R_next = R_prev @
    R_delta``) — the same convention as
    :func:`anvil_shared.ee_transform.ee_rel_forward`.
    """
    R_prev = quat_to_matrix(anvil_state8[3:7])
    pos_delta = R_prev.T @ (next_anvil_state8[:3] - anvil_state8[:3])
    R_delta = R_prev.T @ quat_to_matrix(next_anvil_state8[3:7])
    rot6d_delta = matrix_to_rot6d(R_delta)
    gripper_abs = next_anvil_state8[7]
    return np.concatenate([pos_delta, rot6d_delta, [gripper_abs]]).astype(np.float32)


def convert_episode_delta_hand_actions(anvil_states: np.ndarray) -> np.ndarray:
    """Derive the HAND-frame delta EE action for one episode — Experiment 7
    condition #5 (``hand-n(n-1)``), the one genuinely new condition not
    covered by any prior experiment. Same act-from-obs pairing (state[t] ->
    state[t+1], REAL achieved states — NOT the formal ``goal_states``
    construction) as :func:`convert_episode_delta_actions` (condition #4,
    ``world-n(n-1)``, already covered by the EXISTING ``ee_delta`` arm —
    reused unchanged, no new dataset needed for it).

    Earlier v1 of this experiment built #4/#5 from consecutive FORMAL GOALS
    (``G[t-1]`` -> ``G[t]``, via ``native_delta_to_goal``) instead. That was
    wrong: at eval time the ``ZeroCalActionProcessorStep`` running-target
    accumulator resets to the REAL chunk-anchor state at every chunk start
    (matching :func:`convert_episode_delta_actions`'s own REAL-to-REAL
    anchor), so seeding it with a goal-to-goal delta (whose OWN anchor is a
    formal goal, not a real state) introduced a large, roughly constant
    offset (empirically ~0.26, the rough size of ``native_delta[0]`` — the
    exact "no predecessor" boundary case). Using REAL consecutive states for
    both #4 and #5, exactly like the already-validated ``ee_delta``, avoids
    this mismatch entirely (verified against real episode data with the
    same near-zero error as the ``abs``/n-0 conditions).
    """
    n = anvil_states.shape[0]
    targets = np.vstack([anvil_states[1:], anvil_states[-1:]])  # shift by one, repeat last
    action_delta = np.stack(
        [anvil_state_to_hand_delta_action(anvil_states[t], targets[t]) for t in range(n)]
    )
    return action_delta.astype(np.float32)


def native_action_to_rot6d(native_action7: np.ndarray) -> np.ndarray:
    """Repackage LIBERO's native 7-dim action into a 10-dim shape by
    re-encoding its rotation component as rot6d — a 5th, experimental arm
    used to isolate whether rot6d encoding *itself* explains ee_abs/ee_rel/
    ee_delta's lower closed-loop success vs native, without any of their
    other confounds.

    Unlike ``ee_delta`` (see :func:`anvil_state_to_delta_action`), this does
    NOT convert to physical units or apply any calibration scale.
    ``native_action7[3:6]`` is the controller's own command-scale numbers
    (NOT real radians — that's exactly why ``NATIVE_ROT_SCALE`` exists, to
    approximate the unknown mapping between real angles and these numbers).
    Here we don't care what they physically mean: we just treat them
    formally as an axis-angle vector, build a rotation matrix from it, and
    flatten that into rot6d. ``axis_angle_to_matrix``/``matrix_to_rot6d`` are
    exact, invertible math regardless of what the input numbers represent,
    so this round-trips losslessly (see
    ``anvil_sim.libero_processor.rot6d_action_to_native``, its exact
    inverse) — no calibration coefficient, no approximation error. Position
    and gripper are copied through unchanged (there's no axis-angle vs rot6d
    choice for a 3-vector position or a scalar gripper command).
    """
    pos = native_action7[:3]
    rot6d = matrix_to_rot6d(axis_angle_to_matrix(native_action7[3:6]))
    gripper = native_action7[6]
    return np.concatenate([pos, rot6d, [gripper]]).astype(np.float32)


def native_action_to_hand(native_action7: np.ndarray, ee_axis_angle3: np.ndarray) -> np.ndarray:
    """Rotate LIBERO's native WORLD-frame command into the EE BODY (hand)
    frame — the ONE genuinely new transform for the ``native_hand`` group,
    which isolates the FRAME factor (world vs hand) using the NATIVE command
    representation, changing ONLY the frame vs :func:`_make_native_dataset`'s
    ``native``.

    ``native_action7`` is ``[Δpos(3, world/base), Δaxis-angle(3, world),
    gripper]`` (see ``NATIVE_ACTION_DIM``). Its position and rotation
    components are the linear and angular parts of one spatial command;
    expressing them in the body frame is a pure change of basis by
    ``R_ee.T`` where ``R_ee = axis_angle_to_matrix(ee_axis_angle3)`` is the
    EE's world-frame orientation at this step (from
    ``observation.state[3:6]``). This is the SAME body-frame convention
    ``ee_rel`` uses for translation (``body_delta = R_state.T @ world_delta``,
    see ``anvil_sim.libero_processor``'s calibration note); for the rotation
    command the linear map ``R_ee.T @ aa`` equals the conjugation
    ``axis_angle(R_ee.T @ axis_angle_to_matrix(aa) @ R_ee)`` because
    axis-angle = axis*angle scales linearly and these deltas are well within
    the round-trip range. Gripper is a frame-invariant open/close command,
    passed through.

    Exactly invertible by
    :func:`anvil_sim.libero_processor.hand_action_to_native` with the same
    EE orientation — that inverse, applied at eval time against the CURRENT
    obs EE orientation, reconstructs the world-frame native command for
    ``env.control_mode="relative"`` delivery (identical to ``native``).
    """
    R_ee = axis_angle_to_matrix(ee_axis_angle3)
    pos_hand = R_ee.T @ native_action7[:3]
    rot_hand = R_ee.T @ native_action7[3:6]
    return np.concatenate([pos_hand, rot_hand, [native_action7[6]]]).astype(np.float32)


def convert_episode_native_hand_actions(
    raw_states: np.ndarray, native_actions: np.ndarray
) -> np.ndarray:
    """Rotate every native command in one episode into the hand frame using
    that step's EE orientation (raw LIBERO ``observation.state[3:6]``
    axis-angle) — see :func:`native_action_to_hand`. ``raw_states`` and
    ``native_actions`` are the source ``observation.state`` (8-dim) and
    ``action`` (7-dim) columns, unchanged from ``native``.
    """
    n = raw_states.shape[0]
    return np.stack(
        [native_action_to_hand(native_actions[t], raw_states[t][3:6]) for t in range(n)]
    ).astype(np.float32)


def convert_episode_actions(anvil_states: np.ndarray) -> np.ndarray:
    """Derive the absolute EE action for one episode from its Anvil state trajectory.

    ``action_abs[t] = encode(state[t+1])`` (act-from-obs); the terminal frame
    has no next state, so it holds its own state (a "stay" action) rather
    than dropping the frame.

    Both the abs and rel datasets store this SAME absolute action — the SE(3)
    relative transform is applied exactly once, at load time, by
    ``anvil_trainer.transforms.EERelTransform`` (``--action-type=ee_rel``).
    Pre-relativizing here too would double-transform (the already-relative
    value gets treated as if it were an absolute pose and relativized again),
    which silently corrupts the ee_rel training target while leaving the
    training loss looking deceptively normal.
    """
    n = anvil_states.shape[0]
    targets = np.vstack([anvil_states[1:], anvil_states[-1:]])  # shift by one, repeat last
    action_abs = np.stack([anvil_state_to_abs_action(targets[t]) for t in range(n)])
    return action_abs.astype(np.float32)


LIBERO_IMAGE_RESOLUTION = [256, 256]  # [width, height] — LIBERO's native camera resolution


NATIVE_STATE_DIM = 8  # [pos(3), axis-angle(3), gripper_qpos(2)] -- unchanged from the source
NATIVE_ACTION_DIM = 7  # [Δxyz(3), Δaxis-angle(3), gripper] -- unchanged from the source


def _make_native_dataset(output_dir: Path, repo_id: str) -> LeRobotDataset:
    """Create a local dataset that is a straight, untransformed copy of
    ``lerobot/libero``'s own schema — no Anvil code involved, just
    ``LeRobotDataset.create()`` directly (the plain lerobot mechanism).
    """
    # Camera keys are the ORIGINAL source names ("image"/"image2"), NOT the
    # "agentview"/"wrist" rename used for B/C -- arm A has no custom
    # processor at eval time, so its trained policy's input_features must
    # match what the live LiberoEnv actually outputs (lerobot's own
    # camera_name_mapping in lerobot/envs/libero.py), or lerobot-eval's
    # validate_visual_features_consistency() rejects the checkpoint.
    img_w, img_h = LIBERO_IMAGE_RESOLUTION
    features = {
        "observation.state": {"dtype": "float32", "shape": (NATIVE_STATE_DIM,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (NATIVE_ACTION_DIM,), "names": ["actions"]},
        **{
            source_key: {
                "dtype": "video",
                "shape": (3, img_h, img_w),
                "names": ["channel", "height", "width"],
            }
            for source_key in SOURCE_CAMERA_KEYS.values()
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=str(output_dir),
        robot_type="panda",
        features=features,
        use_videos=True,
    )


def _make_native_rot6d_dataset(output_dir: Path, repo_id: str) -> LeRobotDataset:
    """Create the local dataset for the experimental 5th arm: native's own
    observation schema UNCHANGED (8-dim axis-angle state, original
    ``image``/``image2`` camera keys — no Anvil obs encoding at all, so eval
    can reuse lerobot's stock ``LiberoProcessorStep``), with only the
    ``action`` column repackaged to 10-dim via :func:`native_action_to_rot6d`.
    """
    img_w, img_h = LIBERO_IMAGE_RESOLUTION
    features = {
        "observation.state": {"dtype": "float32", "shape": (NATIVE_STATE_DIM,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (10,), "names": ["actions"]},
        **{
            source_key: {
                "dtype": "video",
                "shape": (3, img_h, img_w),
                "names": ["channel", "height", "width"],
            }
            for source_key in SOURCE_CAMERA_KEYS.values()
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=str(output_dir),
        robot_type="panda",
        features=features,
        use_videos=True,
    )


def _make_goalabs_aa_dataset(output_dir: Path, repo_id: str) -> LeRobotDataset:
    """Create the local dataset for the ``goalabs_aa`` group — the axis-angle
    counterpart of ``goalabs``. Same Anvil EE observation as ``goalabs``
    (8-dim ``[x,y,z,qx,qy,qz,qw,gripper]`` quat state, ``agentview``/``wrist``
    cameras), but the ``action`` column is the 7-dim AXIS-ANGLE encoding of
    the formal goal (see :func:`goal_state_to_axis_angle_action`) instead of
    the 10-dim rot6d one ``goalabs`` stores. Written via
    ``LeRobotDataset.create`` directly (the plain lerobot mechanism), like
    :func:`_make_native_rot6d_dataset`, because the mcap EE ``LeRobotWriter``
    enforces its own 10-dim rot6d action schema.
    """
    img_w, img_h = LIBERO_IMAGE_RESOLUTION
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        **{
            f"observation.images.{cam}": {
                "dtype": "video",
                "shape": (3, img_h, img_w),
                "names": ["channel", "height", "width"],
            }
            for cam in CAMERA_NAMES
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=str(output_dir),
        robot_type="panda",
        features=features,
        use_videos=True,
    )


def _make_writer(output_dir: Path, repo_id: str) -> LeRobotWriter:
    config = DataConfig(
        data_space="ee",
        observation_topics={ARM_ID: "unused"},
        image_resolution=LIBERO_IMAGE_RESOLUTION,
    )
    return LeRobotWriter(
        output_dir=str(output_dir),
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        config=config,
    )


ALL_DATASET_GROUPS = frozenset(
    {
        "native", "native_hand", "native_rot6d", "native_abs", "native_n0",
        "abs", "rel", "delta", "goalabs", "delta_hand", "goalabs_aa",
    }
)
"""All dataset groups :func:`convert` can write. See its ``only`` parameter.

Experiment 7 condition #4 (``world-n(n-1)``) reuses the EXISTING ``delta``
dataset/checkpoint unchanged (no separate ``goal_worldseq`` group — see
``convert_episode_delta_hand_actions``'s docstring for why the earlier v1
design that built it from consecutive formal goals was wrong); only
``goalabs`` (conditions #1/#2/#3) and ``delta_hand`` (condition #5, the one
genuinely new dataset) are Experiment 7 additions.
"""


def convert(
    task_index: int = TASK_INDEX,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_episodes: int | None = None,
    only: set[str] | None = None,
) -> dict:
    """Convert ``task_index``'s episodes into the local Anvil-format datasets.

    Also writes a JSON manifest of the task's global episode indices, for
    arm A (native) to consume directly via
    ``LeRobotDataset("lerobot/libero", episodes=<manifest>)`` — no local copy
    needed for that arm.

    Args:
        only: If given, restrict dataset CREATION to this subset of
            :data:`ALL_DATASET_GROUPS` (e.g. ``{"goalabs", "delta_hand"}``
            to add the Experiment 7 datasets to an output_root that already
            has native/abs/rel/delta/native_rot6d written, without
            re-creating — and erroring on — their existing directories).
            ``None`` (default) writes all groups, matching this function's
            original behavior.

    Returns a summary dict (episode/frame counts, output paths).
    """
    groups = ALL_DATASET_GROUPS if only is None else set(only)
    unknown = groups - ALL_DATASET_GROUPS
    if unknown:
        raise ValueError(f"Unknown dataset group(s) {unknown}, must be subset of {ALL_DATASET_GROUPS}")

    output_root.mkdir(parents=True, exist_ok=True)

    ds_probe = LeRobotDataset(SOURCE_REPO_ID)
    episodes = task_episode_indices(ds_probe, task_index)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    task_text = task_text_for_index(ds_probe, task_index)
    log.info("task_index=%d (%r): %d episodes, writing groups=%s", task_index, task_text, len(episodes), groups)

    episodes_manifest = output_root / f"libero-task{task_index}-episodes.json"
    with open(episodes_manifest, "w") as f:
        json.dump({"task_index": task_index, "task": task_text, "episodes": episodes}, f, indent=2)

    ds = LeRobotDataset(SOURCE_REPO_ID, episodes=episodes)

    native_dir = output_root / f"libero-task{task_index}-native"
    native_hand_dir = output_root / f"libero-task{task_index}-native-hand"
    native_abs_dir = output_root / f"libero-task{task_index}-native-abs"
    native_n0_dir = output_root / f"libero-task{task_index}-native-n0"
    abs_dir = output_root / f"libero-task{task_index}-abs"
    rel_dir = output_root / f"libero-task{task_index}-rel"
    delta_dir = output_root / f"libero-task{task_index}-delta"
    native_rot6d_dir = output_root / f"libero-task{task_index}-native-rot6d"
    goalabs_dir = output_root / f"libero-task{task_index}-goalabs"
    delta_hand_dir = output_root / f"libero-task{task_index}-delta-hand"
    goalabs_aa_dir = output_root / f"libero-task{task_index}-goalabs-aa"

    dataset_native = (
        _make_native_dataset(native_dir, f"anvil/libero-task{task_index}-native")
        if "native" in groups
        else None
    )
    # native_hand shares native's exact schema (8-dim axis-angle state, 7-dim
    # action, original image/image2 camera keys) — only the action VALUES
    # differ (world command rotated into the hand frame), so it reuses
    # _make_native_dataset unchanged.
    dataset_native_hand = (
        _make_native_dataset(native_hand_dir, f"anvil/libero-task{task_index}-native-hand")
        if "native_hand" in groups
        else None
    )
    dataset_native_rot6d = (
        _make_native_rot6d_dataset(native_rot6d_dir, f"anvil/libero-task{task_index}-native-rot6d")
        if "native_rot6d" in groups
        else None
    )
    # native_abs / native_n0 are NATIVE-family control-factor conditions: they
    # share native's exact schema (8-dim native axis-angle state, 7-dim action,
    # original image/image2 camera keys) and train via plain lerobot-train — so
    # they hold observation + trainer FIXED vs `native` and differ from it in
    # exactly ONE thing (abs-vs-rel target; n-0 vs n-(n-1) anchor). Only the
    # ACTION column differs: the 7-dim axis-angle goal (native_abs) / n-0
    # relativized goal (native_n0). Reuses _make_native_dataset unchanged.
    dataset_native_abs = (
        _make_native_dataset(native_abs_dir, f"anvil/libero-task{task_index}-native-abs")
        if "native_abs" in groups
        else None
    )
    dataset_native_n0 = (
        _make_native_dataset(native_n0_dir, f"anvil/libero-task{task_index}-native-n0")
        if "native_n0" in groups
        else None
    )
    dataset_goalabs_aa = (
        _make_goalabs_aa_dataset(goalabs_aa_dir, f"anvil/libero-task{task_index}-goalabs-aa")
        if "goalabs_aa" in groups
        else None
    )
    writer_abs = _make_writer(abs_dir, f"anvil/libero-task{task_index}-abs") if "abs" in groups else None
    writer_rel = _make_writer(rel_dir, f"anvil/libero-task{task_index}-rel") if "rel" in groups else None
    writer_delta = (
        _make_writer(delta_dir, f"anvil/libero-task{task_index}-delta") if "delta" in groups else None
    )
    writer_goalabs = (
        _make_writer(goalabs_dir, f"anvil/libero-task{task_index}-goalabs")
        if "goalabs" in groups
        else None
    )
    writer_delta_hand = (
        _make_writer(delta_hand_dir, f"anvil/libero-task{task_index}-delta-hand")
        if "delta_hand" in groups
        else None
    )
    dataset_abs = writer_abs.create_dataset(joint_names={}, camera_names=CAMERA_NAMES) if writer_abs else None
    dataset_rel = writer_rel.create_dataset(joint_names={}, camera_names=CAMERA_NAMES) if writer_rel else None
    dataset_delta = (
        writer_delta.create_dataset(joint_names={}, camera_names=CAMERA_NAMES) if writer_delta else None
    )
    dataset_goalabs = (
        writer_goalabs.create_dataset(joint_names={}, camera_names=CAMERA_NAMES) if writer_goalabs else None
    )
    dataset_delta_hand = (
        writer_delta_hand.create_dataset(joint_names={}, camera_names=CAMERA_NAMES)
        if writer_delta_hand
        else None
    )

    total_frames = 0
    idx = 0
    n_items = len(ds)
    for ep_num, ep_global_idx in enumerate(episodes):
        ep_items = []
        while idx < n_items:
            item = ds[idx]
            if int(item["episode_index"]) != ep_global_idx:
                break
            ep_items.append(item)
            idx += 1
        if not ep_items:
            raise RuntimeError(f"No frames found for episode {ep_global_idx} (task {task_index})")

        raw_states = np.stack([item["observation.state"].numpy() for item in ep_items])
        native_actions = np.stack([item["action"].numpy() for item in ep_items])
        anvil_states = np.stack([raw_state_to_anvil(s) for s in raw_states])
        # native_hand: native's own command rotated into the per-step hand frame.
        action_native_hand = convert_episode_native_hand_actions(raw_states, native_actions)
        action_abs = convert_episode_actions(anvil_states)
        action_delta = convert_episode_delta_actions(anvil_states)

        # Experiment 7 ("goal" target family) — see native_delta_to_goal.
        goal_states = convert_episode_goal_states(anvil_states, native_actions)
        action_goal_abs = convert_episode_goal_abs_actions(goal_states)
        action_goal_aa = convert_episode_goal_aa_actions(goal_states)
        action_delta_hand = convert_episode_delta_hand_actions(anvil_states)
        # native_abs: absolute goal in axis-angle (identical column to
        # goalabs_aa, but stored alongside NATIVE 8-dim obs). native_n0: that
        # goal relativized per-frame against its own observed EE pose (n-0,
        # world frame) — see convert_episode_goal_n0_aa_actions.
        action_native_n0 = convert_episode_goal_n0_aa_actions(goal_states, anvil_states)

        for t, item in enumerate(ep_items):
            images = {
                f"observation.images.{cam}": item[SOURCE_CAMERA_KEYS[cam]].numpy()
                for cam in CAMERA_NAMES
            }
            images_native = {
                source_key: item[source_key].numpy() for source_key in SOURCE_CAMERA_KEYS.values()
            }
            frame_native = {
                "observation.state": raw_states[t],
                "action": item["action"].numpy(),
                "task": task_text,
                **images_native,
            }
            frame_native_hand = {
                "observation.state": raw_states[t],
                "action": action_native_hand[t],
                "task": task_text,
                **images_native,
            }
            frame_native_rot6d = {
                "observation.state": raw_states[t],
                "action": native_action_to_rot6d(item["action"].numpy()),
                "task": task_text,
                **images_native,
            }
            # native_abs / native_n0: NATIVE 8-dim observation (raw_states,
            # unchanged from native), only the ACTION column is the 7-dim
            # axis-angle goal / n-0 relativized goal.
            frame_native_abs = {
                "observation.state": raw_states[t],
                "action": action_goal_aa[t],
                "task": task_text,
                **images_native,
            }
            frame_native_n0 = {
                "observation.state": raw_states[t],
                "action": action_native_n0[t],
                "task": task_text,
                **images_native,
            }
            frame_abs = {
                "observation.state": anvil_states[t],
                "action": action_abs[t],
                "task": task_text,
                **images,
            }
            frame_rel = {
                "observation.state": anvil_states[t],
                "action": action_abs[t],
                "task": task_text,
                **images,
            }
            frame_delta = {
                "observation.state": anvil_states[t],
                "action": action_delta[t],
                "task": task_text,
                **images,
            }
            # Experiment 7: observation.state is still the ACTUAL observed
            # state trajectory (anvil_states, what the policy conditions on
            # and what n-0 anchor transforms read) — only the action target
            # changes to the reconstructed goal G[t] (see native_delta_to_goal).
            # Unlike frame_abs/frame_rel (act-from-obs, action[t]~state[t+1]),
            # this action is at the SAME time index t as the state it pairs
            # with (G[t] is derived from state[t] itself).
            frame_goalabs = {
                "observation.state": anvil_states[t],
                "action": action_goal_abs[t],
                "task": task_text,
                **images,
            }
            frame_delta_hand = {
                "observation.state": anvil_states[t],
                "action": action_delta_hand[t],
                "task": task_text,
                **images,
            }
            # goalabs_aa: SAME Anvil EE observation + cameras as goalabs (so
            # this is cleanly "goalabs with an axis-angle action"), only the
            # action is the 7-dim axis-angle goal encoding (see
            # convert_episode_goal_aa_actions).
            frame_goalabs_aa = {
                "observation.state": anvil_states[t],
                "action": action_goal_aa[t],
                "task": task_text,
                **images,
            }
            if dataset_native is not None:
                dataset_native.add_frame(frame_native)
            if dataset_native_hand is not None:
                dataset_native_hand.add_frame(frame_native_hand)
            if dataset_native_rot6d is not None:
                dataset_native_rot6d.add_frame(frame_native_rot6d)
            if dataset_native_abs is not None:
                dataset_native_abs.add_frame(frame_native_abs)
            if dataset_native_n0 is not None:
                dataset_native_n0.add_frame(frame_native_n0)
            if dataset_abs is not None:
                dataset_abs.add_frame(frame_abs)
            if dataset_rel is not None:
                dataset_rel.add_frame(frame_rel)
            if dataset_delta is not None:
                dataset_delta.add_frame(frame_delta)
            if dataset_goalabs is not None:
                dataset_goalabs.add_frame(frame_goalabs)
            if dataset_delta_hand is not None:
                dataset_delta_hand.add_frame(frame_delta_hand)
            if dataset_goalabs_aa is not None:
                dataset_goalabs_aa.add_frame(frame_goalabs_aa)

        for dataset in (
            dataset_native, dataset_native_hand, dataset_native_rot6d,
            dataset_native_abs, dataset_native_n0, dataset_abs,
            dataset_rel, dataset_delta, dataset_goalabs, dataset_delta_hand,
            dataset_goalabs_aa,
        ):
            if dataset is not None:
                dataset.save_episode()
        total_frames += len(ep_items)
        log.info(
            "episode %d/%d (global idx %d): %d frames converted",
            ep_num + 1, len(episodes), ep_global_idx, len(ep_items),
        )

    if dataset_native is not None:
        dataset_native.finalize()
    if dataset_native_hand is not None:
        dataset_native_hand.finalize()
    if dataset_native_rot6d is not None:
        dataset_native_rot6d.finalize()
    if dataset_native_abs is not None:
        dataset_native_abs.finalize()
    if dataset_native_n0 is not None:
        dataset_native_n0.finalize()
    if dataset_goalabs_aa is not None:
        dataset_goalabs_aa.finalize()
    if writer_abs is not None:
        writer_abs.finalize(dataset_abs)
    if writer_rel is not None:
        writer_rel.finalize(dataset_rel)
    if writer_delta is not None:
        writer_delta.finalize(dataset_delta)
    if writer_goalabs is not None:
        writer_goalabs.finalize(dataset_goalabs)
    if writer_delta_hand is not None:
        writer_delta_hand.finalize(dataset_delta_hand)

    dir_by_group = {
        "native": native_dir,
        "native_hand": native_hand_dir,
        "native_rot6d": native_rot6d_dir,
        "native_abs": native_abs_dir,
        "native_n0": native_n0_dir,
        "abs": abs_dir,
        "rel": rel_dir,
        "delta": delta_dir,
        "goalabs": goalabs_dir,
        "delta_hand": delta_hand_dir,
        "goalabs_aa": goalabs_aa_dir,
    }
    return {
        "task_index": task_index,
        "task": task_text,
        "num_episodes": len(episodes),
        "num_frames": total_frames,
        **{f"{group}_dataset": str(dir_by_group[group]) for group in groups},
        "episodes_manifest": str(episodes_manifest),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, default=TASK_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Convert only the first N episodes (for quick smoke testing).",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help=(
            "Comma-separated subset of dataset groups to (re-)write "
            f"({sorted(ALL_DATASET_GROUPS)}), e.g. --only=goalabs,delta_hand "
            "to add the Experiment 7 datasets without touching an output_root "
            "that already has the others. Default: write all groups."
        ),
    )
    args = parser.parse_args()
    only = {g.strip() for g in args.only.split(",")} if args.only else None
    summary = convert(
        task_index=args.task_index,
        output_root=args.output_root,
        max_episodes=args.max_episodes,
        only=only,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

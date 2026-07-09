"""libero_ee's GT-replay wiring, injected into the harness's study-agnostic
:func:`anvil_sim.eval_replay.replay` as a :class:`~anvil_sim.study.ReplayAdapter`.

The harness owns the env rollout loop and the generic chunk-anchor state
machine; this module owns everything that touches the study's own encoding
and processors — the per-treatment pre/post pipelines (byte-identical to the
closed-loop eval path, so GT replay reproduces the exact anchor timing), the
stored-action -> policy-output mode map, and the axis-angle rot6d codec.
"""

from __future__ import annotations

from lerobot.processor import PolicyProcessorPipeline

from anvil_sim.studies.libero_ee.eval_libero_ee import (
    _AXIS_ANGLE_ACTION_TYPES,
    _LEGACY_ACTION_TYPES,
    _NATIVE_FRAME_ACTION_TYPES,
    _ZERO_CAL_ACTION_TYPES,
    _make_anvil_env_pre_post_processors,
)
from anvil_sim.studies.libero_ee.libero_processor import (
    AnvilEEObsProcessorStep,
    NativeRot6dActionProcessorStep,
    axis_angle_action_to_rot6d,
    rot6d_action_to_native,
)
from anvil_sim.study import ReplayAdapter

# Action types replayable by the driver, beyond eval_libero_ee's own:
# "native" (raw 7-dim passthrough, identity processors — the baseline) and
# "native_rot6d" (10-dim re-encoded column via NativeRot6dActionProcessorStep).
_EXTRA_ACTION_TYPES = ("native", "native_rot6d")


def _provider_mode(action_type: str) -> str:
    """Map an eval action-type to how the dataset's stored action column
    relates to what a perfect policy would output at each step.

    - "direct": stored value IS the per-step policy output (absolute
      targets, per-step deltas, native commands) — feed as-is.
    - "rel_hand"/"rel_world": stored value is the ABSOLUTE target (the n-0
      relativization happens at train load time, not in the stored data);
      a perfect policy outputs it relativized against the CHUNK ANCHOR, so
      the provider forward-transforms with the same anchor tracking the
      action processor uses.
    """
    if action_type in _EXTRA_ACTION_TYPES:
        return "direct"
    if action_type in _NATIVE_FRAME_ACTION_TYPES:
        # native_hand stores the per-step native command (rotated into the
        # hand frame) — a perfect policy outputs it as-is; the eval action
        # step rotates it back to world against the live obs EE orientation.
        return "direct"
    if action_type in _AXIS_ANGLE_ACTION_TYPES:
        # native_abs / native_n0 are NATIVE-family (lerobot-train raw): the
        # per-frame policy-output form is BAKED into the stored column at
        # convert time (native_abs = absolute goal; native_n0 = the goal
        # ALREADY relativized per-frame against its own obs pose). So a perfect
        # policy outputs the stored value as-is — "direct", NOT a live
        # re-relativization. (The rot6d goalabs world-n0 family, by contrast,
        # stores the ABSOLUTE goal and relativizes at load time, hence its
        # "rel_world" provider below.)
        return "direct"
    if action_type in _ZERO_CAL_ACTION_TYPES:
        mode = _ZERO_CAL_ACTION_TYPES[action_type][1]
        if mode in ("rel_hand", "rel_world"):
            return mode
        return "direct"  # "abs", "rel_world_seq", "rel_hand_seq" — stored form is the output form
    if action_type in _LEGACY_ACTION_TYPES:
        # ee_abs: absolute stored+output. ee_delta: per-step delta stored+output.
        # ee_rel: absolute stored, hand-relativized output (same as rel_hand).
        return "rel_hand" if action_type == "ee_rel" else "direct"
    raise ValueError(f"Unsupported --action-type for replay: {action_type!r}")


def _action_encoding(action_type: str) -> str:
    """"axis_angle" for the native_abs/native_n0 families, else "rot6d"."""
    return "axis_angle" if action_type in _AXIS_ANGLE_ACTION_TYPES else "rot6d"


def _identity_pipelines() -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    return PolicyProcessorPipeline(steps=[]), PolicyProcessorPipeline(steps=[])


def _make_replay_processors(action_type: str, n_action_steps: int):
    """(env_preprocessor, env_postprocessor, obs_step_for_anchor_or_None)."""
    if action_type == "native":
        pre, post = _identity_pipelines()
        return pre, post, None
    if action_type == "native_rot6d":
        pre, _ = _identity_pipelines()
        return pre, PolicyProcessorPipeline(steps=[NativeRot6dActionProcessorStep()]), None
    pre, post = _make_anvil_env_pre_post_processors(action_type, n_action_steps=n_action_steps)
    obs_step = pre.steps[0]
    return pre, post, obs_step


def _make_state_probe() -> AnvilEEObsProcessorStep:
    """Diagnostics-only state extractor, independent of the treatment pipeline
    (feeds the per-episode init-state alignment metric)."""
    return AnvilEEObsProcessorStep(action_type="ee_abs")


def build_replay_adapter() -> ReplayAdapter:
    return ReplayAdapter(
        make_processors=_make_replay_processors,
        make_state_probe=_make_state_probe,
        provider_mode=_provider_mode,
        action_encoding=_action_encoding,
        encode_to_rot6d=axis_angle_action_to_rot6d,
        decode_from_rot6d=rot6d_action_to_native,
    )

"""Closed-loop eval driver for the LIBERO-EE study's non-native action types
(``native_hand``, ``native_abs``, ``native_n0``, and the ``zerocal_goal_*``
"goal" family), on the exact native ``LiberoEnv`` the ``native`` arm also uses.

This mirrors ``lerobot.scripts.lerobot_eval.eval_main`` almost line for
line — same CLI config (``EvalPipelineConfig``), same policy loading, same
``eval_policy_all`` rollout/metrics code — with exactly one line swapped:
instead of ``make_env_pre_post_processors(env_cfg, policy_cfg)`` (which
would attach the stock ``LiberoProcessorStep``, wrong for a checkpoint
trained on Anvil's EE encoding), we build our own env pre/post-processor
pair per action type (see :func:`_make_anvil_env_pre_post_processors`).
Everything else — the sim, the rollout loop, the success-rate aggregation —
is untouched, so results are directly comparable to ``native``'s plain
``lerobot-eval`` numbers.

Usage (mirrors `lerobot-eval`, plus one extra flag)::

    # "goal" family (see _ZERO_CAL_ACTION_TYPES), env.control_mode=relative:
    anvil-eval-libero \\
        --action-type=zerocal_goal_abs \\
        --policy.path=model_zoo/research/libero_ee/<name>/checkpoints/last/pretrained_model \\
        --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \\
        --env.control_mode=relative \\
        --eval.n_episodes=10 --eval.batch_size=1 \\
        --output_dir=research/libero_ee/scratch/libero_zerocal_goal_abs

    # native_hand / native_abs / native_n0 (NATIVE-family control-factor conditions):
    anvil-eval-libero \\
        --action-type=native_n0 \\
        --policy.path=model_zoo/research/libero_ee/<name>/checkpoints/last/pretrained_model \\
        --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \\
        --env.control_mode=relative \\
        --eval.n_episodes=10 --eval.batch_size=1 \\
        --output_dir=research/libero_ee/scratch/libero_native_n0
"""

import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.pipeline import ActionProcessorStep
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from anvil_sim.studies.libero_ee.libero_processor import (
    AnvilEEObsProcessorStep,
    NativeHandActionProcessorStep,
    NativeHandObsProcessorStep,
    NativeRot6dActionProcessorStep,
    ZeroCalActionProcessorStep,
)

# The "goal" target family (see anvil_sim.libero_convert's module docstring):
# action = G[t] = state[t] + native_delta[t], UNSCALED — a purely FORMAL
# composition (see native_delta_to_goal). Recovered as a native-delta
# relative to the REAL current state at eval time (see
# recovered_delta_native_action) and delivered via env.control_mode="relative",
# letting robosuite's own scale_action apply the true physical scale. An
# earlier version scaled this by robosuite's assumed output_max (0.05/0.5)
# and fed it via "absolute" — that caused catastrophic failure (0% across all
# 5 conditions) because the assumed scale was ~4.5x too large (empirically
# verified against real episodes).
#
# Maps each CLI value to (obs_action_type, ZeroCalActionProcessorStep mode,
# deliver, gripper_mode) — gripper_mode is "native_cmd" because the goalabs
# family's gripper is LIBERO's own +/-1 command, not a qpos-scale target (see
# recovered_delta_native_action's docstring for the real bug this fixes).
_ZERO_CAL_ACTION_TYPES = {
    "zerocal_goal_abs": ("ee_abs", "abs", "relative", "native_cmd"),
    "zerocal_goal_world_n0": ("ee_rel", "rel_world", "relative", "native_cmd"),
    "zerocal_goal_hand_n0": ("ee_rel", "rel_hand", "relative", "native_cmd"),
    # NATIVE-family control-factor conditions (obs + trainer FIXED vs `native`;
    # differ in exactly ONE thing). Both use the 8-dim NATIVE passthrough
    # observation (NativeHandObsProcessorStep, see
    # _make_anvil_env_pre_post_processors) and are trained via plain
    # lerobot-train on their own dataset group (native_abs / native_n0).
    # Rotation is stored/predicted as AXIS-ANGLE (native's own format),
    # decoded to rot6d up front by
    # ZeroCalActionProcessorStep(action_encoding="axis_angle") so the abs /
    # rel_world reconstruction is byte-identical to the rot6d goalabs family.
    # - native_abs: dataset action = absolute goal G[t]=state[t]+native_delta[t]
    #   (axis-angle); flips ONLY absolute-vs-relative from `native` (delta
    #   command -> absolute goal). mode "abs".
    # - native_n0: dataset action = that goal relativized per-frame against its
    #   own observed EE pose (n-0, world frame, BAKED at convert time since
    #   lerobot-train applies no load-time transform); flips ONLY the anchor
    #   (n-(n-1) -> chunk-start n-0). mode "rel_world" reconstructs the absolute
    #   goal via ee_rel_world_inverse against the chunk-start anchor.
    # (The first tuple element is unused for these two — obs is always the
    # 8-dim native passthrough, never AnvilEEObsProcessorStep.)
    "native_abs": ("ee_abs", "abs", "relative", "native_cmd"),
    "native_n0": ("ee_rel", "rel_world", "relative", "native_cmd"),
    # native_ctrlgoal — reconstructs the controller's OWN scaled internal
    # goal (see libero_convert.native_delta_to_ctrlgoal: state + native_delta
    # * output_max), a GENUINE absolute pose in the same physical units as
    # observation.state (unlike native_abs's formal, unscaled composition).
    # Delivered "absolute" with ZERO further scaling, since the target is
    # already physical. Diagnostic: separates "is absolute delivery itself
    # sound" from "is the afo_abs target's magnitude/construction the
    # problem" — see research/libero_ee/stage1-closeout.md.
    "native_ctrlgoal": ("ee_abs", "abs", "absolute", "native_cmd"),
    # afo_abs_h{1,5,10} — action-FROM-OBSERVATION absolute-pose family (see
    # libero_convert.convert_episode_afo_abs_actions): the stored action is
    # the REAL observed EE pose h frames ahead (not a formal
    # state+native_delta composition like native_abs), so unlike native_abs
    # it delivers "absolute" directly (zero calibration, no recovered-delta
    # step needed — the target IS a physically achievable pose) rather than
    # "relative". Gripper is the RAW RECORDED native command (UMI
    # decomposition — only the EE pose is obs-derived), same "native_cmd"
    # convention as native_abs/goalabs.
    "afo_abs_h1": ("ee_abs", "abs", "absolute", "native_cmd"),
    "afo_abs_h5": ("ee_abs", "abs", "absolute", "native_cmd"),
    "afo_abs_h10": ("ee_abs", "abs", "absolute", "native_cmd"),
    # afo_abs_rel_h{1,5,10} — SAME afo_abs dataset/target construction (the
    # stored action is still the real observed future EE pose; nothing about
    # what's being LEARNED changes), only the DELIVERY differs: instead of
    # feeding the raw absolute pose to the env (absolute_native_action_from_target),
    # deliver="relative" recovers a delta against the CURRENT real state
    # (recovered_delta_native_action, matrix_to_axis_angle(target_R @
    # current_R.T)) — exactly native_abs's delivery path, reused unchanged.
    # RESULT: GT-replay 0%, WORSE than plain afo_abs's 8.2% — because the
    # recovered delta here is a REAL metres/radians value (unlike native_abs's
    # already-command-scale formal delta), and recovered_delta_native_action
    # applies no rescale, so the controller's own scale_action() double-scales
    # it. (The original comment here attributed afo_abs's 8.2% to an
    # axis-angle theta=pi singularity — directly tested and disproved, see
    # research/libero_ee/stage1-closeout.md; kept only as a historical marker that
    # this variant was built under that now-superseded theory.) mode stays
    # "abs" (no anchor composition either way; only "deliver" flips).
    "afo_abs_rel_h1": ("ee_abs", "abs", "relative", "native_cmd"),
    "afo_abs_rel_h5": ("ee_abs", "abs", "relative", "native_cmd"),
    "afo_abs_rel_h10": ("ee_abs", "abs", "relative", "native_cmd"),
    # native_ctrlgoal_relconv — SAME native_ctrlgoal dataset/target
    # (state[t] + native_delta[t]*output_max), only the DELIVERY differs:
    # deliver="relative_converted" recovers a real metres/radians delta
    # against the LIVE current state, THEN divides by output_max (unlike
    # afo_abs_rel_h*'s deliver="relative", which applies no rescale and
    # double-scales) to produce the normalized command, fed via
    # env.control_mode="relative". Diagnostic: does this two-step physical
    # conversion pipeline itself work — should algebraically recover
    # ~clip(native_delta[t]) and reproduce close to native's own ~60%
    # GT-replay baseline, since native_ctrlgoal's target IS exactly
    # state[t]+native_delta[t]*output_max. See research/libero_ee/stage1-closeout.md.
    "native_ctrlgoal_relconv": ("ee_abs", "abs", "relative_converted", "native_cmd"),
    # afo_relative — SAME afo_abs_h1 dataset/target (the real observed future
    # EE pose, NOT a recorded/reconstructed command), same relative_converted
    # delivery as native_ctrlgoal_relconv above. Isolates "commanded vs.
    # achieved" as a variable independent of delivery-mode bugs: this target
    # is already only ~20-30% of a full commanded motion (the measured
    # per-step convergence ratio), so converting+delivering it is expected to
    # systematically undershoot relative to native_ctrlgoal_relconv — the
    # discount applied a second time by the controller.
    "afo_relative": ("ee_abs", "abs", "relative_converted", "native_cmd"),
}

# Zero-cal goal action types whose stored/predicted action carries its
# rotation as AXIS-ANGLE (7-dim [pos, aa, gripper]) rather than rot6d (10-dim)
# — the ONLY difference from their rot6d counterparts. Consumed by
# _make_anvil_env_pre_post_processors here and by the GT-replay provider in
# eval_replay to decode/re-encode around the shared rot6d n-0 machinery.
_AXIS_ANGLE_ACTION_TYPES = frozenset(
    {
        "native_abs", "native_n0", "native_ctrlgoal", "native_ctrlgoal_relconv",
        "afo_abs_h1", "afo_abs_h5", "afo_abs_h10",
        "afo_abs_rel_h1", "afo_abs_rel_h5", "afo_abs_rel_h10", "afo_relative",
    }
)


def _load_policy_from_checkpoint(cfg: PreTrainedConfig):
    """Load a policy from its own saved checkpoint config, WITHOUT going
    through ``lerobot.policies.factory.make_policy``.

    ``make_policy(cfg, env_cfg=...)`` unconditionally overwrites
    ``cfg.output_features``/``input_features`` from ``env_cfg`` (LIBERO's
    native 7-dim action) even when ``cfg`` was already loaded from a
    pretrained checkpoint with the correct (Anvil, 10-dim) shapes — it's
    designed for "train a new policy for this env", not "load an existing
    checkpoint and run it against a differently-shaped env via a format
    adapter". Loading directly like this trusts the checkpoint's own
    config.json, which is what our env_preprocessor/env_postprocessor
    pair make correct here.
    """
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=cfg.pretrained_path, config=cfg)
    policy.to(cfg.device)
    return policy


# Native-command arms whose action column is the raw LIBERO 7-dim command
# re-expressed in a different FRAME (currently only the EE body/hand frame),
# reconstructed to world at eval time and delivered via
# env.control_mode="relative" exactly like `native`. Isolates the frame
# factor with the native representation — see native_action_to_hand.
_NATIVE_FRAME_ACTION_TYPES = ("native_hand",)
# native_rot6d: observation.state is byte-identical to native's (stock
# lerobot LiberoProcessorStep, no Anvil obs encoding at all); only the
# action side is custom (NativeRot6dActionProcessorStep, an exact
# invertible rot6d<->axis-angle format swap, zero calibration).
_NATIVE_ROT6D_ACTION_TYPES = ("native_rot6d",)
_ALL_ACTION_TYPES = (
    _NATIVE_FRAME_ACTION_TYPES + _NATIVE_ROT6D_ACTION_TYPES + tuple(_ZERO_CAL_ACTION_TYPES)
)


def _pop_action_type() -> str:
    """Extract --action-type=<value> from sys.argv (draccus would otherwise
    reject it as an unrecognized flag), matching the anvil_trainer
    convention for custom CLI flags layered on lerobot CLIs.
    """
    prefix = "--action-type="
    for arg in list(sys.argv):
        if arg.startswith(prefix):
            sys.argv.remove(arg)
            value = arg.split("=", 1)[1]
            if value not in _ALL_ACTION_TYPES:
                raise ValueError(f"--action-type must be one of {_ALL_ACTION_TYPES}, got {value!r}")
            return value
    raise ValueError(f"Missing required --action-type=<one of {_ALL_ACTION_TYPES}>")


def _pop_trace_dir() -> Path | None:
    """Extract the optional --trace-dir=<path> from sys.argv (same custom-flag
    convention as --action-type). When given, every rollout step appends one
    JSONL record — {step, act_raw (policy output), native_cmd (what actually
    went to the env), current_state8} — to <trace-dir>/trace.jsonl, so a
    surprising closed-loop result can be diagnosed mechanically (e.g. compare
    recovered native command magnitudes against the dataset's own action
    distribution) instead of guessed at. Default off; zero behavior change."""
    prefix = "--trace-dir="
    for arg in list(sys.argv):
        if arg.startswith(prefix):
            sys.argv.remove(arg)
            return Path(arg.split("=", 1)[1])
    return None


class _TraceTapStep(ActionProcessorStep):
    """Two pipeline taps sharing one writer: the "raw" tap (before the Anvil
    action step) buffers the policy's raw output; the "native" tap (after)
    writes the paired JSONL record. Pure passthrough for the action itself."""

    def __init__(self, writer: "_TraceWriter", role: str):
        self._writer = writer
        self._role = role

    def action(self, action):
        self._writer.record(self._role, action)
        return action

    def transform_features(self, features):
        return features


class _TraceWriter:
    def __init__(self, trace_dir: Path, obs_step: AnvilEEObsProcessorStep):
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(trace_dir / "trace.jsonl", "w")
        self._obs_step = obs_step
        self._step = 0
        self._pending_raw: list | None = None

    def record(self, role: str, action) -> None:
        arr = (action[0] if action.dim() > 1 else action).detach().cpu().numpy().tolist()
        if role == "raw":
            self._pending_raw = arr
            return
        state = self._obs_step.last_anvil_state
        self._f.write(
            json.dumps(
                {
                    "step": self._step,
                    "act_raw": self._pending_raw,
                    "native_cmd": arr,
                    "current_state8": state.tolist() if state is not None else None,
                }
            )
            + "\n"
        )
        self._step += 1
        self._pending_raw = None

    def close(self) -> None:
        self._f.close()


def _make_anvil_env_pre_post_processors(
    action_type: str, n_action_steps: int, trace_dir: Path | None = None
):
    if action_type in _NATIVE_ROT6D_ACTION_TYPES:
        # native_rot6d's observation.state was never touched at dataset-write
        # time, so there's nothing Anvil-specific to convert on the obs side —
        # reuse lerobot's stock LiberoProcessorStep directly (equivalent to
        # make_env_pre_post_processors(env_cfg, policy_cfg) for a LIBERO env
        # + non-XVLA policy, which just wraps this same step with no other
        # config-dependent behavior). Only the action side is custom.
        from lerobot.processor.env_processor import LiberoProcessorStep

        obs_step = LiberoProcessorStep()
        action_step = NativeRot6dActionProcessorStep()
        env_preprocessor = PolicyProcessorPipeline(steps=[obs_step])
        env_postprocessor = PolicyProcessorPipeline(steps=[action_step])
        return env_preprocessor, env_postprocessor
    if action_type in _NATIVE_FRAME_ACTION_TYPES:
        obs_step = NativeHandObsProcessorStep()
        action_step = NativeHandActionProcessorStep(obs_step=obs_step)
    elif action_type in _AXIS_ANGLE_ACTION_TYPES:
        # native_abs / native_n0 — NATIVE-family control-factor conditions:
        # 8-dim native passthrough observation (SAME as native / native_hand,
        # via NativeHandObsProcessorStep — which also caches last_anvil_state
        # in quat layout for the goal reconstruction), trained via plain
        # lerobot-train. Only the ACTION side runs the ZeroCal goal
        # reconstruction (decode axis-angle -> rot6d -> recover native delta).
        # This is the fix for the obs-dim confound: the old build attached
        # AnvilEEObsProcessorStep here (10-dim rot6d obs), mismatching the
        # 8-dim dataset the policy was actually trained on.
        _, zero_cal_mode, deliver, gripper_mode = _ZERO_CAL_ACTION_TYPES[action_type]
        obs_step = NativeHandObsProcessorStep()
        action_step = ZeroCalActionProcessorStep(
            mode=zero_cal_mode,
            obs_step=obs_step,
            n_action_steps=n_action_steps,
            deliver=deliver,
            gripper_mode=gripper_mode,
            action_encoding="axis_angle",
            # native_n0's column bakes its n-0 relativization per-frame against
            # each frame's OWN pose (libero_convert.goal_state_to_n0_axis_angle_action),
            # since plain lerobot-train applies no chunk-start load-time transform;
            # its reconstruction must therefore anchor per-frame, not to the chunk
            # start (which only matches at n_action_steps=1). native_abs is mode
            # "abs" and ignores the anchor entirely.
            per_frame_anchor=zero_cal_mode in ("rel_world", "rel_hand"),
        )
    else:  # action_type in _ZERO_CAL_ACTION_TYPES (rot6d goalabs family)
        obs_action_type, zero_cal_mode, deliver, gripper_mode = _ZERO_CAL_ACTION_TYPES[action_type]
        obs_step = AnvilEEObsProcessorStep(action_type=obs_action_type)
        action_step = ZeroCalActionProcessorStep(
            mode=zero_cal_mode,
            obs_step=obs_step,
            n_action_steps=n_action_steps,
            deliver=deliver,
            gripper_mode=gripper_mode,
            action_encoding="rot6d",
        )
    post_steps: list = [action_step]
    if trace_dir is not None:
        writer = _TraceWriter(trace_dir, obs_step)
        post_steps = [_TraceTapStep(writer, "raw"), action_step, _TraceTapStep(writer, "native")]
    env_preprocessor = PolicyProcessorPipeline(steps=[obs_step])
    env_postprocessor = PolicyProcessorPipeline(steps=post_steps)
    return env_preprocessor, env_postprocessor


def _install_episode_reset_hook() -> None:
    """Wrap ``lerobot_eval.rollout`` so every episode start also resets our
    stateful env processors — REAL BUG #5: ``rollout()`` calls
    ``policy.reset()`` (the policy replans from episode-local step 0) but
    never resets the env pre/post processors, whose chunk-anchor call
    counters therefore ran on ACROSS episodes. Unless an episode's length
    happened to be a multiple of ``n_action_steps``, every episode after
    the first reconstructed targets against an anchor captured at the wrong
    time — initially one from the PREVIOUS episode's scene. All
    chunk-anchored conditions (the goal n-0 family) measured before this fix
    understate their true success from episode 2 onward. Exposed by the
    GT-replay diagnostic: world-n0 replay
    at n_action_steps=100 scored 20% where the forward/inverse identity
    predicts parity with the abs condition (80%).

    Idempotent; patches the module global so ``eval_policy`` (same module)
    picks it up.
    """
    import lerobot.scripts.lerobot_eval as _le

    if getattr(_le.rollout, "_anvil_episode_reset_hook", False):
        return
    _orig_rollout = _le.rollout

    def _rollout_with_processor_reset(env, policy, env_preprocessor, env_postprocessor, *args, **kwargs):
        for pipeline in (env_preprocessor, env_postprocessor):
            for step in getattr(pipeline, "steps", []):
                if hasattr(step, "reset_episode_state"):
                    step.reset_episode_state()
        return _orig_rollout(env, policy, env_preprocessor, env_postprocessor, *args, **kwargs)

    _rollout_with_processor_reset._anvil_episode_reset_hook = True
    _le.rollout = _rollout_with_processor_reset


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig, action_type: str, trace_dir: Path | None = None) -> None:
    logging.info(pformat(asdict(cfg)))
    logging.info("action_type=%s trace_dir=%s", action_type, trace_dir)
    _install_episode_reset_hook()

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")
    policy = _load_policy_from_checkpoint(cfg.policy)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # The one line that differs from stock lerobot-eval: our own format
    # converters instead of the stock LiberoProcessorStep.
    env_preprocessor, env_postprocessor = _make_anvil_env_pre_post_processors(
        action_type, n_action_steps=policy.config.n_action_steps, trace_dir=trace_dir
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

    close_envs(envs)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


def main() -> None:
    init_logging()
    register_third_party_plugins()
    action_type = _pop_action_type()
    trace_dir = _pop_trace_dir()
    eval_main(action_type=action_type, trace_dir=trace_dir)


if __name__ == "__main__":
    main()

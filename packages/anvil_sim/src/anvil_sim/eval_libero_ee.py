"""Closed-loop eval driver for the ee_abs/ee_rel arms (B/C), on the exact
native ``LiberoEnv`` arm A (native) also uses.

This mirrors ``lerobot.scripts.lerobot_eval.eval_main`` almost line for
line — same CLI config (``EvalPipelineConfig``), same policy loading, same
``eval_policy_all`` rollout/metrics code — with exactly one line swapped:
instead of ``make_env_pre_post_processors(env_cfg, policy_cfg)`` (which
would attach the stock ``LiberoProcessorStep``, wrong for a checkpoint
trained on Anvil's EE encoding), we build our own
``AnvilEEObsProcessorStep``/``AnvilEEActionProcessorStep`` pair. Everything
else — the sim, the rollout loop, the success-rate aggregation — is
untouched, so results are directly comparable to arm A's plain
``lerobot-eval`` numbers.

Usage (mirrors `lerobot-eval`, plus one extra flag)::

    # Legacy (scaled-delta reconstruction, env.control_mode=relative):
    anvil-eval-libero \\
        --action-type=ee_abs \\
        --policy.path=outputs/train/.../checkpoints/last/pretrained_model \\
        --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \\
        --env.control_mode=relative \\
        --eval.n_episodes=10 --eval.batch_size=1 \\
        --output_dir=outputs/eval/libero_ee_abs

    # Zero-calibration re-run (see _ZERO_CAL_ACTION_TYPES), env.control_mode=absolute:
    anvil-eval-libero \\
        --action-type=zerocal_abs \\
        --policy.path=outputs/train/.../checkpoints/last/pretrained_model \\
        --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \\
        --env.control_mode=absolute \\
        --eval.n_episodes=10 --eval.batch_size=1 \\
        --output_dir=outputs/eval/libero_zerocal_abs

    # 7th-round "goal" family (see _ZERO_CAL_GOAL_ACTION_TYPES) -- NOTE the
    # required --env.control_mode differs PER ACTION TYPE within this
    # family: abs/world-n0/hand-n0 need "relative" (deliver="relative"),
    # world-seq/hand-seq need "absolute" (deliver="absolute"):
    anvil-eval-libero \\
        --action-type=zerocal_goal_abs \\
        --policy.path=outputs/train/.../checkpoints/last/pretrained_model \\
        --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \\
        --env.control_mode=relative \\
        --eval.n_episodes=10 --eval.batch_size=1 \\
        --output_dir=outputs/eval/libero_zerocal_goal_abs
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
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from anvil_sim.libero_processor import (
    AnvilEEActionProcessorStep,
    AnvilEEObsProcessorStep,
    ZeroCalActionProcessorStep,
)

# Zero-calibration re-run (env.control_mode="absolute", see
# absolute_native_action_from_target) of the 4 control-variable conditions —
# NOT the same code path as the original scaled-delta ee_abs/ee_rel/ee_delta
# (those remain available, unchanged, under their original --action-type
# values for backward comparison against the pre-fix numbers). Maps each
# CLI value to (obs_action_type, ZeroCalActionProcessorStep mode, deliver).
_ZERO_CAL_ACTION_TYPES = {
    "zerocal_abs": ("ee_abs", "abs", "absolute"),
    "zerocal_rel_world": ("ee_rel", "rel_world", "absolute"),
    "zerocal_rel_hand": ("ee_rel", "rel_hand", "absolute"),
    "zerocal_rel_world_seq": ("ee_abs", "rel_world_seq", "absolute"),
}

# 7th-round "goal" target family (see anvil_sim.libero_convert's module
# docstring). Two sub-families with DIFFERENT deliver modes — this is the
# result of two bugs found and fixed during implementation (see the plan's
# "實作中發現的第二個 bug" section), not a design choice made up front:
#
# - abs/world-n0/hand-n0 (trained on the `goalabs` dataset): action =
#   G[t] = state[t] + native_delta[t], UNSCALED — a purely FORMAL
#   composition (see native_delta_to_goal). Recovered as a native-delta
#   relative to the REAL current state at eval time (see
#   recovered_delta_native_action) and delivered via
#   env.control_mode="relative", letting robosuite's own scale_action apply
#   the true physical scale. An earlier version scaled this by robosuite's
#   assumed output_max (0.05/0.5) and fed it via "absolute" — that caused
#   catastrophic failure (0% across all 5 conditions) because the assumed
#   scale was ~4.5x too large (empirically verified against real episodes).
# - world-seq/hand-seq (trained on the EXISTING `delta`/new `delta-hand`
#   datasets — REAL achieved-state deltas, NOT the formal `goalabs`
#   construction): these targets are already in PHYSICAL units (metres/
#   radians), so they're delivered via env.control_mode="absolute" with
#   zero further scaling — identical to how zerocal_rel_world_seq already
#   works above. world-seq REUSES the existing ee_delta checkpoint
#   unchanged (no new dataset/training needed for it).
_ZERO_CAL_GOAL_ACTION_TYPES = {
    "zerocal_goal_abs": ("ee_abs", "abs", "relative"),
    "zerocal_goal_world_n0": ("ee_rel", "rel_world", "relative"),
    "zerocal_goal_hand_n0": ("ee_rel", "rel_hand", "relative"),
    "zerocal_goal_world_seq": ("ee_abs", "rel_world_seq", "absolute"),
    "zerocal_goal_hand_seq": ("ee_abs", "rel_hand_seq", "absolute"),
}
_ZERO_CAL_ACTION_TYPES = {**_ZERO_CAL_ACTION_TYPES, **_ZERO_CAL_GOAL_ACTION_TYPES}


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


_LEGACY_ACTION_TYPES = ("ee_abs", "ee_rel", "ee_delta")
_ALL_ACTION_TYPES = _LEGACY_ACTION_TYPES + tuple(_ZERO_CAL_ACTION_TYPES)


def _pop_action_type() -> str:
    """Extract --action-type=<value> from sys.argv (draccus would otherwise
    reject it as an unrecognized flag), matching the anvil_trainer
    convention for custom CLI flags layered on lerobot CLIs.

    Legacy values (``ee_abs``/``ee_rel``/``ee_delta``) use the ORIGINAL
    ``NATIVE_POS_SCALE``/``NATIVE_ROT_SCALE`` scaled-delta reconstruction
    with ``env.control_mode="relative"`` -- kept only so the old (pre-fix)
    numbers stay reproducible for comparison. The ``zerocal_*`` values (see
    ``_ZERO_CAL_ACTION_TYPES``) use ``ZeroCalActionProcessorStep`` with
    ``env.control_mode="absolute"`` instead -- zero calibration error, this
    is the corrected re-run.
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


def _make_anvil_env_pre_post_processors(action_type: str, n_action_steps: int):
    if action_type in _ZERO_CAL_ACTION_TYPES:
        obs_action_type, zero_cal_mode, deliver = _ZERO_CAL_ACTION_TYPES[action_type]
        obs_step = AnvilEEObsProcessorStep(action_type=obs_action_type)
        action_step = ZeroCalActionProcessorStep(
            mode=zero_cal_mode, obs_step=obs_step, n_action_steps=n_action_steps, deliver=deliver
        )
    else:
        obs_step = AnvilEEObsProcessorStep(action_type=action_type)
        action_step = AnvilEEActionProcessorStep(
            action_type=action_type, obs_step=obs_step, n_action_steps=n_action_steps
        )
    env_preprocessor = PolicyProcessorPipeline(steps=[obs_step])
    env_postprocessor = PolicyProcessorPipeline(steps=[action_step])
    return env_preprocessor, env_postprocessor


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig, action_type: str) -> None:
    logging.info(pformat(asdict(cfg)))
    logging.info("action_type=%s", action_type)

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
        action_type, n_action_steps=policy.config.n_action_steps
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
    eval_main(action_type=action_type)


if __name__ == "__main__":
    main()

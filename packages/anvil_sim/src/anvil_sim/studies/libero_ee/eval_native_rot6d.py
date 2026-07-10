"""Closed-loop eval driver for the experimental 5th arm ``native_rot6d``, on
the exact native ``LiberoEnv`` arm A (native) also uses.

This is the zero-calibration isolation of rot6d vs axis-angle rotation
encoding (see ``anvil_sim.libero_convert`` module docstring) — its
observation handling is IDENTICAL to arm A's (this arm's dataset never
changes ``observation.state``, so it reuses lerobot's own stock
``LiberoProcessorStep`` via ``make_env_pre_post_processors``, unlike
``eval_libero_ee.py``'s B/C/``ee_delta`` arms which need a custom obs
processor). Only the action side is custom:
``NativeRot6dActionProcessorStep`` decodes the policy's 10-dim
(position/gripper unchanged, rotation packed as rot6d) output back into
LIBERO's native 7-dim action via an exact, invertible format conversion —
no calibration coefficient, no approximation error.

Otherwise mirrors ``eval_libero_ee.py``/``lerobot.scripts.lerobot_eval.eval_main``
line for line: same CLI config, same policy loading (bypassing
``make_policy``'s env-feature override), same ``eval_policy_all``
rollout/metrics code.

Usage (mirrors `lerobot-eval`, no extra flags needed)::

    anvil-eval-native-rot6d \\
        --policy.path=model_zoo/native/libero-task10-native-rot6d/act/checkpoints/last/pretrained_model \\
        --env.type=libero --env.task=libero_goal --env.task_ids='[8]' \\
        --env.control_mode=relative \\
        --eval.n_episodes=10 --eval.batch_size=1 \\
        --output_dir=research/libero_ee/scratch/native-rot6d-act
"""

import json
import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from anvil_sim.studies.libero_ee.libero_processor import NativeRot6dActionProcessorStep


def _load_policy_from_checkpoint(cfg: PreTrainedConfig):
    """Load a policy from its own saved checkpoint config, WITHOUT going
    through ``lerobot.policies.factory.make_policy`` — see
    ``eval_libero_ee.py``'s identical helper for why (``make_policy``'s
    env-feature override would clobber this checkpoint's correct 10-dim
    action shape with LIBERO's native 7-dim).
    """
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=cfg.pretrained_path, config=cfg)
    policy.to(cfg.device)
    return policy


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig) -> None:
    logging.info(pformat(asdict(cfg)))

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

    # Obs side: stock lerobot LiberoProcessorStep (same as arm A) -- this
    # arm's observation.state was never touched at dataset-write time, so
    # there's nothing Anvil-specific to convert here. Action side: our own
    # zero-calibration rot6d decoder.
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    env_postprocessor = PolicyProcessorPipeline(steps=[NativeRot6dActionProcessorStep()])

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
    eval_main()


if __name__ == "__main__":
    main()

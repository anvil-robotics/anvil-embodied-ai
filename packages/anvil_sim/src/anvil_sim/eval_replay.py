"""Open-loop ground-truth replay through the eval processor chain — an
independent health check of the EVAL PATH, with no policy involved.

Motivation (Experiment 7 post-mortem, see README): two real eval-path bugs
(a wrong scale assumption, then a construction/delivery mismatch) each
survived unit tests and a normal-looking training loss, and were only
exposed after burning a full 5x50k-step training sweep on 0% rollouts.
Both would have been caught in minutes by replaying the DATASET'S OWN
ground-truth actions through the exact same env pre/post processors the
policy eval uses: if the ground truth itself cannot succeed through a
treatment's eval path, no checkpoint ever will.

How it works, per episode:

1. Reset the env. ``LiberoEnv`` defaults to FIXED per-task init states
   (``init_states[episode_index]``, striding by ``n_envs`` per reset — see
   ``lerobot/envs/libero.py``), the same 50 states LIBERO demos were
   recorded from, so dataset episode k replayed against env episode k is a
   near-matched-initial-condition open-loop replay (the per-episode
   ``init_state_pos_err`` metric in the output reports how well they
   actually align — the lerobot/libero conversion has 49 episodes for
   task_index=10, so alignment may drift after a dropped demo).
2. Each step: run the live observation through the treatment's obs
   processor (so chunk anchors / ``last_anvil_state`` are maintained
   exactly as in a real eval), take the dataset's stored action for this
   timestep, convert it into what a PERFECT policy would output for this
   treatment (see :class:`GtActionProvider` — identity for abs/seq/native
   families, a forward-relativization against the live chunk anchor for
   the n-0 families), feed it through the treatment's action processor,
   and step the env.
3. Record success, plus a per-step trace (raw GT, provided act, recovered
   native command) to JSONL for magnitude/diagnostic comparison.

The gate consuming this (see the bench runner) is RELATIVE: a treatment's
replay success is compared against the ``native`` replay baseline on the
same episodes/seeds — open-loop replay is inherently brittle to any
init-state mismatch, so the native ceiling, not 100%, is the reference.

Usage::

    anvil-libero-replay \\
        --action-type zerocal_goal_abs \\
        --dataset-root data/datasets/ee-space/libero-task10-goalabs \\
        --control-mode relative \\
        --task libero_goal --task-id 8 \\
        --n-episodes 10 \\
        --output-dir research/libero_ee/replay/goalabs-relative
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from anvil_shared.ee_transform import ee_rel_forward, ee_rel_world_forward
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from anvil_sim.study import ReplayAdapter

log = logging.getLogger(__name__)


@dataclass
class GtActionProvider:
    """Convert the dataset's stored action for step t into what a perfect
    policy would output at step t of a live rollout.

    For the n-0 families this mirrors the action processor's chunk-anchor
    state machine (anchor = live state at every ``n_action_steps``-th call)
    so that provider and processor always agree on the anchor: both read
    ``obs_step.last_anvil_state`` in the same step, and both advance their
    call counters exactly once per env step.
    """

    mode: str  # "direct" | "rel_hand" | "rel_world"
    obs_step: Any | None = None  # study obs step (None for "direct" without anchor needs)
    n_action_steps: int = 1
    # "rot6d" (default) or "axis_angle": for the goalabs_aa family
    # (native_abs / native_n0) the stored action is 7-dim [pos, aa, gripper];
    # the n-0 forward relativization runs on the shared rot6d machinery, so
    # decode -> forward -> re-encode keeps the provided "policy output" in the
    # same axis-angle layout the action step then decodes. The codec itself
    # (``encode``/``decode``) is study-provided so the harness stays agnostic:
    # ``encode`` maps the stored action -> 10-dim rot6d, ``decode`` maps the
    # relativized rot6d back to the stored layout. Both ``None`` == rot6d
    # identity (the default).
    action_encoding: str = "rot6d"
    encode: Callable[[np.ndarray], np.ndarray] | None = None
    decode: Callable[[np.ndarray], np.ndarray] | None = None
    _call_count: int = field(default=0, init=False, repr=False)
    _chunk_anchor: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, stored_action: np.ndarray) -> np.ndarray:
        if self.mode == "direct":
            self._call_count += 1
            return stored_action.astype(np.float32)

        if self.obs_step is None or self.obs_step.last_anvil_state is None:
            raise RuntimeError("GtActionProvider needs a processed observation before the first action.")
        if self._call_count % self.n_action_steps == 0:
            self._chunk_anchor = self.obs_step.last_anvil_state.copy()
        self._call_count += 1

        act10 = self.encode(stored_action) if self.encode is not None else stored_action
        forward = ee_rel_forward if self.mode == "rel_hand" else ee_rel_world_forward
        rel = forward(act10.reshape(1, 10), self._chunk_anchor.reshape(1, 8))[0]
        if self.decode is not None:
            # Re-encode to the layout the study's action step expects.
            return self.decode(rel).astype(np.float32)
        return rel.astype(np.float32)


def load_episode_actions(dataset_root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (episode_actions, episode_first_states) in episode order.

    ``episode_actions[k]`` is an (T_k, action_dim) array of the stored
    action column; ``episode_first_states[k]`` is frame 0's
    ``observation.state`` (for the init-state alignment diagnostic).
    """
    ds = LeRobotDataset(repo_id="local", root=str(dataset_root))
    hf = ds.hf_dataset.select_columns(["episode_index", "action", "observation.state"]).with_format(None)
    episodes: dict[int, list] = {}
    first_states: dict[int, np.ndarray] = {}
    for ep_idx, action, state in zip(
        hf["episode_index"], hf["action"], hf["observation.state"], strict=True
    ):
        ep = int(ep_idx)
        if ep not in episodes:
            episodes[ep] = []
            first_states[ep] = np.asarray(state, dtype=np.float32)
        episodes[ep].append(np.asarray(action, dtype=np.float32))
    order = sorted(episodes)
    return [np.stack(episodes[k]) for k in order], [first_states[k] for k in order]


def replay(
    action_type: str,
    dataset_root: Path,
    control_mode: str,
    task: str,
    task_id: int,
    n_episodes: int,
    n_action_steps: int,
    output_dir: Path,
    adapter: ReplayAdapter,
    max_steps: int | None = None,
) -> dict:
    """Run the open-loop GT replay and return the summary dict (also
    written to ``output_dir/replay_info.json``)."""
    episode_actions, episode_first_states = load_episode_actions(dataset_root)
    n_episodes = min(n_episodes, len(episode_actions))

    env_cfg = LiberoEnvConfig(task=task, task_ids=[task_id], control_mode=control_mode)
    envs = make_env(env_cfg, n_envs=1)
    # make_env returns {suite_name: {task_id: vec_env}} for libero.
    env = next(iter(next(iter(envs.values())).values()))

    env_pre, env_post, obs_step = adapter.make_processors(action_type, n_action_steps)
    # Diagnostics-only state extractor, independent of the treatment pipeline.
    state_probe = adapter.make_state_probe()
    encoding = adapter.action_encoding(action_type)
    provider_mode = adapter.provider_mode(action_type)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    per_episode: list[dict] = []

    with open(trace_path, "w") as trace_f:
        for ep in range(n_episodes):
            provider = GtActionProvider(
                mode=provider_mode,
                obs_step=obs_step,
                n_action_steps=n_action_steps,
                action_encoding=encoding,
                encode=adapter.encode_to_rot6d if encoding == "axis_angle" else None,
                decode=adapter.decode_from_rot6d if encoding == "axis_angle" else None,
            )
            # Real bug #5 regression guard: the action step's chunk counter
            # must re-align at every episode start, exactly like the policy
            # eval's rollout wrapper does (via the study's action processors).
            for step in env_post.steps:
                if hasattr(step, "reset_episode_state"):
                    step.reset_episode_state()
            observation, _ = env.reset(seed=ep)
            success = False
            init_state_pos_err = None
            actions = episode_actions[ep]
            steps_run = 0
            env_max = env.call("_max_episode_steps")[0]
            limit = min(len(actions), max_steps or env_max)

            for t in range(limit):
                obs_proc = preprocess_observation(observation)
                obs_proc = add_envs_task(env, obs_proc)
                if t == 0:
                    # Diagnostics only: how closely does this env reset's fixed
                    # init state match the dataset episode's first frame?
                    state_probe.observation(dict(obs_proc))
                    if state_probe.last_anvil_state is not None:
                        demo_pos = episode_first_states[ep][:3]
                        init_state_pos_err = float(
                            np.linalg.norm(state_probe.last_anvil_state[:3] - demo_pos)
                        )
                obs_proc = env_pre(obs_proc)

                provided = provider(actions[t])
                act_t = torch.from_numpy(provided).unsqueeze(0)
                transition = env_post({ACTION: act_t})
                action_numpy = transition[ACTION].to("cpu").numpy()

                trace_f.write(
                    json.dumps(
                        {
                            "episode": ep,
                            "t": t,
                            "stored": actions[t].tolist(),
                            "provided": provided.tolist(),
                            "native_cmd": action_numpy[0].tolist(),
                        }
                    )
                    + "\n"
                )

                observation, _reward, terminated, truncated, info = env.step(action_numpy)
                steps_run += 1
                if "final_info" in info:
                    success = bool(np.asarray(info["final_info"]["is_success"]).flatten()[0])
                if bool(np.asarray(terminated).flatten()[0]) or bool(np.asarray(truncated).flatten()[0]):
                    break

            per_episode.append(
                {
                    "episode": ep,
                    "success": success,
                    "steps": steps_run,
                    "gt_steps": len(actions),
                    "init_state_pos_err": init_state_pos_err,
                }
            )
            log.info(
                "episode %d/%d: success=%s steps=%d init_pos_err=%s",
                ep + 1, n_episodes, success, steps_run,
                f"{init_state_pos_err:.4f}" if init_state_pos_err is not None else "n/a",
            )

    close_envs(envs)

    pc_success = 100.0 * sum(e["success"] for e in per_episode) / max(len(per_episode), 1)
    summary = {
        "action_type": action_type,
        "dataset_root": str(dataset_root),
        "control_mode": control_mode,
        "task": task,
        "task_id": task_id,
        "n_action_steps": n_action_steps,
        "n_episodes": len(per_episode),
        "pc_success": pc_success,
        "per_episode": per_episode,
        "trace": str(trace_path),
    }
    with open(output_dir / "replay_info.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("GT replay pc_success=%.1f%% (%d episodes) -> %s", pc_success, len(per_episode), output_dir)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from anvil_sim.study import get_study

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="libero_ee")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--control-mode", required=True, choices=["relative", "absolute"])
    parser.add_argument("--task", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    # Resolve the study first so --action-type's choices come from its registry.
    known_study = parser.parse_known_args()[0].study
    study = get_study(known_study)
    parser.add_argument("--action-type", required=True, choices=sorted(study.eval_action_types))
    args = parser.parse_args()
    replay(
        action_type=args.action_type,
        dataset_root=args.dataset_root,
        control_mode=args.control_mode,
        task=args.task,
        task_id=args.task_id,
        n_episodes=args.n_episodes,
        n_action_steps=args.n_action_steps,
        output_dir=args.output_dir,
        adapter=study.replay_adapter,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

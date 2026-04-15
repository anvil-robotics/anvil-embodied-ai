"""CLI entry point for anvil-eval-ros — MCAP replay evaluation via Docker Compose.

Reads split_info.json from checkpoint, maps episode indices to MCAP files,
generates an eval_plan.json, then launches docker-compose.eval.yml.

Usage:
    uv run anvil-eval-ros \\
        --checkpoint outputs/run/checkpoints/000050 \\
        --mcap-root data/raw/placing-block-r1/ \\
        [--output-dir eval_results/ros/] \\
        [--num-eps 5] \\
        [--no-docker]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="anvil-eval-ros: run offline eval by replaying MCAP files through the ROS2 inference node"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint directory (contains pretrained_model/)",
    )
    parser.add_argument(
        "--mcap-root",
        required=True,
        help="Raw MCAP data directory (e.g. data/raw/placing-block-r1/). "
             "MCAP files are sorted to match training episode indices.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: eval_results/{dataset}/{job}/{checkpoint}/ros)",
    )
    parser.add_argument(
        "--episodes",
        help="Manual comma-separated episode indices (overrides split_info.json)",
    )
    parser.add_argument(
        "--num-eps",
        type=int,
        help="Max episodes to sample per split (random, reproducible via --seed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode sampling (default: 42)",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Print eval_plan.json path and docker compose command instead of running it",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=5.0,
        help="Seconds to wait for inference node warmup before first episode (default: 5.0)",
    )
    parser.add_argument(
        "--inference-drain-sec",
        type=float,
        default=3.0,
        help="Seconds to wait after bag ends for inference pipeline to drain (default: 3.0)",
    )
    parser.add_argument(
        "--image-tag",
        default="latest",
        help="Docker image tag (default: latest)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MCAP collection (mirrors mcap_converter.collect_mcap_files)
# ──────────────────────────────────────────────────────────────────────────────

def collect_mcap_files(mcap_root: Path) -> list[Path]:
    """Recursively collect and sort MCAP files — same order as mcap_converter."""
    mcap_paths = []
    for root, _, files in os.walk(mcap_root):
        for file in sorted(files):
            if file.endswith(".mcap"):
                mcap_paths.append(Path(root) / file)
    return sorted(mcap_paths)


def build_episode_map(mcap_root: Path) -> dict[int, Path]:
    """Return {episode_idx: mcap_path} using sorted MCAP discovery order."""
    files = collect_mcap_files(mcap_root)
    return {idx: path for idx, path in enumerate(files)}


# ──────────────────────────────────────────────────────────────────────────────
# Split info loading
# ──────────────────────────────────────────────────────────────────────────────

def load_split_info(checkpoint_path: Path) -> dict:
    """Load split_info.json from checkpoint pretrained_model/."""
    split_path = checkpoint_path / "pretrained_model" / "split_info.json"
    if not split_path.exists():
        # Fallback: job root
        split_path = checkpoint_path.parent.parent / "split_info.json"

    if split_path.exists():
        data = json.loads(split_path.read_text())
        return {
            "train": data.get("train_episodes", []),
            "val": data.get("val_episodes", []),
            "test": data.get("test_episodes", []),
        }

    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Inference config generation — auto-detect arm count from model config.json
# ──────────────────────────────────────────────────────────────────────────────

# Maps arm name → (ros_prefix, arm_key).
# arm_key is the short key used in joint names (e.g. "follower_r_joint1" → key "r").
_ARM_META: dict[str, tuple[str, str]] = {
    "left":  ("follower_l", "l"),
    "right": ("follower_r", "r"),
}


def _read_action_dim(checkpoint_path: Path) -> int | None:
    """Return the model's action output dimension from config.json, or None."""
    config_path = checkpoint_path / "pretrained_model" / "config.json"
    if not config_path.exists():
        return None
    cfg = json.loads(config_path.read_text())
    shape = cfg.get("output_features", {}).get("action", {}).get("shape")
    return shape[0] if shape else None


def _detect_arms_from_conversion_config(mcap_root: Path) -> list[str] | None:
    """Read conversion_config.yaml to find which arms are used for actions.

    Returns ordered list of arm names (e.g. ["right"]) or None if not found.
    The dataset conversion config lives at data/datasets/{dataset_name}/conversion_config.yaml.
    """
    try:
        import yaml
    except ImportError:
        return None

    # mcap_root is data/raw/{dataset_name}/ → datasets dir is two levels up
    dataset_name = mcap_root.name
    config_path = mcap_root.parent.parent / "datasets" / dataset_name / "conversion_config.yaml"
    if not config_path.exists():
        return None

    cfg = yaml.safe_load(config_path.read_text())
    action_topics: dict = cfg.get("action_topics", {})
    if not action_topics:
        return None

    # Return arm names in topic-definition order
    arm_names = [info.get("arm") for info in action_topics.values() if info.get("arm")]
    return arm_names if arm_names else None


def generate_inference_config(
    checkpoint_path: Path,
    base_yaml_path: Path,
    output_dir: Path,
    mcap_root: Path | None = None,
) -> tuple[Path, dict]:
    """Generate a model-aware inference YAML and return (path, arm_info).

    arm_info keys: gt_topics, pred_topics, arm_names (all lists).
    Falls back to base_yaml_path when YAML or model config is unavailable.

    Arm detection priority:
      1. conversion_config.yaml (most accurate — matches training data exactly)
      2. action_dim / joints_per_arm count (fallback, assumes left-first order)
    """
    try:
        import yaml  # PyYAML — available wherever ROS2 tools are installed
    except ImportError:
        log.warning("[anvil-eval-ros] PyYAML not available — using base inference_eval.yaml")
        return base_yaml_path, _default_arm_info()

    action_dim = _read_action_dim(checkpoint_path)
    if action_dim is None:
        log.warning("[anvil-eval-ros] config.json not found — using base inference_eval.yaml")
        return base_yaml_path, _default_arm_info()

    base_cfg = yaml.safe_load(base_yaml_path.read_text())
    joints_per_arm = len(base_cfg.get("joint_names", {}).get("model_joint_order", []))
    if joints_per_arm == 0:
        log.warning("[anvil-eval-ros] model_joint_order empty — using base inference_eval.yaml")
        return base_yaml_path, _default_arm_info()

    n_arms = action_dim // joints_per_arm
    if n_arms < 1:
        log.warning(
            "[anvil-eval-ros] action_dim=%d < joints_per_arm=%d — using base config",
            action_dim, joints_per_arm,
        )
        return base_yaml_path, _default_arm_info()

    # Determine arm names: prefer conversion_config.yaml (exact training mapping)
    arm_names_ordered: list[str] | None = None
    if mcap_root is not None:
        arm_names_ordered = _detect_arms_from_conversion_config(mcap_root)
        if arm_names_ordered:
            log.info(
                "[anvil-eval-ros] Arm config from conversion_config.yaml: %s", arm_names_ordered
            )

    if not arm_names_ordered:
        # Fallback: assume left-first for n_arms arms
        fallback_order = ["left", "right"]
        arm_names_ordered = fallback_order[:n_arms]
        log.warning(
            "[anvil-eval-ros] conversion_config.yaml not found — assuming arm order: %s",
            arm_names_ordered,
        )

    # Trim to n_arms in case conversion_config has more arms than the model
    arm_names_ordered = arm_names_ordered[:n_arms]

    # Build arms section
    new_arms: dict = {}
    gt_topics: list[str] = []
    pred_topics: list[str] = []
    arm_names: list[str] = []

    for i, arm_name in enumerate(arm_names_ordered):
        meta = _ARM_META.get(arm_name)
        if meta is None:
            log.warning("[anvil-eval-ros] Unknown arm name '%s', skipping", arm_name)
            continue
        ros_prefix, _ = meta
        new_arms[arm_name] = {
            "ros_prefix": ros_prefix,
            "command_topic": f"/eval/{ros_prefix}_forward_position_controller/commands",
            "action_start": i * joints_per_arm,
            "action_end": (i + 1) * joints_per_arm,
        }
        gt_topics.append(f"/{ros_prefix}_forward_position_controller/commands")
        pred_topics.append(f"/eval/{ros_prefix}_forward_position_controller/commands")
        arm_names.append(arm_name)

    base_cfg["arms"] = new_arms

    # Trim arm_mapping so state vector dimension matches action_dim.
    # multi_process.py builds observation.state by iterating arm_mapping keys,
    # so arm_mapping must only contain the arms actually used by this model.
    # arm_mapping keys are short keys (e.g. "l", "r"); values are arm names.
    orig_arm_mapping: dict = base_cfg.get("joint_names", {}).get("arm_mapping", {})
    filtered_arm_mapping = {k: v for k, v in orig_arm_mapping.items() if v in set(arm_names)}
    # If conversion_config gave us a specific arm order, rebuild arm_mapping in that order
    # so sorted(arm_mapping.keys()) in multi_process.py matches action slice order.
    ordered_arm_mapping: dict = {}
    for arm_name in arm_names:
        meta = _ARM_META.get(arm_name)
        if meta:
            _, arm_key = meta
            if arm_key in filtered_arm_mapping:
                ordered_arm_mapping[arm_key] = filtered_arm_mapping[arm_key]
    base_cfg.setdefault("joint_names", {})["arm_mapping"] = ordered_arm_mapping or filtered_arm_mapping

    config_path = output_dir / "inference_eval_generated.yaml"
    config_path.write_text(yaml.dump(base_cfg, default_flow_style=False, allow_unicode=True))

    log.info(
        "[anvil-eval-ros] Generated inference config: %d arm(s), action_dim=%d → %s",
        n_arms, action_dim, config_path,
    )

    return config_path, {"gt_topics": gt_topics, "pred_topics": pred_topics, "arm_names": arm_names}


def _default_arm_info() -> dict:
    """Dual-arm fallback matching the static inference_eval.yaml."""
    return {
        "gt_topics": [
            "/follower_l_forward_position_controller/commands",
            "/follower_r_forward_position_controller/commands",
        ],
        "pred_topics": [
            "/eval/follower_l_forward_position_controller/commands",
            "/eval/follower_r_forward_position_controller/commands",
        ],
        "arm_names": ["left", "right"],
    }


def _ros2_list(items: list[str]) -> str:
    """Format a Python list as a ROS2 parameter array string: ["a","b"]."""
    inner = ",".join(f'"{x}"' for x in items)
    return f"[{inner}]"


# ──────────────────────────────────────────────────────────────────────────────
# Output dir resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_output_dir(checkpoint_path: Path, mcap_root: Path) -> Path:
    dataset_name = mcap_root.name
    checkpoint_name = checkpoint_path.name
    parent = checkpoint_path.parent
    job_name = parent.parent.name if parent.name == "checkpoints" else parent.name
    return Path("eval_results") / dataset_name / job_name / checkpoint_name / "ros"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    mcap_root = Path(args.mcap_root).resolve()

    if not checkpoint_path.exists():
        log.error("[anvil-eval-ros] Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)
    if not mcap_root.exists():
        log.error("[anvil-eval-ros] MCAP root not found: %s", mcap_root)
        sys.exit(1)

    # 1. Build episode → MCAP path mapping
    ep_map = build_episode_map(mcap_root)
    if not ep_map:
        log.error("[anvil-eval-ros] No MCAP files found under %s", mcap_root)
        sys.exit(1)
    log.info("[anvil-eval-ros] Found %d MCAP files in %s", len(ep_map), mcap_root)

    # 2. Determine episodes to evaluate
    rng = random.Random(args.seed)

    if args.episodes:
        # Manual override
        manual = [int(x.strip()) for x in args.episodes.split(",")]
        episodes_to_eval = [(idx, "replay") for idx in manual]
    else:
        split_info = load_split_info(checkpoint_path)

        if split_info:
            episodes_to_eval = []
            for split_name in ("train", "val", "test"):
                ep_list = split_info.get(split_name, [])
                if args.num_eps is not None:
                    n = min(len(ep_list), args.num_eps)
                    ep_list = rng.sample(ep_list, n)
                for ep_idx in ep_list:
                    episodes_to_eval.append((ep_idx, split_name))
        else:
            # No split info: compute a default 8:1:1 split from available MCAP files.
            # WARNING: this may not match the actual training split.
            all_eps = sorted(ep_map.keys())
            total = len(all_eps)
            shuffled = rng.sample(all_eps, total)
            n_test = max(1, round(total * 0.1))
            n_val = max(1, round(total * 0.1))
            n_train = total - n_val - n_test
            split_info = {
                "train": shuffled[:n_train],
                "val": shuffled[n_train : n_train + n_val],
                "test": shuffled[n_train + n_val :],
            }
            log.warning(
                "[anvil-eval-ros] split_info.json not found — using default 8:1:1 split "
                "(%d train / %d val / %d test). This may NOT match the actual training split.",
                n_train, n_val, n_test,
            )
            episodes_to_eval = []
            for split_name in ("train", "val", "test"):
                ep_list = split_info.get(split_name, [])
                if args.num_eps is not None:
                    n = min(len(ep_list), args.num_eps)
                    ep_list = rng.sample(ep_list, n)
                for ep_idx in ep_list:
                    episodes_to_eval.append((ep_idx, split_name))

    # Filter to episodes that have corresponding MCAP files
    valid_episodes = []
    skipped = 0
    for ep_idx, split_label in sorted(episodes_to_eval, key=lambda x: x[0]):
        if ep_idx not in ep_map:
            log.warning(
                "[anvil-eval-ros] Episode %d has no MCAP file (only %d files available), skipping",
                ep_idx,
                len(ep_map),
            )
            skipped += 1
            continue
        valid_episodes.append({
            "episode_idx": ep_idx,
            "split_label": split_label,
            "mcap_path": str(ep_map[ep_idx]),
        })

    if not valid_episodes:
        log.error("[anvil-eval-ros] No valid episodes to evaluate")
        sys.exit(1)

    if skipped:
        log.warning("[anvil-eval-ros] Skipped %d episodes with no MCAP file", skipped)

    log.info("[anvil-eval-ros] Evaluating %d episodes", len(valid_episodes))

    # 3. Resolve output dir
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        resolve_output_dir(checkpoint_path, mcap_root).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("[anvil-eval-ros] Output dir: %s", output_dir)

    # 4. Generate eval_plan.json
    eval_plan = {
        "checkpoint_path": str(checkpoint_path),
        "mcap_root": str(mcap_root),
        "output_dir": str(output_dir),
        "episodes": valid_episodes,
    }
    plan_path = output_dir / "eval_plan.json"
    plan_path.write_text(json.dumps(eval_plan, indent=2))
    log.info("[anvil-eval-ros] Eval plan written: %s (%d episodes)", plan_path, len(valid_episodes))

    # 5. Find repo root (for docker-compose.eval.yml path)
    repo_root = Path(__file__).resolve().parents[4]
    compose_file = repo_root / "docker-compose.eval.yml"
    if not compose_file.exists():
        log.error("[anvil-eval-ros] docker-compose.eval.yml not found at %s", compose_file)
        sys.exit(1)

    # 5b. Auto-generate inference config from model's action shape
    base_inference_yaml = repo_root / "configs" / "lerobot_control" / "inference_eval.yaml"
    inference_config_path, arm_info = generate_inference_config(
        checkpoint_path, base_inference_yaml, output_dir, mcap_root=mcap_root
    )

    # 6. Build docker compose command
    env = {
        **os.environ,
        "MODEL_PATH": str(checkpoint_path),
        "MCAP_ROOT": str(mcap_root),
        "OUTPUT_DIR": str(output_dir),
        "EVAL_PLAN_FILE": str(plan_path),
        "IMAGE_TAG": args.image_tag,
        # Pass tuning params to nodes via env (picked up by compose)
        "EVAL_WARMUP_SEC": str(args.warmup_sec),
        "EVAL_DRAIN_SEC": str(args.inference_drain_sec),
        # Auto-generated inference config (arm count derived from model)
        "INFERENCE_CONFIG_FILE": str(inference_config_path),
        # eval-recorder topics derived from arm config
        "EVAL_GT_TOPICS": _ros2_list(arm_info["gt_topics"]),
        "EVAL_PRED_TOPICS": _ros2_list(arm_info["pred_topics"]),
        "EVAL_ARM_NAMES": _ros2_list(arm_info["arm_names"]),
    }

    compose_cmd = [
        "docker", "compose",
        "-f", str(compose_file),
        "up",
        "--build",
        "--abort-on-container-exit",
        "--exit-code-from", "eval-recorder",
    ]

    if args.no_docker:
        log.info("[anvil-eval-ros] --no-docker set. Eval plan: %s", plan_path)
        log.info("[anvil-eval-ros] Run manually:\n  %s", " ".join(compose_cmd))
        return

    # 7. Run Docker Compose
    log.info("[anvil-eval-ros] Starting Docker Compose eval stack...")
    result = subprocess.run(compose_cmd, env=env, cwd=str(repo_root))

    if result.returncode != 0:
        log.error("[anvil-eval-ros] Docker Compose exited with code %d", result.returncode)
        sys.exit(result.returncode)

    # 8. Print summary
    summary_path = output_dir / "metrics_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        overall = summary.get("overall", {})
        log.info(
            "[anvil-eval-ros] === Eval complete ===\n"
            "  Episodes: %s\n"
            "  Mean MAE: %.4f\n"
            "  Mean RMSE: %.4f\n"
            "  Results: %s",
            overall.get("count", "?"),
            overall.get("mean_mae", float("nan")),
            overall.get("mean_rmse", float("nan")),
            output_dir,
        )
    else:
        log.info("[anvil-eval-ros] Eval complete. Results: %s", output_dir)


if __name__ == "__main__":
    main()

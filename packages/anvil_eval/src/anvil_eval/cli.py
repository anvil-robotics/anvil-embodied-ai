"""CLI entry point for anvil-eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import EvalConfig
from .dataset import EvaluationDataset, get_episode_indices
from .evaluator import EpisodeEvaluator, load_model
from .metrics import compute_episode_metrics
from .plotting import plot_episode_joints, plot_summary_box_plot
from .reporting import write_metrics_csv, write_metrics_summary

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure basic logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Anvil offline model evaluation — replay dataset episodes through trained policies"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory (should contain pretrained_model/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the LeRobot v3.0 dataset directory",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        help="Manual comma-separated list of episode indices (overrides --split)",
    )
    parser.add_argument(
        "--num-eps",
        type=int,
        help="Randomly sample N episodes from the selected split(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: eval_results/{dataset_name}/{job_name}/{checkpoint}/raw)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu, default: cuda)",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        help="Task prompt for VLA models (overrides anvil_config.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling episodes (default: 42)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    setup_logging()
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)

    # 1. Load anvil_config.json from checkpoint
    anvil_cfg_path = checkpoint_path / "pretrained_model" / "anvil_config.json"
    anvil_cfg = {}
    if anvil_cfg_path.exists():
        try:
            anvil_cfg = json.loads(anvil_cfg_path.read_text())
            log.info("[anvil-eval] Loaded anvil_config.json")
        except Exception as e:
            log.warning("[anvil-eval] Failed to read anvil_config.json: %s", e)

    # 2. Build EvalConfig
    config = EvalConfig(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        num_episodes=args.num_eps,
        device=args.device,
        task_description=args.task_description or anvil_cfg.get("task_description"),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        seed=args.seed,
    )

    # 3. Initialize Dataset and Split info
    try:
        eval_dataset = EvaluationDataset(dataset_path)
        split_info = eval_dataset.resolve_splits(anvil_cfg, checkpoint_path=checkpoint_path)
        log.info("[anvil-eval] Dataset: %s (%d episodes)", dataset_path.name, eval_dataset.total_episodes)
    except Exception as e:
        log.error("[anvil-eval] Failed to load dataset: %s", e)
        sys.exit(1)

    # 4. Determine episodes to evaluate
    if args.episodes:
        episode_list = [int(idx.strip()) for idx in args.episodes.split(",")]
        # For manual episodes, we'll label them as 'manual' unless we can find their split
        episodes_to_eval = []
        for idx in episode_list:
            label = "manual"
            for s_name, s_list in split_info.items():
                if idx in s_list:
                    label = s_name
                    break
            episodes_to_eval.append((idx, label))
    else:
        episodes_to_eval = get_episode_indices(split_info, config)

    if not episodes_to_eval:
        log.error("[anvil-eval] No episodes selected for evaluation")
        sys.exit(1)

    log.info("[anvil-eval] Selected %d episodes for evaluation", len(episodes_to_eval))

    # 5. Load Model
    log.info("[anvil-eval] Loading model from %s...", checkpoint_path)
    try:
        model, preprocessor, postprocessor, model_type = load_model(str(checkpoint_path), config.device)
    except Exception as e:
        import traceback
        log.error("[anvil-eval] Failed to load model: %s", e)
        traceback.print_exc()
        sys.exit(1)

    # 6. Initialize Evaluator
    evaluator = EpisodeEvaluator(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        model_type=model_type,
        device=config.device,
        anvil_cfg=anvil_cfg,
        task_description=config.task_description,
        joint_names=eval_dataset.joint_names,
    )

    # 7. Create output directory
    if not config.output_dir:
        config.output_dir = config.resolve_output_dir()

    log.info("[anvil-eval] Results will be saved to: %s", config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 8. Run Evaluation loop
    all_metrics = []
    for ep_idx, split_label in episodes_to_eval:
        log.info("[anvil-eval] Evaluating episode %d (%s)...", ep_idx, split_label)
        
        frame_indices = eval_dataset.get_episode_frames(ep_idx)
        result = evaluator.evaluate_episode(eval_dataset.dataset, frame_indices, ep_idx, split_label)
        
        # Compute metrics
        metrics = compute_episode_metrics(
            result.predicted, result.ground_truth, result.joint_names, ep_idx, split_label
        )
        all_metrics.append(metrics)
        
        # Plot episode
        plot_path = plots_dir / f"episode_{ep_idx:04d}_{split_label}.png"
        plot_episode_joints(result.predicted, result.ground_truth, result.joint_names, metrics, plot_path)
        log.info("[anvil-eval] Episode %d MAE: %.4f", ep_idx, metrics.mae)

    # 9. Save Summary Results
    log.info("[anvil-eval] Writing summary reports...")
    write_metrics_summary(all_metrics, config.output_dir / "metrics_summary.json")
    write_metrics_csv(all_metrics, config.output_dir / "metrics_per_episode.csv")
    
    # Summary plot
    plot_summary_box_plot(all_metrics, eval_dataset.joint_names, plots_dir / "summary_per_joint_mae.png")

    log.info("[anvil-eval] Evaluation complete!")


if __name__ == "__main__":
    main()

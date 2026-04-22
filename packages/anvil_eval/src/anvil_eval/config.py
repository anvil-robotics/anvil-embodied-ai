"""Evaluation configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for offline model evaluation."""

    checkpoint_path: Path
    dataset_path: Path
    num_episodes: int = 3
    split: str = "all"
    device: str = "cuda"
    task_description: str | None = None
    output_dir: Path | None = None
    seed: int = 42

    def resolve_output_dir(self) -> Path:
        """Auto-generate output directory if not provided.

        Convention: eval_results/{dataset_name}/{job_name}/{checkpoint}/raw
        where job_name is the training run directory (parent of 'checkpoints/')
        and checkpoint is the step identifier (e.g. '000015' or 'last').
        The /raw suffix distinguishes offline dataset eval from ROS2 replay eval (/ros).
        """
        if self.output_dir:
            return self.output_dir

        dataset_name = self.dataset_path.name

        # Resolve symlinks so 'last' → actual step folder (e.g. '010000')
        resolved = self.checkpoint_path.resolve()
        checkpoint_name = resolved.name

        # Infer job_name from path: .../checkpoints/000015 → parent.parent
        parent = resolved.parent
        if parent.name == "checkpoints":
            job_name = parent.parent.name
        else:
            job_name = parent.name

        return Path("eval_results") / dataset_name / job_name / checkpoint_name / "raw"

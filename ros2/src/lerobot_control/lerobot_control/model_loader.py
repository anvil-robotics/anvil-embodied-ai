"""Model loading utilities for LeRobot models.

Simplified model loader that uses LeRobot's built-in PolicyProcessorPipeline
for pre/post processing instead of custom implementations.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_deterministic_mode(seed: int = 42):
    """
    Set PyTorch to deterministic mode for reproducible inference.

    This should be called before loading the model for consistent results.

    Args:
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def reset_model_state(model):
    """
    Reset model's internal state for reproducible inference.

    For ACT models, this clears the action queue so each inference
    starts fresh with a new action chunk prediction.

    Args:
        model: The policy model (ACTPolicy, DiffusionPolicy, etc.)
    """
    if hasattr(model, "reset"):
        model.reset()


class ModelLoader:
    """
    Load trained LeRobot models from checkpoints.

    Uses LeRobot's built-in PolicyProcessorPipeline for preprocessing
    and postprocessing, eliminating the need for custom implementations.

    Supports:
    - ACT (Action Chunking Transformer)
    - Diffusion Policy
    - SmolVLA (Small Vision-Language-Action)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: str | None = None,
        logger=None,
        deterministic: bool = False,
        seed: int = 42,
        config_overrides: dict = None,
    ):
        """
        Initialize model loader.

        Args:
            model_path: Path to model checkpoint directory
            device: Device for inference ("cuda" or "cpu")
            model_type: Model type ("act", "diffusion", or "smolvla"). None = auto-detect
                        from config.json in the checkpoint.
            logger: Optional ROS2 logger
            deterministic: If True, enable deterministic mode
            seed: Random seed for deterministic mode
            config_overrides: Dict of config values to override at load time
                e.g. {"temporal_ensemble_coeff": 0.01, "n_action_steps": 1}
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model_type = model_type
        self.logger = logger
        self.deterministic = deterministic
        self.seed = seed
        self.config_overrides = config_overrides or {}
        self._model = None
        self._pre_processor = None
        self._post_processor = None
        self._orig_n_action_steps: int | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Auto-detect pretrained_model subdirectory
        pretrained_model_path = self.model_path / "pretrained_model"
        if pretrained_model_path.exists() and (pretrained_model_path / "config.json").exists():
            self._log("info", f"Found pretrained_model subdirectory: {pretrained_model_path}")
            self.model_path = pretrained_model_path

        # Auto-detect model type from checkpoint if not provided
        if self.model_type is None:
            self.model_type = self._detect_model_type()

        # Load anvil_config.json (custom training flags persisted by anvil-trainer)
        self._anvil_config = self._load_anvil_config()

    def _log(self, level: str, msg: str):
        """Log message using ROS2 logger or print."""
        if self.logger:
            try:
                getattr(self.logger, level)(msg)
            except ValueError:
                # ROS2 logger can fail with "severity cannot be changed between calls"
                print(f"[{level.upper()}] {msg}")
        else:
            print(f"[{level.upper()}] {msg}")

    def _detect_model_type(self) -> str | None:
        """Read model type from config.json in the checkpoint directory."""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text()).get("type")
        return None

    def _load_anvil_config(self) -> dict:
        """Read anvil_config.json written by anvil-trainer, if present."""
        path = self.model_path / "anvil_config.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @property
    def anvil_config(self) -> dict:
        """Custom training flags persisted by anvil-trainer (e.g. use_delta_actions)."""
        return self._anvil_config

    @property
    def checkpoint_n_action_steps(self) -> int | None:
        """Original n_action_steps from checkpoint before any overrides were applied."""
        return self._orig_n_action_steps

    @property
    def chunk_size(self) -> int | None:
        """Get model chunk size from config."""
        if (
            self._model
            and hasattr(self._model, "config")
            and hasattr(self._model.config, "chunk_size")
        ):
            return self._model.config.chunk_size
        return None

    @property
    def n_action_steps(self) -> int | None:
        """Get model n_action_steps from config."""
        if (
            self._model
            and hasattr(self._model, "config")
            and hasattr(self._model.config, "n_action_steps")
        ):
            return self._model.config.n_action_steps
        return None

    def load(self):
        """
        Load model from checkpoint.

        Returns:
            Loaded model in eval mode
        """
        self._log("info", f"Loading {self.model_type} model from: {self.model_path}")

        if self.deterministic:
            set_deterministic_mode(self.seed)
            self._log("info", f"Deterministic mode enabled with seed={self.seed}")

        try:
            if self.model_type == "act":
                from lerobot.policies.act.modeling_act import ACTPolicy

                model = ACTPolicy.from_pretrained(str(self.model_path))
            elif self.model_type == "diffusion":
                from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

                model = DiffusionPolicy.from_pretrained(str(self.model_path))
            elif self.model_type == "smolvla":
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

                model = SmolVLAPolicy.from_pretrained(str(self.model_path))
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            model = model.to(self.device)
            model.eval()
            self._model = model
            self._log("info", f"Model loaded successfully on {self.device}")

            # Snapshot checkpoint n_action_steps before any overrides
            if hasattr(model, "config"):
                self._orig_n_action_steps = getattr(model.config, "n_action_steps", None)

            # Apply config overrides after loading
            self._apply_config_overrides(model)

            return model

        except ImportError as e:
            self._log("error", f"lerobot package not installed: {e}")
            raise
        except Exception as e:
            self._log("error", f"Failed to load model: {e}")
            raise

    def _apply_config_overrides(self, model) -> None:
        """Apply config overrides after model is loaded.

        For temporal_ensemble_coeff, we also need to create the ensembler
        since it's only created in __init__ if the coeff was set at load time.
        """
        if not self.config_overrides or not hasattr(model, "config"):
            return

        config = model.config
        overrides_applied = []

        for key, value in self.config_overrides.items():
            if value is not None and hasattr(config, key):
                old_val = getattr(config, key)
                setattr(config, key, value)
                overrides_applied.append(f"{key}: {old_val} -> {value}")

        if overrides_applied:
            self._log("info", "Applied config overrides:")
            for override in overrides_applied:
                self._log("info", f"  - {override}")

        # Special handling: create temporal_ensembler if enabling at runtime
        if "temporal_ensemble_coeff" in self.config_overrides:
            coeff = self.config_overrides["temporal_ensemble_coeff"]
            if coeff is not None and not hasattr(model, "temporal_ensembler"):
                self._create_temporal_ensembler(model, coeff)

    def _create_temporal_ensembler(self, model, coeff: float) -> None:
        """Create temporal ensembler for ACT model."""
        try:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            chunk_size = model.config.chunk_size
            model.temporal_ensembler = ACTTemporalEnsembler(coeff, chunk_size)
            self._log(
                "info", f"Created temporal ensembler (coeff={coeff}, chunk_size={chunk_size})"
            )
        except ImportError as e:
            self._log("error", f"Failed to import ACTTemporalEnsembler: {e}")
        except Exception as e:
            self._log("error", f"Failed to create temporal ensembler: {e}")

    def load_with_processors(self) -> tuple[Any, Any, Any]:
        """
        Load model with LeRobot's built-in processor pipelines.

        Returns:
            Tuple of (model, preprocessor_pipeline, postprocessor_pipeline)
        """
        model = self.load()

        pre_processor = None
        post_processor = None

        try:
            from lerobot.processor import PolicyProcessorPipeline

            # Check if processor files exist
            pre_config = self.model_path / "policy_preprocessor.json"
            post_config = self.model_path / "policy_postprocessor.json"

            if pre_config.exists():
                pre_processor = PolicyProcessorPipeline.from_pretrained(
                    str(self.model_path), config_filename="policy_preprocessor.json"
                )
                # Move to device if the pipeline supports it
                if hasattr(pre_processor, "to") and callable(pre_processor.to):
                    try:
                        pre_processor.to(self.device)
                    except (TypeError, AttributeError):
                        pass  # Pipeline doesn't support device movement
                self._pre_processor = pre_processor
                self._log("info", "Loaded preprocessor pipeline from checkpoint")

            if post_config.exists():
                post_processor = PolicyProcessorPipeline.from_pretrained(
                    str(self.model_path), config_filename="policy_postprocessor.json"
                )
                # Move to device if the pipeline supports it
                if hasattr(post_processor, "to") and callable(post_processor.to):
                    try:
                        post_processor.to(self.device)
                    except (TypeError, AttributeError):
                        pass  # Pipeline doesn't support device movement
                self._post_processor = post_processor
                self._log("info", "Loaded postprocessor pipeline from checkpoint")

        except FileNotFoundError as e:
            self._log("warn", f"Processor pipelines not found: {e}")
        except Exception as e:
            self._log("warn", f"Failed to load processor pipelines: {e}")

        return model, pre_processor, post_processor

    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """
        Validate model checkpoint path.

        Args:
            model_path: Path to check

        Returns:
            True if path appears to be valid model checkpoint
        """
        path = Path(model_path)
        if not path.exists():
            return False

        has_config = (path / "config.json").exists()
        has_model = (path / "pretrained_model").exists() or (path / "model.safetensors").exists()
        return has_config or has_model

"""Frozen LeRobot policy composed with an embodiment adapter artifact."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .artifact import AdapterArtifact


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class EmbodimentAdaptedPolicy:
    """Wrap a frozen HF policy without changing its weights or processors."""

    def __init__(
        self,
        *,
        model: Any,
        preprocessor: Any,
        postprocessor: Any,
        artifact: AdapterArtifact,
        device: str | torch.device,
    ):
        if preprocessor is None or postprocessor is None:
            raise ValueError("the frozen policy requires both processor pipelines")
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.artifact = artifact
        self.device = torch.device(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.artifact.residual.eval()

    @torch.no_grad()
    def predict_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        self._previous_reference_state: np.ndarray | None = None
        if "observation.state" not in observation:
            raise ValueError("OpenArm 2 observation.state is required")
        raw_state = _to_numpy(observation["observation.state"])
        if raw_state.ndim == 2 and raw_state.shape[0] == 1:
            raw_state = raw_state[0]
        if raw_state.shape != (16,):
            raise ValueError(f"expected one 16-D state, got {raw_state.shape}")

        mapped_state = self.artifact.bridge.target_state_to_policy(
            raw_state, self._previous_reference_state
        ).values
        self._previous_reference_state = mapped_state.copy()
        policy_input = dict(observation)
        policy_input["observation.state"] = torch.as_tensor(
            mapped_state, dtype=torch.float32
        ).unsqueeze(0)
        processed = _move_to_device(self.preprocessor(policy_input), self.device)
        reference_chunk = self.model.predict_action_chunk(processed)
        post_input = (
            reference_chunk.squeeze(0)
            if isinstance(reference_chunk, torch.Tensor) and reference_chunk.ndim == 3
            else reference_chunk
        )
        reference_absolute = self.postprocessor.process_action(post_input)
        reference_absolute = _to_numpy(reference_absolute)
        if reference_absolute.ndim == 3 and reference_absolute.shape[0] == 1:
            reference_absolute = reference_absolute[0]

        bridge_chunk = self.artifact.bridge.policy_chunk_to_target(
            reference_absolute, raw_state
        ).values
        state_tensor = torch.as_tensor(
            raw_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        bridge_tensor = torch.as_tensor(
            bridge_chunk, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        corrected, _ = self.artifact.residual(state_tensor, bridge_tensor)
        return corrected

    def train(self, mode: bool = True) -> EmbodimentAdaptedPolicy:
        # Only the residual may be trained explicitly by the adapter trainer.
        self.model.eval()
        self.artifact.residual.train(mode)
        return self

    def reset(self) -> None:
        """Clear policy and IK continuity state at an episode boundary."""
        self._previous_reference_state = None
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

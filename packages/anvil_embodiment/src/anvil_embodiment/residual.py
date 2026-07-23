"""Bounded residual chunk adapter and its supervised loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from anvil_shared.embodiment import ResidualAdapterContract
from torch import Tensor, nn

from .kinematics import RobotModelSpec, torch_forward_kinematics

ACTIVE_JOINT_INDICES = tuple(index for index in range(16) if index not in (7, 15))


class ResidualChunkAdapter(nn.Module):
    """Small post-bridge network that cannot command unbounded corrections."""

    active_indices: Tensor
    correction_bounds: Tensor
    target_lower: Tensor
    target_upper: Tensor

    def __init__(
        self,
        config: ResidualAdapterContract,
        correction_bounds: Tensor,
        target_ranges: Tensor,
    ):
        super().__init__()
        if correction_bounds.shape != (16,):
            raise ValueError("correction_bounds must have shape [16]")
        if target_ranges.shape != (16, 2):
            raise ValueError("target_ranges must have shape [16, 2]")
        self.config = config
        self.gru = nn.GRU(
            input_size=50,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, len(ACTIVE_JOINT_INDICES))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer(
            "active_indices",
            torch.tensor(ACTIVE_JOINT_INDICES, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "correction_bounds",
            correction_bounds.detach().to(dtype=torch.float32),
        )
        self.register_buffer("target_lower", target_ranges[:, 0].detach().to(dtype=torch.float32))
        self.register_buffer("target_upper", target_ranges[:, 1].detach().to(dtype=torch.float32))

    def forward(self, current_state: Tensor, bridge_chunk: Tensor) -> tuple[Tensor, Tensor]:
        if current_state.ndim != 2 or current_state.shape[-1] != 16:
            raise ValueError("current_state must have shape [B, 16]")
        if bridge_chunk.ndim != 3 or bridge_chunk.shape[-1] != 16:
            raise ValueError("bridge_chunk must have shape [B, T, 16]")
        if bridge_chunk.shape[0] != current_state.shape[0]:
            raise ValueError("current_state and bridge_chunk batch sizes differ")

        steps = bridge_chunk.shape[1]
        current = current_state[:, None, :].expand(-1, steps, -1)
        delta = bridge_chunk - current
        phase = torch.linspace(
            0.0,
            1.0,
            steps,
            dtype=bridge_chunk.dtype,
            device=bridge_chunk.device,
        )
        horizon = torch.stack([torch.sin(torch.pi * phase), torch.cos(torch.pi * phase)], dim=-1)
        horizon = horizon[None].expand(bridge_chunk.shape[0], -1, -1)
        features = torch.cat([current, bridge_chunk, delta, horizon], dim=-1)
        hidden, _ = self.gru(features)
        active_raw = self.output(hidden)

        residual = torch.zeros_like(bridge_chunk)
        active_bounds = self.correction_bounds[self.active_indices].to(dtype=bridge_chunk.dtype)
        residual[..., self.active_indices] = torch.tanh(active_raw) * active_bounds
        corrected = torch.maximum(
            torch.minimum(bridge_chunk + residual, self.target_upper), self.target_lower
        )
        effective_residual = corrected - bridge_chunk
        return corrected, effective_residual


@dataclass(frozen=True)
class AdapterLossWeights:
    joint: float = 1.0
    pose: float = 0.25
    velocity: float = 0.05
    residual: float = 0.01
    huber_beta: float = 0.05
    position_scale_m: float = 0.05
    orientation_scale_rad: float = 0.25


def _smooth_zero(value: Tensor, beta: float) -> Tensor:
    return F.smooth_l1_loss(value, torch.zeros_like(value), beta=beta)


def _geodesic_angle(predicted: Tensor, target: Tensor) -> Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = torch.clamp(
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    vee = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
    return torch.atan2(sine, cosine)


def compute_adapter_loss(
    *,
    corrected: Tensor,
    residual: Tensor,
    target: Tensor,
    target_ranges: Tensor,
    target_model: RobotModelSpec,
    correction_bounds: Tensor,
    weights: AdapterLossWeights = AdapterLossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Loss in target-robot coordinates; no HF/reference joint angles are compared."""
    if corrected.shape != target.shape or corrected.shape != residual.shape:
        raise ValueError("corrected, residual, and target shapes must match")
    if corrected.ndim != 3 or corrected.shape[-1] != 16:
        raise ValueError("adapter loss expects [B, T, 16] chunks")

    active = torch.tensor(ACTIVE_JOINT_INDICES, device=corrected.device)
    widths = (target_ranges[:, 1] - target_ranges[:, 0]).clamp_min(1e-6)
    normalized_error = (corrected - target) / widths
    joint_loss = F.smooth_l1_loss(
        normalized_error[..., active],
        torch.zeros_like(normalized_error[..., active]),
        beta=weights.huber_beta,
    )

    pose_terms = []
    for side_index, side in enumerate(("right", "left")):
        start = side_index * 8
        predicted_position, predicted_rotation = torch_forward_kinematics(
            target_model, side, corrected[..., start : start + 7]
        )
        target_position, target_rotation = torch_forward_kinematics(
            target_model, side, target[..., start : start + 7]
        )
        position_error = (predicted_position - target_position) / weights.position_scale_m
        orientation_error = (
            _geodesic_angle(predicted_rotation, target_rotation) / weights.orientation_scale_rad
        )
        pose_terms.append(_smooth_zero(position_error, weights.huber_beta))
        pose_terms.append(_smooth_zero(orientation_error, weights.huber_beta))
    pose_loss = torch.stack(pose_terms).mean()

    if corrected.shape[1] > 1:
        predicted_velocity = corrected[:, 1:] - corrected[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        velocity_error = (predicted_velocity - target_velocity) / widths
        velocity_loss = F.smooth_l1_loss(
            velocity_error[..., active],
            torch.zeros_like(velocity_error[..., active]),
            beta=weights.huber_beta,
        )
    else:
        velocity_loss = corrected.new_zeros(())

    safe_bounds = correction_bounds.clamp_min(1e-6)
    residual_loss = torch.mean((residual[..., active] / safe_bounds[active]) ** 2)
    total = (
        weights.joint * joint_loss
        + weights.pose * pose_loss
        + weights.velocity * velocity_loss
        + weights.residual * residual_loss
    )
    return total, {
        "loss": total.detach(),
        "joint_loss": joint_loss.detach(),
        "pose_loss": pose_loss.detach(),
        "velocity_loss": velocity_loss.detach(),
        "residual_loss": residual_loss.detach(),
    }

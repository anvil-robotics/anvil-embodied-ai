"""RTC compatibility adapter for LeRobot 0.6's VLA-JEPA policy."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION
from torch import Tensor


def sample_vla_jepa_actions(
    action_head: Any,
    conditioning_tokens: Tensor,
    state: Tensor | None = None,
    *,
    noise: Tensor | None = None,
    rtc_processor: RTCProcessor | None = None,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
) -> Tensor:
    """Run VLA-JEPA's Euler flow sampler with optional RTC prefix guidance."""
    batch_size = conditioning_tokens.shape[0]
    if noise is None:
        actions = torch.randn(
            batch_size,
            action_head.action_horizon,
            action_head.config.action_dim,
            dtype=conditioning_tokens.dtype,
            device=conditioning_tokens.device,
        )
    else:
        actions = noise.to(
            device=conditioning_tokens.device,
            dtype=conditioning_tokens.dtype,
        ).clone()

    num_steps = int(action_head.num_inference_timesteps)
    if num_steps <= 0:
        raise ValueError(f"num_inference_timesteps must be positive, got {num_steps}")
    dt = 1.0 / num_steps
    use_rtc = rtc_processor is not None and (
        inference_delay is not None or prev_chunk_left_over is not None
    )

    def predict_velocity(seq: Tensor, timesteps: Tensor) -> Tensor:
        hidden_states = action_head._build_inputs(  # noqa: SLF001
            conditioning_tokens,
            seq,
            state,
            timesteps,
        )
        pred = action_head.model(
            hidden_states=hidden_states,
            encoder_hidden_states=conditioning_tokens,
            timestep=timesteps,
        )
        return action_head.action_decoder(pred[:, -action_head.action_horizon :])

    for step in range(num_steps):
        t = step / float(num_steps)
        timestep = int(t * action_head.config.action_num_timestep_buckets)
        timesteps = torch.full(
            (batch_size,),
            timestep,
            device=conditioning_tokens.device,
            dtype=torch.long,
        )

        if use_rtc:
            assert rtc_processor is not None
            # RTCProcessor follows the PI0 convention (time 1 -> 0 and the
            # opposite velocity sign). VLA-JEPA integrates noise -> action from
            # t=0 -> 1, so map conventions exactly as EVO1 does upstream.
            guided = rtc_processor.denoise_step(
                x_t=actions,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=int(inference_delay or 0),
                time=1.0 - t,
                original_denoise_step_partial=lambda seq, ts=timesteps: -predict_velocity(seq, ts),
                execution_horizon=execution_horizon,
            )
            velocity = -guided
        else:
            with torch.no_grad():
                velocity = predict_velocity(actions, timesteps)

        actions = actions + dt * velocity

    return actions


@torch.no_grad()
def _rtc_predict_action_chunk(
    policy: Any,
    batch: dict[str, Tensor],
    noise: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """RTC-compatible replacement for VLAJEPAPolicy.predict_action_chunk."""
    policy.eval()
    policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])  # noqa: SLF001
    inputs = policy._prepare_model_inputs(batch, training=False)  # noqa: SLF001

    with torch.no_grad():
        conditioning_tokens, _ = policy.model._encode_qwen(  # noqa: SLF001
            inputs["images"],
            inputs["instructions"],
            need_action_tokens=False,
        )

    state = inputs.get("state")
    actions = sample_vla_jepa_actions(
        policy.model.action_model,
        conditioning_tokens.float(),
        state.float() if state is not None else None,
        noise=noise,
        rtc_processor=policy.rtc_processor,
        inference_delay=kwargs.get("inference_delay"),
        prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
        execution_horizon=kwargs.get("execution_horizon"),
    )
    return actions.to(device=policy.config.device, dtype=torch.float32)


def install_vla_jepa_rtc(policy: Any) -> None:
    """Attach RTC sampling to one loaded VLA-JEPA policy instance."""
    if getattr(policy, "_anvil_vla_jepa_rtc_installed", False):
        return

    rtc_config = getattr(policy.config, "rtc_config", None)
    if rtc_config is None:
        raise ValueError("VLA-JEPA RTC requires policy.config.rtc_config")

    required_policy = ("eval", "_prepare_model_inputs", "model", "_queues")
    missing_policy = [name for name in required_policy if not hasattr(policy, name)]
    action_head = getattr(getattr(policy, "model", None), "action_model", None)
    required_head = (
        "_build_inputs",
        "model",
        "action_decoder",
        "action_horizon",
        "num_inference_timesteps",
    )
    missing_head = [name for name in required_head if not hasattr(action_head, name)]
    if missing_policy or missing_head:
        details = ", ".join(missing_policy + [f"action_head.{name}" for name in missing_head])
        raise RuntimeError(
            f"Installed LeRobot VLA-JEPA is incompatible with Anvil RTC adapter; missing: {details}"
        )

    policy.rtc_processor = RTCProcessor(rtc_config)
    policy.predict_action_chunk = MethodType(_rtc_predict_action_chunk, policy)
    policy._anvil_vla_jepa_rtc_installed = True

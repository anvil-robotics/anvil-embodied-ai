"""Policy capability registry for Anvil's LeRobot integration."""

from __future__ import annotations

SUPPORTED_POLICIES = {
    "act",
    "diffusion",
    "smolvla",
    "pi0",
    "pi05",
    "molmoact2",
    "groot",
    "multi_task_dit",
    "evo1",
    "fastwam",
    "vla_jepa",
}

LANGUAGE_CONDITIONED_POLICIES = {
    "smolvla",
    "pi0",
    "pi05",
    "molmoact2",
    "groot",
    "multi_task_dit",
    "evo1",
    "fastwam",
    "vla_jepa",
}

RTC_CHUNK_POLICIES = {
    "smolvla",
    "pi0",
    "pi05",
    "molmoact2",
    "groot",
    "evo1",
}

# Policies whose upstream implementation can be adapted to the RTC runtime but
# which retain synchronous inference by default for backwards compatibility.
OPTIONAL_RTC_CHUNK_POLICIES = {
    "vla_jepa",
}

RTC_CAPABLE_POLICIES = RTC_CHUNK_POLICIES | OPTIONAL_RTC_CHUNK_POLICIES

SYNC_CHUNK_POLICIES = {
    "multi_task_dit",
    "fastwam",
    "vla_jepa",
    "pi05",
}


def is_supported_policy(model_type: str | None) -> bool:
    return model_type in SUPPORTED_POLICIES


def is_language_conditioned(model_type: str | None) -> bool:
    return model_type in LANGUAGE_CONDITIONED_POLICIES


def uses_rtc_inference(model_type: str | None) -> bool:
    return model_type in RTC_CHUNK_POLICIES


def supports_rtc_inference(model_type: str | None) -> bool:
    return model_type in RTC_CAPABLE_POLICIES


def resolve_rtc_inference(model_type: str | None, enabled: bool | None = None) -> bool:
    """Resolve the selected runtime without changing existing policy defaults."""
    if enabled is None:
        return uses_rtc_inference(model_type)
    if enabled and not supports_rtc_inference(model_type):
        raise ValueError(f"Policy '{model_type}' does not support RTC inference")
    return enabled


def uses_sync_chunk_inference(model_type: str | None) -> bool:
    return model_type in SYNC_CHUNK_POLICIES

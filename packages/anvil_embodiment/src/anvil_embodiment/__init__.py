"""Frozen-policy embodiment bridge and residual adapter."""

from .artifact import AdapterArtifact, load_adapter_artifact
from .bridge import BridgeError, BridgeResult, KinematicEmbodimentBridge
from .policy import EmbodimentAdaptedPolicy
from .residual import ResidualChunkAdapter

__all__ = [
    "AdapterArtifact",
    "BridgeError",
    "BridgeResult",
    "EmbodimentAdaptedPolicy",
    "KinematicEmbodimentBridge",
    "ResidualChunkAdapter",
    "load_adapter_artifact",
]

__version__ = "0.1.0"

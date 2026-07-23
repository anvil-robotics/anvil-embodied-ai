"""Shared pure-Python utilities used across anvil packages."""
from anvil_shared.provenance import git_provenance
from anvil_shared.splits import (
    compute_split_episodes,
    load_split_info,
    save_split_info,
)

__version__ = "0.1.0"

from .embodiment import (
    AdapterVectorContract,
    EmbodimentAdapterSpec,
    EmbodimentContract,
    EmbodimentError,
    ExperimentContract,
    GripperCalibration,
    IKContract,
    KinematicModelContract,
    PolicyBinding,
    ResidualAdapterContract,
    normalize_from_limits,
)

__all__ = [
    "AdapterVectorContract",
    "EmbodimentAdapterSpec",
    "EmbodimentContract",
    "EmbodimentError",
    "ExperimentContract",
    "GripperCalibration",
    "IKContract",
    "KinematicModelContract",
    "PolicyBinding",
    "ResidualAdapterContract",
    "compute_split_episodes",
    "git_provenance",
    "load_split_info",
    "normalize_from_limits",
    "save_split_info",
]

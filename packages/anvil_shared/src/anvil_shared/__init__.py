"""Shared pure-Python utilities used across anvil packages."""
from anvil_shared.action_types import (
    ACTION_TYPE_ALIASES,
    normalize_action_type,
)
from anvil_shared.dataset_config import (
    read_conversion_config,
    resolve_action_encoding,
    resolve_action_type,
    resolve_data_space,
    resolve_observation_encoding,
)
from anvil_shared.ee_encodings import (
    IMPLEMENTED_ACTION_ENCODINGS,
    OBSERVATION_ROTATION_LAYOUTS,
    VALID_ACTION_ENCODINGS,
    VALID_OBSERVATION_ENCODINGS,
    encode_rotation,
    observation_state_dim_per_arm,
    observation_state_names_per_arm,
)
from anvil_shared.ee_transform import ee_obs_abs_forward
from anvil_shared.provenance import git_provenance
from anvil_shared.rotation import (
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
    rot6d_to_matrix,
)
from anvil_shared.splits import (
    compute_split_episodes,
    load_split_info,
    save_split_info,
)

__version__ = "0.1.0"

__all__ = [
    "ACTION_TYPE_ALIASES",
    "normalize_action_type",
    "read_conversion_config",
    "resolve_action_encoding",
    "resolve_action_type",
    "resolve_data_space",
    "resolve_observation_encoding",
    "IMPLEMENTED_ACTION_ENCODINGS",
    "OBSERVATION_ROTATION_LAYOUTS",
    "VALID_ACTION_ENCODINGS",
    "VALID_OBSERVATION_ENCODINGS",
    "encode_rotation",
    "observation_state_dim_per_arm",
    "observation_state_names_per_arm",
    "compute_split_episodes",
    "load_split_info",
    "matrix_to_quat",
    "matrix_to_rot6d",
    "quat_to_matrix",
    "rot6d_to_matrix",
    "save_split_info",
    "git_provenance",
    "ee_obs_abs_forward",
]

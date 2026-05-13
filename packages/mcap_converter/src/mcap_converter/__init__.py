"""
MCAP to LeRobot Dataset Converter

A modular conversion pipeline for transforming ROS2 MCAP recordings
into LeRobot v3.0 format datasets.
"""

__version__ = "0.1.0"

from .config import ActionSource, DataConfig, load_config
from .core import LeRobotWriter, McapReader
from .exceptions import (
    ConfigurationError,
    DataExtractionError,
    DatasetWriteError,
    McapConverterError,
    McapReadError,
)

__all__ = [
    "__version__",
    # Core modules
    "McapReader",
    "LeRobotWriter",
    # Config
    "ActionSource",
    "DataConfig",
    "load_config",
    # Exceptions
    "McapConverterError",
    "ConfigurationError",
    "McapReadError",
    "DataExtractionError",
    "DatasetWriteError",
]

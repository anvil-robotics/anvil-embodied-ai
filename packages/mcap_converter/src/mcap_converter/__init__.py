"""
MCAP to LeRobot Dataset Converter

A modular conversion pipeline for transforming ROS2 MCAP recordings
into LeRobot v3.0 format datasets.
"""

__version__ = "0.1.0"

# Core modules
from .config import DEFAULT_DATA_CONFIG, ConfigLoader, DataConfig
from .core import DataExtractor, LeRobotWriter, McapReader, TimeAligner
from .exceptions import (
    ConfigurationError,
    DataExtractionError,
    DatasetWriteError,
    McapConverterError,
    McapReadError,
    TimeAlignmentError,
)

__all__ = [
    "__version__",
    # Core modules
    "McapReader",
    "DataExtractor",
    "TimeAligner",
    "LeRobotWriter",
    # Config
    "ConfigLoader",
    "DataConfig",
    "DEFAULT_DATA_CONFIG",
    # Exceptions
    "McapConverterError",
    "ConfigurationError",
    "McapReadError",
    "DataExtractionError",
    "TimeAlignmentError",
    "DatasetWriteError",
]

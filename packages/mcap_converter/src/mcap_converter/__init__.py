"""
MCAP to LeRobot Dataset Converter

A modular conversion pipeline for transforming ROS2 MCAP recordings
into LeRobot v3.0 format datasets.
"""

__version__ = "0.1.0"

# Core modules
from .core import McapReader, DataExtractor, TimeAligner, LeRobotWriter
from .config import ConfigLoader, DataConfig, DEFAULT_DATA_CONFIG
from .exceptions import (
    McapConverterError,
    ConfigurationError,
    McapReadError,
    DataExtractionError,
    TimeAlignmentError,
    DatasetWriteError,
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

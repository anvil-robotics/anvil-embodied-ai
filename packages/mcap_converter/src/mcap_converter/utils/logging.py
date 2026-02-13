"""Centralized logging utilities for mcap_converter package."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger for the package.

    Args:
        name: Logger name (will be prefixed with 'mcap_converter.')
        level: Optional logging level. Defaults to INFO.

    Returns:
        Configured logger instance.

    Example:
        logger = get_logger("cli.convert")
        logger.info("Starting conversion...")
    """
    logger = logging.getLogger(f"mcap_converter.{name}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level or logging.INFO)

    return logger


def set_log_level(level: int) -> None:
    """
    Set log level for all mcap_converter loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logging.getLogger("mcap_converter").setLevel(level)

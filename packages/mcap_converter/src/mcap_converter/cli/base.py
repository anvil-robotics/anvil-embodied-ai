"""Shared CLI utilities for mcap_converter package."""

import argparse
import sys
from typing import Optional

from ..utils.logging import get_logger


class BaseCLI:
    """
    Base class for CLI tools with common utilities.

    Provides:
    - Consistent logging
    - Standard output formatting
    - Common argument handling

    Example:
        class ConvertCLI(BaseCLI):
            def __init__(self):
                super().__init__("convert", "Convert MCAP to LeRobot dataset")

            def setup_args(self):
                self.parser.add_argument("-i", "--input", required=True)
    """

    def __init__(self, name: str, description: str):
        """
        Initialize CLI base.

        Args:
            name: CLI tool name (for logging)
            description: Tool description (for help)
        """
        self.name = name
        self.logger = get_logger(f"cli.{name}")
        self.parser = argparse.ArgumentParser(description=description)

    def print_header(self, title: str, width: int = 70) -> None:
        """Print formatted header."""
        print("=" * width)
        print(title)
        print("=" * width)

    def print_step(self, step: int, total: int, message: str) -> None:
        """Print progress step."""
        print(f"[{step}/{total}] {message}")

    def print_success(self, message: str) -> None:
        """Print success message."""
        print(f"[OK] {message}")

    def print_error(self, message: str) -> None:
        """Print error message."""
        print(f"[ERROR] {message}", file=sys.stderr)

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"[WARNING] {message}")

    def run(self, args: Optional[list] = None) -> int:
        """
        Run the CLI tool.

        Args:
            args: Command line arguments (defaults to sys.argv)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            parsed_args = self.parser.parse_args(args)
            return self.execute(parsed_args)
        except KeyboardInterrupt:
            self.print_warning("Interrupted by user")
            return 130
        except Exception as e:
            self.print_error(f"Error: {e}")
            self.logger.exception("Unhandled exception")
            return 1

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the CLI command. Override in subclass.

        Args:
            args: Parsed command line arguments

        Returns:
            Exit code
        """
        raise NotImplementedError("Subclass must implement execute()")

"""
Logging Configuration
=====================

Provides standardized logging setup for all auto_roto modules.
"""

import logging
from typing import Optional


def setup_logging(
    verbose: bool = False,
    name: str = "AutoRoto",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO
        name: Logger name (default: "AutoRoto")
        log_file: Optional file path to write logs to
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO

    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(message)s'

    # Configure root logger if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format=format_string,
            datefmt='%H:%M:%S'
        )

    # Get or create named logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            format_string,
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "AutoRoto") -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

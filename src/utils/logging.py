# src/utils/logging.py

"""
Simple logging helpers for the benchmark suite.

Usage:

    from src.utils.logging import setup_logger

    logger = setup_logger("benchmark.runner", log_level="INFO")
    logger.info("Starting benchmark...")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "inferbench",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name (module / component name).
        log_level: Logging level as a string, e.g. "INFO", "DEBUG".
        log_file: Optional path to a log file. If provided, logs go to both
                  console and file; otherwise only console.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if setup_logger is called multiple times.
    if logger.handlers:
        return logger

    # Resolve log level string to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Common formatter for all handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

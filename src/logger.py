"""Centralized logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str | None = "logs/autochem.log",
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """Configure loguru logger for Auto-ChemInstruct.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. None disables file logging.
        rotation: When to rotate log files.
        retention: How long to keep rotated logs.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level="DEBUG",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            rotation=rotation,
            retention=retention,
        )

    logger.debug("Logging configured: level={}, file={}", level, log_file)


def get_logger():
    """Get the configured logger instance."""
    return logger

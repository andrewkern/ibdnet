"""Logging utilities for consistent CLI logging."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a concise formatter."""

    if logging.getLogger().handlers:
        # Assume another configuration already exists.
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger after ensuring configuration."""

    configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]

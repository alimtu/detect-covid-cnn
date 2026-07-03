"""Simple console logging configuration."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "covid", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger with a consistent format.

    The root logging configuration is applied only once so repeated calls do
    not attach duplicate handlers.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        A configured :class:`logging.Logger`.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)

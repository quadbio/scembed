"""Logging setup for the package."""

import logging
import os
from typing import Literal


def _setup_logger() -> logging.Logger:
    from rich.console import Console
    from rich.logging import RichHandler

    logger = logging.getLogger(__name__)
    level = os.environ.get("LOGLEVEL", logging.INFO)
    logger.setLevel(level=level)
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=logger.level == logging.DEBUG)
    logger.addHandler(ch)

    # this prevents double outputs
    logger.propagate = False
    return logger


def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | Literal[10, 20, 30, 40, 50],
) -> None:
    """Set the logging level for the scembed logger.

    Parameters
    ----------
    level
        Logging level. Can be a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or logging constants (logging.DEBUG=10, logging.INFO=20, logging.WARNING=30,
        logging.ERROR=40, logging.CRITICAL=50).

    Examples
    --------
    >>> import scembed.logging
    >>> scembed.logging.set_log_level("DEBUG")
    >>> scembed.logging.set_log_level(logging.INFO)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)
    # Update handlers to ensure they respect the new level
    for handler in logger.handlers:
        handler.setLevel(level)


logger = _setup_logger()

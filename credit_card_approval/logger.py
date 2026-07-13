"""
logger.py
---------
Centralised logging setup.  Import ``get_logger`` anywhere in the project.

Usage
-----
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Training started")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from config import LOG_FILE


def get_logger(name: str = "credit_approval") -> logging.Logger:
    """Return a logger with both console and rotating-file handlers.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the function is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (INFO and above) ──────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── Rotating file handler (DEBUG and above, max 5 MB × 3 backups) ─────
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

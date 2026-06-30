"""CBPS package logging configuration.

Provides a package-level logger that users can control via standard
Python logging mechanisms.

Examples
--------
>>> from cbps import set_verbosity
>>> set_verbosity(1)  # Enable INFO-level progress messages
>>> set_verbosity(2)  # Enable DEBUG-level diagnostics
>>> set_verbosity(0)  # Restore default (WARNING only)
"""
import logging

# Package-level logger
logger = logging.getLogger('cbps')
logger.addHandler(logging.NullHandler())  # No output by default


def set_verbosity(level: int = 0):
    """Set CBPS package verbosity level.

    Parameters
    ----------
    level : int
        0 = WARNING only (default, production)
        1 = INFO (progress messages)
        2 = DEBUG (detailed diagnostics)
    """
    if level == 0:
        logger.setLevel(logging.WARNING)
    elif level == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    # Add console handler if not already present
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in logger.handlers
    )
    if not has_console:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[CBPS] %(levelname)s: %(message)s'))
        logger.addHandler(handler)

"""Shared logger: every module logs through `rich` so messages are coloured and carry the source line."""

import logging

from rich.logging import RichHandler

# Configured once on import; modules then only call `get_logger(__name__)`.
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

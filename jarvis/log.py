"""Leveled logging.

Replaces ad-hoc `print("[prefix] ...")` diagnostics with the stdlib logging
module, so users can raise verbosity or add timestamps without editing code:

    JARVIS_LOG_LEVEL=DEBUG
    JARVIS_LOG_FORMAT="%(asctime)s %(levelname)s %(name)s: %(message)s"

The default format is just the message, so existing console output is
unchanged. User-facing UI lines (rich console, [USER]/[NAME] prefixes) are
deliberately NOT routed through here.
"""
import logging
import os
import sys

_configured = False


def log(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        level = os.getenv("JARVIS_LOG_LEVEL", "INFO").upper()
        fmt = os.getenv("JARVIS_LOG_FORMAT", "%(message)s")
        logging.basicConfig(stream=sys.stdout,
                            level=getattr(logging, level, logging.INFO),
                            format=fmt)
        _configured = True
    return logging.getLogger(name)

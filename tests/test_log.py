"""Logging helper."""
from jarvis.log import log


def test_log_returns_named_logger():
    lg = log("router")
    assert lg.name == "router"
    lg.info("[router] test line")   # must not raise
    lg.warning("[router] warn line")

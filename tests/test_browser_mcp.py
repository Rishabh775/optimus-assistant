"""Browser MCP tool: real-Chrome control through the local MCP server."""
import pytest

from jarvis import config
from jarvis.tools import browser_mcp as mcp


class FakeClient:
    def __init__(self, reply="✓ navigated"):
        self.reply = reply
        self.calls = []

    def call_browser(self, commands):
        self.calls.append(commands)
        return self.reply


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_BROWSER_MCP_ENABLED", "1")
    old = mcp._client
    mcp._client = None
    yield
    mcp._client = old


def test_run_delegates_to_client(monkeypatch):
    fake = FakeClient()
    mcp._client = fake
    assert mcp.run("snapshot -i") == "✓ navigated"
    assert fake.calls == ["snapshot -i"]


def test_run_rejects_empty_commands():
    with pytest.raises(ValueError):
        mcp.run("   ")


def test_run_respects_disable_switch(monkeypatch):
    monkeypatch.setenv("JARVIS_BROWSER_MCP_ENABLED", "0")
    mcp._client = None
    with pytest.raises(RuntimeError):
        mcp.run("snapshot -i")


def test_tool_wrapper_and_registration(monkeypatch):
    from jarvis import builtin_tools as bt
    assert "browser_mcp" in bt.TOOL_FUNCS
    assert any(s["name"] == "browser_mcp" for s in bt.TOOL_SCHEMAS)

    def boom(commands):
        raise RuntimeError("extension not connected")
    monkeypatch.setattr(mcp, "run", boom)
    out = bt.TOOL_FUNCS["browser_mcp"](commands="snapshot -i")
    assert out.startswith("error:")


def test_confirmation_only_for_actions():
    from jarvis.builtin_tools import needs_confirmation
    assert needs_confirmation("browser_mcp", {"commands": "snapshot -i"}) is None
    assert needs_confirmation(
        "browser_mcp", {"commands": "navigate https://x\nclick @e3"})
    assert needs_confirmation(
        "browser_mcp", {"commands": "type hello\npress Enter"})

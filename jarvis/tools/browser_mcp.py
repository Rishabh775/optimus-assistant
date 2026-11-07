"""Browser MCP client: let Soundwave drive the user's real Chrome.

The Browser MCP server (browsermcp.io) is a local stdio MCP server:
    npx -y @browsermcp/mcp@latest
It exposes one tool, `browser`, whose `commands` argument is a newline
separated mini-language (navigate / snapshot / click / type / press /
get text / ...). This module speaks MCP JSON-RPC over the child process's
stdin/stdout and keeps one server alive for the process lifetime.

Why: the existing Playwright tools run a separate headless browser with no
logins. Browser MCP uses the user's real Chrome (via their extension), so
sites they are signed into work. It is still read-mostly: click/type calls
are confirmation-gated in builtin_tools.

Config:
    JARVIS_BROWSER_MCP_ENABLED=1
    JARVIS_BROWSER_MCP_CMD="npx -y @browsermcp/mcp@latest"
"""
import atexit
import json
import queue
import shlex
import subprocess
import threading
import time

from .. import config
from ..log import log as _mk_log

log = _mk_log("browser_mcp")

DEFAULT_CMD = "npx -y @browsermcp/mcp@latest"
_PROTOCOL = "2024-11-05"


class _Client:
    """One long-lived MCP stdio session."""

    def __init__(self, cmd: str):
        self.cmd = cmd
        self.proc = None
        self.q = queue.Queue()
        self._lock = threading.RLock()
        self._next_id = 1
        self._reader = None

    # -- process/IO ------------------------------------------------------
    def _start(self) -> None:
        self.proc = subprocess.Popen(
            shlex.split(self.cmd), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        self._reader = threading.Thread(
            target=self._read, name="browser-mcp-reader", daemon=True)
        self._reader.start()
        self._request("initialize", {
            "protocolVersion": _PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "soundwave", "version": "0.1"},
        }, timeout=60)
        self._notify("notifications/initialized")
        log.info("browser MCP ready (%s)", self.cmd)

    def _read(self) -> None:
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.q.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except Exception:
            return

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def _request(self, method: str, params: dict = None,
                 timeout: float = 90) -> dict:
        with self._lock:
            if not self.proc or self.proc.poll() is not None:
                self._start()
            rid = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                        "params": params or {}})
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Browser MCP timed out after {timeout:.0f}s "
                        f"({method})")
                try:
                    msg = self.q.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            "Browser MCP server exited — is the Chrome "
                            "extension connected?")
                    continue
                if msg.get("id") != rid:
                    continue                 # unrelated notification
                if "error" in msg:
                    err = msg["error"]
                    raise RuntimeError(
                        f"Browser MCP error: {err.get('message', err)}")
                return msg.get("result") or {}

    # -- public ----------------------------------------------------------
    def call_browser(self, commands: str) -> str:
        result = self._request("tools/call", {
            "name": "browser", "arguments": {"commands": commands},
        })
        text = "\n".join(c.get("text", "") for c in result.get("content", [])
                         if c.get("type") == "text").strip()
        if result.get("isError"):
            raise RuntimeError(text or "browser command failed")
        return text or "(no output)"

    def close(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        self.proc = None


_client = None
_client_lock = threading.Lock()


def _get() -> _Client:
    global _client
    if not config.get_bool("JARVIS_BROWSER_MCP_ENABLED"):
        raise RuntimeError("Browser MCP is disabled "
                           "(set JARVIS_BROWSER_MCP_ENABLED=1)")
    with _client_lock:
        if _client is None:
            _client = _Client(config.get("JARVIS_BROWSER_MCP_CMD")
                              or DEFAULT_CMD)
    return _client


def run(commands: str) -> str:
    """Execute Browser MCP commands against the user's real Chrome."""
    commands = str(commands or "").strip()
    if not commands:
        raise ValueError("commands is empty")
    try:
        return _get().call_browser(commands)
    except RuntimeError as e:
        # a dead server usually just needs one cold restart
        if "exited" in str(e).lower() or "timed out" in str(e).lower():
            log.warning("browser MCP restarting after: %s", e)
            global _client
            with _client_lock:
                if _client:
                    _client.close()
                _client = None
            return _get().call_browser(commands)
        raise


atexit.register(lambda: _client and _client.close())

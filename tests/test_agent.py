"""Agent tool loop: confirmation gating of consequential actions."""
from types import SimpleNamespace

from jarvis import builtin_tools as bt
from jarvis.agent import Agent
from jarvis.router import Route


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _TC:
    def __init__(self, name, args):
        self.id = "1"
        self.function = SimpleNamespace(name=name, arguments=args)


class _Resp:
    def __init__(self, msg):
        self.choices = [SimpleNamespace(message=msg)]


class _Route(SimpleNamespace):
    provider, model, reason = "cloud", "fake", "test"


class _FakeLLM:
    """Returns a tool call, then a final text answer."""
    def __init__(self, tool_name, tool_args):
        self.tool_name, self.tool_args = tool_name, tool_args
        self.calls = 0
    def chat(self, messages, tools=None, route_text=None):
        self.calls += 1
        if self.calls == 1:
            return _Resp(_Msg(tool_calls=[_TC(self.tool_name, self.tool_args)])), _Route()
        return _Resp(_Msg(content="done")), _Route()


class _NoMem:
    def query(self, *a, **k): return ""
    def add(self, *a, **k): return ""


def _agent(confirm, tool_name, tool_args):
    a = Agent.__new__(Agent)
    a.llm = _FakeLLM(tool_name, tool_args)
    a.memory = _NoMem()
    a.history = [{"role": "system", "content": "s"}]
    a.confirm = confirm
    return a


def test_denied_blocks_tool(monkeypatch):
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "shell_run",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "ran")
    a = _agent(confirm=lambda reason, fn, args: False,
               tool_name="shell_run", tool_args='{"cmd": "rm -rf build"}')
    a.ask("delete build")
    assert ran["n"] == 0            # not executed


def test_confirmed_runs_tool(monkeypatch):
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "shell_run",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "ran")
    a = _agent(confirm=lambda reason, fn, args: True,
               tool_name="shell_run", tool_args='{"cmd": "rm -rf build"}')
    a.ask("delete build")
    assert ran["n"] == 1


def test_no_hook_allows(monkeypatch):
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "shell_run",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "ran")
    a = _agent(confirm=None, tool_name="shell_run", tool_args='{"cmd": "rm -rf build"}')
    a.ask("delete build")
    assert ran["n"] == 1


def test_safe_command_not_gated(monkeypatch):
    asked = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "shell_run", lambda **k: "ok")
    a = _agent(confirm=lambda *x: asked.__setitem__("n", asked["n"] + 1) or True,
               tool_name="shell_run", tool_args='{"cmd": "ls -la"}')
    a.ask("list files")
    assert asked["n"] == 0          # safe -> no prompt

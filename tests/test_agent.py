"""Agent tool loop: confirmation gating of consequential actions."""
import json
import os
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
    monkeypatch.delenv("JARVIS_AUTO", raising=False)
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "shell_run",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "ran")
    a = _agent(confirm=lambda reason, fn, args, mandatory=False: False,
               tool_name="shell_run", tool_args='{"cmd": "rm -rf build"}')
    a.ask("delete build")
    assert ran["n"] == 0            # not executed


def test_confirmed_runs_tool(monkeypatch):
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "shell_run",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "ran")
    a = _agent(confirm=lambda reason, fn, args, mandatory=False: True,
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


def test_out_of_workspace_write_denied_without_hook(monkeypatch):
    monkeypatch.delenv("JARVIS_AUTO", raising=False)
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "write_file",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "wrote")
    target = os.path.expanduser("~/jarvis_agent_out_of_ws.txt")
    a = _agent(confirm=None, tool_name="write_file",
               tool_args=json.dumps({"path": target, "content": "x"}))
    a.ask("write a file")
    assert ran["n"] == 0            # mandatory action, no hook -> denied


def test_out_of_workspace_write_runs_when_confirmed(monkeypatch):
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "write_file",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "wrote")
    target = os.path.expanduser("~/jarvis_agent_out_of_ws.txt")
    a = _agent(confirm=lambda *x: True, tool_name="write_file",
               tool_args=json.dumps({"path": target, "content": "x"}))
    a.ask("write a file")
    assert ran["n"] == 1


def test_workspace_write_not_gated(monkeypatch):
    asked = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "write_file", lambda **k: "wrote")
    a = _agent(confirm=lambda *x: asked.__setitem__("n", asked["n"] + 1) or True,
               tool_name="write_file",
               tool_args='{"path": "agent_tmp_newfile_xyz.txt", "content": "x"}')
    a.ask("write a file")
    assert asked["n"] == 0          # workspace -> no prompt


def test_auto_mode_bypasses_confirmation(monkeypatch):
    monkeypatch.setenv("JARVIS_AUTO", "1")
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "write_file",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "wrote")
    target = os.path.expanduser("~/jarvis_agent_out_of_ws.txt")
    a = _agent(confirm=lambda *x: False, tool_name="write_file",
               tool_args=json.dumps({"path": target, "content": "x"}))
    a.ask("write a file")
    assert ran["n"] == 1            # mandatory + hooked, but AUTO overrides


def test_auto_mode_bypasses_denying_hook(monkeypatch):
    monkeypatch.setenv("JARVIS_AUTO", "1")
    ran = {"n": 0}
    monkeypatch.setitem(bt.TOOL_FUNCS, "shell_run",
                        lambda **k: ran.__setitem__("n", ran["n"] + 1) or "ran")
    a = _agent(confirm=lambda *x: False, tool_name="shell_run",
               tool_args='{"cmd": "rm -rf build"}')
    a.ask("delete build")
    assert ran["n"] == 1            # hook says no, AUTO wins


def test_auto_slash_command_toggle(monkeypatch):
    from jarvis import __main__ as jm
    monkeypatch.delenv("JARVIS_AUTO", raising=False)
    monkeypatch.setattr(jm.console, "print", lambda *a, **k: None)

    consumed = jm._handle_auto_command("/auto")   # toggle -> on
    assert consumed
    assert os.environ.get("JARVIS_AUTO") == "1"

    consumed = jm._handle_auto_command("/auto off")
    assert consumed and os.environ.get("JARVIS_AUTO") == "0"

    consumed = jm._handle_auto_command("/auto on")
    assert consumed and os.environ.get("JARVIS_AUTO") == "1"

    # non-command passes through
    assert not jm._handle_auto_command("list files")


def test_auto_slash_command_ignored_for_regular_text(monkeypatch):
    from jarvis import __main__ as jm
    assert not jm._handle_auto_command("what is 2+2")

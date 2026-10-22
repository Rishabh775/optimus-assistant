"""Streaming (on_delta) materialization tests for router + agent plumbing."""
from types import SimpleNamespace

import pytest
import os

from jarvis.router import LLM
from jarvis import config


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tc(index, tid=None, name=None, args=None):
    return SimpleNamespace(index=index, id=tid,
                           function=SimpleNamespace(name=name,
                                                    arguments=args))


class _FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.seen = []

    def create(self, **kwargs):
        self.seen.append(kwargs)
        assert "on_delta" not in kwargs  # never reaches the API layer
        assert "stream" in kwargs
        return iter(self.chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


@pytest.fixture()
def llm(monkeypatch):
    os.environ.setdefault("OPENCODE_API_KEY", "test-key")
    return LLM()


def test_content_stream_deltas_in_order(llm):
    chunks = [_chunk("hel"), _chunk("lo "), _chunk("world")]
    llm.cloud = _FakeClient(chunks)
    deltas = []
    resp, route = llm.chat([{"role": "user", "content": "hi"}],
                           route_text="hi", on_delta=deltas.append)
    assert deltas == ["hel", "lo ", "world"]
    assert resp.choices[0].message.content == "hello world"
    assert resp.choices[0].message.tool_calls is None
    assert route.provider == "cloud"


def test_tool_call_stream_assembled(llm):
    chunks = [
        _chunk(tool_calls=[_tc(0, "call_1", "shell", '{"cmd":"ls')]),
        _chunk(tool_calls=[_tc(0, None, None, ' -la"}')]),
        _chunk(tool_calls=[_tc(1, "call_2", "py_read", '{"path":"a"}')]),
    ]
    llm.cloud = _FakeClient(chunks)
    deltas = []
    resp, _route = llm.chat([{"role": "user", "content": "do it"}],
                            route_text="do it", on_delta=deltas.append)
    msg = resp.choices[0].message
    assert deltas == []            # tool assembly is silent
    assert msg.content is None     # tool-only turn -> no content, like real API
    assert [tc.function.name for tc in msg.tool_calls] == ["shell", "py_read"]
    assert msg.tool_calls[0].function.arguments == '{"cmd":"ls -la"}'


def test_mixed_content_and_tool_stream(llm):
    chunks = [_chunk("Let me check."),
              _chunk(tool_calls=[_tc(0, "c1", "shell", '{"cmd":"x"}')])]
    llm.cloud = _FakeClient(chunks)
    deltas = []
    resp, _route = llm.chat([{"role": "user", "content": "go"}],
                            route_text="go", on_delta=deltas.append)
    msg = resp.choices[0].message
    assert deltas == ["Let me check."]
    assert msg.content == "Let me check."
    assert msg.tool_calls[0].function.name == "shell"


def test_agent_forwards_on_delta_to_llm(monkeypatch):
    """Agent.ask passes the callback through (no stream when absent)."""
    import jarvis.agent as agent_mod

    seen = {}

    class _FakeRouter:
        def chat(self, messages, tools=None, route_text=None, **kw):
            seen.update(kw)
            msg = SimpleNamespace(content="done", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)]), \
                SimpleNamespace(provider="cloud", model="x", reason="r")

    agent = agent_mod.Agent.__new__(agent_mod.Agent)
    agent.llm = _FakeRouter()
    agent.history = [{"role": "system", "content": "system stub"}]
    agent.memory = SimpleNamespace(query=lambda t: None,
                                   add=lambda *a, **k: None)
    monkeypatch.setattr(agent_mod, "facts", SimpleNamespace(text=lambda: ""),
                        raising=False)
    out = agent.ask("hello", on_delta=lambda d: None)
    assert out == "done"
    assert seen.get("on_delta") is not None
    seen.clear()
    agent.ask("hello again")
    assert "on_delta" not in seen

"""Routing: intent classification, model validation, circuit breaker."""
import os
import pytest

from jarvis import router
from jarvis.router import Route, LLM, pick_route, needs_tools, _supported


def test_word_boundary_matching():
    # "capital" must NOT match "api"; "digital" must NOT match "git"
    assert "code" not in pick_route("what is the capital of France").reason
    assert "code" not in pick_route("my digital photos").reason
    assert "code" not in pick_route("send an email to bob").reason


def test_code_intent_beats_action_verbs():
    assert "code" in pick_route("write a python function to sort a list").reason
    assert "code" in pick_route("fix this bug in my script").reason
    assert "action" in pick_route("list my files").reason


def test_action_and_default_routing():
    assert "action" in pick_route("open youtube").reason
    assert "default" in pick_route("hey how's it going").reason


def test_needs_tools():
    assert needs_tools("run a command")
    assert needs_tools("can you check the weather?")
    assert not needs_tools("tell me a joke")


def test_supported_falls_back():
    assert _supported("mimo-v2.6-flash") == "mimo-v2.6-flash"
    assert _supported("not-a-real-model") == router.DEFAULT_CLOUD_MODEL


class _Comp:
    def __init__(self, fail_models=()):
        self.fail_models = set(fail_models)
        self.tried = []
    def create(self, **kw):
        self.tried.append(kw["model"])
        if kw["model"] in self.fail_models:
            raise RuntimeError("model down")
        class M: pass
        msg = M(); msg.content = f"ans by {kw['model']}"; msg.tool_calls = None
        ch = M(); ch.message = msg; ch.choices = [ch]
        r = M(); r.choices = [ch]
        return r


def _llm_with(comp):
    llm = LLM()
    llm.cloud = type("C", (), {})()
    llm.cloud.chat = type("Ch", (), {})()
    llm.cloud.chat.completions = comp
    return llm


def test_fallback_on_primary_failure():
    comp = _Comp(fail_models={"longcat-2.5-preview-free"})
    llm = _llm_with(comp)
    resp, route = llm.chat([{"role": "user", "content": "hi"}], route_text="hi")
    assert route.model == "mimo-v2.6-flash"
    assert comp.tried == ["longcat-2.5-preview-free", "mimo-v2.6-flash"]


def test_circuit_breaker_skips_dead_primary():
    comp = _Comp(fail_models={"longcat-2.5-preview-free"})
    llm = _llm_with(comp)
    llm.chat([{"role": "user", "content": "hi"}], route_text="hi")   # sets cooldown
    comp.tried.clear()
    llm.chat([{"role": "user", "content": "hi"}], route_text="hi")   # within cooldown
    assert comp.tried == ["mimo-v2.6-flash"]      # primary skipped


def test_circuit_breaker_recovers_after_cooldown():
    comp = _Comp(fail_models={"longcat-2.5-preview-free"})
    llm = _llm_with(comp)
    llm.chat([{"role": "user", "content": "hi"}], route_text="hi")
    router._cloud_down_until = 0.0                 # cooldown expires
    comp.tried.clear()
    llm.chat([{"role": "user", "content": "hi"}], route_text="hi")
    assert comp.tried[0] == "longcat-2.5-preview-free"

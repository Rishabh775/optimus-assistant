"""Wake conversation flow: echo guard, stop phrases, empty-command cooldown."""
import pytest

from jarvis import __main__ as jm
from jarvis import wake, voice


def test_is_stop_normalization():
    for s in ["stop", "Thanks!", "that's all.", "  THANK YOU  ", "goodbye", "never-mind"]:
        assert jm._is_stop(s), s
    for s in ["stop light is red", "show my calendar", "thanks for nothing more please"]:
        assert not jm._is_stop(s), s


def test_is_echo():
    spoken = "Just say the word when you want me to send something."
    assert jm._is_echo("say the word when you want me to send something", spoken)
    assert not jm._is_echo("what's the weather", spoken)
    assert not jm._is_echo("short", spoken)   # < 8 chars


def test_wake_flow(monkeypatch):
    import threading

    class FakeAgent:
        def __init__(self):
            self.calls = []
        def ask(self, q):
            self.calls.append(q)
            return f"answer:{q}"

    agent = FakeAgent()

    def fake_wait_for_wake(*a, **k):
        for i in range(3):
            yield f"hey_soundwave_{i}"
        raise KeyboardInterrupt

    cmds = ["open youtube", "", "what is the capital"]
    ci = {"i": 0}
    def fake_listen_after(phrase_limit=30):
        q = cmds[ci["i"]] if ci["i"] < len(cmds) else ""
        ci["i"] += 1
        return q

    fu = ["answer:open youtube", "thanks"]   # echo, then stop phrase
    fi = {"i": 0}
    def fake_vlisten(timeout=6, phrase_limit=30, energy_threshold=None, model_name=None):
        q = fu[fi["i"]] if fi["i"] < len(fu) else ""
        fi["i"] += 1
        return q

    monkeypatch.setattr(wake, "wait_for_wake", fake_wait_for_wake)
    monkeypatch.setattr(wake, "listen_after_wake", fake_listen_after)
    monkeypatch.setattr(voice, "listen", fake_vlisten)
    monkeypatch.setattr(jm, "_speak_with_interrupt", lambda text, agent: None)
    monkeypatch.setattr(jm, "_greet", lambda: None)
    monkeypatch.setattr(jm.sounds, "ack", lambda: None)
    monkeypatch.setattr(jm.time, "sleep", lambda s: None)

    jm.wake_loop(agent)
    # empty command skipped (no cooldown crash), echo ignored, only genuine asks
    assert agent.calls == ["open youtube", "what is the capital"]

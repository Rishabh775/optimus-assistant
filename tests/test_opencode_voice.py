"""Soundwave voice bridge (opencode_voice.py) - parser + command assembly."""
import subprocess  # noqa: F401  (imported for monkeypatching)
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opencode_voice as bridge  # noqa: E402


TEXT_EVT = ('{"type":"text","part":{"type":"text",'
            '"text":"The build passes now.","time":{"end":1}}}')
STEP = ('{"type":"step_start","part":{"id":"p1"}}')
TOOL = ('{"type":"tool_use","part":{"type":"tool","tool":"bash","state":'
        '{"status":"completed","output":"ok","title":"print"}}}')


def test_parse_collects_all_text_events():
    raw = "\n".join([STEP, TOOL, TEXT_EVT, TEXT_EVT, STEP])
    out = bridge.parse_jsonl(raw)
    assert out == "The build passes now.\n\nThe build passes now."


def test_parse_error_without_text():
    raw = '{"type":"error","error":{"name":"LimitError","data":{"message":"free limit reached"}}}'
    out = bridge.parse_jsonl(raw)
    assert out.startswith("OpenCode reported an error:")
    assert "free limit reached" in out


def test_parse_error_losses_to_text_when_both():
    raw = "\n".join([TEXT_EVT, '{"type":"error","error":{"message":"gone"}}'])
    assert "The build passes now." in bridge.parse_jsonl(raw)


def test_parse_plaintext_fallback_strips_ansi():
    raw = "\x1b[32mhello \x1b[0mfrom plain mode\n\ntool noise line"
    assert "hello from plain mode" in bridge.parse_jsonl(raw)
    assert "\x1b[" not in bridge.parse_jsonl(raw)


def test_parse_empty():
    assert bridge.parse_jsonl("") == ""
    assert bridge.parse_jsonl('{"type":"step_start","part":{}}') == ""


def test_run_opencode_builds_command_and_parses(tmp_path, monkeypatch):
    saved = {}

    def fake_run(cmd, capture_output, text, timeout, cwd=None):
        saved["cmd"] = cmd
        saved["cwd"] = cwd
        class P:
            stdout = TEXT_EVT
            stderr = ""
        return P()

    monkeypatch.setsubprocess = None  # noqa
    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(bridge, "opencode_dir", lambda: str(tmp_path))
    out = bridge.run_opencode("what changed", session="ses_test")
    assert "The build passes now." in out
    cmd = saved["cmd"]
    assert cmd[:3] == ["opencode", "run", "what changed"]
    assert "ses_test" in cmd
    assert saved["cwd"] == str(tmp_path)   # project dir via subprocess cwd
    assert "--auto" not in cmd      # safe default
    assert "--format" in cmd and "json" in cmd
    assert "--dir" not in cmd       # v2.1.3 has no --dir; cwd is used instead


@pytest.mark.parametrize("phrase,expected", [
    ("soundwave reset", ""),
    ("reset session and list my todos", "and list my todos"),
    ("new session tell me about hubris", "tell me about hubris"),
    ("tell me about hubris", None),
])
def test_maybe_reset(phrase, expected):
    if expected is None:
        assert bridge._maybe_reset(phrase) == phrase
    else:
        assert bridge._maybe_reset(phrase) == expected

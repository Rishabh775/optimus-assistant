"""Central config: typed getters, defaults, validation."""
import os
import pytest

from jarvis import config


def test_typed_getters(monkeypatch):
    monkeypatch.setenv("JARVIS_WAKE_THRESHOLD", "0.85")
    assert config.get_float("JARVIS_WAKE_THRESHOLD") == 0.85
    monkeypatch.setenv("JARVIS_WAKE_DEBOUNCE", "2")
    assert config.get_int("JARVIS_WAKE_DEBOUNCE") == 2
    monkeypatch.setenv("JARVIS_FOLLOWUP", "0")
    assert config.get_bool("JARVIS_FOLLOWUP") is False
    monkeypatch.setenv("JARVIS_FOLLOWUP", "1")
    assert config.get_bool("JARVIS_FOLLOWUP") is True


def test_defaults_used_when_unset(monkeypatch):
    monkeypatch.delenv("JARVIS_STT_GATE_MAX", raising=False)
    assert config.get_float("JARVIS_STT_GATE_MAX") == 6000.0


def test_bad_value_falls_back_not_crashes(monkeypatch):
    monkeypatch.setenv("JARVIS_WAKE_DEBOUNCE", "not-a-number")
    assert config.get_int("JARVIS_WAKE_DEBOUNCE") == int(float(config.DEFAULTS["JARVIS_WAKE_DEBOUNCE"]))


def test_validate_flags_out_of_range(monkeypatch):
    monkeypatch.setenv("JARVIS_WAKE_THRESHOLD", "5.0")
    monkeypatch.setenv("JARVIS_STT_END_SILENCE", "abc")
    problems = config.validate()
    assert any("JARVIS_WAKE_THRESHOLD" in p for p in problems)
    assert any("JARVIS_STT_END_SILENCE" in p for p in problems)


def test_validate_clean(monkeypatch):
    # defaults are all in range
    for k in list(config._RANGES):
        monkeypatch.delenv(k, raising=False)
    assert config.validate() == []

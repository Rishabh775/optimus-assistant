"""Shared audio helpers."""
import numpy as np

from jarvis import audio


def test_apply_gain_identity():
    a = np.array([-1000, 0, 1000], dtype=np.int16)
    assert np.array_equal(audio.apply_gain(a, 1.0), a)


def test_apply_gain_clips():
    out = audio.apply_gain(np.array([10000, 20000, -20000], dtype=np.int16), 2.0)
    assert out.tolist() == [20000, 32767, -32768]


def test_energy_gate(monkeypatch):
    monkeypatch.setenv("JARVIS_MIC_GAIN", "1.0")
    assert audio.energy_gate(np.full(1000, 5000, dtype=np.int16), 1000)
    assert not audio.energy_gate(np.full(1000, 50, dtype=np.int16), 1000)


def test_energy_gate_with_gain(monkeypatch):
    monkeypatch.setenv("JARVIS_MIC_GAIN", "3.0")
    # raw peak 400 < 1000, boosted 1200 > 1000
    assert audio.energy_gate(np.full(1000, 400, dtype=np.int16), 1000)

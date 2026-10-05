"""Voice pipeline: gain, confidence gating, calibration, capture, TTS."""
import os
import time
import threading
import numpy as np
import pytest

import jarvis.voice as voice


# ---- gain ----
def test_apply_gain_identity_and_clip():
    arr = np.array([-1000, 0, 1000, 32000, -32000], dtype=np.int16)
    assert np.array_equal(voice._apply_gain(arr, 1.0), arr)
    out = voice._apply_gain(np.array([10000, 20000, -20000], dtype=np.int16), 2.0)
    assert out.tolist() == [20000, 32767, -32768]


# ---- confidence gating ----
@pytest.mark.parametrize("text,lp,nsp,expect", [
    ("hello there", -0.3, 0.1, "hello there"),     # confident -> keep
    ("thank you.", -1.6, 0.92, ""),                # both bad -> reject
    ("what time is it", -1.5, 0.2, "what time is it"),  # no_speech low -> keep
    ("hey", -0.1, 0.9, "hey"),                     # logprob ok -> keep
])
def test_confidence_gate(fake_mic, monkeypatch, text, lp, nsp, expect):
    fake_mic.script = [("Q", 5), ("S", 20), ("Q", 20)]
    voice._ENERGY_GATE = 400.0
    monkeypatch.setattr(voice, "_calibrate_energy", lambda: 400.0)
    monkeypatch.setattr(voice, "sounds", type("S", (), {"end": staticmethod(lambda: None)}))
    monkeypatch.setattr(voice, "_transcribe_scored",
                        lambda a, m=None, vad=True: (text, lp, nsp))
    assert voice.listen(timeout=2) == expect


def test_vad_fallback_recovers(monkeypatch, fake_mic):
    fake_mic.script = [("Q", 5), ("S", 20), ("Q", 20)]
    voice._ENERGY_GATE = 400.0
    monkeypatch.setattr(voice, "_calibrate_energy", lambda: 400.0)
    monkeypatch.setattr(voice, "sounds", type("S", (), {"end": staticmethod(lambda: None)}))
    calls = {"n": 0}
    def scored(a, m=None, vad=True):
        calls["n"] += 1
        return ("", 0.0, 1.0) if vad else ("recovered", -0.2, 0.1)
    monkeypatch.setattr(voice, "_transcribe_scored", scored)
    assert voice.listen(timeout=2) == "recovered"
    assert calls["n"] == 2


def test_listen_timeout_invalidates_gate(monkeypatch, fake_mic):
    fake_mic.script = [("Q", 500)]
    voice._ENERGY_GATE = 800
    monkeypatch.setattr(voice, "_calibrate_energy", lambda: 800.0)
    assert voice.listen(timeout=1) == ""
    assert voice._ENERGY_GATE is None   # self-heal


# ---- calibration robustness ----
def _cal_with(peaks):
    it = {"i": 0}
    class S:
        def read(self, n):
            p = peaks[it["i"] % len(peaks)]; it["i"] += 1
            return np.full(n, p, dtype=np.int16).reshape(-1, 1), False
        def __enter__(self): return self
        def __exit__(self, *a): return False
    import types
    return types.SimpleNamespace(InputStream=lambda **k: S())


def test_calibration_ignores_loud_transient(monkeypatch):
    voice._ENERGY_GATE = None
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", _cal_with([50, 60, 30000, 55, 65]))
    assert voice._calibrate_energy() == 400.0   # median 60 -> floor 400


def test_calibration_clamped_in_loud_room(monkeypatch):
    voice._ENERGY_GATE = None
    monkeypatch.setitem(__import__("sys").modules, "sounddevice",
                        _cal_with([12000, 13000, 12500, 12900, 12100]))
    assert voice._calibrate_energy() == 6000.0


# ---- TTS pipeline ----
def test_split_sentences():
    chunks = voice._split_sentences("First one here. Second one follows! Third.")
    assert 1 <= len(chunks) <= 3
    assert all(isinstance(c, str) and c for c in chunks)


def test_speak_no_overlap_no_leak(fake_audio_out, tmp_path):
    base = threading.active_count()
    for _ in range(10):
        voice.speak("Alpha one. Beta two. Gamma three. Delta four.")
    time.sleep(0.2)
    assert fake_audio_out.overlaps == 0
    # every synthesized temp file cleaned up by _play
    leftovers = list(tmp_path.glob("*.mp3"))
    assert leftovers == []
    assert threading.active_count() <= base


def test_speak_interrupted_retires_threads(fake_audio_out):
    base = threading.active_count()
    for _ in range(8):
        t = threading.Thread(target=voice.speak,
                             args=("Long sentence one. Long sentence two. Long three.",))
        t.start()
        time.sleep(0.02)
        voice.stop_tts()
        t.join(timeout=2)
    time.sleep(0.2)
    assert fake_audio_out.overlaps == 0
    assert threading.active_count() <= base + 1

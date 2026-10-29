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
                        _cal_with([18000, 19000, 18500, 18900, 18100]))
    assert voice._calibrate_energy() == 15000.0


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


# ---- offline (Piper) TTS fallback ----
def test_synth_piper_writes_stereo_wav(monkeypatch, tmp_path):
    import wave
    import types as T
    chunk = T.SimpleNamespace(sample_rate=22050,
                              audio_float_array=np.zeros(1000, dtype=np.float32))
    class FakeVoice:
        def synthesize(self, text): return [chunk]
    class FakePV:
        @staticmethod
        def load(path): return FakeVoice()
    sub = T.ModuleType("piper.voice"); sub.PiperVoice = FakePV
    monkeypatch.setitem(__import__("sys").modules, "piper", T.ModuleType("piper"))
    monkeypatch.setitem(__import__("sys").modules, "piper.voice", sub)
    monkeypatch.setattr(voice, "_piper_model_path", lambda: "/fake/model.onnx")
    voice._PIPER = None
    out = voice._synth_piper("hello")
    try:
        with wave.open(out) as w:
            assert w.getnchannels() == 2
            assert w.getframerate() == 22050
            assert w.getnframes() == 1000          # mono duplicated
    finally:
        os.unlink(out)
        voice._PIPER = None


def test_speak_falls_back_to_piper_on_edge_tts_error(fake_audio_out, monkeypatch, tmp_path):
    made = []
    def fake_piper(text, out_wav=None):
        p = tmp_path / f"piper_{len(made)}.wav"; p.write_bytes(b"x")
        made.append(text); return str(p)
    monkeypatch.setattr(voice, "_synth_piper", fake_piper)
    monkeypatch.setattr(voice, "_synth",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("403")))
    voice.speak("Fallback alpha. Fallback beta.", blocking=True)
    assert made                          # Piper was used
    assert fake_audio_out.order          # and its audio was played


def test_speak_no_fallback_when_disabled(fake_audio_out, monkeypatch, tmp_path):
    called = []
    monkeypatch.setenv("JARVIS_TTS_FALLBACK", "off")
    monkeypatch.setattr(voice, "_synth_piper",
                        lambda *a, **k: called.append(1) or "x")
    monkeypatch.setattr(voice, "_synth",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("403")))
    voice.speak("Fallback alpha. Fallback beta.", blocking=True)
    assert called == []


def _capture_dur(fake_mic, monkeypatch):
    """Run listen() and return the captured duration in seconds."""
    monkeypatch.setattr(voice, "_calibrate_energy", lambda: voice._ENERGY_GATE)
    monkeypatch.setattr(voice, "sounds",
                        type("S", (), {"end": staticmethod(lambda: None)}))
    box = {}
    def scored(a, m=None, vad=True):
        box["dur"] = len(a) / 16000.0
        return ("text", -0.2, 0.1)
    monkeypatch.setattr(voice, "_transcribe_scored", scored)
    voice.listen(timeout=2)
    return box.get("dur", 0.0)


def test_hysteresis_survives_soft_dip(fake_mic, monkeypatch):
    # loud room: gate 2000; speech 4000; a soft dip to 1500 for 2.5s mid-sentence
    voice._ENERGY_GATE = 2000
    fake_mic.script = [("Q", 3), (4000, 5), (1500, 25), (4000, 3), ("Q", 40)]
    dur = _capture_dur(fake_mic, monkeypatch)
    # the dip alone is 2.5s; proving capture continued through it (old code cut ~1.1s)
    assert dur > 2.5, dur


def test_hysteresis_still_stops_on_silence(fake_mic, monkeypatch):
    voice._ENERGY_GATE = 1000
    fake_mic.script = [(3000, 10), ("Q", 60)]   # speech then real silence
    dur = _capture_dur(fake_mic, monkeypatch)
    assert 0.8 < dur < 2.5, dur   # ends ~1s of speech + end-silence (300ms test)


def test_transcribe_file_falls_back_when_pyav_breaks(monkeypatch):
    """PyAV >= 19 rejects faster-whisper's metadata_errors -> ffmpeg path."""
    import jarvis.voice as voice

    class Seg:
        text = " hello there "

    class FakeModel:
        def transcribe(self, audio, **kw):
            if isinstance(audio, str):
                raise TypeError(
                    "open() got an unexpected keyword argument 'metadata_errors'")
            return iter([Seg()]), None

    monkeypatch.setattr(voice, "get_stt", lambda *a, **k: FakeModel())
    monkeypatch.setattr(voice, "_stt_backend", lambda: "faster")
    monkeypatch.setattr(voice, "_decode_audio_f32", lambda p: [0.0] * 16000)
    assert voice.transcribe_file("/tmp/voice.webm") == "hello there"

"""Shared pytest fixtures and fakes for the jarvis test suite.

Everything runs offline: fake microphone (sounddevice), fake audio player
(subprocess), temp data dir. No network, no API keys, no real hardware.
"""
import os
import sys
import types
import tempfile
import threading
import time

import numpy as np
import pytest

# --- isolate from the user's real .env / data BEFORE jarvis is imported ---
_DATA = tempfile.mkdtemp(prefix="jarvis-test-")
os.environ.update({
    "JARVIS_DATA_DIR": _DATA,
    "JARVIS_SOUNDS": "0",
    "JARVIS_WAKE_WORDS": "hey_soundwave",
    "JARVIS_WAKE_THRESHOLD": "0.7",
    "JARVIS_WAKE_DEBOUNCE": "1",
    "JARVIS_WAKE_CHUNK_MS": "1200",
    "JARVIS_CLOUD_MODEL": "longcat-2.5-preview-free",
    "JARVIS_CLOUD_FALLBACK": "mimo-v2.6-flash",
    "JARVIS_CODE_MODEL": "kimi-k2.7-code",
    "JARVIS_TTS_VOICE": "en-US-AndrewMultilingualNeural",
    "JARVIS_STT_END_SILENCE": "300",   # keep tests fast
    "JARVIS_STT_TAIL": "0",
    "JARVIS_FACTS_MAX": "100",
})


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module-global mutable state between tests."""
    import jarvis.voice as voice
    import jarvis.router as router
    voice._ENERGY_GATE = None
    voice._SPEAKING.clear()
    voice._TTS_STOP_EVENT.clear()
    router._cloud_down_until = 0.0
    yield
    voice._SPEAKING.clear()
    voice._TTS_STOP_EVENT.clear()


class MockMic:
    """Controls a fake sounddevice module. Set .script = [('Q'|'S', count), ...]."""
    def __init__(self):
        self.script = [("Q", 500)]
        self.module = types.ModuleType("sounddevice")
        self.module.InputStream = lambda **kw: self._stream()

    def _stream(self):
        ctl = self
        chunks = []
        for kind, cnt in ctl.script:
            chunks += [kind] * cnt
        state = {"i": 0}

        class S:
            def read(self, n):
                kind = chunks[state["i"]] if state["i"] < len(chunks) else "Q"
                state["i"] += 1
                if isinstance(kind, (int, float)):
                    peak = int(kind)          # explicit amplitude
                elif kind == "S":
                    peak = 8000
                else:                          # "Q" = quiet room
                    peak = 60
                arr = np.full(n, peak, dtype=np.int16)
                return arr.reshape(-1, 1), False
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return S()


@pytest.fixture
def fake_mic(monkeypatch):
    ctl = MockMic()
    monkeypatch.setitem(sys.modules, "sounddevice", ctl.module)
    return ctl


class PlayerRecorder:
    """Fake audio player: records playback order and detects overlap."""
    def __init__(self, duration=0.05):
        self.active = 0
        self.overlaps = 0
        self.order = []
        self.duration = duration
        self.lock = threading.Lock()

    def popen(self, args, **kw):
        rec = self
        path = args[-1]

        class P:
            def __init__(self):
                with rec.lock:
                    rec.order.append(path)
                    if rec.active > 0:
                        rec.overlaps += 1
                    rec.active += 1
                self.done = False
            def wait(self, timeout=None):
                time.sleep(rec.duration)
                with rec.lock:
                    if not self.done:
                        rec.active -= 1
                        self.done = True
                return 0
            def poll(self):
                return 0
            def kill(self):
                with rec.lock:
                    if not self.done:
                        rec.active -= 1
                        self.done = True
        return P()


@pytest.fixture
def fake_audio_out(monkeypatch, tmp_path):
    """Replace TTS synth + the player subprocess. Returns a PlayerRecorder."""
    import asyncio
    import jarvis.voice as voice

    rec = PlayerRecorder()

    async def fake_synth(text, v, rate="+0%"):
        await asyncio.sleep(0)   # no real network
        p = tmp_path / f"chunk{len(rec.order)}_{abs(hash(text))}.mp3"
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(voice, "_synth", fake_synth)
    monkeypatch.setattr(voice.subprocess, "Popen", rec.popen)
    monkeypatch.setattr(voice, "PLAYER", "paplay")
    return rec

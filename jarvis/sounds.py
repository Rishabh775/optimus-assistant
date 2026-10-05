"""Audible feedback sounds -- generated locally, no assets, works offline.

Solves the "did it hear me?" problem:
- ack(): rising two-tone chirp -> wake word recognized, "I'm listening"
- end(): soft single tone      -> speech captured, processing your request

Sounds are synthesized once into JARVIS_DATA_DIR/sounds/ and replayed via the
same player used for TTS. Disable with JARVIS_SOUNDS=0.
"""
import os
import subprocess
import wave
from pathlib import Path
import numpy as np

DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", Path.home() / "jarvis" / "data"))
SOUNDS_DIR = DATA_DIR / "sounds"
PLAYER = os.getenv(
    "JARVIS_TTS_PLAYER",
    "ffplay -hide_banner -loglevel error -nodisp -autoexit").split()
_SR = 16000


def enabled() -> bool:
    return os.getenv("JARVIS_SOUNDS", "1") == "1"


def _write_wav(path: Path, samples: np.ndarray):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(pcm.tobytes())


def _tone(segments) -> np.ndarray:
    """segments: [(freq_hz, duration_ms), ...] with 5ms fade in/out."""
    parts = []
    for freq, ms in segments:
        n = int(_SR * ms / 1000)
        t = np.arange(n) / _SR
        part = 0.5 * np.sin(2 * np.pi * freq * t)
        fade = int(_SR * 0.005)
        env = np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n))
                         / max(fade, 1))
        parts.append(part * env)
    return np.concatenate(parts) if parts else np.zeros(1)


def ensure_sounds():
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    ack, end = SOUNDS_DIR / "ack.wav", SOUNDS_DIR / "end.wav"
    if not ack.exists():
        _write_wav(ack, _tone([(660, 80), (880, 120)]))   # rising "listening"
    if not end.exists():
        _write_wav(end, _tone([(520, 140)]))             # low "got it"
    return ack, end


def play(path: Path):
    if not enabled():
        return
    try:
        subprocess.Popen(PLAYER + [str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # no player / no audio device: stay silent, never crash


def ack():
    """Wake word recognized -- chirp immediately so the user knows it's live."""
    if not enabled():
        return
    p, _ = ensure_sounds()
    play(p)


def end():
    """Speech captured -- confirm the command was heard before processing."""
    if not enabled():
        return
    _, p = ensure_sounds()
    play(p)

"""Wake-word detection with three modes:
  - ONNX (openWakeWord): instant, low CPU, but only for pre-trained words
  - Custom verifier: trained classifier on openWakeWord 96-dim embeddings, instant
  - Whisper-based: works for ANY phrase, ~1.5s latency

Auto-falls-back: ONNX -> custom verifier -> whisper.
Includes silence gate to skip inference when audio is quiet (CPU savings).
"""
import os
import json
from pathlib import Path
import numpy as np

from .audio import apply_gain as _apply_gain
from .audio import energy_gate as _energy_gate
from .audio import mic_gain as _mic_gain
from .audio import silence_warnings as _silence_warnings
from . import config

_MODEL_DIR = Path(os.getenv(
    "JARVIS_WAKE_MODELS_DIR",
    str(Path(os.path.dirname(__file__)).parent
        / ".venv/lib/python3.14/site-packages/openwakeword/resources/models")
))
_AVAILABLE_ONNX = {p.stem.replace("_v0.1", "")
                  for p in _MODEL_DIR.glob("*_v0.1.onnx")}

_DEFAULT_WORDS = ["hey_jarvis"]
_MODEL = None
_WHISPER = None
_CUSTOM_MODEL = None
_CUSTOM_META = None
_CUSTOM_AF = None
_last_wake_log = [0.0]   # throttle for near-miss telemetry


def _load_onnx():
    global _MODEL
    if _MODEL is None:
        from openwakeword.model import Model
        words = os.getenv("JARVIS_WAKE_WORDS",
                          ",".join(_DEFAULT_WORDS)).split(",")
        paths = [str(_MODEL_DIR / f"{w}_v0.1.onnx") for w in words]
        paths = [p for p in paths if Path(p).exists()]
        if not paths:
            return None
        _MODEL = Model(wakeword_model_paths=paths, enable_speex_noise_suppression=False)
    return _MODEL


def _has_onnx(words: list) -> bool:
    return any((_MODEL_DIR / f"{w}_v0.1.onnx").exists() for w in words)


def _has_custom(words: list) -> bool:
    for w in words:
        meta = _MODEL_DIR / f"{w}.json"
        if meta.exists():
            try:
                return json.loads(meta.read_text()).get("framework") == "pytorch"
            except Exception:
                pass
    return False


def _load_custom(words: list):
    global _CUSTOM_MODEL, _CUSTOM_META, _CUSTOM_AF
    if _CUSTOM_MODEL is None:
        import warnings
        import onnxruntime as ort
        for w in words:
            meta_p = _MODEL_DIR / f"{w}.json"
            onnx_p = _MODEL_DIR / f"{w}.onnx"
            if meta_p.exists() and onnx_p.exists():
                meta = json.loads(meta_p.read_text())
                if meta.get("framework") != "pytorch":
                    continue
                # openwakeword/onnxruntime reset warning filters internally,
                # so the global CUDA-provider filter in __main__ is bypassed
                # here. Scope the suppression to this one construction.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _CUSTOM_MODEL = ort.InferenceSession(
                        str(onnx_p), providers=["CPUExecutionProvider"])
                    _CUSTOM_META = meta
                    from openwakeword.utils import AudioFeatures
                    _CUSTOM_AF = AudioFeatures()
                return True
    return _CUSTOM_MODEL is not None


def _load_whisper():
    global _WHISPER
    if _WHISPER is None:
        from .voice import get_stt
        model_name = os.getenv("JARVIS_WAKE_WHISPER_MODEL", "tiny.en")
        _WHISPER = get_stt(model_name)  # persistent, cached per process
    return _WHISPER


def _match_phrase(transcript: str, phrases: list) -> str | None:
    import re
    t = re.sub(r"[^a-z0-9 ]", " ", transcript.lower())
    t = re.sub(r"\s+", " ", t).strip()
    for p in phrases:
        needle = p.lower().replace("_", " ")
        if needle in t:
            return p
    return None


def _name() -> str:
    return os.getenv("JARVIS_NAME", "Optimus").upper()


def _threshold() -> float:
    return config.get_float("JARVIS_WAKE_THRESHOLD")


def _debounce_frames() -> int:
    """Number of consecutive frames above threshold to confirm wake."""
    return config.get_int("JARVIS_WAKE_DEBOUNCE")


def wait_for_wake(threshold: float = None, chunk: int = 1280):
    """Generator yielding wake-word name. Auto-chooses best available backend.

    Never arms while our own TTS is playing (speaker->mic echo caused
    false wakes right after replies).
    """
    import time
    from .voice import is_speaking
    while is_speaking():
        time.sleep(0.1)
    time.sleep(0.3)  # speaker tail margin
    words = config.get("JARVIS_WAKE_WORDS").split(",")
    if threshold is None:
        threshold = _threshold()
    if _has_onnx(words):
        yield from _wait_onnx(words, threshold, chunk)
    elif _has_custom(words) and _load_custom(words):
        yield from _wait_custom(threshold)
    else:
        yield from _wait_whisper(words)


def _wait_onnx(words: list, threshold: float, chunk: int):
    import sounddevice as sd
    import time
    _silence_warnings()
    model = _load_onnx()
    sample_rate = 16000
    print(f"[{_name()}] listening for wake-word (onnx, threshold={threshold})...")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        blocksize=chunk) as stream:
        hit = None
        while hit is None:
            audio = stream.read(chunk)[0].flatten()
            audio = audio.astype(np.int16)
            pred = model.predict(audio)
            for n, s in pred.items():
                if s > threshold:
                    print(f"[{_name()}] wake: {n} ({s:.2f})")
                    hit = n
                    break
    # stream closed here: mic is free for listen_after_wake() and not
    # listening while TTS plays (no self-wake / buffer overflow)
    time.sleep(0.2)
    yield hit


def _wait_custom(threshold: float = 0.7):
    """Custom-trained classifier with silence gate + debouncing + adaptive buffer."""
    import sounddevice as sd
    _silence_warnings()
    sample_rate = 16000
    chunk_ms = config.get_int("JARVIS_WAKE_CHUNK_MS")
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    energy_min = config.get_int("JARVIS_WAKE_ENERGY_MIN")
    debounce_n = _debounce_frames()
    af = _CUSTOM_AF
    meta = _CUSTOM_META
    print(f"[{_name()}] listening for wake-word (custom '{meta.get('model_name')}', "
          f"threshold={threshold}, debounce={debounce_n}, chunk={chunk_ms}ms)...")
    audio_buffer = np.empty(0, dtype=np.int16)
    consecutive_hits = 0
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        blocksize=chunk_samples) as stream:
        hit = None
        while hit is None:
            audio = stream.read(chunk_samples)[0].flatten().astype(np.int16)
            audio_buffer = np.concatenate([audio_buffer, audio])
            if len(audio_buffer) < chunk_samples:
                continue
            process_buffer = audio_buffer
            audio_buffer = np.empty(0, dtype=np.int16)
            if not _energy_gate(process_buffer, energy_min):
                consecutive_hits = 0
                continue
            emb = af._get_embeddings(process_buffer)
            if len(emb) == 0:
                consecutive_hits = 0
                continue
            rows = (emb - meta["mean"]) / (meta["std"] + 1e-6)
            rows = rows.astype(np.float32)
            outs = _CUSTOM_MODEL.run(None, {"input": rows})[0]
            peak = float(outs.max())
            if peak > threshold:
                consecutive_hits += 1
                if consecutive_hits >= debounce_n:
                    print(f"[{_name()}] wake: {meta['model_name']} ({peak:.2f})")
                    hit = meta["model_name"]
                else:
                    print(f"[wake] match {consecutive_hits}/{debounce_n} "
                          f"(score {peak:.2f}) - need {debounce_n} in a row")
            else:
                # near-miss telemetry: shows your real score so we can tune
                import time as _t
                if peak >= 0.25 and _t.time() - _last_wake_log[0] > 1.5:
                    _last_wake_log[0] = _t.time()
                    print(f"[wake] near miss (score {peak:.2f}, "
                          f"need >{threshold:.2f})")
                consecutive_hits = 0
    # stream closed: mic free for listen_after_wake(), no self-wake
    yield hit


def _wait_whisper(phrases: list):
    """Wake-phrase detection via streaming Whisper (any phrase, ~1.5s latency).

    CPU-heavy on 4-core machines: use as fallback/diagnostic only, not 24/7.
    Silence gate runs on RAW audio BEFORE normalization (normalizing makes
    any nonzero signal peak at 1.0, so the old post-normalize check never fired).
    """
    import sounddevice as sd
    model = _load_whisper()
    from .voice import _stt_backend, _transcribe
    sample_rate = 16000
    chunk_seconds = float(os.getenv("JARVIS_WAKE_CHUNK_SECONDS", "1.5"))
    samples = int(sample_rate * chunk_seconds)
    silence_peak = int(os.getenv("JARVIS_WAKE_SILENCE_PEAK", "300"))
    phrases_pretty = [p.replace("_", " ") for p in phrases]
    print(f"[{_name()}] listening for wake-phrase (whisper): "
          f"{phrases_pretty}")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        blocksize=samples) as stream:
        hit = None
        while hit is None:
            audio = stream.read(samples)[0].flatten().astype(np.int16)
            if np.abs(audio).max() < silence_peak:  # gate BEFORE normalize
                continue
            f32 = audio.astype(np.float32) / 32768.0
            if _stt_backend() == "faster":
                text = _transcribe(f32, os.getenv(
                    "JARVIS_WAKE_WHISPER_MODEL", "tiny.en"))
            else:
                result = model.transcribe(f32, language="en", fp16=False,
                                          without_timestamps=True,
                                          beam_size=1, best_of=1)
                text = result.get("text", "").strip()
            if not text:
                continue
            hit = _match_phrase(text, phrases_pretty)
            if hit:
                print(f"[{_name()}] wake: '{hit}' (heard: '{text}')")
    yield hit


def listen_after_wake(phrase_limit: int = 30) -> str:
    """Capture the spoken command after a wake hit (persistent STT model)."""
    from .voice import listen
    model = os.getenv("JARVIS_COMMAND_MODEL",
                      os.getenv("JARVIS_WHISPER_MODEL", "tiny.en"))
    return listen(timeout=8, phrase_limit=phrase_limit, model_name=model)
"""Voice I/O: faster-whisper STT (CPU int8, persistent) + edge-tts TTS.

STT: faster-whisper with a module-level persistent model (no reload per call,
int8 quantization for 4-core CPUs). Falls back to openai-whisper if
faster-whisper is unavailable. Ambient-noise energy is calibrated ONCE and
cached (speech_recognition re-calibrated on every call = 0.3-0.5s wasted).

TTS: edge-tts, played per-sentence so long answers start speaking sooner.
TTS can be interrupted: stop_tts() aborts current playback.

Offline fallback: if edge-tts (a cloud service) fails -- e.g. Microsoft
rotates its auth token and old clients get HTTP 403 -- we fall back to a local
Piper voice so the assistant never goes silent. Set JARVIS_TTS_FALLBACK=off to
disable, JARVIS_PIPER_MODEL to point at a specific .onnx voice.
"""
import os
import asyncio
import tempfile
import subprocess
import threading
import re
import numpy as np

from . import sounds
from . import config
from .log import log
from .audio import apply_gain as _apply_gain
from .audio import silence_warnings as _silence_warnings

VOICE_DEFAULT = "en-US-AndrewMultilingualNeural"  # most natural male; closest
# to ChatGPT-style speech. Others to try (set JARVIS_TTS_VOICE):
#   en-US-BrianMultilingualNeural   (young male, expressive)
#   en-US-AvaMultilingualNeural     (female, very natural)
#   en-US-GuyNeural                 (older male, flatter)
#   en-GB-RyanNeural                (British male)
VOICE_FEMALE = "en-US-JennyNeural"
VOICE_BRITISH = "en-GB-RyanNeural"
VOICE_DEEP = "en-US-EricNeural"
VOICE_CASUAL = "en-US-TonyNeural"  # Casual male
VOICE_NEWS = "en-US-DavisNeural"  # News-anchor male

# Voice-pack presets (JARVIS_TTS_EFFECT): DSP transforms applied to the
# synthesized audio via ffmpeg -- no redistributable celebrity voice packs
# exist, so we recreate the character colour on top of any neural voice.
VOICES_EFFECTS = {
    # Optimus Prime: deep, noble, warm - NOT radio-processed (that made the
    # voice muffled). Just pitch down ~14% + slight volume lift. Keep the
    # base voice deep (JARVIS_TTS_VOICE=en-US-EricNeural) for the full effect.
    "optimus": (
        "asetrate=24000*0.86,atempo=1.163,aresample=24000,volume=1.35"
    ),
    # Soundwave (Transformers): monotoneCommunicator - pitched down, radio
    # band-pass, mild drive, slow amplitude hum. Reads clearly, sounds cold.
    "soundwave": (
        "asetrate=24000*0.72,atempo=1.389,aresample=24000,"
        "volume=2.5,asoftclip=threshold=0.35,volume=1.2,"
        "highpass=f=110,lowpass=f=3300,tremolo=f=23:d=0.55,volume=1.3"
    ),
    # Plain robotic (more aggressive): deeper + ring-mod-ish tremolo
    "robot": (
        "asetrate=24000*0.60,atempo=1.5,aresample=24000,"
        "tremolo=f=8:d=0.7,volume=1.4"
    ),
    # Deep, slow narrator (Throne-room voice)
    "deep": "asetrate=24000*0.80,atempo=1.15,aresample=24000,volume=1.2",
}

# Default to paplay when available: it decodes edge-tts MP3 and BLOCKS until
# audio is fully drained from PulseAudio's buffer (ffplay's Popen.wait() can
# return before ALSA/Pulse finishes, which caused chunk overlap/cut-off).
# Do NOT use --no-remix: edge-tts is mono and without remixing it only plays
# on one channel (left). Let PulseAudio upmix to the sink's stereo layout.
PLAYER = os.getenv("JARVIS_TTS_PLAYER")
if not PLAYER:
    if os.path.exists("/usr/bin/paplay"):
        PLAYER = "paplay --latency-msec=20"
    else:
        PLAYER = "ffplay -hide_banner -loglevel error -nodisp -autoexit"

_TTS_LOCK = threading.Lock()
_TTS_STOP_EVENT = threading.Event()
_CURRENT_PROCESS = None
_PROC_LOCK = threading.Lock()    # guards _CURRENT_PROCESS across threads
_SPEAKING = threading.Event()   # set while our TTS is playing


def is_speaking() -> bool:
    """True while the assistant is speaking (mic must stay closed)."""
    return _SPEAKING.is_set()

# ---- Persistent STT (loaded once per process) ----
_STT_MODELS = {}          # model_name -> engine object
_STT_BACKEND = None       # "faster" | "openai"
_ENERGY_GATE = None       # cached ambient-energy gate (int16 rms)
_SAMPLE_RATE = 16000


def _stt_backend() -> str:
    global _STT_BACKEND
    if _STT_BACKEND is None:
        try:
            import faster_whisper  # noqa: F401
            _STT_BACKEND = "faster"
        except ImportError:
            _STT_BACKEND = "openai"
    return _STT_BACKEND


def get_stt(model_name: str = None):
    """Load (once) and cache the STT model for this process."""
    if model_name is None:
        model_name = config.get("JARVIS_WHISPER_MODEL")
    if model_name not in _STT_MODELS:
        if _stt_backend() == "faster":
            from faster_whisper import WhisperModel
            _STT_MODELS[model_name] = WhisperModel(
                model_name,
                device="cpu",
                compute_type=config.get("JARVIS_STT_COMPUTE"),
                cpu_threads=config.get_int("JARVIS_STT_THREADS"),
            )
        else:
            import whisper
            _STT_MODELS[model_name] = whisper.load_model(model_name)
    return _STT_MODELS[model_name]


def _stt_decode_kwargs(faster: bool = True) -> dict:
    """Decoding options tuned to recognize the user's actual speech:

    - language lock (skip auto-detect -> fewer mis-detections)
    - beam search (>1 -> more accurate than greedy)
    - initial_prompt / hotwords bias toward domain vocabulary (HTML, CSS,
      DOM manipulation, file paths, ...) that Whisper otherwise mis-hears.
    """
    kw = {"beam_size": config.get_int("JARVIS_STT_BEAM_SIZE")}
    lang = config.get("JARVIS_STT_LANGUAGE")
    if lang:
        kw["language"] = lang
    prompt = config.get("JARVIS_STT_INITIAL_PROMPT")
    if prompt:
        kw["initial_prompt"] = prompt
    hot = config.get("JARVIS_STT_HOTWORDS")
    if hot and faster:          # openai-whisper has no `hotwords` parameter
        kw["hotwords"] = hot
    return kw


def _transcribe(audio_f32: np.ndarray, model_name: str = None,
                vad: bool = True) -> str:
    """Transcribe float32 [-1,1] 16kHz audio with the persistent model."""
    if model_name is None:
        model_name = config.get("JARVIS_WHISPER_MODEL")
    model = get_stt(model_name)
    if _stt_backend() == "faster":
        segments, _info = model.transcribe(
            audio_f32, vad_filter=vad, without_timestamps=True,
            **_stt_decode_kwargs())
        return " ".join(s.text.strip() for s in segments).strip()
    result = model.transcribe(audio_f32, fp16=False, without_timestamps=True,
                              **_stt_decode_kwargs(faster=False))
    return result.get("text", "").strip()


def _transcribe_scored(audio_f32: np.ndarray, model_name: str = None,
                       vad: bool = True):
    """Like _transcribe but also returns (text, avg_logprob, no_speech_prob).

    faster-whisper exposes per-segment confidence; we use it to reject
    silence/hallucinated transcripts (pattern from computah / Isolaee).
    Falls back to (text, 0.0, 0.0) when scores aren't available.
    """
    if model_name is None:
        model_name = config.get("JARVIS_WHISPER_MODEL")
    model = get_stt(model_name)
    if _stt_backend() == "faster":
        segments, _info = model.transcribe(
            audio_f32, vad_filter=vad, without_timestamps=True,
            **_stt_decode_kwargs())
        segs = list(segments)
        if not segs:
            return "", 0.0, 1.0
        text = " ".join(s.text.strip() for s in segs).strip()
        lp = float(sum(getattr(s, "avg_logprob", 0.0) for s in segs) / len(segs))
        nsp = float(max(getattr(s, "no_speech_prob", 0.0) for s in segs))
        return text, lp, nsp
    result = model.transcribe(audio_f32, fp16=False, without_timestamps=True,
                              **_stt_decode_kwargs(faster=False))
    return result.get("text", "").strip(), 0.0, 0.0


def _calibrate_energy() -> int:
    """One-time ambient noise calibration -> int16 PEAK gate (cached).

    Peak-based on purpose: the wake detector's energy gate is peak-based too
    (np.abs(audio).max() > 1000) and provably hears the user on this mic.
    Uses the MEDIAN peak over several short windows (robust to a single loud
    transient, e.g. speaker bleed) and clamps the result so a noisy calibration
    can never make the assistant deaf for the whole session.
    """
    global _ENERGY_GATE
    if _ENERGY_GATE is None:
        import sounddevice as sd
        _silence_warnings()
        gain = config.get_float("JARVIS_MIC_GAIN")
        win = int(_SAMPLE_RATE * 0.1)   # 100ms windows
        n_win = 5                        # ~0.5s total
        try:
            peaks = []
            with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1,
                                dtype="int16", blocksize=win) as stream:
                for _ in range(n_win):
                    a = stream.read(win)[0].flatten().astype(np.int16)
                    peaks.append(float(np.abs(_apply_gain(a, gain)).max()))
            ambient_peak = float(np.median(peaks))
        except Exception:
            ambient_peak = 300.0
        mult = config.get_float("JARVIS_STT_ENERGY_MULT")
        gate_max = config.get_float("JARVIS_STT_GATE_MAX")
        _ENERGY_GATE = min(max(ambient_peak * mult, 400.0), gate_max)
        msg = (f"[stt] mic gate calibrated: {_ENERGY_GATE:.0f} "
               f"(ambient peak {ambient_peak:.0f}")
        if gain != 1.0:
            msg += f", gain x{gain}"
        msg += ")"
        if ambient_peak * mult > gate_max:
            msg += f"  [capped at {gate_max:.0f} - room is loud]"
        log("stt").info(msg)
    return _ENERGY_GATE


def _wait_for_quiet(max_wait: float = 30.0) -> None:
    """Block while our own TTS is playing (kills the speaker->mic echo loop),
    then ALWAYS wait a tail margin so we never sample the speaker's fading
    audio (which previously poisoned the one-time ambient calibration)."""
    import time
    waited = 0.0
    while is_speaking() and waited < max_wait:
        time.sleep(0.1)
        waited += 0.1
    time.sleep(config.get_float("JARVIS_STT_TAIL"))


def listen(timeout: int = 6, phrase_limit: int = 30,
           energy_threshold: int = None, model_name: str = None) -> str:
    """Record mic until end-of-speech, then transcribe.

    - Mic stays CLOSED while our own TTS is playing (no echo feedback loop)
    - Energy gate calibrated once (not per call)
    - End-of-speech: stops after ~1.5s of silence once speech started
    - Returns "" on silence/timeout/unintelligible audio
    """
    import sounddevice as sd
    _silence_warnings()
    _wait_for_quiet()
    gate = energy_threshold or _calibrate_energy()
    chunk_ms = 100
    chunk = int(_SAMPLE_RATE * chunk_ms / 1000)
    # How long a mid-speech pause is tolerated before deciding you finished.
    # Too low cuts users off mid-question while they think; ~1.5-2s feels
    # natural (cost: that much extra latency after you stop speaking).
    end_silence_ms = config.get_int("JARVIS_STT_END_SILENCE")
    max_chunks = int(phrase_limit * 1000 / chunk_ms)
    wait_chunks = int(timeout * 1000 / chunk_ms)

    name = os.getenv("JARVIS_NAME", "OPTIMUS")
    print(f"[{name}] listening...")
    frames, voiced, quiet_run, waited = [], False, 0, 0
    gain = config.get_float("JARVIS_MIC_GAIN")
    # Hysteresis + adaptive stop threshold: the (higher) gate decides when you
    # START talking; once talking, capture ends only when audio drops well below
    # YOUR speech level (ref), not below the room gate. This stops the "cuts me
    # off mid-sentence" behaviour on soft syllables / short pauses.
    stop_ratio = config.get_float("JARVIS_STT_STOP_RATIO")
    stop_floor = config.get_float("JARVIS_STT_STOP_FLOOR")
    min_speech_chunks = max(
        1, int(config.get_float("JARVIS_STT_MIN_SPEECH") * 1000 / chunk_ms))
    ref = 0.0
    speech_chunks = 0
    try:
        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1,
                            dtype="int16", blocksize=chunk) as stream:
            while len(frames) < max_chunks:
                audio = stream.read(chunk)[0].flatten().astype(np.int16)
                audio = _apply_gain(audio, gain)
                peak = int(np.abs(audio).max())   # peak, matches wake gate
                if not voiced:
                    waited += 1
                    if waited > wait_chunks:
                        # Recalibrate next time: the cached gate may have been
                        # poisoned by a loud calibration moment.
                        global _ENERGY_GATE
                        _ENERGY_GATE = None
                        print(f"[{name}] (no speech above gate {gate:.0f} "
                              f"in {timeout}s - recalibrating; speak AFTER the greeting)")
                        return ""
                    if peak < gate:
                        continue
                    voiced = True  # speech started
                    ref = float(peak)
                    speech_chunks = 1
                    quiet_run = 0
                else:
                    speech_chunks += 1
                    ref = max(float(peak), ref * 0.98)   # track speech level
                    stop_gate = max(stop_floor, ref * stop_ratio)
                    if peak < stop_gate and speech_chunks >= min_speech_chunks:
                        quiet_run += 1
                        if quiet_run * chunk_ms >= end_silence_ms:
                            break  # end of speech
                    else:
                        quiet_run = 0
                frames.append(audio)
    except Exception as e:
        print(f"mic error: {e}")
        return ""
    if not frames:
        return ""
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    if audio.max() < 0.01:
        return ""
    try:
        text, lp, nsp = _transcribe_scored(audio, model_name)
        if not text:
            # Silero VAD can strip quiet/short utterances entirely; the energy
            # gate already confirmed speech, so retry once without VAD
            text, lp, nsp = _transcribe_scored(audio, model_name, vad=False)
        if text:
            max_nsp = config.get_float("JARVIS_STT_MAX_NO_SPEECH")
            min_lp = config.get_float("JARVIS_STT_MIN_LOGPROB")
            # Confidence gate: reject silence/noise decodes. Following
            # faster-whisper's rule, a confident decode is never rejected for
            # low logprob alone -- both must look bad.
            if nsp > max_nsp and lp < min_lp:
                log("stt").info(f"[stt] rejected low-confidence transcript '{text[:40]}' "
                            f"(no_speech={nsp:.2f}, logprob={lp:.2f})")
                return ""
            print(f"[USER] {text}")
            sounds.end()  # audible: command captured, processing starts
        return text
    except Exception as e:
        print(f"STT error: {e}")
        return ""


def transcribe_file(path: str) -> str:
    model_name = config.get("JARVIS_WHISPER_MODEL")
    if _stt_backend() == "faster":
        model = get_stt(model_name)
        segments, _info = model.transcribe(
            path, vad_filter=True, without_timestamps=True,
            **_stt_decode_kwargs())
        return " ".join(s.text.strip() for s in segments).strip()
    import whisper
    model = get_stt(model_name)
    return model.transcribe(path, **_stt_decode_kwargs(faster=False))["text"]


# ---- TTS ----

def _apply_voice_fx(path: str) -> str:
    """Apply the JARVIS_TTS_EFFECT preset to an audio file. Returns the new
    (or unchanged, if no effect) path; the original temp file is consumed."""
    chain = VOICES_EFFECTS.get(
        (config.get("JARVIS_TTS_EFFECT") or "").strip().lower(), "")
    if not chain:
        return path
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-af", chain, "-ar", "24000", out],
            check=True, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log("tts").warning(f"voice effect failed ({e}); playing unmodified")
        return path
    try:
        if out != path:
            os.unlink(path)
    except OSError:
        pass
    return out


def _split_sentences(text: str) -> list:
    """Split into speakable chunks so playback can start on the first one."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, buf = [], ""
    for p in parts:
        buf = f"{buf} {p}".strip()
        if len(buf) >= 60:  # ~1 sentence
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks or [""]


async def _synth(text: str, voice: str, rate: str = "+0%") -> str:
    import edge_tts
    fd, mp3 = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(mp3)
    # edge-tts outputs MONO; convert to stereo (duplicate to L+R) so playback
    # is never routed to a single channel. Falls back to the mono file if
    # ffmpeg is unavailable.
    try:
        wav = mp3[:-4] + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
             "-af", "pan=stereo|c0=c0|c1=c0", "-ar", "24000", wav],
            check=True, timeout=20,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(mp3)
        return wav
    except Exception:
        return mp3


_PIPER = None       # cached PiperVoice (model load is ~1.5s)


def _piper_model_path() -> str:
    """Resolve the Piper .onnx voice: JARVIS_PIPER_MODEL, else the first model
    in $JARVIS_DATA_DIR/piper/."""
    env = config.get("JARVIS_PIPER_MODEL")
    if env:
        return os.path.expanduser(env)
    data_dir = config.get("JARVIS_DATA_DIR") or os.path.expanduser("~/jarvis/data")
    d = os.path.join(data_dir, "piper")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".onnx"):
                return os.path.join(d, f)
    return ""


def _synth_piper(text: str, out_wav: str = None) -> str:
    """Offline fallback TTS via Piper. Returns a stereo WAV path."""
    global _PIPER
    from piper.voice import PiperVoice
    if _PIPER is None:
        model = _piper_model_path()
        if not model:
            raise RuntimeError("no Piper voice (set JARVIS_PIPER_MODEL)")
        _PIPER = PiperVoice.load(model)
    chunks = list(_PIPER.synthesize(text))
    if not chunks:
        raise RuntimeError("Piper produced no audio")
    import wave
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    if out_wav is None:
        fd, out_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)                 # duplicate mono -> stereo (paplay)
        w.setsampwidth(2)
        w.setframerate(chunks[0].sample_rate)
        w.writeframes(np.repeat(pcm, 2).tobytes())
    return out_wav


def stop_tts():
    """Stop any ongoing TTS playback immediately (thread-safe)."""
    global _CURRENT_PROCESS
    _TTS_STOP_EVENT.set()
    with _PROC_LOCK:
        proc, _CURRENT_PROCESS = _CURRENT_PROCESS, None
    if proc and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass


_URL_RE = re.compile(r"https?://\S+")
_MD_CHARS = re.compile(r"[*_`#>|]+")


def _clean_for_tts(text: str) -> str:
    """Strip markdown/URLs the TTS engine would mangle or read aloud."""
    text = _URL_RE.sub(" link ", text)
    text = _MD_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tts_player_args() -> list:
    """Player args for TTS playback (paplay by default; paplay already drains
    correctly so we don't need the silenceremove filter ffplay required)."""
    return PLAYER.split()


def speak(text: str, blocking: bool = True):
    name = os.getenv("JARVIS_NAME", "OPTIMUS")
    text = _clean_for_tts(text)
    voice = config.get("JARVIS_TTS_VOICE") or VOICE_DEFAULT
    rate = config.get("JARVIS_TTS_RATE")
    _TTS_STOP_EVENT.clear()
    shown = threading.Event()

    def _show_once():
        # Print the reply text only when audio actually starts, so text and
        # sound appear together instead of text-then-silent-delay.
        if not shown.is_set():
            print(f"[{name}] {text}")
            shown.set()

    def _play(path: str, player_args: list) -> None:
        global _CURRENT_PROCESS
        if _TTS_STOP_EVENT.is_set():
            return
        _show_once()
        proc = subprocess.Popen(
            player_args + [path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with _PROC_LOCK:
            _CURRENT_PROCESS = proc
        try:
            proc.wait()
        finally:
            with _PROC_LOCK:
                if _CURRENT_PROCESS is proc:
                    _CURRENT_PROCESS = None
            try:
                os.unlink(path)
            except OSError:
                pass

    def _run():
        """Pipelined TTS: sentence N+1 is synthesized WHILE sentence N plays,
        so full stops only cost a short natural pause -- not synth latency."""
        import queue
        _SPEAKING.set()
        q = queue.Queue(maxsize=2)

        def producer():
            fallback = config.get("JARVIS_TTS_FALLBACK").strip().lower() \
                not in ("", "0", "off", "no")
            use_piper = False
            try:
                for chunk in _split_sentences(text):
                    if _TTS_STOP_EVENT.is_set():
                        break
                    path = None
                    if not use_piper:
                        try:
                            path = asyncio.run(_synth(chunk, voice, rate))
                        except Exception as e:
                            log("tts").warning(f"edge-tts failed: {e}")
                            use_piper = fallback
                    if path is None and use_piper:
                        try:
                            path = _synth_piper(chunk)
                        except Exception as e:
                            log("tts").error(f"Piper fallback failed: {e}")
                    if not path:
                        break
                    path = _apply_voice_fx(path)
                    q.put(path)
                q.put(None)
            except Exception as e:
                print(f"TTS error: {e}")
                q.put(None)

        def drain():
            """Best-effort cleanup of queued temp files on stop."""
            while True:
                try:
                    p = q.get_nowait()
                except queue.Empty:
                    return
                if p is None:
                    return
                try:
                    os.unlink(p)
                except OSError:
                    pass

        threading.Thread(target=producer, daemon=True).start()
        player_args = _tts_player_args()
        try:
            while True:
                path = q.get()
                if path is None or _TTS_STOP_EVENT.is_set():
                    if path:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                    drain()
                    return
                _play(path, player_args)
        except Exception as e:
            _show_once()   # still surface the reply text if audio failed
            print(f"TTS error: {e}")
            drain()
        finally:
            _SPEAKING.clear()

    if blocking:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


def speak_interruptible(text: str):
    """Speak text but return immediately so caller can monitor for interrupts."""
    speak(text, blocking=False)


async def _list_voices():
    import edge_tts
    return await edge_tts.list_voices()


def list_voices(filter_lang: str = "en") -> list:
    voices = asyncio.run(_list_voices())
    return [v for v in voices if v["Locale"].startswith(filter_lang)]

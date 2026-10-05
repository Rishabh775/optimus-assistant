"""Voice I/O: faster-whisper STT (CPU int8, persistent) + edge-tts TTS.

STT: faster-whisper with a module-level persistent model (no reload per call,
int8 quantization for 4-core CPUs). Falls back to openai-whisper if
faster-whisper is unavailable. Ambient-noise energy is calibrated ONCE and
cached (speech_recognition re-calibrated on every call = 0.3-0.5s wasted).

TTS: edge-tts, played per-sentence so long answers start speaking sooner.
TTS can be interrupted: stop_tts() aborts current playback.

Offline note: edge-tts is a cloud service. For fully-offline TTS install
piper and wrap it here (JARVIS_TTS_ENGINE=piper, future work).
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
        model_name = os.getenv("JARVIS_WHISPER_MODEL", "base")
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


def _transcribe(audio_f32: np.ndarray, model_name: str = None,
                vad: bool = True) -> str:
    """Transcribe float32 [-1,1] 16kHz audio with the persistent model."""
    if model_name is None:
        model_name = os.getenv("JARVIS_WHISPER_MODEL", "base")
    model = get_stt(model_name)
    if _stt_backend() == "faster":
        segments, _info = model.transcribe(
            audio_f32, beam_size=1, vad_filter=vad,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()
    result = model.transcribe(audio_f32, fp16=False, beam_size=1,
                              without_timestamps=True)
    return result.get("text", "").strip()


def _transcribe_scored(audio_f32: np.ndarray, model_name: str = None,
                       vad: bool = True):
    """Like _transcribe but also returns (text, avg_logprob, no_speech_prob).

    faster-whisper exposes per-segment confidence; we use it to reject
    silence/hallucinated transcripts (pattern from computah / Isolaee).
    Falls back to (text, 0.0, 0.0) when scores aren't available.
    """
    if model_name is None:
        model_name = os.getenv("JARVIS_WHISPER_MODEL", "base")
    model = get_stt(model_name)
    if _stt_backend() == "faster":
        segments, _info = model.transcribe(
            audio_f32, beam_size=1, vad_filter=vad, without_timestamps=True)
        segs = list(segments)
        if not segs:
            return "", 0.0, 1.0
        text = " ".join(s.text.strip() for s in segs).strip()
        lp = float(sum(getattr(s, "avg_logprob", 0.0) for s in segs) / len(segs))
        nsp = float(max(getattr(s, "no_speech_prob", 0.0) for s in segs))
        return text, lp, nsp
    result = model.transcribe(audio_f32, fp16=False, beam_size=1,
                              without_timestamps=True)
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
        print(msg)
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
                    quiet_run = 0
                else:
                    if peak < gate:
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
                print(f"[stt] rejected low-confidence transcript '{text[:40]}' "
                      f"(no_speech={nsp:.2f}, logprob={lp:.2f})")
                return ""
            print(f"[USER] {text}")
            sounds.end()  # audible: command captured, processing starts
        return text
    except Exception as e:
        print(f"STT error: {e}")
        return ""


def transcribe_file(path: str) -> str:
    model_name = os.getenv("JARVIS_WHISPER_MODEL", "base")
    if _stt_backend() == "faster":
        model = get_stt(model_name)
        segments, _info = model.transcribe(path, vad_filter=True,
                                           without_timestamps=True)
        return " ".join(s.text.strip() for s in segments).strip()
    import whisper
    model = get_stt(model_name)
    return model.transcribe(path)["text"]


# ---- TTS ----

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


def stop_tts():
    """Stop any ongoing TTS playback immediately."""
    global _CURRENT_PROCESS
    _TTS_STOP_EVENT.set()
    if _CURRENT_PROCESS and _CURRENT_PROCESS.poll() is None:
        try:
            _CURRENT_PROCESS.kill()
            _CURRENT_PROCESS.wait(timeout=1)
        except Exception:
            pass
    _CURRENT_PROCESS = None


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
        _CURRENT_PROCESS = subprocess.Popen(
            player_args + [path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            _CURRENT_PROCESS.wait()
        finally:
            if _CURRENT_PROCESS:
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
            try:
                for chunk in _split_sentences(text):
                    if _TTS_STOP_EVENT.is_set():
                        break
                    q.put(asyncio.run(_synth(chunk, voice, rate)))
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

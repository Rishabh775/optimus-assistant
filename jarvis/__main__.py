"""Optimus main entrypoint with multiple interaction modes."""
import os
import random
import warnings
import threading
import argparse
from dotenv import load_dotenv

# Suppress third-party library noise before any of them get imported:
#  - sounddevice/NumPy 2.5 `data.shape = -1, channels` DeprecationWarning spam
#  - onnxruntime "CUDAExecutionProvider not in available providers" UserWarning
#  - HuggingFace progress bars + "unauthenticated requests" warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Specified provider 'CUDA'.*")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Models are cached after the first run -- offline mode skips the
# "unauthenticated requests" warning and speeds up startup. Set to 0 in
# this shell to re-enable model downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

load_dotenv()

from rich.console import Console
from rich.prompt import Prompt

from .agent import Agent
from .voice import speak, listen, stop_tts
from . import sounds
from . import config
import time

console = Console()
NAME = config.get("JARVIS_NAME").upper()


def text_loop(agent: Agent):
    console.print(f"[bold cyan]{NAME} (text mode)[/bold cyan] - type 'quit' to exit")
    while True:
        try:
            q = Prompt.ask("[bold green]you[/bold green]")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"quit", "exit"}:
            break
        if not q.strip():
            continue
        ans = agent.ask(q)
        console.print(f"[bold magenta]{NAME.lower()}[/bold magenta]: {ans}\n")


def voice_loop(agent: Agent):
    console.print(f"[bold cyan]{NAME} (voice mode)[/bold cyan] - Ctrl+C to exit")
    while True:
        try:
            q = listen()
            if not q:
                continue
            ans = agent.ask(q)
            speak(ans)
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")


def _parse_hotkey(spec: str) -> list:
    """Parse "ctrl+alt+j" into the matching pynput Key sequence.

    Recognised: ctrl/control, alt, shift, cmd/super/win/meta, and a single
    char (a-z, 0-9). Returns the pynput Key list (in order) or the default
    ctrl+alt+j if nothing parses."""
    from pynput import keyboard
    name_map = {
        "ctrl": keyboard.Key.ctrl, "control": keyboard.Key.ctrl,
        "alt": keyboard.Key.alt,
        "shift": keyboard.Key.shift,
        "cmd": keyboard.Key.cmd, "super": keyboard.Key.cmd,
        "win": keyboard.Key.cmd, "meta": keyboard.Key.cmd,
    }
    seq = []
    for raw in spec.lower().split("+"):
        raw = raw.strip()
        if not raw:
            continue
        if raw in name_map:
            seq.append(name_map[raw])
        elif len(raw) == 1:
            seq.append(keyboard.KeyCode.from_char(raw))
        else:
            return _parse_hotkey("ctrl+alt+j")  # unrecognised -> default
    return seq or _parse_hotkey("ctrl+alt+j")


def hotkey_loop(agent: Agent):
    from pynput import keyboard

    def on_activate():
        stop_tts()  # stop any current speech
        sounds.ack()  # instant local chirp (cloud "Yes?" cost ~1s of latency)
        time.sleep(0.25)  # let the chime clear the mic
        _greet()
        q = listen()
        if q:
            ans = agent.ask(q)
            speak(ans)
            if config.get_bool("JARVIS_FOLLOWUP"):
                _followup_loop(agent, last_answer=ans)  # continued conversation

    target = _parse_hotkey(os.getenv("JARVIS_HOTKEY", "ctrl+alt+j"))
    state = {"pressed": []}

    def on_press(key):
        state["pressed"].append(key)
        state["pressed"] = state["pressed"][-5:]
        if (len(state["pressed"]) >= len(target)
                and state["pressed"][-len(target):] == target):
            threading.Thread(target=on_activate, daemon=True).start()

    label = os.getenv("JARVIS_HOTKEY", "ctrl+alt+j").replace("+", "+")
    console.print(f"[bold cyan]{NAME} hotkey mode[/bold cyan] - press {label}")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def _stdin_watchable() -> bool:
    """True only if stdin is a real selectable terminal. IDEs/pty wrappers can
    report isatty() without a working fileno(); watching those crashes."""
    import sys
    try:
        return sys.stdin is not None and sys.stdin.isatty() \
            and sys.stdin.fileno() >= 0
    except Exception:
        return False


def _interrupt_monitor(agent: Agent, stop_flag: threading.Event):
    """Listen for ENTER key during TTS playback. Press Enter to interrupt."""
    import select
    import sys
    while not stop_flag.is_set():
        # Non-blocking check for stdin (Enter key)
        try:
            readable = select.select([sys.stdin], [], [], 0.05)[0]
        except Exception:          # stdin went bad mid-speech: stop watching
            stop_flag.set()
            return
        if sys.stdin in readable:
            line = sys.stdin.readline()
            if not line:   # EOF (non-interactive stdin): stop watching, no interrupt
                stop_flag.set()
                return
            stop_tts()
            stop_flag.set()
            return


def _speak_with_interrupt(text: str, agent: Agent):
    """Speak text. Press Enter to interrupt (avoids mic feedback loop).

    Opt-in via JARVIS_BARGE_IN=1. OFF by default: on a TTY terminal, stray
    Enter keystrokes during TTS would kill the rest of the reply (a
    reproducible bug -- text printed in full, audio cut after a few words).
    With voice-only hardware there's no keyboard to press anyway.
    """
    stop_flag = threading.Event()
    if config.get_bool("JARVIS_BARGE_IN") and _stdin_watchable():
        monitor = threading.Thread(target=_interrupt_monitor,
                                   args=(agent, stop_flag), daemon=True)
        monitor.start()
    speak(text)
    was_interrupted = stop_flag.is_set()   # Enter pressed during speech?
    stop_flag.set()          # retire the monitor (no leaked thread per answer)
    return was_interrupted


FOLLOWUP_STOP = {
    "stop", "goodbye", "good bye", "bye", "cancel", "quit", "never mind",
    "nevermind", "that's all", "thats all", "that's it", "thats it",
    "thank you", "thanks", "no thanks", "nothing else", "that's everything",
    "thats everything", "go to sleep", "sleep", "end conversation",
}


def _norm(q: str) -> str:
    import re
    q = q.lower().replace("'", "").replace("-", " ")  # that's -> thats
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _is_stop(q: str) -> bool:
    return _norm(q) in FOLLOWUP_STOP


def _norm_words(t: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def _is_echo(q: str, spoken: str) -> bool:
    """True if the captured text is our own TTS tail coming back in the mic
    (speaker echo). Cheap check: normalized capture is a substring of what we
    just said."""
    if not spoken:
        return False
    nq, nsp = _norm_words(q), _norm_words(spoken)
    return len(nq) >= 8 and nq in nsp


# Keep greetings to 1-3 words: the mic is CLOSED while the greeting speaks
# (echo guard), so every extra word delays when the user's command can start.
DEFAULT_GREETINGS = [
    "Yes, boss?",
    "Go ahead.",
    "Boss?",
    "Listening.",
    "I'm here.",
    "Shoot.",
]


def _greet():
    """Short spoken greeting on wake (chirp = instant ack, this = personality).

    Keep phrases SHORT: the mic stays closed while speaking (echo guard),
    so a long greeting delays when your command can start.
    Set JARVIS_GREET=0 to skip; customize with JARVIS_GREETINGS="a|b|c".
    """
    if not config.get_bool("JARVIS_GREET"):
        return
    pool = os.getenv("JARVIS_GREETINGS", "")
    choices = [g.strip() for g in pool.split("|") if g.strip()] or DEFAULT_GREETINGS
    speak(random.choice(choices))


def _followup_loop(agent: Agent, last_answer: str = ""):
    """Continued conversation: after an answer, keep listening for follow-ups.

    - Speak within the window (JARVIS_FOLLOWUP_SECONDS, default 6) -> same
      conversation context (agent history carries over)
    - Silence -> exit back to wake-word mode
    - Stop phrase ("stop", "thanks", "that's all", ...) -> exit
    - Own-speech echo captured by the mic is ignored, not answered
    The window resets after every exchange.
    """
    from .voice import listen
    window = config.get_float("JARVIS_FOLLOWUP_SECONDS")
    spoken = last_answer
    while True:
        q = listen(timeout=window, phrase_limit=30)
        if not q or not _norm(q):
            console.print("[dim]conversation ended (silence) "
                          "- say the wake word to resume[/dim]")
            return
        if _is_stop(q):
            console.print("[dim]conversation ended (stop phrase) "
                          "- say the wake word to resume[/dim]")
            return
        if _is_echo(q, spoken):
            console.print("[dim](own speech echo ignored)[/dim]")
            continue
        ans = agent.ask(q)
        spoken = ans
        _speak_with_interrupt(ans, agent)


def wake_loop(agent: Agent):
    from .wake import wait_for_wake, listen_after_wake
    phrases = config.get("JARVIS_WAKE_WORDS").replace("_", " ").split(",")
    console.print(f"[bold cyan]{NAME} (wake mode)[/bold cyan] - say '{phrases[0]}' then your query")
    console.print("[dim]Press ENTER during speech to interrupt[/dim]")
    while True:
        try:
            for _ in wait_for_wake():
                sounds.ack()          # audible: wake word heard, now listening
                time.sleep(0.25)     # let the chime clear the mic
                _greet()             # "What's up, boss?" (mic opens after)
                q = listen_after_wake()
                if not q:
                    # false wake (noise/speech tripped the detector and nobody
                    # spoke): cool down before re-arming so we don't ping-pong
                    time.sleep(2)
                    continue
                ans = agent.ask(q)
                if not _speak_with_interrupt(ans, agent):
                    pass
                if config.get_bool("JARVIS_FOLLOWUP"):
                    _followup_loop(agent, last_answer=ans)  # continued conversation
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")


def main():
    p = argparse.ArgumentParser(prog="optimus")
    p.add_argument("--mode", choices=["text", "voice", "hotkey", "wake"],
                   default="text")
    p.add_argument("--query", help="one-shot query")
    args = p.parse_args()

    for problem in config.validate():
        console.print(f"[yellow]config warning:[/yellow] {problem}")

    agent = Agent()

    if args.query:
        print(agent.ask(args.query))
        return

    if args.mode == "text":
        text_loop(agent)
    elif args.mode == "voice":
        voice_loop(agent)
    elif args.mode == "hotkey":
        hotkey_loop(agent)
    elif args.mode == "wake":
        wake_loop(agent)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Soundwave mode: drive OpenCode by voice, reusing Optimus's audio stack.

Pipeline: wake word (OpenWakeWord, "hey soundwave") -> Optimus STT
(faster-whisper) -> `opencode run <prompt> --format json` one persistent
session -> Optimus edge-tts reads the answer aloud.

OpenCode stays untouched - this is a sidecar bridge. Run it from the

  source .venv/bin/activate
  python opencode_voice.py                # wake-word loop
  python opencode_voice.py --text "what does this repo do"
                                          # one-turn stdin/debug mode

Environment (all optional):
  JARVIS_OPENCODE_DIR     project dir opencode runs in (default: cwd)
  JARVIS_OPENCODE_SESSION session id; must start with "ses" (default
                          ses_soundwave_voice) - reused every turn, so
                          follow-ups like "now make it blue" keep context
  JARVIS_OPENCODE_AUTO    1 = pass --auto (auto-approve permissions).
                          Default 0: without it permission requests are
                          auto-REJECTED by opencode, and it tells you what
                          it would have run - safer under voice control.
  JARVIS_OPENCODE_MODEL   pin a model (provider/id), else opencode default
  JARVIS_OPENCODE_TIMEOUT run timeout seconds (default 600)
Wake/STT/TTS knobs come from jarvis/config (JARVIS_WAKE_*,
JARVIS_TTS_*), plus say "soundwave reset" mid-conversation to start a
fresh session.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

from jarvis import sounds, wake, voice

DEFAULT_SESSION = "ses_soundwave_voice"
RESET_PREFIXES = ("soundwave reset", "reset session", "new session")
EXIT_PHRASES = ("soundwave stop", "stop soundwave", "soundwave exit",
                "soundwave bye", "exit soundwave", "soundwave shutdown")


def _env(name, default=""):
    from jarvis import config
    val = config.get(name)
    return val or default


def opencode_dir() -> str:
    return os.path.expanduser(_env("JARVIS_OPENCODE_DIR") or os.getcwd())


def session_id() -> str:
    sid = _env("JARVIS_OPENCODE_SESSION") or DEFAULT_SESSION
    return sid if sid.startswith("ses") else f"ses_{sid}"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def parse_jsonl(raw: str) -> str:
    """Collect complete text events from `opencode run --format json` NDJSON.

    Falls back to RAW text (ansi-stripped, tool lines omitted) when stdout
    isn't event-shaped, so the bridge survives CLI version drift.
    """
    texts, errors = [], []
    saw_event = False
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        saw_event = True
        etype = evt.get("type", "")
        if etype == "text" and evt.get("part", {}).get("text"):
            texts.append(evt["part"]["text"].strip())
        elif etype == "error":
            err = evt.get("error", {})
            msg = (err.get("data", {}) or {}).get("message") \
                or err.get("message") or json.dumps(err)
            errors.append(msg)

    if saw_event and texts:
        return "\n\n".join(texts).strip()
    if errors:
        return f"OpenCode reported an error: {'; '.join(errors[-2:])}"
    if saw_event:
        return ""   # event stream w/o text (e.g. only step_start) - no reply

    out = _strip_ansi(raw).strip()
    if not out:
        return ""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return "\n".join(lines)


def run_opencode(prompt: str, session: str = None) -> str:
    """Send one prompt in a persistent session; returns the final text."""
    text = prompt.strip()
    if not text:
        return "I didn't catch a prompt."

    cmd = ["opencode", "run", text,
           "--session", session or session_id(),
           "--format", "json"]
    if _env("JARVIS_OPENCODE_AUTO") == "1":
        cmd.append("--auto")
    model = _env("JARVIS_OPENCODE_MODEL")
    if model:
        cmd += ["--model", model]

    timeout = float(_env("JARVIS_OPENCODE_TIMEOUT", "600"))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=opencode_dir())
    except subprocess.TimeoutExpired:
        return (f"OpenCode hit the {int(timeout)} second timeout. "
                "Ask me to continue it in the terminal if it needs longer.")
    except FileNotFoundError:
        return "opencode CLI not found on PATH. Install it, then retry."

    answer = parse_jsonl(proc.stdout or "")
    if not answer and proc.stderr:
        answer = _strip_ansi(proc.stderr).strip()[:400]
    return answer or "OpenCode finished without any reply text."


def _maybe_reset(text: str) -> str:
    """'soundwave reset ...' -> fresh session; returns the cleaned prompt."""
    global _SESSION
    low = text.strip().lower()
    for p in RESET_PREFIXES:
        if low.startswith(p):
            _SESSION = f"{DEFAULT_SESSION}_{int(time.time())}"
            voice.speak("Starting a fresh session.", blocking=True)
            return text.strip()[len(p):].strip()
    return text


def _session() -> str:
    global _SESSION
    if _SESSION is None:
        _SESSION = session_id()
    return _SESSION


_SESSION = None


def voice_loop(once_with: str = None):
    try:
        print("=== Soundwave mode for OpenCode ===")
        print(f"    dir      : {opencode_dir()}")
        print(f"    session  : {_session()}")
        print(f"    auto-ok  : {_env('JARVIS_OPENCODE_AUTO') == '1'}")
        print('    say "soundwave reset" for a fresh session, or exit phrases to quit')
        print()

        if once_with is not None:
            answer = run_opencode(once_with)
            print(f">> {answer}")
            voice.speak(answer)
            return

        for wake_name in wake.wait_for_wake():
            sounds.ack()
            heard = wake.listen_after_wake()
            if not heard:
                print(f"[{_wake_name()}] heard nothing usable back")
                continue
            print(f"> you: {heard}")

            low = heard.strip().lower()
            if any(p in low for p in EXIT_PHRASES):
                voice.speak("Soundwave signing off. Talk soon!", blocking=True)
                return

            sounds.end()
            prompt = _maybe_reset(heard)
            if not prompt:
                continue
            answer = run_opencode(prompt)
            print(f"> opencode: {answer[:300]}{' ...' if len(answer) > 300 else ''}")
            voice.speak(answer)   # non-wake arms only after speaking finishes
    except KeyboardInterrupt:
        voice.stop_tts()
        print("\nInterrupted - good luck on the build.")


def _wake_name() -> str:
    from jarvis import config
    return (config.get("JARVIS_NAME") or "soundwave").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", metavar="PROMPT",
                    help="skip wake word; run this single prompt instead")
    args = ap.parse_args()
    voice_loop(once_with=args.text)


if __name__ == "__main__":
    main()

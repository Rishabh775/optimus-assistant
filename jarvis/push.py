"""Phone push + speaker delivery for fired reminders (proactive loop p2).

Delivery chain, tried in order, per fire:
  1. ntfy phone push   -- when JARVIS_NTFY_URL + JARVIS_NTFY_TOPIC are set.
     Start with the public server ("https://ntfy.sh", topic = your secret
     channel name; install the ntfy app on the phone and subscribe to the
     same topic). Flip to self-hosted later with a one-line .env change:
     JARVIS_NTFY_URL=http://<tailnet-ip>:2580
     Optional JARVIS_NTFY_TOKEN for servers that require auth.
  2. Speaker announce  -- ONLY when this process is already running a voice
     mode (jarvis.voice already imported). Never loads the STT stack just
     for a reminder, so text/wake-in-another-terminal stays silent-skipped.

Everything here is best-effort: a dead ntfy server or a busy speaker must
never crash the scheduler or block other listeners.
"""
import json
import sys
import urllib.request

from . import config, scheduler
from .log import log as _mk_log

log = _mk_log("push")


def push_configured() -> bool:
    return bool(config.get("JARVIS_NTFY_URL") and config.get("JARVIS_NTFY_TOPIC"))


def _post(url: str, topic: str, title: str, message: str,
          token: str = "", timeout: float = 5.0) -> None:
    """POST one ntfy message (JSON body so title/priority ride along)."""
    body = json.dumps({
        "topic": topic,
        "title": title,
        "message": message,
        "priority": "high",
        "tags": ["bell"],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{url.rstrip('/')}/{topic}", data=body, headers=headers,
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def send(title: str, message: str) -> bool:
    """Push one notification. Returns True when a push was sent."""
    if not push_configured():
        return False
    try:
        _post(config.get("JARVIS_NTFY_URL"),
              config.get("JARVIS_NTFY_TOPIC"),
              title, message,
              token=config.get("JARVIS_NTFY_TOKEN"))
        return True
    except Exception as e:
        log.warning("ntfy push failed: %s", e)
        return False


# ------------------------------------------------------------ listeners ---

def speak_local(text: str) -> None:
    """Announce text on THIS machine's speakers, voice modes only.

    No-op when this process is not running a voice mode -- we never load
    the STT stack just for an announcement."""
    if "jarvis.voice" not in sys.modules:
        return
    try:
        from .voice import speak
        speak(text, blocking=False)
        log.info("announced on speaker: %s", text[:60])
    except Exception:
        log.exception("speaker announcement failed")


def _deliver(r: dict) -> None:
    """One fire -> phone push + (voice modes) speaker announce."""
    mark = "Missed reminder" if r.get("missed") else "Reminder"
    if send(mark, r["message"]):
        log.info("reminder pushed: %s", r["message"])
    speak_local(f"{mark}: {r['message']}")


def install() -> bool:
    """Register the push/announce listeners with the scheduler. Idempotent.

    Returns whether phone push is currently enabled (handy for logs and
    self_status); the listeners themselves degrade to TTS/log-only when not.
    """
    scheduler.add_listener(_deliver)
    return push_configured()

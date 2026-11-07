"""Telegram Bot API channel (official, free, no public webhook needed).

Why Telegram for a personal assistant:
  - official API, no business verification, no phone-number registration
  - no 24-hour window / approved-template rules: you can message yourself
    any time (reminders, job alerts, briefings)
  - long polling means the laptop needs no public HTTPS endpoint
  - buttons (inline keyboards) can later carry approve/deny for gated tools

Setup: @BotFather -> /newbot -> copy token into JARVIS_TELEGRAM_BOT_TOKEN,
set JARVIS_TELEGRAM_ENABLED=1. Message the bot /start; it replies with your
chat id; put that in JARVIS_TELEGRAM_ALLOW (comma-separated) and restart.

Config:
    JARVIS_TELEGRAM_ENABLED=1
    JARVIS_TELEGRAM_BOT_TOKEN=123:ABC
    JARVIS_TELEGRAM_ALLOW=123456789
"""
import json
import threading
import time
import urllib.request
from pathlib import Path

from . import config
from .log import log as _mk_log

log = _mk_log("telegram")

_OFFSET_FILE = Path(__file__).resolve().parents[1] / "data" / "phone-app" / \
    "telegram_offset.json"
_thread = None


def _token() -> str:
    return (config.get("JARVIS_TELEGRAM_BOT_TOKEN") or "").strip()


def configured() -> bool:
    return config.get_bool("JARVIS_TELEGRAM_ENABLED") and bool(_token())


def _allowed() -> list:
    raw = config.get("JARVIS_TELEGRAM_ALLOW") or ""
    return [x.strip() for x in raw.split(",") if x.strip()]


def _call(method: str, payload: dict, timeout: float = 35):
    url = f"https://api.telegram.org/bot{_token()}/{method}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _target(chat_id: str = "") -> str:
    if chat_id:
        return str(chat_id)
    allow = _allowed()
    return allow[0] if allow else ""


def send(text: str, chat_id: str = "") -> bool:
    """Send a text message. True when Telegram accepted every chunk."""
    if not configured():
        return False
    target = _target(chat_id)
    if not target:
        log.warning("telegram recipient missing (JARVIS_TELEGRAM_ALLOW)")
        return False
    body = str(text or "")
    chunks = [body[i:i + 4000] for i in range(0, max(1, len(body)), 4000)]
    try:
        for chunk in chunks:
            _call("sendMessage", {"chat_id": target, "text": chunk,
                                  "disable_web_page_preview": True})
        return True
    except Exception as e:
        log.warning("telegram send failed: %s", e)
        return False


def notify(title: str, message: str) -> bool:
    return send(f"{title}\n{message}")


# --------------------------------------------------------------- polling --

def _load_offset() -> int:
    try:
        return int(json.loads(_OFFSET_FILE.read_text()).get("offset", 0))
    except Exception:
        return 0


def _save_offset(offset: int) -> None:
    try:
        _OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _OFFSET_FILE.write_text(json.dumps({"offset": offset}))
    except Exception:
        pass


def start(handler) -> bool:
    """Start long-polling. handler(chat_id, text) -> reply string."""
    global _thread
    if not configured():
        return False
    if _thread and _thread.is_alive():
        return True
    _thread = threading.Thread(target=_loop, args=(handler,),
                               name="telegram-poll", daemon=True)
    _thread.start()
    log.info("telegram polling started")
    return True


def _loop(handler) -> None:
    offset = _load_offset()
    while True:
        try:
            data = _call("getUpdates", {
                "timeout": 25, "offset": offset + 1,
                "allowed_updates": ["message"],
            }, timeout=40)
            for update in data.get("result") or []:
                offset = max(offset, int(update.get("update_id", 0)))
                _save_offset(offset)
                msg = update.get("message") or {}
                chat_id = str((msg.get("chat") or {}).get("id", ""))
                text = (msg.get("text") or "").strip()
                if not chat_id or not text:
                    continue
                allow = _allowed()
                if not allow:
                    send(f"Your Telegram chat id is {chat_id}.\n"
                         "Add it to JARVIS_TELEGRAM_ALLOW in .env and "
                         "restart Soundwave.", chat_id=chat_id)
                    continue
                if chat_id not in allow:
                    send(f"This bot is private. Your chat id: {chat_id}",
                         chat_id=chat_id)
                    continue
                try:
                    reply = handler(chat_id, text)
                except Exception as e:
                    reply = f"error: {e}"
                if reply:
                    send(reply, chat_id=chat_id)
        except Exception as e:
            log.warning("telegram poll failed: %s", e)
            time.sleep(5)

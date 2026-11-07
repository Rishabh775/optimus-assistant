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


def send(text: str, chat_id: str = "", reply_markup: dict = None) -> bool:
    """Send a text message (chunked, retried). True when all chunks landed."""
    if not configured():
        return False
    target = _target(chat_id)
    if not target:
        log.warning("telegram recipient missing (JARVIS_TELEGRAM_ALLOW)")
        return False
    body = str(text or "")
    chunks = [body[i:i + 4000] for i in range(0, max(1, len(body)), 4000)]
    for idx, chunk in enumerate(chunks):
        payload = {"chat_id": target, "text": chunk,
                   "disable_web_page_preview": True}
        if reply_markup and idx == 0:
            payload["reply_markup"] = reply_markup
        for attempt in range(3):
            try:
                _call("sendMessage", payload)
                break
            except Exception as e:
                log.warning("telegram send failed (attempt %d): %s",
                            attempt + 1, e)
                if attempt == 2:
                    return False
                time.sleep(1.5 * (attempt + 1))
    return True


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
                "allowed_updates": ["message", "callback_query"],
            }, timeout=40)
            for update in data.get("result") or []:
                offset = max(offset, int(update.get("update_id", 0)))
                _save_offset(offset)

                cb = update.get("callback_query")
                if cb:
                    message = cb.get("message") or {}
                    chat = str((message.get("chat") or {}).get("id", ""))
                    if chat in _allowed():
                        outcome = _resolve_callback(cb.get("data", ""))
                        try:
                            _call("answerCallbackQuery", {
                                "callback_query_id": cb.get("id"),
                                "text": outcome or "expired"})
                            if outcome and message.get("message_id"):
                                _call("editMessageReplyMarkup", {
                                    "chat_id": chat,
                                    "message_id": message["message_id"],
                                    "reply_markup": {"inline_keyboard": []}})
                            send("✅ Approved" if outcome == "approved"
                                 else "❌ Denied", chat_id=chat)
                        except Exception:
                            pass
                    continue

                msg = update.get("message") or {}
                chat_id = str((msg.get("chat") or {}).get("id", ""))
                text = (msg.get("text") or "").strip()
                if not chat_id or not text:
                    continue
                allow = _allowed()
                if not allow:
                    log.info("telegram onboarding chat id: %s", chat_id)
                    send(f"Your Telegram chat id is {chat_id}.\n"
                         "Add it to JARVIS_TELEGRAM_ALLOW in .env and "
                         "restart Soundwave.", chat_id=chat_id)
                    continue
                if chat_id not in allow:
                    log.info("telegram rejected chat id: %s", chat_id)
                    send(f"This bot is private. Your chat id: {chat_id}",
                         chat_id=chat_id)
                    continue
                threading.Thread(target=_handle_message,
                                 args=(handler, chat_id, text),
                                 daemon=True).start()
        except Exception as e:
            log.warning("telegram poll failed: %s", e)
            time.sleep(5)


# ------------------------------------------------------ approvals --------

_pending = {}
_pending_lock = threading.Lock()
_next_confirm = 0


def ask_confirmation(chat_id: str, reason: str, fn_name: str, args: dict,
                     mandatory: bool = False, timeout: float = None) -> bool:
    """Send Approve/Deny buttons and block this agent turn until answered."""
    global _next_confirm
    timeout = float(timeout or config.get_int("JARVIS_APP_CONFIRM_TIMEOUT")
                    or 120)
    with _pending_lock:
        _next_confirm += 1
        cid = f"c{_next_confirm}"
        event = threading.Event()
        box = {"value": False}
        _pending[cid] = (event, box)
    text = (f"Approval needed{' (mandatory)' if mandatory else ''}\n"
            f"{reason}\n\nTool: {fn_name}")
    keyboard = {"inline_keyboard": [[
        {"text": "✅ Approve", "callback_data": f"ok:{cid}"},
        {"text": "❌ Deny", "callback_data": f"no:{cid}"},
    ]]}
    try:
        if not send(text, chat_id=chat_id, reply_markup=keyboard):
            return False
        if not event.wait(timeout):
            send(f"⏱ Approval timed out ({int(timeout)}s) — denied.",
                 chat_id=chat_id)
            return False
        return bool(box["value"])
    finally:
        with _pending_lock:
            _pending.pop(cid, None)


def _resolve_callback(data: str):
    """Handle 'ok:c3'/'no:c3'. Returns 'approved'/'denied'/None."""
    if not data or ":" not in data:
        return None
    action, cid = data.split(":", 1)
    if action not in ("ok", "no"):
        return None
    with _pending_lock:
        entry = _pending.get(cid)
        if not entry:
            return None
        event, box = entry
        box["value"] = action == "ok"
        event.set()
    return "approved" if action == "ok" else "denied"


def _handle_message(handler, chat_id: str, text: str) -> None:
    """Run one agent turn off the poll thread (so approval buttons work)."""
    try:
        reply = handler(chat_id, text)
        if reply:
            send(reply, chat_id=chat_id)
    except Exception as e:
        log.warning("telegram handler failed: %s", e)
        send(f"error: {e}", chat_id=chat_id)

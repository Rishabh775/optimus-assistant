"""Durable reminders + the proactive scheduler loop.

Reminders live in SQLite (JARVIS_REMINDER_DB, under the git-ignored data/)
so they survive reboots. On startup, anything already past-due fires
immediately with missed=True rather than silently disappearing.

Phase 1 of the proactive loop: the scheduler only DELIVERS text -- it never
runs the agent autonomously. Delivery is language-neutral through listener
hooks (see add_listener); phase 2 adds ntfy push, phone-app WS notes and
local TTS on top.

Time parsing accepts what the LLM naturally produces, plus quick human
forms: ISO datetime ("2026-10-24T15:30", signed offsets honoured), bare
"HH:MM" (today, or tomorrow if already past), and relative ("in 20
minutes", "in 1h30m"). Naive datetimes are read in the local timezone
(JARVIS_TZ overrides, e.g. "Asia/Kolkata").

Multiple jarvis processes can run at once (voice + phone-app server); the
fire path CLAIMS each row with a conditional UPDATE, so a reminder is
delivered exactly once no matter who wins.
"""
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from .log import log as _mk_log

log = _mk_log("scheduler")

_REPEATS = ("daily", "weekly")
# how late a fire may legally be before it is flagged as a restart catch-up
_MISSED_GRACE = 120          # seconds
_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id         TEXT PRIMARY KEY,
    message    TEXT NOT NULL,
    due_at     REAL NOT NULL,
    repeat     TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    fired_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at);
CREATE TABLE IF NOT EXISTS daily_runs (
    name      TEXT PRIMARY KEY,
    last_date TEXT NOT NULL,
    last_ts   REAL NOT NULL
);
"""
_DAILY = {}
_DAILY_LOCK = threading.Lock()
_HHMM_RE = re.compile(r"(\d{1,2}):(\d{2})")


def add_daily(name: str, at: str, fn) -> bool:
    """Register a scheduled daily job ("07:30" local). Runs at most once per
    day, persisted across restarts in daily_runs. fn() -> any."""
    m = _HHMM_RE.fullmatch(at.strip())
    if not m or not (0 <= int(m.group(1)) < 24 and 0 <= int(m.group(2)) < 60):
        raise ValueError(f"bad daily time {at!r} -- use 'HH:MM'")
    with _DAILY_LOCK:
        _DAILY[name] = (int(m.group(1)), int(m.group(2)), fn)
    return True


def db_path() -> Path:
    """Resolve the DB path from config (same rules as server.store)."""
    raw = config.get("JARVIS_REMINDER_DB") or \
        config.DEFAULTS["JARVIS_REMINDER_DB"]
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / raw
    return p


def _connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=10)
    conn.executescript(_SCHEMA)   # IF NOT EXISTS, near-free after first
    return conn


# ---------------------------------------------------------------- time ----

_UNIT_SECONDS = {
    "d": 86400, "day": 86400, "days": 86400,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
}
_IN_RE = re.compile(r"^in\s+((?:\d+\s*[a-z]+\s*)+)$", re.I)
_IN_PART_RE = re.compile(r"(\d+)\s*([a-z]+)", re.I)


def _local_tz():
    name = config.get("JARVIS_TZ")
    if name:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    return datetime.now().astimezone().tzinfo


def _local_now():
    return datetime.now(_local_tz())


def _parse_when(at: str) -> float:
    """Parse a reminder time into an epoch timestamp. Raises ValueError."""
    raw = str(at).strip()
    if not raw:
        raise ValueError("reminder time is empty -- use ISO, 'HH:MM' or 'in N minutes'")
    due = None
    m = _IN_RE.match(raw.lower())
    if m:
        total = 0
        for n, unit in _IN_PART_RE.findall(m.group(1)):
            if unit not in _UNIT_SECONDS:
                raise ValueError(f"unknown time unit {unit!r} in {raw!r}")
            total += int(n) * _UNIT_SECONDS[unit]
        if total <= 0:
            raise ValueError(f"{raw!r} must be in the future")
        due = time.time() + total
    elif re.fullmatch(r"\d{1,2}:\d{2}", raw):
        hh, mm = map(int, raw.split(":"))
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError(f"bad time-of-day {raw!r}")
        local = _local_now().replace(
            hour=hh, minute=mm, second=0, microsecond=0)
        if local.timestamp() <= time.time():
            local += timedelta(days=1)          # "9" said at 9:01 -> tomorrow
        due = local.timestamp()
    else:
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            raise ValueError(
                f"cannot parse time {raw!r} -- use an ISO datetime"
                f" ('2026-10-24T15:30'), 'HH:MM', or relative ('in 20 minutes')"
            ) from None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_local_tz())     # naive -> local
        due = dt.timestamp()
    if due <= time.time():
        raise ValueError(f"{raw!r} is in the past -- the reminder must be in the future")
    return due


def _fmt_when(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=_local_tz()) \
        .strftime("%Y-%m-%d %H:%M")


def _norm_repeat(repeat: str) -> str:
    r = str(repeat or "").strip().lower()
    if r and r not in _REPEATS:
        raise ValueError(f"repeat must be '', 'daily' or 'weekly' (got {repeat!r})")
    return r


def _next_repeat_epoch(epoch: float, repeat: str, after: float) -> float:
    """Advance a repeating reminder's epoch until it sits after `after`."""
    step = timedelta(days=1 if repeat == "daily" else 7)
    dt = datetime.fromtimestamp(epoch, tz=_local_tz())
    while dt.timestamp() <= after:
        dt += step
    return dt.timestamp()


# ----------------------------------------------------------------- CRUD ---

def add(message: str, at: str, repeat: str = "") -> dict:
    """Persist a reminder; returns the dict (id, due_at, ...)."""
    if not str(message or "").strip():
        raise ValueError("reminder message is empty")
    repeat = _norm_repeat(repeat)
    due = _parse_when(at)   # validates future-ness itself
    rid = uuid.uuid4().hex[:12]
    now = time.time()
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO reminders (id, message, due_at, repeat,"
                " status, created_at) VALUES (?,?,?,?, 'pending', ?)",
                (rid, str(message).strip(), due, repeat, now))
    finally:
        conn.close()
    return {"id": rid, "message": str(message).strip(), "due_at": due,
            "repeat": repeat, "status": "pending"}


def _row(r) -> dict:
    return {"id": r[0], "message": r[1], "due_at": r[2],
            "repeat": r[3], "status": r[4]}


def list_pending(limit: int = 50) -> list:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, message, due_at, repeat, status FROM reminders"
            " WHERE status='pending' ORDER BY due_at LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()
    return [_row(r) for r in rows]


def cancel(rid: str) -> bool:
    """Cancel a pending reminder. False for unknown / already-settled ids."""
    conn = _connect()
    try:
        with conn:
            n = conn.execute(
                "UPDATE reminders SET status='cancelled'"
                " WHERE id=? AND status='pending'", (str(rid),)).rowcount
    finally:
        conn.close()
    return n > 0


# ------------------------------------------------------------- delivery ---

_LISTENERS = []
_LISTENERS_LOCK = threading.Lock()


def add_listener(fn) -> None:
    """Register a delivery hook: fn(reminder_dict). Idempotent."""
    with _LISTENERS_LOCK:
        if fn not in _LISTENERS:
            _LISTENERS.append(fn)


def _deliver(r: dict) -> None:
    for fn in list(_LISTENERS):
        try:
            fn(r)
        except Exception:
            log.exception("reminder listener failed")


def _claim(conn, r, now: float) -> bool:
    """Atomically claim one due row (exact-once across processes).

    One-shot rows flip to 'fired'. Repeating rows jump their due_at to the
    next cycle while staying 'pending' -- whoever claims first delivers.
    """
    with conn:
        if r["repeat"]:
            nxt = _next_repeat_epoch(r["due_at"], r["repeat"], now)
            n = conn.execute(
                "UPDATE reminders SET due_at=? WHERE id=?"
                " AND status='pending' AND due_at<=?", (nxt, r["id"], now)
            ).rowcount
        else:
            n = conn.execute(
                "UPDATE reminders SET status='fired', fired_at=?"
                " WHERE id=? AND status='pending' AND due_at<=?",
                (now, r["id"], now)).rowcount
    return n == 1


def _tick(now: float = None) -> list:
    """Fire every pending reminder that is due. Returns what it delivered.

    `now` is injectable for tests; production calls _tick() and lets it use
    the wall clock.
    """
    now = time.time() if now is None else now
    conn = _connect()
    fired = []
    try:
        rows = conn.execute(
            "SELECT id, message, due_at, repeat, status FROM reminders"
            " WHERE status='pending' AND due_at<=? ORDER BY due_at",
            (now,)).fetchall()
        for r in rows:
            d = _row(r)
            if _claim(conn, d, now):
                    d["missed"] = (now - d["due_at"]) > _MISSED_GRACE
                    if d["missed"]:
                        log.info("missed reminder (restart catch-up): %s",
                                 d["message"])
                    _deliver(d)
                    fired.append(d)
        _sweep_daily(now)
    finally:
        conn.close()
    return fired


# ------------------------------------------------- daily scheduled jobs ---

def _daily_run_date(name: str) -> str:
    conn = _connect()
    try:
        r = conn.execute("SELECT last_date FROM daily_runs WHERE name=?",
                         (name,)).fetchone()
    finally:
        conn.close()
    return r[0] if r else ""


def _sweep_daily(now: float) -> None:
    """Run any registered daily job whose local time has come (once/day)."""
    lt = datetime.fromtimestamp(now, tz=_local_tz())
    today = lt.strftime("%Y-%m-%d")
    for name, (hh, mm, fn) in list(_DAILY.items()):
        target = lt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if lt.timestamp() < target.timestamp() or _daily_run_date(name) == today:
            continue
        conn = _connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO daily_runs (name, last_date, last_ts)"
                    " VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET"
                    " last_date=?, last_ts=?", (name, today, now, today, now))
        finally:
            conn.close()
        try:
            fn()
        except Exception:
            log.exception("daily job %s failed", name)


# -------------------------------------------------------------- the loop --

_thread = None


def _console_listener(r: dict) -> None:
    """Default phase-1 delivery: log line (+ later phases hook in more)."""
    log.info("reminder%s: %s",
             " (missed)" if r.get("missed") else "", r["message"])


def start() -> None:
    """Idempotently launch the background scheduler thread (daemon)."""
    global _thread
    if _thread and _thread.is_alive():
        return
    add_listener(_console_listener)
    _thread = threading.Thread(
        target=_loop, name="jarvis-scheduler", daemon=True)
    _thread.start()
    log.info("scheduler started (tick=%ss)", max(2, config.get_int("JARVIS_SCHED_TICK")))


def _loop() -> None:
    while True:
        try:
            _tick()
        except Exception:
            log.exception("scheduler tick failed")
        time.sleep(max(2, config.get_int("JARVIS_SCHED_TICK")))

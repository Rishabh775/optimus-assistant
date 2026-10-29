"""SQLite chat-history store for the phone app.

Two tables:
  sessions  id, title, created_at, updated_at, pinned, archived
  messages  id, session_id, role(user|assistant), content, meta(JSON), ts

Only user/assistant turns are stored -- tool exchanges live inside the
Agent's in-RAM history (the summary lands in the assistant message anyway).
The DB lives under data/phone-app/ (already git-ignored); set
JARVIS_APP_DB="" to keep the session list in RAM only.
"""
import json
import sqlite3
import time
import uuid
from pathlib import Path

from .. import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id        TEXT PRIMARY KEY,
    title     TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    pinned    INTEGER NOT NULL DEFAULT 0,
    archived  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    meta       TEXT,
    ts         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_sid_ts ON messages(session_id, ts);
"""


def db_path() -> Path:
    """Resolve the DB path from config (always falls back to the default)."""
    raw = config.get("JARVIS_APP_DB") or config.DEFAULTS["JARVIS_APP_DB"]
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / raw
    return p


def _connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=10)
    conn.executescript(_SCHEMA)   # IF NOT EXISTS, near-free after first
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init() -> None:
    _connect().close()


def _new(sid: str, title: str) -> dict:
    return {"id": sid, "title": title, "pinned": 0, "archived": 0}


def new_session(title: str = "") -> dict:
    sid = uuid.uuid4().hex[:16]
    now = time.time()
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at)"
                " VALUES (?,?,?,?)", (sid, title, now, now))
    finally:
        conn.close()
    s = _new(sid, title)
    s["created_at"] = s["updated_at"] = now
    return s


def session_exists(sid: str) -> bool:
    conn = _connect()
    try:
        row = conn.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone()
    finally:
        conn.close()
    return row is not None


def get_session(sid: str) -> dict | None:
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT id, title, created_at, updated_at, pinned, archived"
            " FROM sessions WHERE id=?", (sid,)).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    return {"id": r[0], "title": r[1], "created_at": r[2],
            "updated_at": r[3], "pinned": bool(r[4]), "archived": bool(r[5])}


def list_sessions() -> list:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT s.id, s.title, s.updated_at, s.archived, s.pinned, "
            " (SELECT content FROM messages m WHERE m.session_id = s.id"
            "   ORDER BY m.ts DESC, m.id DESC LIMIT 1) AS last"
            " FROM sessions s WHERE s.archived = 0"
            " ORDER BY s.pinned DESC, s.updated_at DESC").fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "title": r[1], "updated_at": r[2],
             "archived": bool(r[3]), "pinned": bool(r[4]),
             "last": r[5] or ""} for r in rows]


def get_messages(sid: str, limit: int = None) -> list:
    conn = _connect()
    try:
        q = ("SELECT role, content, ts FROM messages WHERE session_id=?"
             " ORDER BY ts, id")
        rows = conn.execute(q, (sid,)).fetchall()
    finally:
        conn.close()
    if limit:
        rows = rows[-limit:]
    return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]


def add_message(sid: str, role: str, content: str, meta: dict = None) -> None:
    now = time.time()
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, meta, ts)"
                " VALUES (?,?,?,?,?)",
                (sid, role, content,
                 json.dumps(meta) if meta else None, now))
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?",
                         (now, sid))
    finally:
        conn.close()


def rename_session(sid: str, title: str) -> None:
    conn = _connect()
    try:
        with conn:
            conn.execute("UPDATE sessions SET title=? WHERE id=?",
                         (title, sid))
    finally:
        conn.close()


def update_session(sid: str, **fields) -> None:
    cols, vals = [], []
    for k, v in fields.items():
        if k not in ("pinned", "archived", "title"):
            raise ValueError(f"bad field {k}")
        cols.append(f"{k}=?")
        vals.append(int(v) if k != "title" else v)
    conn = _connect()
    try:
        with conn:
            conn.execute(f"UPDATE sessions SET {', '.join(cols)} WHERE id=?",
                         (*vals, sid))
    finally:
        conn.close()


def delete_session(sid: str) -> bool:
    conn = _connect()
    try:
        with conn:
            n = conn.execute("DELETE FROM messages WHERE session_id=?",
                             (sid,)).rowcount
            conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    finally:
        conn.close()
    return n > 0


def prune_empty_sessions(max_age_seconds: int = 3600) -> int:
    """Drop sessions that were created but never used (no messages).

    These come from opening "+ New" and walking away; they clutter the rail
    forever otherwise. Only sessions older than an hour are removed."""
    cutoff = time.time() - max_age_seconds
    conn = _connect()
    try:
        with conn:
            n = conn.execute(
                "DELETE FROM sessions WHERE created_at < ? AND id NOT IN"
                " (SELECT DISTINCT session_id FROM messages)", (cutoff,)
            ).rowcount
    finally:
        conn.close()
    return n

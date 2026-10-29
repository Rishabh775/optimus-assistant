"""SQLite store for discovered jobs + application tracking.

jobs.db tables:
  jobs        one row per posting (UNIQUE url; company+title dedupe)
  job_events  append-only trail (discovered, status changes, notes)

Statuses are a small pipeline the agent and phone app share:
  new -> shortlisted -> drafting -> applied -> interview -> offer
  (plus terminal skipped / rejected)
Nothing is ever auto-submitted by the tracker; it only records state.
"""
import sqlite3
import time
import uuid
from pathlib import Path

from .. import config

STATUSES = ("new", "shortlisted", "drafting", "applied",
            "interview", "offer", "rejected", "skipped")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    company     TEXT NOT NULL,
    title       TEXT NOT NULL,
    location    TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    posted_at   REAL,
    salary      TEXT NOT NULL DEFAULT '',
    score       REAL NOT NULL DEFAULT 0,
    score_notes TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'new',
    channel     TEXT NOT NULL DEFAULT '',
    docs        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    applied_at  REAL
);
CREATE TABLE IF NOT EXISTS job_events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    kind   TEXT NOT NULL,
    note   TEXT NOT NULL DEFAULT '',
    ts     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_score ON jobs(score DESC);
"""


def db_path() -> Path:
    raw = config.get("JARVIS_JOBS_DB") or config.DEFAULTS["JARVIS_JOBS_DB"]
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / raw
    return p


def _connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=10)
    conn.executescript(_SCHEMA)
    return conn


def init() -> None:
    _connect().close()


def _row(r) -> dict:
    keys = ("id", "source", "company", "title", "location", "url",
            "description", "posted_at", "salary", "score", "score_notes",
            "status", "channel", "docs", "created_at", "updated_at",
            "applied_at")
    return dict(zip(keys, r))


def upsert(job: dict) -> str | None:
    """Insert a discovered job. Returns its id, or None when already known."""
    url = str(job.get("url") or "").strip()
    title = str(job.get("title") or "").strip()
    company = str(job.get("company") or "").strip()
    if not (url and title and company):
        return None
    conn = _connect()
    try:
        with conn:
            dupe = conn.execute(
                "SELECT id FROM jobs WHERE url=? OR (lower(company)=? AND"
                " lower(title)=?)", (url, company.lower(), title.lower())
            ).fetchone()
            if dupe:
                conn.execute(
                    "UPDATE jobs SET score=MAX(score, ?), score_notes=?"
                    " WHERE id=?", (float(job.get("score") or 0),
                                    str(job.get("score_notes") or ""),
                                    dupe[0]))
                return None
            jid = uuid.uuid4().hex[:12]
            now = time.time()
            conn.execute(
                "INSERT INTO jobs (id, source, company, title, location, url,"
                " description, posted_at, salary, score, score_notes, status,"
                " created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?, 'new', ?,?)",
                (jid, str(job.get("source") or ""), company, title,
                 str(job.get("location") or ""), url,
                 str(job.get("description") or "")[:8000],
                 job.get("posted_at"), str(job.get("salary") or ""),
                 float(job.get("score") or 0),
                 str(job.get("score_notes") or ""), now, now))
            conn.execute("INSERT INTO job_events (job_id, kind, note, ts)"
                         " VALUES (?, 'discovered', ?, ?)",
                         (jid, job.get("source") or "", now))
        return jid
    finally:
        conn.close()


def get(job_id: str) -> dict | None:
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT id, source, company, title, location, url, description,"
            " posted_at, salary, score, score_notes, status, channel, docs,"
            " created_at, updated_at, applied_at FROM jobs WHERE id=?",
            (job_id,)).fetchone()
    finally:
        conn.close()
    return _row(r) if r else None


def list_jobs(status: str = "", limit: int = 50, min_score: float = 0) -> list:
    q = ("SELECT id, source, company, title, location, url, description,"
         " posted_at, salary, score, score_notes, status, channel, docs,"
         " created_at, updated_at, applied_at FROM jobs WHERE score>=?")
    args = [float(min_score)]
    if status:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY score DESC, created_at DESC LIMIT ?"
    args.append(max(1, min(int(limit), 200)))
    conn = _connect()
    try:
        rows = conn.execute(q, args).fetchall()
    finally:
        conn.close()
    return [_row(r) for r in rows]


def update_status(job_id: str, status: str, note: str = "") -> bool:
    if status not in STATUSES:
        raise ValueError(f"status must be one of: {', '.join(STATUSES)}")
    conn = _connect()
    try:
        with conn:
            n = conn.execute(
                "UPDATE jobs SET status=?, updated_at=?,"
                " applied_at=CASE WHEN ?='applied' AND applied_at IS NULL"
                " THEN ? ELSE applied_at END WHERE id=?",
                (status, time.time(), status, time.time(), job_id)).rowcount
            if n:
                conn.execute(
                    "INSERT INTO job_events (job_id, kind, note, ts)"
                    " VALUES (?, ?, ?, ?)",
                    (job_id, f"status:{status}", note or "", time.time()))
    finally:
        conn.close()
    return n > 0


def events(job_id: str, limit: int = 20) -> list:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT kind, note, ts FROM job_events WHERE job_id=?"
            " ORDER BY ts DESC LIMIT ?", (job_id, limit)).fetchall()
    finally:
        conn.close()
    return [{"kind": r[0], "note": r[1], "ts": r[2]} for r in rows]


def stats() -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        conn.close()
    by = {s: 0 for s in STATUSES}
    by.update({r[0]: r[1] for r in rows})
    return {"total": total, "by_status": by}

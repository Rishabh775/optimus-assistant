"""Soundwave companion phone-app server.

A small FastAPI app that lives next to the voice modes and exposes the same
Agent pipeline to a phone (PWA) over the tailnet / LAN:

  REST (Bearer token):
    GET  /api/health              liveness probe (no token)
    POST /api/login               token check for the login screen
    GET  /api/status              basic machine info (+ reminders health)
    POST /api/sessions            new chat session
    GET  /api/sessions            sidebar list (newest first)
    GET  /api/sessions/{sid}/messages
    PATCH /api/sessions/{sid}     {title}
    DELETE /api/sessions/{sid}    delete + cascade messages
    GET  /api/reminders           pending reminders (scheduler/proactive.py)
    POST /api/reminders           {message, at, repeat} -> schedule
    DELETE /api/reminders/{rid}   cancel a pending reminder
    POST /api/briefing            composed daily briefing (sync in pool)
    POST /api/upload              multipart file (photo/doc) -> saved path
    POST /api/transcribe          multipart audio (voice note) -> text (STT)
    GET  /api/jobs                tracked jobs (optional ?status=)
    GET  /api/jobs/stats          counts per status
    GET  /api/jobs/profile        job-search profile + configured flag
    POST /api/jobs/sweep          discover + score + digest (read-only)
    PATCH /api/jobs/{id}          {status, note} pipeline update

  WS   /ws?token=     chat channel (streaming):
       {"type":"chat","session_id":...,"text":...,"attachments":[...]}
       -> {"type":"hello"|"delta"|"reply"|"confirm"|"note"|"error"}
       reply/confirm semantics unchanged; deltas stream the final answer.
       Fired reminders surface live as {"type":"note"} on every open chat.
       attachments: files previously returned by /api/upload; their paths
       are added to the agent turn (marker lines) so tools can read them.

Security:
  - every endpoint except /api/health requires the auth token
  - the token is read from JARVIS_APP_TOKEN; if unset it is generated and
    appended to the repo .env on first start
  - agent confirmations (including mandatory ones) are bridged over the
    socket, so gated writes work from the phone instead of being denied

History: sessions/messages live in SQLite (JARVIS_APP_DB); the Agent's
in-RAM history for a session is rebuilt from the DB per connection, so
reopening the PWA continues the conversation.
"""
import asyncio
import os
import re
import secrets
import socket
import tempfile
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from pathlib import Path

from dotenv import load_dotenv, set_key
from fastapi import (FastAPI, File, Header, HTTPException, UploadFile,
                     WebSocket, status)
from fastapi.websockets import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config
from .. import briefing, jobs, push, scheduler
from ..agent import Agent
from . import store

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
_STATIC = Path(__file__).resolve().parent / "static"

# Keep the Agent's in-RAM history bounded like the voice loop does.
_MSG_WINDOW = 40

_UPLOAD_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_AUDIO_SUFFIXES = {".webm", ".m4a", ".mp3", ".wav", ".ogg", ".opus",
                   ".aac", ".flac", ".3gp"}
_MAX_ATTACHMENTS = 5


def _uploads_dir() -> Path:
    raw = config.get("JARVIS_APP_UPLOAD_DIR")
    if not raw:
        raise RuntimeError("JARVIS_APP_UPLOAD_DIR is empty")
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO_ROOT / raw
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_name(name: str) -> str:
    base = Path(str(name or "file")).name
    cleaned = _UPLOAD_NAME_RE.sub("_", base).strip("._") or "file"
    return cleaned[-80:]


def _upload_allowed(raw: str) -> Path | None:
    """Only paths returned by /api/upload may be attached from the phone."""
    try:
        p = Path(str(raw)).resolve()
        root = _uploads_dir().resolve()
        p.relative_to(root)              # raises if outside the uploads dir
    except (ValueError, OSError, RuntimeError):
        return None
    return p if p.is_file() else None


def ensure_token() -> str:
    """Return the app auth token, generating one into .env if it's missing."""
    tok = config.get("JARVIS_APP_TOKEN")
    if tok:
        return tok
    tok = secrets.token_urlsafe(24)
    try:
        if _ENV_FILE.exists():
            set_key(str(_ENV_FILE), "JARVIS_APP_TOKEN", tok)
        else:
            _ENV_FILE.write_text(f"JARVIS_APP_TOKEN={tok}\n")
        os.environ["JARVIS_APP_TOKEN"] = tok
    except OSError:
        return tok
    return tok


class _TokenIn(BaseModel):
    token: str


class _TitleIn(BaseModel):
    title: str


class _ReminderIn(BaseModel):
    message: str
    at: str
    repeat: str = ""


class _JobPatchIn(BaseModel):
    status: str
    note: str = ""


def create_app() -> FastAPI:
    app = FastAPI(title="Soundwave", docs_url=None, redoc_url=None)
    token = ensure_token()
    store.init()
    scheduler.start()                        # reminders, same rows as CLI modes
    notify_on = push.install()               # phone push off => TTS/log-only
    briefing.install()                       # scheduled daily briefing
    jobs.install()                           # scheduled job sweep (opt-in)

    # live WS connections get proactive events (fired reminders) as notes
    _live_ws: list = []                      # (loop, threadsafeSender)

    def _proactive_note(r: dict) -> None:
        mark = "Missed reminder" if r.get("missed") else "Reminder"
        for loop, send in list(_live_ws):
            try:
                send({"type": "note", "text": f"{mark}: {r['message']}"})
            except Exception:
                pass

    scheduler.add_listener(_proactive_note)

    def _token_ok(candidate: str | None) -> bool:
        return bool(candidate) and secrets.compare_digest(candidate, token)

    def _bearer(authorization: str | None) -> str:
        scheme, param = get_authorization_scheme_param(authorization or "")
        if scheme.lower() != "bearer" or not _token_ok(param):
            raise HTTPException(status_code=403, detail="bad token")
        return param

    @app.get("/api/health")
    def health():
        return {"ok": True, "machine": socket.gethostname()}

    @app.post("/api/login")
    def login(body: _TokenIn):
        if not _token_ok(body.token):
            raise HTTPException(status_code=403, detail="bad token")
        return {"ok": True}

    @app.get("/api/status")
    def st(authorization: str | None = Header(default=None)):
        _bearer(authorization)
        return {"ok": True, "machine": socket.gethostname(),
                "system": f"{os.uname().sysname} {os.uname().release}",
                "reminders": {"pending": len(scheduler.list_pending()),
                              "push": notify_on}}

    # ---- sessions (ChatGPT-style right rail) ----

    @app.post("/api/sessions")
    def new_session(title: str = "", authorization: str | None = Header(default=None)):
        _bearer(authorization)
        return store.new_session(title)

    @app.get("/api/sessions")
    def list_sessions(authorization: str | None = Header(default=None)):
        _bearer(authorization)
        return store.list_sessions()

    @app.get("/api/sessions/{sid}/messages")
    def session_messages(sid: str, limit: int = 0,
                         authorization: str | None = Header(default=None)):
        _bearer(authorization)
        if not store.session_exists(sid):
            raise HTTPException(status_code=404, detail="no such session")
        return store.get_messages(sid, limit=limit or None)

    @app.patch("/api/sessions/{sid}")
    def rename_session(sid: str, body: _TitleIn,
                       authorization: str | None = Header(default=None)):
        _bearer(authorization)
        if not store.session_exists(sid):
            raise HTTPException(status_code=404, detail="no such session")
        store.rename_session(sid, body.title.strip() or "untitled")
        return {"ok": True}

    @app.delete("/api/sessions/{sid}")
    def del_session(sid: str, authorization: str | None = Header(default=None)):
        _bearer(authorization)
        store.delete_session(sid)
        return {"ok": True}

    # ---- proactive loop: reminders + briefing on demand ----

    @app.get("/api/reminders")
    def list_reminders(authorization: str | None = Header(default=None)):
        _bearer(authorization)
        return scheduler.list_pending()

    @app.post("/api/reminders")
    def add_reminder(body: _ReminderIn,
                     authorization: str | None = Header(default=None)):
        _bearer(authorization)
        try:
            return scheduler.add(body.message, body.at, body.repeat or "")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/reminders/{rid}")
    def del_reminder(rid: str,
                     authorization: str | None = Header(default=None)):
        _bearer(authorization)
        if not scheduler.cancel(rid):
            raise HTTPException(status_code=404,
                                detail="no pending reminder with that id")
        return {"ok": True}

    @app.post("/api/briefing")
    def briefing_ep(authorization: str | None = Header(default=None)):
        # plain def -> FastAPI runs it in the worker pool (IMAP can be slow)
        _bearer(authorization)
        return {"title": "Briefing", "text": briefing.build()}

    # ---- jobs skill (discovery + tracker; nothing is submitted here) ----

    @app.get("/api/jobs")
    def jobs_list(status: str = "", limit: int = 50,
                  authorization: str | None = Header(default=None)):
        _bearer(authorization)
        if status and status not in jobs.tracker.STATUSES:
            raise HTTPException(status_code=400,
                                detail=f"bad status {status!r}")
        rows = jobs.tracker.list_jobs(status=status, limit=limit)
        keep = ("id", "source", "company", "title", "location", "url",
                "salary", "score", "score_notes", "status", "created_at",
                "applied_at")
        return {"jobs": [{k: r.get(k) for k in keep} for r in rows]}

    @app.get("/api/jobs/stats")
    def jobs_stats(authorization: str | None = Header(default=None)):
        _bearer(authorization)
        return jobs.tracker.stats()

    @app.get("/api/jobs/profile")
    def jobs_profile(authorization: str | None = Header(default=None)):
        _bearer(authorization)
        prof = jobs.profile.load()
        return {"configured": bool(prof.get("roles")), "profile": prof,
                "resume": jobs.resume.ready()}

    @app.post("/api/jobs/sweep")
    def jobs_sweep(authorization: str | None = Header(default=None)):
        # plain def -> worker pool: board downloads take a while
        _bearer(authorization)
        result = jobs.sweep()
        if not result["ok"]:
            raise HTTPException(status_code=400, detail=result["reason"])
        return {"ok": True, "found": result["found"],
                "inserted": result["inserted"]}

    @app.patch("/api/jobs/{jid}")
    def jobs_patch(jid: str, body: _JobPatchIn,
                   authorization: str | None = Header(default=None)):
        _bearer(authorization)
        try:
            ok = jobs.tracker.update_status(jid, body.status, body.note)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not ok:
            raise HTTPException(status_code=404, detail="no such job")
        return {"ok": True}

    @app.post("/api/upload")
    def upload_file(file: UploadFile = File(...),
                    authorization: str | None = Header(default=None)):
        """Save a phone photo/document. Returns the path to attach to a chat."""
        _bearer(authorization)
        max_mb = config.get_int("JARVIS_APP_UPLOAD_MAX_MB") or 25
        limit = max_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(status_code=413,
                                detail=f"file exceeds {max_mb} MB")
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        name = _safe_name(file.filename or "file")
        dest = _uploads_dir() / time.strftime("%Y%m%d") / \
            f"{uuid.uuid4().hex[:8]}_{name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"path": str(dest), "name": name, "size": len(data),
                "mime": file.content_type or ""}

    @app.post("/api/transcribe")
    def transcribe_audio(file: UploadFile = File(...),
                         authorization: str | None = Header(default=None)):
        """Voice note -> text via the existing faster-whisper stack."""
        _bearer(authorization)
        suffix = Path(file.filename or "note.webm").suffix.lower()
        if suffix not in _AUDIO_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported audio type {suffix or '(none)'}")
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=str(_uploads_dir()))
        try:
            with os.fdopen(fd, "wb") as fh:
                while chunk := file.file.read(1024 * 1024):
                    fh.write(chunk)
            from ..voice import transcribe_file
            text = (transcribe_file(tmp) or "").strip()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"transcription failed: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if not text:
            raise HTTPException(status_code=422,
                                detail="no speech detected in the recording")
        return {"text": text}

    # ---- PWA ----

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        if not _token_ok(ws.query_params.get("token")):
            await ws.send_json({"type": "error", "text": "bad token"})
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        loop = asyncio.get_running_loop()
        pending: dict[str, Future] = {}
        state = {"busy": False}
        agents: dict[str, Agent] = {}   # session_id -> Agent (hist cache)

        async def _send(obj: dict) -> None:
            await ws.send_json(obj)

        def _send_threadsafe(obj: dict) -> None:
            asyncio.run_coroutine_threadsafe(_send(obj), loop)

        confirm_timeout = config.get_int("JARVIS_APP_CONFIRM_TIMEOUT")

        def confirm_hook(reason: str, fn_name: str, args: dict,
                         mandatory: bool = False) -> bool:
            """Ask the phone for a tool confirmation (runs in agent thread)."""
            cid = uuid.uuid4().hex[:12]
            fut: Future = Future()
            pending[cid] = fut
            _send_threadsafe({"type": "confirm", "id": cid, "reason": reason,
                              "tool": fn_name, "args": args,
                              "mandatory": bool(mandatory)})
            try:
                return bool(fut.result(timeout=confirm_timeout))
            except FutureTimeoutError:
                return False
            finally:
                pending.pop(cid, None)

        await _send({"type": "hello", "machine": socket.gethostname()})

        entry = (loop, _send_threadsafe)
        _live_ws.append(entry)
        def _agent_for(sid: str) -> Agent:
            """Reuse a session's Agent, rebuilding its history from the DB."""
            a = agents.get(sid)
            if a is None:
                a = Agent(confirm=confirm_hook)
                hist = [{"role": m["role"], "content": m["content"]}
                        for m in store.get_messages(sid, limit=_MSG_WINDOW)]
                if hist:
                    base = a.history[:1]   # keep the system stub
                    a.history = ([*base, *hist] if base else hist)
                agents[sid] = a
            else:
                a.confirm = confirm_hook   # rebind to this connection
            return a

        async def run_chat(sid: str, text: str) -> None:
            def _send_delta(piece: str) -> None:
                _send_threadsafe({"type": "delta", "sid": sid, "text": piece})

            def _job() -> str:
                return _agent_for(sid).ask(text, on_delta=_send_delta)

            try:
                reply = await loop.run_in_executor(None, _job)
                store.add_message(sid, "assistant", reply)
                await _send({"type": "reply", "sid": sid, "text": reply})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                try:
                    await _send({"type": "reply", "sid": sid,
                                 "text": f"error: {e}"})
                except Exception:
                    pass
            finally:
                state["busy"] = False

        try:
            while True:
                try:
                    msg = await ws.receive_json()
                except WebSocketDisconnect:
                    return
                except ValueError:  # non-JSON frame
                    continue
                kind = msg.get("type")
                if kind == "chat":
                    text = (msg.get("text") or "").strip()
                    atts = []
                    for a in (msg.get("attachments") or [])[:_MAX_ATTACHMENTS]:
                        if not isinstance(a, dict):
                            continue
                        p = _upload_allowed(a.get("path", ""))
                        if p:
                            atts.append({"path": str(p),
                                         "name": a.get("name") or p.name,
                                         "mime": a.get("mime") or ""})
                    if not text and not atts:
                        await _send({"type": "note", "text": "empty"})
                    elif state["busy"]:
                        await _send({"type": "note", "text": "busy"})
                    else:
                        sid = msg.get("session_id")
                        created = not sid or not store.session_exists(sid)
                        if created:
                            sid = store.new_session()["id"]
                        else:
                            existing = store.get_session(sid)
                            if existing and not existing["title"]:
                                created = True   # untitled -> auto-title
                        state["busy"] = True
                        body = text
                        if atts:
                            body += ("\n\n[Attached files - use read_file or "
                                     "shell_run to inspect]\n")
                            body += "\n".join(
                                f"📎 {a['name']} ({a['mime'] or 'file'}): "
                                f"{a['path']}" for a in atts)
                        body = body.strip()
                        # persist the user turn *before* the long ask call
                        store.add_message(sid, "user", body)
                        if created:
                            title = text or (atts[0]["name"] if atts else "")
                            store.rename_session(sid, title[:38])
                        asyncio.create_task(run_chat(sid, body))
                elif kind == "confirm":
                    fut = pending.get(msg.get("id"))
                    if fut and not fut.done():
                        fut.set_result(bool(msg.get("answer")))
                elif kind == "ping":
                    await _send({"type": "pong"})
                else:
                    await _send({"type": "note", "text": "unknown type"})
        finally:
            if entry in _live_ws:
                _live_ws.remove(entry)

    return app

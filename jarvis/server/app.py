"""Soundwave companion phone-app server.

A small FastAPI app that lives next to the voice modes and exposes the same
Agent pipeline to a phone (PWA) over the tailnet / LAN:

  REST (Bearer token):
    GET  /api/health              liveness probe (no token)
    POST /api/login               token check for the login screen
    GET  /api/status              basic machine info
    POST /api/sessions            new chat session
    GET  /api/sessions            sidebar list (newest first)
    GET  /api/sessions/{sid}/messages
    PATCH /api/sessions/{sid}     {title}
    DELETE /api/sessions/{sid}    delete + cascade messages

  WS   /ws?token=     chat channel (streaming):
       {"type":"chat","session_id":...,"text":...}
       -> {"type":"hello"|"delta"|"reply"|"confirm"|"note"|"error"}
       reply/confirm semantics unchanged; deltas stream the final answer.

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
import secrets
import socket
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from pathlib import Path

from dotenv import load_dotenv, set_key
from fastapi import FastAPI, Header, HTTPException, WebSocket, status
from fastapi.websockets import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config
from .. import briefing, push, scheduler
from ..agent import Agent
from . import store

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
_STATIC = Path(__file__).resolve().parent / "static"

# Keep the Agent's in-RAM history bounded like the voice loop does.
_MSG_WINDOW = 40


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


def create_app() -> FastAPI:
    app = FastAPI(title="Soundwave", docs_url=None, redoc_url=None)
    token = ensure_token()
    store.init()
    scheduler.start()                        # reminders, same rows as CLI modes
    notify_on = push.install()               # phone push off => TTS/log-only
    briefing.install()                       # scheduled daily briefing

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
                if not text:
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
                    # persist the user turn *before* the long ask call
                    store.add_message(sid, "user", text)
                    if created:
                        store.rename_session(sid, text[:38])
                    asyncio.create_task(run_chat(sid, text))
            elif kind == "confirm":
                fut = pending.get(msg.get("id"))
                if fut and not fut.done():
                    fut.set_result(bool(msg.get("answer")))
            else:
                await _send({"type": "note", "text": "unknown type"})

    return app

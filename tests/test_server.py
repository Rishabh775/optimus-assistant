"""Tests for the Soundwave companion app server (sessions + streaming WS)."""
import pytest
from fastapi.testclient import TestClient

import jarvis.server.app as srv
from jarvis.server.app import create_app

TOK = "test-token-12345"


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_APP_DB", str(tmp_path / "chats.db"))
    monkeypatch.setenv("JARVIS_REMINDER_DB", str(tmp_path / "reminders.db"))
    monkeypatch.setenv("JARVIS_APP_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JARVIS_JOBS_DB", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("JARVIS_JOBS_PROFILE", str(tmp_path / "profile.json"))
    monkeypatch.setenv("JARVIS_JOBS_ENABLED", "0")
    monkeypatch.setenv("JARVIS_JOBS_ALERTS", "0")
    for k in ("JARVIS_NTFY_URL", "JARVIS_NTFY_TOPIC", "JARVIS_BRIEFING_TIME"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("JARVIS_APP_TOKEN", TOK)
    return TestClient(create_app())


def _auth(client):
    return {"Authorization": f"Bearer {TOK}"}


class _FakeAgent:
    history = []

    def __init__(self, confirm=None):
        self.confirm = confirm

    def ask(self, text, max_steps=None, on_delta=None):
        if on_delta:
            for piece in ("PONG ", "…"):
                on_delta(piece)
        return f"PONG finished {text}"


class _ConfirmingAgent:
    history = []

    def __init__(self, confirm=None):
        self.confirm = confirm

    def ask(self, text, max_steps=None, on_delta=None):
        if self.confirm("test reason", "py_write", {"path": "/tmp/x"}):
            return "DID IT"
        return "DENIED"


# ---- health / login / status ----

def test_health_no_token_needed(client):
    r = client.get("/api/health")
    assert r.status_code == 200


def test_login_rejects_wrong_token(client):
    assert client.post("/api/login", json={"token": "wrong"}).status_code == 403


def test_login_accepts_right_token(client):
    assert client.post("/api/login", json={"token": TOK}).status_code == 200


def test_status_requires_bearer_token(client):
    assert client.get("/api/status").status_code == 403
    assert client.get("/api/status",
                      headers={"Authorization": "Bearer wrong"}).status_code == 403
    assert client.get("/api/status", headers=_auth(client)).status_code == 200


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Soundwave" in r.text


def test_static_assets_served(client):
    for path in ("/static/manifest.webmanifest", "/static/sw.js",
                 "/static/icons/icon-192.png", "/static/icons/icon-512.png"):
        assert client.get(path).status_code == 200, path


# ---- sessions REST ----

def test_sessions_crud_flow(client):
    r = client.post("/api/sessions", json={}, headers=_auth(client))
    assert r.status_code == 200
    sid = r.json()["id"]
    r = client.post("/api/sessions", json={})   # no token
    assert r.status_code == 403

    rows = client.get("/api/sessions", headers=_auth(client)).json()
    assert rows[-1]["id"] == sid

    msgs = client.get(f"/api/sessions/{sid}/messages",
                      headers=_auth(client)).json()
    assert msgs == []
    assert client.get("/api/sessions/nope/messages",
                      headers=_auth(client)).status_code == 404

    r = client.patch(f"/api/sessions/{sid}", json={"title": "renamed"},
                     headers=_auth(client))
    assert r.status_code == 200
    rows = client.get("/api/sessions", headers=_auth(client)).json()
    assert rows[-1]["title"] == "renamed"

    r = client.delete(f"/api/sessions/{sid}", headers=_auth(client))
    assert r.status_code == 200
    rows = client.get("/api/sessions", headers=_auth(client)).json()
    assert all(r["id"] != sid for r in rows)


# ---- websocket chat ----

def test_ws_chat_streams_and_persists(client, monkeypatch):
    monkeypatch.setattr(srv, "Agent", _FakeAgent)
    sid, deltas, reply = None, [], None
    with client.websocket_connect(f"/ws?token={TOK}") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "chat", "text": "ping"})
        evt = ws.receive_json()
        while evt["type"] != "reply":
            deltas.append((evt["type"], evt.get("text")))
            evt = ws.receive_json()
        reply, sid = evt["text"], evt["sid"]
    assert deltas == [("delta", "PONG "), ("delta", "…")]
    assert reply == "PONG finished ping"
    assert sid
    msgs = client.get(f"/api/sessions/{sid}/messages",
                      headers=_auth(client)).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[-1]["content"] == reply
    assert client.get("/api/sessions", headers=_auth(client)).json()[-1][
        "title"].startswith("ping")


def test_ws_missing_session_id_is_created(client, monkeypatch):
    monkeypatch.setattr(srv, "Agent", _FakeAgent)
    with client.websocket_connect(f"/ws?token={TOK}") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "hey"})
        evt = ws.receive_json()
        while evt["type"] != "reply":
            evt = ws.receive_json()
        assert evt["sid"]


def test_ws_bad_token_rejected(client):
    with client.websocket_connect("/ws?token=nope") as ws:
        r = ws.receive_json()
        assert r == {"type": "error", "text": "bad token"}


def test_ws_confirm_flow_allowed(client, monkeypatch):
    monkeypatch.setattr(srv, "Agent", _ConfirmingAgent)
    with client.websocket_connect(f"/ws?token={TOK}") as ws:
        ws.receive_json()  # hello
        ws.send_json({"type": "chat", "text": "write the file"})
        c = ws.receive_json()
        assert c["type"] == "confirm" and c["tool"] == "py_write"
        assert c.get("mandatory") is False   # risk flag reaches the client
        ws.send_json({"type": "confirm", "id": c["id"], "answer": True})
        r = ws.receive_json()
        assert r["type"] == "reply" and r["text"] == "DID IT"


def test_ws_confirm_flow_denied(client, monkeypatch):
    monkeypatch.setattr(srv, "Agent", _ConfirmingAgent)
    with client.websocket_connect(f"/ws?token={TOK}") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "write the file"})
        c = ws.receive_json()
        ws.send_json({"type": "confirm", "id": c["id"], "answer": False})
        r = ws.receive_json()
        assert r["type"] == "reply" and r["text"] == "DENIED"


# ---- proactive loop endpoints (reminders + briefing) ----

def test_reminders_require_token(client):
    assert client.get("/api/reminders").status_code == 403
    assert client.post("/api/reminders",
                       json={"message": "x", "at": "in 1 hour"}).status_code == 403
    assert client.delete("/api/reminders/nope").status_code == 403
    assert client.post("/api/briefing").status_code == 403


def test_reminders_crud_roundtrip(client):
    assert client.get("/api/reminders", headers=_auth(client)).json() == []
    r = client.post("/api/reminders", headers=_auth(client),
                    json={"message": "test ping", "at": "in 5 minutes"})
    assert r.status_code == 200
    rid = r.json()["id"]
    assert r.json() == {"id": rid, "message": "test ping",
                        "due_at": r.json()["due_at"], "repeat": "",
                        "status": "pending"}
    assert len(client.get("/api/reminders", headers=_auth(client)).json()) == 1
    r = client.post("/api/reminders", headers=_auth(client),
                    json={"message": "bad", "at": "not a time"})
    assert r.status_code == 400
    assert "cannot parse time" in r.json()["detail"]
    assert client.delete(f"/api/reminders/{rid}",
                         headers=_auth(client)).json() == {"ok": True}
    assert client.get("/api/reminders", headers=_auth(client)).json() == []
    assert client.delete(f"/api/reminders/{rid}",
                         headers=_auth(client)).status_code == 404


def test_briefing_on_demand(client, monkeypatch):
    import jarvis.briefing as b
    monkeypatch.setattr(b, "build", lambda: "ORDERED LINE")
    r = client.post("/api/briefing", headers=_auth(client))
    assert r.status_code == 200
    assert r.json() == {"title": "Briefing", "text": "ORDERED LINE"}


def test_status_reports_reminders(client):
    d = client.get("/api/status", headers=_auth(client)).json()
    assert d["reminders"]["pending"] == 0
    assert d["reminders"]["push"] is False


# ---- uploads / voice notes (Android + PWA attachments) ----

def test_upload_requires_token(client):
    r = client.post("/api/upload", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 403
    r = client.post("/api/transcribe", files={"file": ("a.webm", b"x", "audio/webm")})
    assert r.status_code == 403


def test_upload_saves_file_and_returns_path(client):
    r = client.post("/api/upload", headers=_auth(client),
                    files={"file": ("my photo.jpg", b"JPEGDATA", "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert d["name"] == "my_photo.jpg"          # sanitized for the filesystem
    assert d["size"] == 8 and d["mime"] == "image/jpeg"
    from pathlib import Path
    p = Path(d["path"])
    assert p.is_file() and p.read_bytes() == b"JPEGDATA"


def test_upload_rejects_oversize(client, monkeypatch):
    monkeypatch.setenv("JARVIS_APP_UPLOAD_MAX_MB", "1")
    big = b"x" * (1024 * 1024 + 1)
    r = client.post("/api/upload", headers=_auth(client),
                    files={"file": ("big.bin", big, "application/octet-stream")})
    assert r.status_code == 413


def test_upload_allowed_confines_to_uploads_dir(tmp_path):
    import jarvis.server.app as srv
    assert srv._upload_allowed("/etc/passwd") is None
    assert srv._upload_allowed(str(tmp_path / "nope.jpg")) is None


def test_transcribe_happy_path(client, monkeypatch):
    seen = {}

    def fake(path):
        seen["path"] = path
        assert path.endswith(".webm")
        return "  remind me to buy milk  "

    import jarvis.voice as voice
    monkeypatch.setattr(voice, "transcribe_file", fake)
    r = client.post("/api/transcribe", headers=_auth(client),
                    files={"file": ("note.webm", b"AUDIO", "audio/webm")})
    assert r.status_code == 200
    assert r.json() == {"text": "remind me to buy milk"}
    assert not __import__("os").path.exists(seen["path"])   # temp cleaned


def test_transcribe_empty_and_bad_type(client, monkeypatch):
    import jarvis.voice as voice
    monkeypatch.setattr(voice, "transcribe_file", lambda p: "")
    r = client.post("/api/transcribe", headers=_auth(client),
                    files={"file": ("note.webm", b"AUDIO", "audio/webm")})
    assert r.status_code == 422
    r = client.post("/api/transcribe", headers=_auth(client),
                    files={"file": ("note.exe", b"X", "application/octet-stream")})
    assert r.status_code == 400


def test_chat_attachment_markers_reach_agent_and_db(client, monkeypatch):
    got = {}

    class Capture:
        history = []

        def __init__(self, confirm=None):
            pass

        def ask(self, text, max_steps=None, on_delta=None):
            got["text"] = text
            return "ok"

    monkeypatch.setattr(srv, "Agent", Capture)
    up = client.post("/api/upload", headers=_auth(client),
                     files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")})
    att = up.json()
    with client.websocket_connect(f"/ws?token={TOK}") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "summarize this",
                      "attachments": [att]})
        r = ws.receive_json()
        assert r["type"] == "reply"
    assert "[Attached files" in got["text"]
    assert att["path"] in got["text"]
    sid = client.get("/api/sessions", headers=_auth(client)).json()[0]["id"]
    msgs = client.get(f"/api/sessions/{sid}/messages", headers=_auth(client)).json()
    assert msgs[0]["role"] == "user"
    assert att["path"] in msgs[0]["content"]


def test_ws_ping_pong_keeps_connection_alive(client):
    with client.websocket_connect(f"/ws?token={TOK}") as ws:
        ws.receive_json()                       # hello
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


# ---- jobs skill endpoints ----

def test_jobs_require_token(client):
    assert client.get("/api/jobs").status_code == 403
    assert client.get("/api/jobs/stats").status_code == 403
    assert client.get("/api/jobs/profile").status_code == 403
    assert client.post("/api/jobs/sweep").status_code == 403
    assert client.patch("/api/jobs/x", json={"status": "new"}).status_code == 403


def test_jobs_empty_and_profile(client):
    assert client.get("/api/jobs", headers=_auth(client)).json() == {"jobs": []}
    st = client.get("/api/jobs/stats", headers=_auth(client)).json()
    assert st["total"] == 0 and st["by_status"]["new"] == 0
    prof = client.get("/api/jobs/profile", headers=_auth(client)).json()
    assert prof["configured"] is False


def test_jobs_sweep_without_profile_is_400(client):
    r = client.post("/api/jobs/sweep", headers=_auth(client))
    assert r.status_code == 400
    assert "profile" in r.json()["detail"]


def test_jobs_patch_lifecycle(client):
    from jarvis.jobs import tracker
    tracker.upsert({"source": "test", "company": "Acme",
                    "title": "Frontend Engineer", "url": "https://x/1",
                    "score": 80})
    jobs = client.get("/api/jobs", headers=_auth(client)).json()["jobs"]
    assert len(jobs) == 1 and jobs[0]["status"] == "new"
    jid = jobs[0]["id"]

    r = client.patch(f"/api/jobs/{jid}", headers=_auth(client),
                     json={"status": "shortlisted", "note": "nice fit"})
    assert r.status_code == 200
    assert client.get("/api/jobs?status=shortlisted",
                      headers=_auth(client)).json()["jobs"][0]["id"] == jid

    assert client.patch(f"/api/jobs/{jid}", headers=_auth(client),
                        json={"status": "banana"}).status_code == 400
    assert client.patch("/api/jobs/nope", headers=_auth(client),
                        json={"status": "applied"}).status_code == 404


def test_jobs_sweep_endpoint_calls_skill(client, monkeypatch):
    import jarvis.jobs as jobs_skill
    monkeypatch.setattr(jobs_skill, "sweep",
                        lambda: {"ok": True, "found": 3, "inserted": 2,
                                 "digest": "x"})
    r = client.post("/api/jobs/sweep", headers=_auth(client))
    assert r.status_code == 200
    assert r.json() == {"ok": True, "found": 3, "inserted": 2}


def test_jobs_min_score_filter(client):
    from jarvis.jobs import tracker
    tracker.upsert({"source": "test", "company": "Low", "title": "A",
                    "url": "https://x/1", "score": 10})
    tracker.upsert({"source": "test", "company": "High", "title": "B",
                    "url": "https://x/2", "score": 80})
    rows = client.get("/api/jobs?min_score=50", headers=_auth(client)).json()["jobs"]
    assert [r["company"] for r in rows] == ["High"]
    assert client.get("/api/jobs?min_score=500",
                      headers=_auth(client)).status_code == 400
    assert client.get("/api/jobs/stats", headers=_auth(client)).json()["min_score"] == 25


def test_push_test_endpoint(client, monkeypatch):
    import jarvis.server.app as srv
    monkeypatch.setattr(srv.push, "push_configured", lambda: False)
    assert client.post("/api/push/test", headers=_auth(client)).status_code == 400

    monkeypatch.setattr(srv.push, "push_configured", lambda: True)
    monkeypatch.setattr(srv.push, "send", lambda t, m: False)
    assert client.post("/api/push/test", headers=_auth(client)).status_code == 502

    monkeypatch.setattr(srv.push, "send", lambda t, m: True)
    r = client.post("/api/push/test", headers=_auth(client))
    assert r.status_code == 200 and r.json() == {"ok": True}


# ---- WhatsApp Cloud API webhook ----

class _EchoAgent:
    history = []

    def __init__(self, confirm=None):
        self.confirm = confirm

    def ask(self, text, max_steps=None, on_delta=None):
        return f"echo {text}"


PAYLOAD = {"object": "whatsapp_business_account", "entry": [{"changes": [{
    "field": "messages", "value": {"messaging_product": "whatsapp",
    "metadata": {"phone_number_id": "123"},
    "messages": [{"from": "919999999999", "id": "wamid.1",
                  "type": "text", "text": {"body": "ping"}}]}}]}]}


def test_wa_extract_parses_text_only():
    from jarvis.server.app import _wa_extract
    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"from": "9199", "type": "text", "text": {"body": " hi "}},
        {"from": "9199", "type": "image", "image": {}},
    ]}}]}]}
    assert _wa_extract(payload) == [("9199", "hi")]


def test_wa_webhook_verification(client, monkeypatch):
    monkeypatch.setenv("JARVIS_WA_VERIFY_TOKEN", "verify-me")
    ok = client.get("/api/whatsapp/webhook", params={
        "hub.mode": "subscribe", "hub.verify_token": "verify-me",
        "hub.challenge": "42"})
    assert ok.status_code == 200 and ok.text == "42"
    bad = client.get("/api/whatsapp/webhook", params={
        "hub.mode": "subscribe", "hub.verify_token": "nope",
        "hub.challenge": "42"})
    assert bad.status_code == 403


def test_wa_webhook_signature(client, monkeypatch):
    import hashlib
    import hmac
    import json
    monkeypatch.setenv("JARVIS_WA_APP_SECRET", "appsecret")
    raw = json.dumps(PAYLOAD).encode()
    good = "sha256=" + hmac.new(b"appsecret", raw, hashlib.sha256).hexdigest()
    r = client.post("/api/whatsapp/webhook", content=raw,
                    headers={"content-type": "application/json",
                             "x-hub-signature-256": good})
    assert r.status_code == 200
    r = client.post("/api/whatsapp/webhook", content=raw,
                    headers={"content-type": "application/json",
                             "x-hub-signature-256": "sha256=bad"})
    assert r.status_code == 403


def test_wa_webhook_replies_and_stores(client, monkeypatch):
    monkeypatch.setenv("JARVIS_WA_APP_SECRET", "")
    monkeypatch.setattr(srv, "Agent", _EchoAgent)
    import jarvis.whatsapp as wa
    sent = []
    monkeypatch.setattr(wa, "send",
                        lambda text, to="": sent.append((to, text)) or True)
    r = client.post("/api/whatsapp/webhook", json=PAYLOAD)
    assert r.status_code == 200
    assert sent == [("919999999999", "echo ping")]

    sessions = client.get("/api/sessions", headers=_auth(client)).json()
    wa_chat = next(x for x in sessions if x["title"] == "WhatsApp")
    msgs = client.get(f"/api/sessions/{wa_chat['id']}/messages",
                      headers=_auth(client)).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]

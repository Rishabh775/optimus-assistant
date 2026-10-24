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

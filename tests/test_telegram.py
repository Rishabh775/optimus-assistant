"""Telegram Bot API client + notification fan-out."""
import json

import pytest

from jarvis import push, telegram


class Recorder:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, req, timeout=None):
        if self.fail:
            raise OSError("telegram down")
        self.calls.append({"url": req.full_url,
                           "body": json.loads(req.data.decode())})
        import contextlib

        @contextlib.contextmanager
        def resp():
            class R:
                def read(self):
                    return b'{"ok": true}'
            yield R()
        return resp()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("JARVIS_TELEGRAM_ENABLED", "JARVIS_TELEGRAM_BOT_TOKEN",
              "JARVIS_TELEGRAM_ALLOW", "JARVIS_WHATSAPP_ENABLED",
              "JARVIS_NTFY_URL", "JARVIS_NTFY_TOPIC"):
        monkeypatch.delenv(k, raising=False)


def _enable(monkeypatch, allow="42"):
    monkeypatch.setenv("JARVIS_TELEGRAM_ENABLED", "1")
    monkeypatch.setenv("JARVIS_TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("JARVIS_TELEGRAM_ALLOW", allow)


def test_disabled_by_default():
    assert telegram.configured() is False
    assert telegram.send("hi") is False


def test_send_uses_first_allowed_chat(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(telegram.urllib.request, "urlopen", rec)
    _enable(monkeypatch)
    assert telegram.send("hello") is True
    c = rec.calls[0]
    assert c["url"].endswith("/bot123:ABC/sendMessage")
    assert c["body"]["chat_id"] == "42"
    assert c["body"]["text"] == "hello"


def test_send_explicit_chat_and_chunking(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(telegram.urllib.request, "urlopen", rec)
    _enable(monkeypatch, allow="")
    assert telegram.send("x" * 8500, chat_id="99") is True
    assert [c["body"]["chat_id"] for c in rec.calls] == ["99", "99", "99"]
    assert sum(len(c["body"]["text"]) for c in rec.calls) == 8500


def test_notify_plain_text(monkeypatch):
    seen = []
    monkeypatch.setattr(telegram, "send",
                        lambda text, chat_id="": seen.append(text) or True)
    telegram.notify("Job alert", "Acme — Frontend")
    assert seen == ["Job alert\nAcme — Frontend"]


def test_send_failure(monkeypatch):
    monkeypatch.setattr(telegram.urllib.request, "urlopen", Recorder(fail=True))
    _enable(monkeypatch)
    assert telegram.send("hi") is False


def test_push_falls_back_to_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(telegram, "configured", lambda: True)
    monkeypatch.setattr(telegram, "notify",
                        lambda t, m: sent.append((t, m)) or True)
    assert push.push_configured() is True
    assert push.send("Reminder", "stand up") is True
    assert sent == [("Reminder", "stand up")]

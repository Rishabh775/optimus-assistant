"""WhatsApp bridge client + push integration."""
import json

import pytest

from jarvis import push, whatsapp


class Recorder:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, req, timeout=None):
        if self.fail:
            raise OSError("bridge down")
        self.calls.append({"url": req.full_url,
                           "headers": dict(req.header_items()),
                           "body": json.loads(req.data.decode())})
        import contextlib

        @contextlib.contextmanager
        def resp():
            class R:
                def read(self):
                    return b"ok"
            yield R()
        return resp()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("JARVIS_WHATSAPP_ENABLED", "JARVIS_WHATSAPP_BRIDGE_URL",
              "JARVIS_WHATSAPP_BRIDGE_TOKEN", "JARVIS_WHATSAPP_TO",
              "JARVIS_NTFY_URL", "JARVIS_NTFY_TOPIC"):
        monkeypatch.delenv(k, raising=False)


def test_disabled_by_default():
    assert whatsapp.configured() is False
    assert whatsapp.send("hi") is False


def test_send_posts_to_bridge(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(whatsapp.urllib.request, "urlopen", rec)
    monkeypatch.setenv("JARVIS_WHATSAPP_ENABLED", "1")
    monkeypatch.setenv("JARVIS_WHATSAPP_BRIDGE_URL", "http://127.0.0.1:8090")
    monkeypatch.setenv("JARVIS_WHATSAPP_BRIDGE_TOKEN", "tok")
    assert whatsapp.send("hello") is True
    c = rec.calls[0]
    assert c["url"] == "http://127.0.0.1:8090/send"
    assert c["headers"]["Authorization"] == "Bearer tok"
    assert c["body"] == {"text": "hello", "to": ""}


def test_notify_formats_title(monkeypatch):
    seen = []
    monkeypatch.setattr(whatsapp, "send", lambda text, to="": seen.append(text) or True)
    whatsapp.notify("Job alert", "Acme — Frontend Engineer")
    assert seen == ["*Job alert*\nAcme — Frontend Engineer"]


def test_push_uses_whatsapp_when_ntfy_absent(monkeypatch):
    sent = []
    monkeypatch.setenv("JARVIS_WHATSAPP_ENABLED", "1")
    monkeypatch.setenv("JARVIS_WHATSAPP_BRIDGE_URL", "http://127.0.0.1:8090")
    monkeypatch.setenv("JARVIS_WHATSAPP_BRIDGE_TOKEN", "tok")
    monkeypatch.setattr(whatsapp, "notify",
                        lambda t, m: sent.append((t, m)) or True)
    assert push.push_configured() is True
    assert push.send("Reminder", "stand up") is True
    assert sent == [("Reminder", "stand up")]

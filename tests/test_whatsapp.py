"""WhatsApp Cloud API client (official Graph API)."""
import json

import pytest

from jarvis import push, whatsapp


class Recorder:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, req, timeout=None):
        if self.fail:
            raise OSError("graph down")
        self.calls.append({"url": req.full_url,
                           "headers": dict(req.header_items()),
                           "body": json.loads(req.data.decode())})
        import contextlib

        @contextlib.contextmanager
        def resp():
            class R:
                def read(self):
                    return b'{"messages":[{"id":"wamid.x"}]}'
            yield R()
        return resp()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("JARVIS_WHATSAPP_ENABLED", "JARVIS_WA_PHONE_ID",
              "JARVIS_WA_TOKEN", "JARVIS_WA_TO", "JARVIS_WA_TEMPLATE_NOTIFY",
              "JARVIS_NTFY_URL", "JARVIS_NTFY_TOPIC"):
        monkeypatch.delenv(k, raising=False)


def _enable(monkeypatch, template=""):
    monkeypatch.setenv("JARVIS_WHATSAPP_ENABLED", "1")
    monkeypatch.setenv("JARVIS_WA_PHONE_ID", "123456")
    monkeypatch.setenv("JARVIS_WA_TOKEN", "tok")
    monkeypatch.setenv("JARVIS_WA_TO", "919999999999")
    if template:
        monkeypatch.setenv("JARVIS_WA_TEMPLATE_NOTIFY", template)


def test_disabled_by_default():
    assert whatsapp.configured() is False
    assert whatsapp.send("hi") is False


def test_send_text_payload(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(whatsapp.urllib.request, "urlopen", rec)
    _enable(monkeypatch)
    assert whatsapp.send("hello") is True
    c = rec.calls[0]
    assert c["url"].endswith("/v23.0/123456/messages")
    assert c["headers"]["Authorization"] == "Bearer tok"
    assert c["body"] == {"messaging_product": "whatsapp",
                         "recipient_type": "individual",
                         "to": "919999999999", "type": "text",
                         "text": {"preview_url": False, "body": "hello"}}


def test_send_template_payload(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(whatsapp.urllib.request, "urlopen", rec)
    _enable(monkeypatch, template="soundwave_alert")
    assert whatsapp.send_template("Job alert", "Acme — Frontend") is True
    body = rec.calls[0]["body"]
    assert body["type"] == "template"
    assert body["template"]["name"] == "soundwave_alert"
    params = body["template"]["components"][0]["parameters"]
    assert [p["text"] for p in params] == ["Job alert", "Acme — Frontend"]


def test_notify_prefers_template_then_text(monkeypatch):
    calls = []
    monkeypatch.setattr(whatsapp, "send_template",
                        lambda t, m: calls.append(("template", t)) or True)
    monkeypatch.setattr(whatsapp, "send",
                        lambda t: calls.append(("text", t)) or True)
    whatsapp.notify("Job alert", "body")
    assert calls == [("template", "Job alert")]

    monkeypatch.setattr(whatsapp, "send_template", lambda t, m: False)
    whatsapp.notify("Reminder", "stand up")
    assert calls[-1] == ("text", "*Reminder*\nstand up")


def test_send_failure_is_false(monkeypatch):
    monkeypatch.setattr(whatsapp.urllib.request, "urlopen", Recorder(fail=True))
    _enable(monkeypatch)
    assert whatsapp.send("hi") is False


def test_push_falls_back_to_whatsapp(monkeypatch):
    sent = []
    monkeypatch.setattr(whatsapp, "configured", lambda: True)
    monkeypatch.setattr(whatsapp, "notify",
                        lambda t, m: sent.append((t, m)) or True)
    assert push.push_configured() is True
    assert push.send("Reminder", "stand up") is True
    assert sent == [("Reminder", "stand up")]

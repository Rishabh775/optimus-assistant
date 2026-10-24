"""Phone push delivery (proactive loop, phase 2).

Runs offline: urllib POSTs captured via a fake urlopen; no network.
"""
import json

import pytest

from jarvis import push, scheduler


class PostRecorder:
    """Fake urllib.request.urlopen: captures ntfy POSTs."""
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, req, timeout=None):
        if self.fail:
            raise OSError("ntfy down")
        body = json.loads(req.data.decode())
        self.calls.append({"url": req.full_url, "body": body,
                           "headers": dict(req.header_items())})
        import contextlib

        @contextlib.contextmanager
        def resp():
            class R:
                def read(self):
                    return b"ok"
            yield R()
        return resp()


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    """Isolate: tmp reminders DB, push disabled, listeners restored."""
    monkeypatch.setenv("JARVIS_REMINDER_DB", str(tmp_path / "reminders.db"))
    old = list(scheduler._LISTENERS)
    scheduler._LISTENERS.clear()
    monkeypatch.delenv("JARVIS_NTFY_URL", raising=False)
    monkeypatch.delenv("JARVIS_NTFY_TOPIC", raising=False)
    monkeypatch.delenv("JARVIS_NTFY_TOKEN", raising=False)
    yield
    scheduler._LISTENERS[:] = old


# ------------------------------------------------------------- config ----

def test_disabled_without_url_or_topic(monkeypatch):
    monkeypatch.delenv("JARVIS_NTFY_URL", raising=False)
    monkeypatch.delenv("JARVIS_NTFY_TOPIC", raising=False)
    assert push.push_configured() is False
    assert push.send("t", "m") is False            # no crash, no push
    monkeypatch.setenv("JARVIS_NTFY_URL", "https://ntfy.sh")
    assert push.push_configured() is False          # topic still missing
    monkeypatch.setenv("JARVIS_NTFY_TOPIC", "secret-ch")
    assert push.push_configured() is True


def test_send_posts_json_with_title_and_priority(monkeypatch):
    rec = PostRecorder()
    monkeypatch.setattr(push.urllib.request, "urlopen", rec)
    monkeypatch.setenv("JARVIS_NTFY_URL", "https://ntfy.sh")
    monkeypatch.setenv("JARVIS_NTFY_TOPIC", "secret-ch")
    assert push.send("Reminder", "water the plants") is True
    c = rec.calls[0]
    assert c["url"] == "https://ntfy.sh/secret-ch"
    assert c["body"] == {"topic": "secret-ch", "title": "Reminder",
                         "message": "water the plants",
                         "priority": "high", "tags": ["bell"]}
    assert "Authorization" not in c["headers"]      # no token configured


def test_send_with_token(monkeypatch):
    rec = PostRecorder()
    monkeypatch.setattr(push.urllib.request, "urlopen", rec)
    monkeypatch.setenv("JARVIS_NTFY_URL", "http://100.115.236.88:2580/")
    monkeypatch.setenv("JARVIS_NTFY_TOPIC", "ch")
    monkeypatch.setenv("JARVIS_NTFY_TOKEN", "tk_secret")
    assert push.send("t", "m") is True
    assert rec.calls[0]["headers"]["Authorization"] == "Bearer tk_secret"
    assert rec.calls[0]["url"] == "http://100.115.236.88:2580/ch"  # rstrip '/'


def test_send_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(push.urllib.request, "urlopen", PostRecorder(fail=True))
    monkeypatch.setenv("JARVIS_NTFY_URL", "https://ntfy.sh")
    monkeypatch.setenv("JARVIS_NTFY_TOPIC", "ch")
    assert push.send("t", "m") is False


# ---------------------------------------------------- scheduler wiring ---

def _fire(monkeypatch, rec, missed=False):
    monkeypatch.setattr(push.urllib.request, "urlopen", rec)
    monkeypatch.setenv("JARVIS_NTFY_URL", "https://ntfy.sh")
    monkeypatch.setenv("JARVIS_NTFY_TOPIC", "ch")
    push.install()
    r = scheduler.add("stretch legs", "in 1 minute")
    import time
    # +90s: past due (fires) but within _MISSED_GRACE -> not flagged missed
    scheduler._tick(time.time() + 3600 if missed else time.time() + 90)
    return r


def test_fire_pushes_exactly_once(monkeypatch):
    rec = PostRecorder()
    _fire(monkeypatch, rec)
    assert len(rec.calls) == 1
    assert rec.calls[0]["body"]["message"] == "stretch legs"
    assert rec.calls[0]["body"]["title"] == "Reminder"


def test_restart_catch_up_is_marked_missed(monkeypatch):
    rec = PostRecorder()
    _fire(monkeypatch, rec, missed=True)
    assert rec.calls[0]["body"]["title"] == "Missed reminder"


def test_fire_push_disabled_is_safe_noop(monkeypatch):
    """No push config -> listener no-ops (voice module not loaded: no TTS)."""
    push.install()                          # push off
    import time
    scheduler.add("quiet", "in 1 minute")
    fired = scheduler._tick(time.time() + 3600)   # must not raise
    assert len(fired) == 1


def test_install_idempotent(monkeypatch):
    rec = PostRecorder()
    monkeypatch.setattr(push.urllib.request, "urlopen", rec)
    monkeypatch.setenv("JARVIS_NTFY_URL", "https://ntfy.sh")
    monkeypatch.setenv("JARVIS_NTFY_TOPIC", "ch")
    push.install()
    push.install()                          # second call must not double-fires
    import time
    scheduler.add("once only", "in 1 minute")
    scheduler._tick(time.time() + 3600)
    assert len(rec.calls) == 1


def test_missed_flag_changes_push_title(monkeypatch):
    rec = PostRecorder()
    monkeypatch.setattr(push.urllib.request, "urlopen", rec)
    monkeypatch.setenv("JARVIS_NTFY_URL", "https://ntfy.sh")
    monkeypatch.setenv("JARVIS_NTFY_TOPIC", "ch")
    push._deliver({"message": "old one", "missed": True})
    assert rec.calls[0]["body"]["title"] == "Missed reminder"

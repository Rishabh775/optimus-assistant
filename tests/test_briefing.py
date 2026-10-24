"""Weather + daily briefing (proactive loop, phase 3).

Runs offline: open-meteo responses are faked at urlopen; calendar/email
sections stubbed at the tools layer; the daily job clock is injected
through scheduler._tick(now).
"""
import time
from datetime import datetime, timedelta, timezone

import pytest

from jarvis import briefing, config, scheduler


class GeoRecorder:
    """Fake urlopen that answers geocode + forecast from canned JSON."""
    def __init__(self, geo=None, forecast=None, fail=False):
        self.geo, self.forecast, self.fail = geo, forecast, fail
        self.urls = []

    def __call__(self, url, timeout=None):
        if self.fail:
            raise OSError("net down")
        self.urls.append(url)
        import io
        import json as _json
        if "geocoding" in url:
            body = {"results": self.geo} if self.geo is not None else {}
        else:
            body = self.forecast or {}
        return io.BytesIO(_json.dumps(body).encode())

FORECAST = {
    "current": {"temperature_2m": 21.4, "weather_code": 61,
                "wind_speed_10m": 12.0},
    "daily": {"time": ["2026-10-24", "2026-10-25"],
              "weather_code": [61, 3],
              "temperature_2m_max": [24.0, 20.0],
              "temperature_2m_min": [18.0, 12.0]},
}

GEO = [{"name": "Hyderabad", "country": "India",
        "latitude": 17.38, "longitude": 78.48}]


# -------------------------------------------------------------- weather --

@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_REMINDER_DB", str(tmp_path / "r.db"))
    old = list(scheduler._LISTENERS)
    scheduler._LISTENERS.clear()
    for k in ("JARVIS_LAT", "JARVIS_LON", "JARVIS_WEATHER_CITY",
              "JARVIS_BRIEFING_TIME", "JARVIS_NTFY_URL",
              "JARVIS_NTFY_TOPIC"):
        monkeypatch.delenv(k, raising=False)
    yield
    scheduler._LISTENERS[:] = old
    with scheduler._DAILY_LOCK:
        scheduler._DAILY.pop("briefing", None)


def test_weather_by_city(monkeypatch):
    rec = GeoRecorder(geo=GEO, forecast=FORECAST)
    monkeypatch.setattr(briefing.urllib.request, "urlopen", rec)
    out = briefing.weather("hyderabad")
    assert "Hyderabad" in out and "21C" in out and "rain" in out
    assert "Today 10-24: 18-24C" in out
    assert "Tomorrow 10-25: 12-20C" in out


def test_weather_by_home_coords(monkeypatch):
    monkeypatch.setenv("JARVIS_LAT", "17.4")
    monkeypatch.setenv("JARVIS_LON", "78.5")
    rec = GeoRecorder(forecast=FORECAST)
    monkeypatch.setattr(briefing.urllib.request, "urlopen", rec)
    out = briefing.weather()
    assert "your location" in out and "21C" in out
    assert rec.urls and "geocoding" not in rec.urls[0]


def test_weather_unconfigured_is_helpful_error(monkeypatch):
    out = briefing.weather()
    assert "no location" in out


def test_weather_network_failure_is_soft(monkeypatch):
    monkeypatch.setenv("JARVIS_LAT", "17.4")
    monkeypatch.setenv("JARVIS_LON", "78.5")
    monkeypatch.setattr(briefing.urllib.request, "urlopen",
                        GeoRecorder(fail=True))
    assert briefing.weather().startswith("ERROR:")


def test_wmo_known_and_unknown():
    assert briefing._wmo_desc(0) == "clear sky"
    assert briefing._wmo_desc(95) == "thunderstorm"
    assert "unusual" in briefing._wmo_desc(999)


# ------------------------------------------------------------- briefing --

def _stub_sections(monkeypatch):
    import jarvis.tools.calendar as cal
    import jarvis.tools.email as em
    monkeypatch.setattr(cal, "calendar_today",
                        lambda: "14:00-15:00 standup (Zoom)")
    monkeypatch.setattr(em, "email_triage",
                        lambda n=15, unread_only=False:
                        "HIGH: invoice due tomorrow (score 60)")
    return None


def test_build_happy_path(monkeypatch):
    monkeypatch.setenv("JARVIS_LAT", "17.4")
    monkeypatch.setenv("JARVIS_LON", "78.5")
    _stub_sections(monkeypatch)
    monkeypatch.setattr(briefing, "weather",
                        lambda city="": "Weather: 21C, rain")
    monkeypatch.setattr(briefing.scheduler, "list_pending",
                        lambda *a, **k: [
                            {"id": "x", "message": "call the bank",
                             "due_at": time.time() + 1800, "repeat": ""}])
    out = briefing.build()
    assert "Good" in out and "Weather: 21C" in out
    assert "standup" in out and "invoice due" in out
    assert "call the bank" in out


def test_build_silently_skips_dead_sections(monkeypatch):
    _stub_sections(monkeypatch)   # then break them: raise inside imports
    import jarvis.tools.calendar as cal
    import jarvis.tools.email as em
    monkeypatch.setattr(cal, "calendar_today",
                        lambda: "ERROR: connection reset")
    monkeypatch.setattr(em, "email_triage",
                        lambda n=15, unread_only=False: "imap not configured")
    out = briefing.build()
    assert "Calendar" not in out and "Email" not in out
    # weather degrades the same way (unconfigured -> one-liner absent)
    assert "Weather" not in out


def test_build_never_empty(monkeypatch):
    out = briefing.build()      # nothing configured
    assert "nothing scheduled" in out or "Good" in out


def test_run_scheduled_pushes_and_speaks(monkeypatch):
    pushes = []
    monkeypatch.setattr(briefing.push, "send",
                        lambda t, m: pushes.append((t, m)) or False)
    monkeypatch.setattr(briefing.push, "speak_local", lambda t: None)
    monkeypatch.setattr(briefing, "build", lambda: "brief text")
    briefing.run_scheduled()
    assert pushes == [("Morning briefing", "brief text")]


# ------------------------------------------------- daily job scheduling --

def test_install_registers_when_configured(monkeypatch):
    monkeypatch.setenv("JARVIS_BRIEFING_TIME", "07:30")
    briefing.install()
    assert "briefing" in scheduler._DAILY
    assert scheduler._DAILY["briefing"][:2] == (7, 30)


def test_install_noop_when_unset(monkeypatch):
    briefing.install()
    assert "briefing" not in scheduler._DAILY


def test_daily_job_fires_once_per_day(monkeypatch):
    import datetime as dt
    tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    fixed = dt.datetime(2026, 10, 24, 8, 0, tzinfo=tz)
    monkeypatch.setattr(scheduler, "_local_now", lambda: fixed)
    monkeypatch.setattr(scheduler, "_local_tz", lambda: tz)
    monkeypatch.setenv("JARVIS_BRIEFING_TIME", "07:30")
    briefing.install()
    runs = []
    monkeypatch.setattr(briefing, "run_scheduled", lambda: runs.append(1))
    scheduler._tick(fixed.timestamp())        # 08:00 local -> after 07:30
    assert len(runs) == 1
    scheduler._tick((fixed + dt.timedelta(minutes=5)).timestamp())
    assert len(runs) == 1                     # same day -> no re-run
    scheduler._tick((fixed + dt.timedelta(days=1)).timestamp())
    assert len(runs) == 2                     # next day -> fires again


def test_daily_job_not_before_target(monkeypatch):
    import datetime as dt
    tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    fixed = dt.datetime(2026, 10, 24, 6, 0, tzinfo=tz)
    monkeypatch.setattr(scheduler, "_local_now", lambda: fixed)
    monkeypatch.setattr(scheduler, "_local_tz", lambda: tz)
    monkeypatch.setenv("JARVIS_BRIEFING_TIME", "07:30")
    briefing.install()
    runs = []
    monkeypatch.setattr(briefing, "run_scheduled", lambda: runs.append(1))
    scheduler._tick(fixed.timestamp())        # 06:00 -> before 07:30
    assert runs == []


def test_daily_tool_wiring():
    from jarvis.builtin_tools import TOOL_FUNCS, TOOL_SCHEMAS
    names = {s["name"] for s in TOOL_SCHEMAS}
    for n in ("weather", "daily_briefing"):
        assert n in names and n in TOOL_FUNCS

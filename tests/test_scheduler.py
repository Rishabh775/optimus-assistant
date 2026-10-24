"""Reminders + scheduler (proactive loop, phase 1).

Runs offline: SQLite in tmp_path via JARVIS_REMINDER_DB, wall clock injected
into _tick(), listeners captured and restored.
"""
import time
from datetime import datetime, timedelta, timezone

import pytest

from jarvis import scheduler


@pytest.fixture(autouse=True)
def _sd(tmp_path, monkeypatch):
    """Isolate scheduler state: tmp DB, no inherited listeners."""
    monkeypatch.setenv("JARVIS_REMINDER_DB", str(tmp_path / "reminders.db"))
    old = list(scheduler._LISTENERS)
    scheduler._LISTENERS.clear()
    yield
    scheduler._LISTENERS[:] = old


@pytest.fixture
def fired():
    """Listener capture: fired() returns the list of delivered dicts."""
    out = []
    scheduler.add_listener(lambda r: out.append(r))
    return out


# ------------------------------------------------------------- parsing ---

def test_parse_relative_minutes():
    t = scheduler._parse_when("in 90 minutes")
    assert abs(t - (time.time() + 90 * 60)) < 2


def test_parse_relative_combo_units():
    t = scheduler._parse_when("in 1h 30m")
    assert abs(t - (time.time() + 5400)) < 2


def test_parse_iso_with_offset_is_exact():
    t = scheduler._parse_when("2027-01-01T10:00:00+05:30")
    want = datetime(2027, 1, 1, 10, 0,
                    tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert t == want.timestamp()


def test_parse_hhmm_tomorrow_when_past(monkeypatch):
    # 10:00 said at 12:00 local -> tomorrow 10:00
    import datetime as dt
    tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    fixed = dt.datetime(2026, 10, 24, 12, 0, 0, tzinfo=tz)
    monkeypatch.setattr(scheduler, "_local_now", lambda: fixed)
    monkeypatch.setattr(scheduler.time, "time", lambda: fixed.timestamp())
    t = scheduler._parse_when("10:00")
    assert t == (fixed + dt.timedelta(hours=22)).timestamp()


def test_parse_hhmm_today_when_future(monkeypatch):
    import datetime as dt
    tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    fixed = dt.datetime(2026, 10, 24, 12, 0, 0, tzinfo=tz)
    monkeypatch.setattr(scheduler, "_local_now", lambda: fixed)
    monkeypatch.setattr(scheduler.time, "time", lambda: fixed.timestamp())
    t = scheduler._parse_when("15:30")
    want = dt.datetime(2026, 10, 24, 15, 30, 0, tzinfo=tz)
    assert t == want.timestamp()


def test_parse_rejects_garbage_and_past_and_empty():
    with pytest.raises(ValueError):
        scheduler._parse_when("sometime next week")
    with pytest.raises(ValueError):
        scheduler._parse_when("2026-01-01T00:00")   # past
    with pytest.raises(ValueError):
        scheduler._parse_when("")


# ---------------------------------------------------------------- CRUD ---

def test_add_lists_and_rejects_past():
    r = scheduler.add("check the oven", "in 5 minutes")
    assert r["status"] == "pending"
    assert len(scheduler.list_pending()) == 1
    with pytest.raises(ValueError):
        scheduler.add("x", "in 0 minutes")      # <= now
    with pytest.raises(ValueError):
        scheduler.add("", "in 5 minutes")
    with pytest.raises(ValueError):
        scheduler.add("x", "in 5 minutes", repeat="hourly")   # not supported


def test_cancel_lifecycle():
    r = scheduler.add("water the plants", "in 1 hour")
    assert scheduler.cancel(r["id"]) is True
    assert scheduler.list_pending() == []
    assert scheduler.cancel(r["id"]) is False    # already settled
    assert scheduler.cancel("nope") is False


# ------------------------------------------------------------- firing ----

def test_tick_fires_one_shot_exactly_once(fired):
    scheduler.add("take tea out", "in 1 minute")
    now = time.time() + 3600           # sweep an hour later
    assert len(scheduler._tick(now)) == 1
    assert fired[0]["message"] == "take tea out"
    assert fired[0]["missed"] is True   # >_MISSED_GRACE late
    assert scheduler.list_pending() == []
    assert scheduler._tick(now) == []   # claimed -> never re-fires


def test_tick_repeating_row_stays_pending(fired):
    r = scheduler.add("stand up", "in 10 minutes", repeat="daily")
    now = time.time() + 3600
    scheduler._tick(now)
    pending = scheduler.list_pending()
    assert len(pending) == 1 and pending[0]["id"] == r["id"]
    assert pending[0]["due_at"] > now + scheduler._UNIT_SECONDS["d"] / 2


def test_tick_skips_early(fired):
    scheduler.add("later", "in 2 hours")
    assert scheduler._tick(time.time() + 60) == []
    assert fired == []


# ------------------------------------------------------------ listener ---

def test_bad_listener_does_not_break_fire(fired):
    def boom(_):
        raise RuntimeError("boom")
    scheduler.add_listener(boom)
    scheduler.add("sweep", "in 1 minute")
    now = time.time() + 3600
    assert len(scheduler._tick(now)) == 1
    assert len(fired) == 1              # good listener still runs


def test_duplicate_listener_registered_once():
    out = []
    fn = lambda r: out.append(r)
    scheduler.add_listener(fn)
    scheduler.add_listener(fn)      # idempotent
    scheduler.add("one", "in 1 minute")
    scheduler._tick(time.time() + 3600)
    assert len(out) == 1


# ------------------------------------------- tools surface (end to end) --

def test_tool_wiring():
    from jarvis.builtin_tools import TOOL_FUNCS, TOOL_SCHEMAS
    names = {s["name"] for s in TOOL_SCHEMAS}
    for n in ("remind_me", "list_reminders", "cancel_reminder"):
        assert n in names and n in TOOL_FUNCS


def test_remind_tools_end_to_end(monkeypatch):
    from jarvis import builtin_tools as bt
    out = bt.remind_me("call mom", "in 30 minutes")
    assert out.startswith("Reminder #")
    out = bt.remind_me("", "in 1 hour")
    assert out.startswith("error:")
    out = bt.remind_me("bad", "in 1 hour", repeat="sobriety")
    assert out.startswith("error:")
    listing = bt.list_reminders()
    assert "call mom" in listing
    rid = listing.split("#")[1].split()[0]
    assert bt.cancel_reminder(rid) == "cancelled"
    assert bt.cancel_reminder(rid).startswith("error:")
    assert bt.list_reminders() == "no pending reminders"


def test_reminders_not_confirm_gated():
    from jarvis.builtin_tools import needs_confirmation
    assert needs_confirmation("remind_me", {}) is None
    assert needs_confirmation("cancel_reminder", {}) is None

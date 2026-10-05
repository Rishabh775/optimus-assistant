"""Calendar via CalDAV (works with iCloud, Nextcloud, Fastmail, Yahoo, etc.)
For Google Calendar, use OAuth + google-api-python-client (not yet implemented here).
"""
import os
from datetime import datetime, timedelta


def _client():
    url = os.getenv("JARVIS_CALDAV_URL")
    user = os.getenv("JARVIS_CALDAV_USER")
    pw = os.getenv("JARVIS_CALDAV_PASS")
    if not all([url, user, pw]):
        return None, "caldav not configured (set JARVIS_CALDAV_URL/USER/PASS)"
    try:
        import caldav
        return caldav.DAVClient(url=url, username=user, password=pw), None
    except Exception as e:
        return None, f"caldav error: {e}"


def _cal(client):
    cal_name = os.getenv("JARVIS_CALDAV_CALENDAR")
    principal = client.principal()
    if cal_name:
        return principal.calendar(cal_name=cal_name)
    return principal.calendars()[0]


def calendar_list(days: int = 7) -> str:
    c, err = _client()
    if err:
        return err
    try:
        cal = _cal(c)
        start = datetime.now()
        end = start + timedelta(days=days)
        events = cal.date_search(start=start, end=end, expand=True)
        out = []
        for e in events:
            v = e.vobject_instance.vevent
            summary = str(v.summary.value) if hasattr(v, "summary") else "(no title)"
            dt = v.dtstart.value
            dtend = v.dtend.value if hasattr(v, "dtend") else None
            loc = str(v.location.value) if hasattr(v, "location") else ""
            out.append(f"- {summary}\n  When: {dt} - {dtend}\n  Where: {loc}")
        return "\n\n".join(out) or "(no events in range)"
    except Exception as e:
        return f"ERROR: {e}"


def calendar_add(title: str, start: str, duration_minutes: int = 60,
                 description: str = "", location: str = "") -> str:
    """start in ISO format e.g. 2026-10-03T15:00"""
    c, err = _client()
    if err:
        return err
    try:
        from icalendar import Calendar, Event
        cal = _cal(c)
        start_dt = datetime.fromisoformat(start)
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        event = Event()
        event.add("summary", title)
        event.add("dtstart", start_dt)
        event.add("dtend", end_dt)
        if description:
            event.add("description", description)
        if location:
            event.add("location", location)
        cal.save_event(event)
        return f"ADDED: {title} @ {start_dt}"
    except Exception as e:
        return f"ERROR: {e}"


def calendar_today() -> str:
    return calendar_list(days=1)
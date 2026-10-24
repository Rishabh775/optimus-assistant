"""Daily briefing + weather (proactive loop p3).

Composes the morning summary from what is actually configured and working:
calendar (CalDAV), email triage (IMAP), today's reminders, and weather
(Open-Meteo, no API key). Every section degrades to a one-line note instead
of failing the briefing -- the send must never die over one integration.

Delivery = push (ntfy) + speaker if a voice mode is running, reusing the
phase-2 helpers. "brief me" / "what's my day" also works on demand: the
agent calls the daily_briefing tool, which returns the same text.
"""
import json
import urllib.request
from datetime import datetime

from . import config, push, scheduler
from .log import log as _mk_log

log = _mk_log("briefing")

_WMO = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def _wmo_desc(code) -> str:
    try:
        return _WMO.get(int(code), f"unusual weather ({code})")
    except (TypeError, ValueError):
        return "unusual weather"


# ----------------------------------------------------------- weather ----

def _get_json(url: str, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _geocode(city: str) -> dict:
    data = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search?name="
        f"{urllib.request.quote(city)}&count=1&language=en&format=json")
    hits = data.get("results") or []
    if not hits:
        raise ValueError(f"unknown place {city!r}")
    return hits[0]


def weather(city: str = "") -> str:
    """One-line-ish current+today+tomorrow forecast for a place.

    Location resolution: explicit `city` arg > JARVIS_LAT/JARVIS_LON >
    JARVIS_WEATHER_CITY. Returns a readable error string when unconfigured
    (same style as the email/calendar tools)."""
    lat, lon = config.get("JARVIS_LAT"), config.get("JARVIS_LON")
    name = "your location"
    if str(city or "").strip():
        try:
            hit = _geocode(city.strip())
        except Exception as e:
            return f"ERROR: {e}"
        lat, lon, name = hit["latitude"], hit["longitude"], \
            f"{hit.get('name', city)}{', ' + hit.get('country') if hit.get('country') else ''}"
    elif lat and lon:
        lat, lon = lat.strip(), lon.strip()
    elif config.get("JARVIS_WEATHER_CITY"):
        return weather(config.get("JARVIS_WEATHER_CITY"))
    else:
        return ("ERROR: no location -- pass a city, or set JARVIS_LAT/"
                "JARVIS_LON; JARVIS_WEATHER_CITY also works")
    try:
        d = _get_json(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,weather_code,wind_speed_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min"
            "&forecast_days=2&timezone=auto")
    except Exception as e:
        return f"ERROR: weather lookup failed ({e})"
    cur = d["current"]
    day = d["daily"]
    now_desc = _wmo_desc(cur["weather_code"])
    out = (f"Weather {name}: {cur['temperature_2m']:.0f}C, {now_desc},"
           f" wind {cur['wind_speed_10m']:.0f} km/h")
    for i, dt in enumerate(day.get("time", [])[:2]):
        out += (f"\n{'Today' if i == 0 else 'Tomorrow'} {dt[5:]}: "
                f"{day['temperature_2m_min'][i]:.0f}-{day['temperature_2m_max'][i]:.0f}C,"
                f" {_wmo_desc(day['weather_code'][i])}")
    return out


# -------------------------------------------------------- the briefing --

def _greeting() -> str:
    h = scheduler._local_now().hour
    if h < 12:
        return "Good morning"
    if h < 17:
        return "Good afternoon"
    return "Good evening"


def _clean_section(text: str, err_markers=("error", "caldav", "imap", "not configured"),
                   max_lines: int = 5) -> str:
    """Trim a tool's output; collapse failure to a short note."""
    low = (text or "").lower()
    if any(m in low[:60] for m in err_markers):
        return None            # integration unavailable -> skip silently
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


def build() -> str:
    """Compose the briefing text from what is configured and working."""
    name = config.get("JARVIS_NAME") or "Optimus"
    lines = [_greeting(), ""]
    # weather first (sets the day), then calendar, email, reminders
    if config.get("JARVIS_LAT") or config.get("JARVIS_WEATHER_CITY"):
        lines.append(_clean_section(weather()) or "")
    try:
        from .tools.calendar import calendar_today
        cal = _clean_section(calendar_today(), max_lines=6)
        if cal:
            lines += ["--- Calendar", cal, ""]
    except Exception as e:
        log.debug("calendar section skipped: %s", e)
    try:
        from .tools.email import email_triage
        mail = _clean_section(email_triage(n=8, unread_only=True), max_lines=4)
        if mail:
            lines += ["--- Email (top unread)", mail, ""]
    except Exception as e:
        log.debug("email section skipped: %s", e)
    try:
        today = scheduler._local_now().strftime("%Y-%m-%d")
        due_today = [r for r in scheduler.list_pending()
                     if datetime.fromtimestamp(r["due_at"],
                                               tz=scheduler._local_tz())
                     .strftime("%Y-%m-%d") == today]
        if due_today:
            rl = "\n".join(f"- {r['message']} ("
                           f"{datetime.fromtimestamp(r['due_at'], tz=scheduler._local_tz()).strftime('%H:%M')})"
                           for r in due_today[:6])
            lines += ["--- Reminders today", rl, ""]
    except Exception as e:
        log.debug("reminders section skipped: %s", e)
    out = "\n".join(ln for ln in lines if ln is not None).strip()
    return out or f"{_greeting()}. All quiet today -- nothing scheduled."


# ----------------------------------------------------- scheduled trigger --

def run_scheduled() -> None:
    """Daily-job entrypoint: compose + push + (voice modes) speak."""
    text = build()
    push.send("Morning briefing", text)
    push.speak_local(text)
    log.info("briefing delivered (%d chars)", len(text))


def install() -> None:
    """Register the daily briefing job when JARVIS_BRIEFING_TIME is set."""
    at = config.get("JARVIS_BRIEFING_TIME")
    if at:
        # late binding so tests (and reloads) can swap run_scheduled
        scheduler.add_daily("briefing", at, lambda: run_scheduled())
        log.info("briefing scheduled daily at %s", at)

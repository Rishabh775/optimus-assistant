"""Central configuration: one source of defaults, typed accessors, validation.

Env var NAMES are unchanged, so an existing .env keeps working. Modules should
read tunables through here rather than scattering `os.getenv(..., "default")`
literals, so defaults live in one place and can be validated at startup.

Accessors read the environment live (env changes at runtime are honoured).
"""
import os

# ---- single source of documented defaults ----
DEFAULTS = {
    # routing / models
    "JARVIS_NAME": "Optimus",
    "JARVIS_WHISPER_MODEL": "small.en",
    "JARVIS_CLOUD_MODEL": "glm-5.3-flash",
    "JARVIS_CLOUD_FALLBACK": "deepseek-v4-flash",
    "JARVIS_CODE_MODEL": "kimi-k2.7-code",
    "JARVIS_LOCAL_MODEL": "llama3.2:3b",
    "JARVIS_LOCAL_OK": "0",
    "JARVIS_LLM_TIMEOUT": "60",
    "JARVIS_LLM_RETRIES": "2",
    "JARVIS_CLOUD_COOLDOWN": "90",
    "JARVIS_MAX_STEPS": "12",
    # wake word
    "JARVIS_WAKE_WORDS": "hey_soundwave",
    "JARVIS_WAKE_THRESHOLD": "0.7",
    "JARVIS_WAKE_DEBOUNCE": "1",
    "JARVIS_WAKE_CHUNK_MS": "1200",
    "JARVIS_WAKE_ENERGY_MIN": "1000",
    # voice / STT
    "JARVIS_MIC_GAIN": "1.0",
    "JARVIS_STT_COMPUTE": "int8",
    "JARVIS_STT_THREADS": "4",
    "JARVIS_STT_LANGUAGE": "en",
    "JARVIS_STT_BEAM_SIZE": "5",
    "JARVIS_STT_INITIAL_PROMPT": (
        "Voice commands to a local assistant about files, folders, HTML, CSS, "
        "JavaScript, DOM manipulation, workspace, email, calendar."),
    "JARVIS_STT_HOTWORDS": "DOM manipulation HTML CSS JavaScript workspace",
    "JARVIS_STT_END_SILENCE": "1800",
    "JARVIS_STT_ENERGY_MULT": "1.0",
    "JARVIS_STT_GATE_MAX": "15000",
    "JARVIS_STT_STOP_RATIO": "0.35",
    "JARVIS_STT_STOP_FLOOR": "120",
    "JARVIS_STT_MIN_SPEECH": "0.5",
    "JARVIS_STT_TAIL": "0.35",
    "JARVIS_STT_MAX_NO_SPEECH": "0.6",
    "JARVIS_STT_MIN_LOGPROB": "-1.0",
    # tts
    "JARVIS_TTS_VOICE": "en-US-AndrewMultilingualNeural",
    "JARVIS_TTS_RATE": "+0%",
    "JARVIS_TTS_EFFECT": "off",
    "JARVIS_TTS_FALLBACK": "piper",
    "JARVIS_PIPER_MODEL": "",
    # memory / facts
    "JARVIS_MEMORY_MAX_DIST": "0.55",
    "JARVIS_MEMORY_MAX_ENTRIES": "500",
    "JARVIS_FACTS_MAX": "100",
    # ux
    "JARVIS_FOLLOWUP": "1",
    "JARVIS_FOLLOWUP_SECONDS": "6",
    "JARVIS_GREET": "1",
    "JARVIS_SOUNDS": "1",
    "JARVIS_BARGE_IN": "1",
    "JARVIS_CONFIRM": "1",
    # JARVIS_AUTO=1: run everything without asking (skips ALL confirmations,
    # including mandatory ones). Hard tool blocks (blocklist, secret-file
    # reads) still apply. Use only on a trusted setup.
    "JARVIS_AUTO": "0",
    # phone-app server (python -m jarvis.server)
    "JARVIS_APP_HOST": "0.0.0.0",
    "JARVIS_APP_PORT": "8080",
    "JARVIS_APP_CONFIRM_TIMEOUT": "120",
    # SQLite chat store (relative to the repo root); JARVIS_APP_DB="" keeps
    # history in RAM only (wallet, sessions die with the process).
    "JARVIS_APP_DB": "data/phone-app/chats.db",
    # uploads from the phone (photos/documents/voice notes) land here
    "JARVIS_APP_UPLOAD_DIR": "data/phone-app/uploads",
    "JARVIS_APP_UPLOAD_MAX_MB": "25",
    # JARVIS_APP_TOKEN is generated into .env on first run if unset.
    # persona of the assistant: soundwave | jarvis | glados | off
    # (or free text: any other value is used verbatim as a personality hint)
    "JARVIS_PERSONALITY": "off",
    # proactive loop (scheduler.py): durable reminders
    # SQLite store (relative to repo root). Startup catch-up fires anything
    # past-due as "(missed)".
    "JARVIS_REMINDER_DB": "data/phone-app/reminders.db",
    # seconds between scheduler sweeps (min 2)
    "JARVIS_SCHED_TICK": "20",
    # IANA timezone for reminder/briefing times; empty = system local
    "JARVIS_TZ": "",
    # proactive p2: phone push for fired reminders (push.py). URL+topic both
    # required: "https://ntfy.sh" to start, self-hosted later via
    # http://<tailnet-ip>:2580. Token optional (auth-forcing ntfy servers).
    "JARVIS_NTFY_URL": "",
    "JARVIS_NTFY_TOPIC": "",
    "JARVIS_NTFY_TOKEN": "",
    # proactive p3: daily briefing. Empty = off; "07:30" delivers a composed
    # summary (weather/calendar/email/reminders) at that local time.
    "JARVIS_BRIEFING_TIME": "",
    # location for the weather section/tool
    "JARVIS_LAT": "",
    "JARVIS_LON": "",
    "JARVIS_WEATHER_CITY": "",
    # jobs skill: discovery/matching/tracking (application engine comes later)
    "JARVIS_JOBS_ENABLED": "0",
    "JARVIS_JOBS_DB": "data/jobs/jobs.db",
    "JARVIS_JOBS_PROFILE": "data/jobs/profile.json",
    "JARVIS_JOBS_SWEEP_TIME": "08:30",
    "JARVIS_JOBS_DIGEST_SIZE": "10",
    "JARVIS_JOBS_MAX_AGE_DAYS": "21",
    # don't show weak matches in the digest (they stay in the tracker)
    "JARVIS_JOBS_MIN_SCORE": "25",
    # remote scope: "any" = show every remote role (geo-restricted ones are
    # labeled), "eligible" = hide roles restricted to other regions
    "JARVIS_JOBS_REMOTE_SCOPE": "any",
    # fast alerts: poll every N minutes and push new roles immediately
    "JARVIS_JOBS_ALERTS": "1",
    "JARVIS_JOBS_ALERT_INTERVAL_MIN": "20",
    "JARVIS_JOBS_ALERT_MIN_SCORE": "45",
    "JARVIS_JOBS_ALERT_MAX_PER_RUN": "4",
    # hide remote roles whose posting never states an eligible region
    "JARVIS_JOBS_HIDE_UNCLEAR_REMOTE": "0",
    "JARVIS_JOBS_COUNTRY": "in",
    # public, no-auth ATS boards: "provider:company" comma-separated
    "JARVIS_JOBS_BOARDS": (
        "greenhouse:stripe,greenhouse:vercel,greenhouse:figma,"
        "greenhouse:gitlab,greenhouse:reddit,greenhouse:pinterest,"
        "greenhouse:dropbox,greenhouse:duolingo,greenhouse:brex,"
        "greenhouse:canonical,greenhouse:elastic,greenhouse:mongodb,"
        "ashby:linear,ashby:supabase,ashby:posthog,ashby:resend,"
        "lever:cred"),
    # optional Adzuna search API (free key; broad search incl. India)
    "JARVIS_ADZUNA_APP_ID": "",
    "JARVIS_ADZUNA_APP_KEY": "",
    # free tier is 250 calls/day -> don't poll it more often than this
    "JARVIS_ADZUNA_MIN_INTERVAL_MIN": "60",
    # aggregator feeds have no ETag -> refresh them hourly, ATS boards often
    "JARVIS_JOBS_FEEDS_INTERVAL_MIN": "60",
}


def get(name: str) -> str:
    return os.getenv(name, DEFAULTS.get(name, ""))


def get_int(name: str) -> int:
    try:
        return int(float(get(name)))
    except (TypeError, ValueError):
        return int(float(DEFAULTS.get(name, 0)))


def get_float(name: str) -> float:
    try:
        return float(get(name))
    except (TypeError, ValueError):
        return float(DEFAULTS.get(name, 0.0))


def get_bool(name: str) -> bool:
    return str(get(name)).strip().lower() in ("1", "true", "yes", "on")


# Critical numeric settings and their sane ranges. validate() is called at
# startup so a typo in .env is reported instead of silently using a default.
_RANGES = {
    "JARVIS_WAKE_THRESHOLD": (0.0, 1.0),
    "JARVIS_WAKE_DEBOUNCE": (1, 10),
    "JARVIS_WAKE_ENERGY_MIN": (0, 32767),
    "JARVIS_STT_END_SILENCE": (200, 8000),
    "JARVIS_STT_ENERGY_MULT": (0.1, 10.0),
    "JARVIS_STT_GATE_MAX": (400, 32767),
    "JARVIS_STT_MAX_NO_SPEECH": (0.0, 1.0),
    "JARVIS_LLM_TIMEOUT": (5, 600),
}


def validate() -> list:
    """Return a list of human-readable problems (empty = all good)."""
    problems = []
    for name, (lo, hi) in _RANGES.items():
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            v = float(raw)
        except ValueError:
            problems.append(f"{name}={raw!r} is not numeric")
            continue
        if not (lo <= v <= hi):
            problems.append(f"{name}={raw} out of range [{lo}, {hi}]")
    # threshold/blink: a value that can never fire
    if get_float("JARVIS_WAKE_THRESHOLD") <= 0:
        problems.append("JARVIS_WAKE_THRESHOLD <= 0 will never fire")
    return problems

"""Tools Jarvis can call.

Security model:
- shell_run: runs in WORKSPACE with a SCRUBBED environment (API keys,
  tokens, passwords removed) and a destructive-command blocklist
- read_file: restricted to $HOME + /tmp; secrets (.env, keys, tokens) blocked
- write_file: restricted to WORKSPACE by default (JARVIS_ALLOW_ANY_PATH=1
  lifts to $HOME), never outside $HOME

The blocklist is not a real sandbox. If the agent handles untrusted content
(emails, web pages), a prompt injection can still reach shell_run; for
defense in depth run the assistant as a dedicated user or wrap shell_run in
firejail/bubblewrap.
"""
import os
import re
import subprocess
from pathlib import Path

WORKSPACE = Path.home() / "jarvis" / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)

ALLOW_ANY_PATH = os.getenv("JARVIS_ALLOW_ANY_PATH", "0") == "1"

# Deceptive/destructive shell patterns (checked case-insensitively)
_DANGER_PATTERNS = [
    r"\bmkfs\b", r"\bfdisk\b", r"\bparted\b", r"\bwipefs\b",
    r"\bdd\b[^;|]*of=/dev/",                               # dd writing devices
    r":\s*\(\)\s*\{",                                      # fork bomb
    r">\s*/dev/sd", r">\s*/dev/nvme", r"/dev/mem\b",
    r"\bchmod\s+-R\s+[^ ]+\s+/\b", r"\bchown\s+-R\s+[^ ]+\s+/\b",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b", r"\bpoweroff\b",
    r"\binit\s+[06]\b",
    r"(curl|wget)[^;|]*\|\s*(sudo\s+)?(ba)?sh\b",         # remote script exec
    r"\|\s*(sudo\s+)?(ba)?sh\b",                           # any pipe to sh/bash
    r"\bmodprobe\b", r"\bnsenter\b", r"\bunshare\b",
]


def _rm_targets_blocked(cmd: str) -> bool:
    """rm with -r/-f is only allowed on RELATIVE paths (land in WORKSPACE)."""
    for m in re.finditer(r"(?:^|[;&|]\s*|\bsudo\s+)rm\s+([^;&|]+)", cmd):
        flags, targets = m.group(1).split(maxsplit=1) if " " in m.group(1) \
            else (m.group(1), "")
        if re.search(r"[rf]", flags) or "--force" in m.group(1):
            if re.search(r"(^|\s)(/|~|\$HOME|\.\.)", targets or ""):
                return True
    return False

# Secret-ish files the agent must never read (blocks key exfiltration)
_SECRET_FILE = re.compile(
    r"(\.env($|[./])|id_rsa|id_ed25519|\.pem$|\.key$|credentials|token|"
    r"\.aws/|\.ssh/|\.netrc|\.git-credentials|secret)", re.I)

# Environment keys scrubbed from child processes (never leak to shell output)
_SENSITIVE_ENV = re.compile(r"(PASS|TOKEN|SECRET|KEY|CREDENTIAL)", re.I)


def _scrubbed_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if not _SENSITIVE_ENV.search(k)}


def _is_dangerous(cmd: str) -> str | None:
    for pat in _DANGER_PATTERNS:
        if re.search(pat, cmd, re.I):
            return f"BLOCKED: matches dangerous pattern '{pat}'"
    if _rm_targets_blocked(cmd):
        return "BLOCKED: rm with -r/-f outside WORKSPACE"
    return None


def shell_run(cmd: str, timeout: int = 30) -> str:
    """Run a shell command in WORKSPACE with a scrubbed environment."""
    blocked = _is_dangerous(cmd)
    if blocked:
        return blocked
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=str(WORKSPACE), env=_scrubbed_env(),
        )
        return (out.stdout + out.stderr).strip()[:8000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def _check_read(path: Path) -> str | None:
    allowed_roots = [Path.home(), Path("/tmp")]
    if not any(str(path).startswith(str(r) + "/") or path == r
               for r in allowed_roots):
        return f"BLOCKED: reads are limited to {allowed_roots}"
    if _SECRET_FILE.search(str(path)) and not ALLOW_ANY_PATH:
        return "BLOCKED: reading secret files is not allowed"
    return None


def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = WORKSPACE / p           # relative paths land in WORKSPACE
    if not p.exists():
        return f"NOT FOUND: {p}"
    blocked = _check_read(p)
    if blocked:
        return blocked
    try:
        return p.read_text(errors="replace")[:16000]
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = WORKSPACE / p          # relative paths land in WORKSPACE
    in_ws = str(p).startswith(str(WORKSPACE))
    in_home = str(p).startswith(str(Path.home()) + "/")
    if not in_ws and not (ALLOW_ANY_PATH and in_home):
        return (f"BLOCKED: writes are limited to {WORKSPACE} "
                f"(set JARVIS_ALLOW_ANY_PATH=1 for $HOME)")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"WROTE {len(content)} bytes to {p}"
    except Exception as e:
        return f"ERROR: {e}"


def self_status() -> str:
    """Report this assistant's own live configuration/state (introspection)."""
    import datetime
    import platform

    def g(k, d=""):
        return os.getenv(k, d)

    key = "configured" if os.getenv("OPENCODE_API_KEY") else "MISSING"
    return "\n".join([
        f"assistant: {g('JARVIS_NAME', 'Optimus')}",
        f"time: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"host: {platform.node()} ({platform.system()} {platform.machine()})",
        f"cloud model: {g('JARVIS_CLOUD_MODEL')}  fallback: {g('JARVIS_CLOUD_FALLBACK', 'mimo-v2.6-flash')}",
        f"code model: {g('JARVIS_CODE_MODEL')}",
        f"local model: {g('JARVIS_LOCAL_MODEL')} (enabled={g('JARVIS_LOCAL_OK', '0')})",
        f"wake word: {g('JARVIS_WAKE_WORDS', 'hey_optimus')} (threshold {g('JARVIS_WAKE_THRESHOLD', '0.7')}, debounce {g('JARVIS_WAKE_DEBOUNCE', '1')})",
        f"tts voice: {g('JARVIS_TTS_VOICE')}  rate {g('JARVIS_TTS_RATE', '+0%')}",
        f"cloud api key: {key}",
        f"tools: {len(TOOL_FUNCS)} registered",
    ])


def web_search(query: str, n: int = 5) -> str:
    """Search the web. Returns titles + URLs + snippets."""
    from duckduckgo_search import DDGS
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n))
        return "\n".join(f"- {r['title']}\n  {r['href']}\n  {r['body'][:150]}"
                         for r in results)
    except Exception as e:
        return f"ERROR: {e}"


# ---- Integrations ----

def browser_open(url: str) -> str:
    from .tools.browser import web_open
    return web_open(url)


def browser_extract(url: str = "", max_chars: int = 4000) -> str:
    from .tools.browser import web_extract
    return web_extract(url or None, max_chars)


def browser_screenshot(path: str = "") -> str:
    from .tools.browser import web_screenshot
    return web_screenshot(path or None)


def email_list(n: int = 10, unread_only: bool = False) -> str:
    from .tools.email import email_list as f
    return f(n=n, unread_only=unread_only)


def email_triage(n: int = 15, unread_only: bool = False) -> str:
    from .tools.email import email_triage as f
    return f(n=n, unread_only=unread_only)


def email_read(mail_id: str) -> str:
    from .tools.email import email_read as f
    return f(mail_id)


def email_send(to: str, subject: str, body: str) -> str:
    from .tools.email import email_send as f
    return f(to, subject, body)


def email_unread_count() -> int:
    from .tools.email import email_unread_count
    return email_unread_count()


def calendar_list(days: int = 7) -> str:
    from .tools.calendar import calendar_list as f
    return f(days=days)


def calendar_add(title: str, start: str, duration_minutes: int = 60,
                 description: str = "", location: str = "") -> str:
    from .tools.calendar import calendar_add as f
    return f(title, start, duration_minutes, description, location)


def calendar_today() -> str:
    from .tools.calendar import calendar_today
    return calendar_today()


def ha_list_entities(domain: str = "") -> str:
    from .tools.homeassistant import ha_list_entities
    return ha_list_entities(domain or None)


def ha_state(entity_id: str) -> str:
    from .tools.homeassistant import ha_state
    return ha_state(entity_id)


def ha_call(service: str, entity_id: str = "", data: str = "") -> str:
    from .tools.homeassistant import ha_call
    import json
    return f(service, entity_id or None, json.loads(data) if data else None)


def ha_search(query: str) -> str:
    from .tools.homeassistant import ha_search
    return ha_search(query)


TOOL_SCHEMAS = [
    {
        "name": "self_status",
        "description": "Report your own live configuration/state (models, wake word, voice). Use when asked about yourself.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "shell_run",
        "description": "Run a shell command. Use for system tasks, running scripts, checking info.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file's contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web. Returns titles, URLs, snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "browser_open",
        "description": "Open a URL in headless browser. Returns page title.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "browser_extract",
        "description": "Extract text content from a webpage.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL (empty = current page)"},
                "max_chars": {"type": "integer"},
            },
        },
    },
    {
        "name": "browser_screenshot",
        "description": "Take a screenshot of current page.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    },
    {
        "name": "email_triage",
        "description": "Rank recent emails by deterministic importance (VIP senders + keyword weights) and explain why. Use for 'what's important in my email'.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "unread_only": {"type": "boolean"},
            },
        },
    },
    {
        "name": "email_list",
        "description": "List recent emails with sender, subject, UNREAD flag and a preview snippet. Use this to triage which emails matter.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "unread_only": {"type": "boolean"},
            },
        },
    },
    {
        "name": "email_read",
        "description": "Read a specific email by its ID.",
        "parameters": {
            "type": "object",
            "properties": {"mail_id": {"type": "string"}},
            "required": ["mail_id"],
        },
    },
    {
        "name": "email_send",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
    },
    },
    {
        "name": "email_unread_count",
        "description": "Get count of unread emails.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "calendar_list",
        "description": "List events in the next N days.",
        "parameters": {
            "type": "object",
            "properties": {"days": {"type": "integer"}},
        },
    },
    {
        "name": "calendar_add",
        "description": "Add a calendar event. start in ISO format e.g. 2026-10-03T15:00.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "description": {"type": "string"},
                "location": {"type": "string"},
            },
            "required": ["title", "start"],
        },
    },
    {
        "name": "calendar_today",
        "description": "List today's events.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "ha_list_entities",
        "description": "List Home Assistant entities, optionally filtered by domain (light, switch, sensor...).",
        "parameters": {
            "type": "object",
            "properties": {"domain": {"type": "string"}},
        },
    },
    {
        "name": "ha_state",
        "description": "Get state and attributes of a Home Assistant entity.",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    {
        "name": "ha_call",
        "description": "Call a Home Assistant service. service='light.turn_on', entity_id='light.kitchen'. data is JSON string.",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "entity_id": {"type": "string"},
                "data": {"type": "string"},
            },
            "required": ["service"],
        },
    },
    {
        "name": "ha_search",
        "description": "Search Home Assistant entities by name.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


TOOL_FUNCS = {
    "self_status": self_status,
    "shell_run": shell_run,
    "read_file": read_file,
    "write_file": write_file,
    "web_search": web_search,
    "browser_open": browser_open,
    "browser_extract": browser_extract,
    "browser_screenshot": browser_screenshot,
    "email_list": email_list,
    "email_triage": email_triage,
    "email_read": email_read,
    "email_send": email_send,
    "email_unread_count": email_unread_count,
    "calendar_list": calendar_list,
    "calendar_add": calendar_add,
    "calendar_today": calendar_today,
    "ha_list_entities": ha_list_entities,
    "ha_state": ha_state,
    "ha_call": ha_call,
    "ha_search": ha_search,
}
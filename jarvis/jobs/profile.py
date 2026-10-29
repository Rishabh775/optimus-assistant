"""Job-search profile: the single source of truth the jobs skill reads.

Stored as JSON under data/jobs/profile.json (git-ignored, local only). The
assistant can read and update it conversationally (jobs_profile_get /
jobs_profile_update), and the phone app shows what is configured. Nothing
here is sent anywhere except as search keywords / application fields you
explicitly approve.

Fields
  name, email, phone       contact details used on applications
  location                 city or "Remote"
  remote_ok                include remote roles
  roles                    target titles/keywords ("frontend engineer", ...)
  skills                   stack keywords for matching/tailoring
  min_years                your experience (used to flag over-senior roles)
  exclude_companies        never show / never apply
  resume_path              absolute or repo-relative path to the resume
  links                    linkedin / github / portfolio
  ai_disclosure            add the honest "assisted by an AI assistant" line
"""
import json
import os
from pathlib import Path

from .. import config

DEFAULTS = {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "job_locations": [],
    "country": "",
    "remote_ok": True,
    "roles": [],
    "skills": [],
    "min_years": 0,
    "exclude_companies": [],
    "resume_path": "",
    "links": {"linkedin": "", "github": "", "portfolio": ""},
    "ai_disclosure": True,
}
_LIST_FIELDS = ("roles", "skills", "exclude_companies", "job_locations")
_UPDATABLE = set(DEFAULTS)


def path() -> Path:
    raw = config.get("JARVIS_JOBS_PROFILE") or \
        config.DEFAULTS["JARVIS_JOBS_PROFILE"]
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / raw
    return p


def _norm_list(value) -> list:
    if isinstance(value, str):
        value = [v for v in value.split(",")]
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    out = []
    for item in value:
        s = str(item).strip()
        if s and s.lower() not in (o.lower() for o in out):
            out.append(s)
    return out[:40]


def load() -> dict:
    p = path()
    prof = json.loads(json.dumps(DEFAULTS))       # deep copy
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        for k, v in data.items():
            if k in DEFAULTS:
                prof[k] = v
        prof["links"] = {**DEFAULTS["links"], **(prof.get("links") or {})}
    return prof


def save(prof: dict) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(prof, indent=2) + "\n")
    os.replace(tmp, p)


def ensure() -> dict:
    prof = load()
    if not path().exists():
        save(prof)
    return prof


def update(patch: dict) -> dict:
    """Validate and merge a partial profile. Raises ValueError on bad input."""
    if not isinstance(patch, dict) or not patch:
        raise ValueError("patch must be a non-empty object")
    unknown = [k for k in patch if k not in _UPDATABLE]
    if unknown:
        raise ValueError(f"unknown profile field(s): {', '.join(sorted(unknown))}")
    prof = load()
    for key, value in patch.items():
        if key in _LIST_FIELDS:
            prof[key] = _norm_list(value)
        elif key == "min_years":
            try:
                n = int(value)
            except (TypeError, ValueError):
                raise ValueError("min_years must be an integer") from None
            if not 0 <= n <= 60:
                raise ValueError("min_years out of range (0-60)")
            prof[key] = n
        elif key == "remote_ok" or key == "ai_disclosure":
            prof[key] = bool(value)
        elif key == "links":
            if not isinstance(value, dict):
                raise ValueError("links must be an object")
            prof["links"] = {**prof["links"],
                             **{str(k): str(v) for k, v in value.items()}}
        else:
            prof[key] = str(value).strip()
    if prof["email"] and "@" not in prof["email"]:
        raise ValueError("email does not look like an email address")
    save(prof)
    return prof


def summary() -> str:
    p = load()
    lines = [
        f"profile: {path()}",
        f"name: {p['name'] or '(not set)'}  email: {p['email'] or '(not set)'}",
        f"location: {p['location'] or '(not set)'}"
        f"  country: {p['country'] or '(not set)'}"
        f"  remote ok: {p['remote_ok']}",
        "job locations: " + (", ".join(p["job_locations"]) or
                             ("(remote only)" if p["remote_ok"] else "(not set)")),        "roles: " + (", ".join(p["roles"]) or "(not set)"),
        "skills: " + (", ".join(p["skills"]) or "(not set)"),
        f"experience: {p['min_years']}y  resume: {p['resume_path'] or '(not set)'}",        f"ai disclosure: {p['ai_disclosure']}",
    ]
    return "\n".join(lines)

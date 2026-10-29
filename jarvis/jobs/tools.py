"""Agent-facing tools for the jobs skill.

Kept deliberately small: discovery/digest/tracker plus profile access.
Submitting applications is NOT here yet -- it lands as job_apply behind a
mandatory confirmation gate in the next phase, so nothing can leave the
house until that gate exists.
"""
from . import digest, profile, tracker

TOOL_SCHEMAS = [
    {
        "name": "jobs_profile_get",
        "description": ("Show the job-search profile Soundwave uses (roles, "
                        "skills, location, experience, resume path)."),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "jobs_profile_update",
        "description": ("Update the job-search profile. Use when the user "
                        "describes what they're looking for: roles, skills, "
                        "location, remote_ok, min_years, email, name, phone, "
                        "resume_path, exclude_companies, links. Patch is a "
                        "partial object; lists replace the old value."),
        "parameters": {
            "type": "object",
            "properties": {"patch": {"type": "object"}},
            "required": ["patch"],
        },
    },
    {
        "name": "jobs_sweep",
        "description": ("Scan configured job boards (Greenhouse/Lever/Ashby/"
                        "Remotive/Adzuna), score matches against the profile, "
                        "store new roles and push a digest to the phone. Runs "
                        "daily if JARVIS_JOBS_ENABLED=1."),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "jobs_digest",
        "description": "Show the highest-scoring new roles, newest sweep first.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
    {
        "name": "jobs_list",
        "description": ("List tracked jobs (id, score, company, title, status, "
                        "url). Optionally filter by status: new, shortlisted, "
                        "drafting, applied, interview, offer, rejected, skipped."),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "job_update",
        "description": ("Update a tracked job's status/notes. Use when the "
                        "user says they applied, got an interview, want to "
                        "skip a role, etc."),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["id", "status"],
        },
    },
]


def _line(j: dict) -> str:
    loc = f" ({j['location']})" if j.get("location") else ""
    return (f"#{j['id']} [{j['score']:.0f}] {j['company']} — {j['title']}{loc}"
            f"  [{j['status']}]\n    {j['url']}")


def jobs_profile_get() -> str:
    return profile.summary()


def jobs_profile_update(patch: dict) -> str:
    try:
        profile.update(patch if isinstance(patch, dict) else {})
    except ValueError as e:
        return f"error: {e}"
    return "profile updated\n" + profile.summary()


def jobs_sweep() -> str:
    result = digest.sweep()
    if not result["ok"]:
        return f"error: {result['reason']}"
    head = (f"sweep done: {result['found']} matched, "
            f"{result['inserted']} new")
    return head + ("\n\n" + result["digest"] if result["digest"] else "")


def jobs_digest(limit: int = 0) -> str:
    rows = digest.top(limit or None)
    if not rows:
        return ("no new roles yet — set your jobs profile, then run "
                "jobs_sweep")
    return digest.render(rows)


def jobs_list(status: str = "", limit: int = 20) -> str:
    rows = tracker.list_jobs(status=(status or "").strip(), limit=int(limit or 20))
    if not rows:
        return "no tracked jobs" + (f" with status {status}" if status else "")
    return "\n".join(_line(j) for j in rows)


def job_update(id: str, status: str, note: str = "") -> str:
    try:
        ok = tracker.update_status(str(id), str(status), str(note or ""))
    except ValueError as e:
        return f"error: {e}"
    return "updated" if ok else f"error: no job #{id}"


TOOL_FUNCS = {
    "jobs_profile_get": jobs_profile_get,
    "jobs_profile_update": jobs_profile_update,
    "jobs_sweep": jobs_sweep,
    "jobs_digest": jobs_digest,
    "jobs_list": jobs_list,
    "job_update": job_update,
}

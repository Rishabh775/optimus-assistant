"""Agent-facing tools for the jobs skill.

Kept deliberately small: discovery/digest/tracker plus profile access.
Submitting applications is NOT here yet -- it lands as job_apply behind a
mandatory confirmation gate in the next phase, so nothing can leave the
house until that gate exists.
"""
import re
import time

from . import digest, profile, resume, tracker

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
                        "job_locations (target cities), location, country, "
                        "remote_ok, min_years, email, name, phone, "
                        "resume_path, exclude_companies, links. Patch is a "
                        "partial object; lists replace the old value."),
        "parameters": {
            "type": "object",
            "properties": {"patch": {"type": "object"}},
            "required": ["patch"],
        },
    },
    {
        "name": "jobs_resume_get",
        "description": ("Read the user's resume text (from jobs profile "
                        "resume_path). ALWAYS call this before comparing a "
                        "job to the user, judging fit, or drafting an "
                        "application/cover letter."),
        "parameters": {
            "type": "object",
            "properties": {"max_chars": {"type": "integer"}},
        },
    },
    {
        "name": "jobs_ingest_email",
        "description": ("Read job-alert emails (Indeed/LinkedIn/Naukri/"
                        "Glassdoor alerts you created) and add their roles to "
                        "the tracker. No scraping: uses your own inbox."),
        "parameters": {
            "type": "object",
            "properties": {"days": {"type": "integer"}},
        },
    },
    {
        "name": "jobs_add_url",
        "description": ("Add a job posting by URL (Indeed/LinkedIn/company "
                        "page). Use when the user pastes a link they found. "
                        "Optionally pass title and company for a better match."),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "title": {"type": "string"},
                "company": {"type": "string"},
            },
            "required": ["url"],
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
        "name": "jobs_job_get",
        "description": ("Read one tracked job in full: description excerpt, "
                        "score notes, status and history. Call this BEFORE "
                        "assessing fit, comparing to the resume, or drafting "
                        "an application for a specific job."),
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
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
    st = resume.ready()
    status = (f"resume: {st['path']} ({st['chars']} chars)" if st["exists"]
              else f"resume: NOT READABLE ({st['error']})")
    return profile.summary() + "\n" + status


def jobs_resume_get(max_chars: int = 6000) -> str:
    st = resume.ready()
    if not st["exists"]:
        return (f"error: resume not readable ({st['error']}) — set "
                "resume_path via jobs_profile_update")
    body = resume.text(max_chars=max(500, min(int(max_chars or 6000), 12000)))
    return f"resume: {st['path']} ({st['chars']} chars total)\n\n{body}"


def jobs_profile_update(patch: dict) -> str:
    try:
        profile.update(patch if isinstance(patch, dict) else {})
    except ValueError as e:
        return f"error: {e}"
    return "profile updated\n" + profile.summary()


def jobs_ingest_email(days: int = 2) -> str:
    from . import email_ingest
    r = email_ingest.ingest(days=max(1, min(int(days or 2), 14)))
    if not r["ok"]:
        return f"error: {r['error']}"
    if not r["new"]:
        return f"no new roles in alert emails ({r['found']} links seen)"
    lines = [f"{r['inserted']} new roles from email alerts:"]
    for j in r["new"][:10]:
        lines.append(f"[{j['score']:.0f}] {j['company']} — {j['title']}"
                     f"\n  {j['url']}")
    return "\n".join(lines)


def jobs_add_url(url: str, title: str = "", company: str = "") -> str:
    from . import email_ingest, match
    u = str(url or "").strip()
    if not u.startswith(("http://", "https://")):
        return "error: pass a full http(s) job URL"
    u = email_ingest.canonical_url(u)
    if not title:
        slug = u.rstrip("/").split("/")[-1]
        words = [w for w in re.split(r"[-_%?=&.]+", slug) if w][:8]
        title = " ".join(words).title() or "Untitled role"
    row = {"source": "manual", "company": company or "Unknown",
           "title": title, "location": "", "url": u, "description": "",
           "posted_at": time.time(), "salary": "",
           "remote": any(k in u.lower() for k in ("indeed", "linkedin"))}
    score, notes = match.score_job(row, profile.load(), resume.keywords())
    row["score"], row["score_notes"] = score, notes
    jid = tracker.upsert(row)
    if not jid:
        return "already tracked"
    return (f"added #{jid} [{score:.0f}] {row['company']} — {title}\n"
            f"  {u}\n  ({notes})")


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


def jobs_job_get(id: str) -> str:
    job = tracker.get(str(id))
    if not job:
        return f"error: no job #{id}"
    desc = (job.get("description") or "").strip()
    head = [
        f"#{job['id']}  [{job['score']:.0f}]  {job['company']} — {job['title']}",
        f"location: {job['location'] or 'n/a'}  salary: {job['salary'] or 'n/a'}",
        f"status: {job['status']}  source: {job['source']}",
        f"url: {job['url']}",
        f"score notes: {job['score_notes'] or 'n/a'}",
    ]
    ev = tracker.events(job["id"], limit=6)
    if ev:
        head.append("history: " + "; ".join(f"{e['kind']}" for e in ev))
    body = desc[:4000] + ("…" if len(desc) > 4000 else "")
    return "\n".join(head) + "\n\n" + (body or "(no description stored)")


def job_update(id: str, status: str, note: str = "") -> str:
    try:
        ok = tracker.update_status(str(id), str(status), str(note or ""))
    except ValueError as e:
        return f"error: {e}"
    return "updated" if ok else f"error: no job #{id}"


TOOL_FUNCS = {
    "jobs_profile_get": jobs_profile_get,
    "jobs_profile_update": jobs_profile_update,
    "jobs_resume_get": jobs_resume_get,
    "jobs_ingest_email": jobs_ingest_email,
    "jobs_add_url": jobs_add_url,
    "jobs_sweep": jobs_sweep,
    "jobs_digest": jobs_digest,
    "jobs_list": jobs_list,
    "jobs_job_get": jobs_job_get,
    "job_update": job_update,
}

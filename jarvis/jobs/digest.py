"""Sweep: discover -> score -> store -> digest.

`sweep()` is both an agent tool and a scheduled daily job (see install()).
It never applies to anything; it just fills the tracker and pushes a
shortlist to the phone. Applying is a separate, confirmation-gated step.
"""
from .. import config, push
from ..log import log as _mk_log
from . import discovery, match, profile, resume, tracker

log = _mk_log("jobs")


def render(rows: list, header: str = "New roles") -> str:
    if not rows:
        return ""
    lines = [header, ""]
    for r in rows:
        loc = f" — {r['location']}" if r.get("location") else ""
        lines.append(f"[{r['score']:.0f}] {r['company']} — {r['title']}{loc}")
        lines.append(f"    {r['url']}")
        if r.get("salary"):
            lines.append(f"    {r['salary']}")
        if r.get("score_notes"):
            lines.append(f"    ({r['score_notes']})")
    return "\n".join(lines)


def top(limit: int = None, status: str = "new") -> list:
    n = limit or int(config.get_int("JARVIS_JOBS_DIGEST_SIZE") or 10)
    floor = config.get_float("JARVIS_JOBS_MIN_SCORE")
    return tracker.list_jobs(status=status, limit=n, min_score=floor)


def sweep(limit: int = None) -> dict:
    prof = profile.load()
    if not prof.get("roles"):
        return {"ok": False,
                "reason": ("jobs profile has no target roles — ask Soundwave "
                           "to set them (e.g. \"set my jobs profile: frontend "
                           "engineer, react, typescript, Hyderabad, 2 years\")"),
                "found": 0, "inserted": 0, "digest": ""}

    found = discovery.search_all(prof)
    resume_kw = resume.keywords()
    inserted = []
    for job in found:
        score, notes = match.score_job(job, prof, resume_kw)
        job["score"], job["score_notes"] = score, notes
        jid = tracker.upsert(job)
        if jid:
            inserted.append(jid)

    rows = top(limit)
    text = render(rows, f"Job digest — {len(inserted)} new of {len(found)} matched")
    if inserted and text:
        push.send("Job digest", text)
        log.info("job sweep: %d found, %d new", len(found), len(inserted))

    return {"ok": True, "found": len(found), "inserted": len(inserted),
            "digest": text}


def install() -> bool:
    """Register the daily sweep when JARVIS_JOBS_ENABLED=1. Idempotent."""
    from .. import scheduler
    if not config.get_bool("JARVIS_JOBS_ENABLED"):
        return False
    at = config.get("JARVIS_JOBS_SWEEP_TIME") or "08:30"
    try:
        scheduler.add_daily("jobs_sweep", at, lambda: sweep())
    except ValueError as e:
        log.warning("jobs sweep not scheduled: %s", e)
        return False
    log.info("jobs sweep scheduled daily at %s", at)
    return True

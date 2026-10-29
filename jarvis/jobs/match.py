"""Deterministic fit scoring.

No LLM calls here on purpose: scoring thousands of postings should be free,
fast and explainable. Each point comes with a human-readable note so the
digest/agent can say *why* a role scored well ("react, typescript in title;
remote; posted 2d ago"). The agent can still add qualitative judgement on
top when you ask it to.
"""
import re

_YEARS_RE = re.compile(r"(\d+)\s*\+?\s*(?:years?|yrs?)", re.I)
_REMOTE_RE = re.compile(r"\b(remote|worldwide|anywhere|work from home)\b", re.I)
_NEGATIVE_TITLES = (("principal", 15), ("staff", 13), ("head of", 15),
                    ("director", 15), ("manager", 12), ("lead", 12),
                    ("senior", 10))
_JUNIOR_TITLES = (("intern", 20), ("trainee", 15), ("fresher", 15),
                  ("junior", 6))
_CITY_ALIASES = {
    "bengaluru": "bangalore", "bangalore": "bengaluru",
    "gurugram": "gurgaon", "gurgaon": "gurugram",
    "mumbai": "bombay", "bombay": "mumbai",
}


def loc_matches(text: str, targets: list) -> bool:
    text = (text or "").lower()
    for t in targets:
        t = str(t).strip().lower()
        if t and t in text:
            return True
        alias = _CITY_ALIASES.get(t)
        if alias and alias in text:
            return True
    return False


def is_remote(job: dict) -> bool:
    """Location field wins over prose: 'Hybrid - SF' is not remote even if
    the description mentions remote culture."""
    loc = (job.get("location") or "").lower()
    if re.search(r"\b(on-?site|hybrid|in[- ]office)\b", loc):
        return False
    if _REMOTE_RE.search(loc):
        return True
    return bool(job.get("remote")) or bool(
        _REMOTE_RE.search(job.get("description") or ""))


def required_years(text: str):
    m = _YEARS_RE.search(text or "")
    return int(m.group(1)) if m else None


def score_job(job: dict, prof: dict,
              resume_keywords: set = None) -> tuple[float, str]:
    """Return (0-100 score, notes). Pure function, easy to test."""
    title = (job.get("title") or "").lower()
    text = f"{title} {job.get('description') or ''}".lower()
    notes = []
    score = 0.0

    roles = [r.lower() for r in prof.get("roles") or []]
    skills = [s.lower() for s in prof.get("skills") or []]

    # 1) title relevance (40): full phrase beats a single keyword
    if any(r in title for r in roles if r):
        score += 40
        notes.append("title match")
    else:
        words = {w for r in roles for w in r.split()
                 if len(w) > 3} - {"engineer", "developer", "software",
                                   "senior", "junior", "staff", "lead"}
        if any(w in title for w in words):
            score += 25
            notes.append("title keyword")

    # 2) stack coverage (30) against your core skills
    core = skills[:8]
    if core:
        hits = [s for s in core if s in text]
        score += 30.0 * len(hits) / len(core)
        if hits:
            notes.append("stack: " + ", ".join(hits[:5]))
    else:
        score += 15

    # 3) location fit (15): target cities (job_locations) or remote
    remote = is_remote(job)
    targets = [str(t).strip().lower()
               for t in (prof.get("job_locations") or []) if str(t).strip()]
    if not targets and prof.get("location"):
        targets = [str(prof["location"]).strip().lower()]
    loc_text = f"{job.get('location', '')} {job.get('description', '')}"
    if remote and prof.get("remote_ok", True):
        score += 15
        notes.append("remote")
    elif targets and loc_matches(loc_text, targets):
        score += 15
        notes.append("target city")
    elif not targets:
        score += 7
    elif not prof.get("remote_ok", True):
        score -= 20
        notes.append("not in target location")

    # 4) resume overlap bonus (10): distinctive resume terms present in the ad
    if resume_keywords:
        hits = [k for k in resume_keywords if k in text]
        if hits:
            score += min(10.0, 10.0 * len(hits) / 6)
            notes.append(f"resume fit {len(hits)}")

    # 4) freshness (10)
    posted = job.get("posted_at")
    if posted:
        import time
        age_days = max(0.0, (time.time() - float(posted)) / 86400)
        if age_days <= 7:
            score += 10
            notes.append("fresh")
        elif age_days <= 14:
            score += 5

    # 5) seniority sanity checks
    years = required_years(text)
    mine = int(prof.get("min_years") or 0)
    if years and years > mine + 2:
        score -= 12
        notes.append(f"asks {years}+ yrs")
    for word, penalty in _NEGATIVE_TITLES:
        if word in title and mine < 4:
            score -= penalty
            notes.append(f"{word} role")
            break
    for word, penalty in _JUNIOR_TITLES:
        if word in title and mine >= 2:
            score -= penalty
            notes.append(f"{word} role")
            break

    # eligibility transparency: never let an unstated or foreign region look
    # confirmed; in "any" mode these are shown but labeled and ranked lower
    loc_raw = (job.get("location") or "").strip().lower()
    scope = job.get("remote_scope")
    if scope == "blocked":
        score -= 8
        notes.append("⚠ region-restricted")
    elif scope == "unknown":
        notes.append("⚠ verify region")
    elif loc_raw in ("", "n/a"):
        notes.append("⚠ location not stated")

    return max(0.0, min(100.0, round(score, 1))), "; ".join(notes)

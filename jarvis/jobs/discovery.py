"""Job discovery from public, automation-friendly sources.

Deliberately excludes LinkedIn / Indeed / Naukri: their terms prohibit
automated access, and a discovery bot is not worth an account ban. What we
do use:

  * Greenhouse / Lever / Ashby public job-board APIs (no auth) for a
    curated list of companies -- set JARVIS_JOBS_BOARDS to "gd:acme,..."
  * Adzuna search API (optional, free key; covers India) for broad search
  * Remotive's public API for remote roles

Everything normalises to:
    source, company, title, location, url, description, posted_at,
    salary, remote
Discovery only READS. It never submits an application (that is a separate
step behind a confirmation gate).
"""
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .. import config
from ..log import log as _mk_log
from . import match

log = _mk_log("jobs.discovery")

_UA = ("soundwave-jobs/1.0 (personal job search; "
       "+https://github.com/Rishabh775/optimus-assistant)")
_MAX_PER_BOARD = 8
_MAX_PER_FEED = 15
_MAX_TOTAL = 60
_GENERIC_WORDS = {"engineer", "engineers", "developer", "developers",
                  "software", "senior", "junior", "staff", "lead", "the",
                  "and", "for", "with", "role", "native"} 


def _get_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _get_text(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _epoch(iso: str):
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def boards() -> list:
    """Parse JARVIS_JOBS_BOARDS: "greenhouse:stripe, ashby:linear"."""
    out = []
    for item in (config.get("JARVIS_JOBS_BOARDS") or "").split(","):
        item = item.strip()
        if ":" not in item:
            continue
        provider, slug = item.split(":", 1)
        provider, slug = provider.strip().lower(), slug.strip()
        if provider in ("greenhouse", "lever", "ashby") and slug:
            out.append((provider, slug))
    return out


def _company(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


# ------------------------------------------------------- ATS providers ---

def fetch_greenhouse(slug: str) -> list:
    data = _get_json(f"https://boards-api.greenhouse.io/v1/boards/"
                     f"{urllib.parse.quote(slug)}/jobs")
    out = []
    for j in data.get("jobs", []):
        out.append({
            "source": f"greenhouse:{slug}",
            "company": _company(slug),
            "title": j.get("title", ""),
            "location": (j.get("location") or {}).get("name", ""),
            "url": j.get("absolute_url", ""),
            "description": j.get("content", ""),
            "posted_at": None,   # updated_at is not a posting date; detail has it
            "salary": "",
            "remote": False,
            "_detail": f"https://boards-api.greenhouse.io/v1/boards/"
                       f"{urllib.parse.quote(slug)}/jobs/{j.get('id')}",
        })
    return out


def fetch_lever(slug: str) -> list:
    data = _get_json(f"https://api.lever.co/v0/postings/"
                     f"{urllib.parse.quote(slug)}?mode=json")
    out = []
    for j in (data if isinstance(data, list) else []):
        cats = j.get("categories") or {}
        out.append({
            "source": f"lever:{slug}",
            "company": _company(slug),
            "title": j.get("text", ""),
            "location": cats.get("location", ""),
            "url": j.get("hostedUrl", ""),
            "description": j.get("descriptionPlain", ""),
            "posted_at": (j.get("createdAt") or 0) / 1000 or None,
            "salary": "",
            "remote": False,
        })
    return out


def fetch_ashby(slug: str) -> list:
    data = _get_json("https://api.ashbyhq.com/posting-api/job-board/"
                     f"{urllib.parse.quote(slug)}?includeCompensation=true")
    out = []
    for j in (data.get("jobs") or []):
        comp = (j.get("compensation") or {}).get("compensationTierSummary") or ""
        out.append({
            "source": f"ashby:{slug}",
            "company": _company(slug),
            "title": j.get("title", ""),
            "location": j.get("location", ""),
            "url": j.get("jobUrl") or j.get("applyUrl", ""),
            "description": j.get("descriptionPlain", ""),
            "posted_at": _epoch(j.get("publishedAt")),
            "salary": comp,
            "remote": bool(j.get("isRemote")),
        })
    return out


# -------------------------------------------------- aggregator sources ---

def fetch_himalayas(limit: int = 100) -> list:
    """Himalayas: remote jobs with explicit location restrictions."""
    data = _get_json(f"https://himalayas.app/jobs/api?limit={limit}")
    out = []
    for j in (data.get("jobs") or [])[:limit]:
        locs = j.get("locationRestrictions") or []
        salary = ""
        lo, hi = j.get("minSalary"), j.get("maxSalary")
        if lo or hi:
            cur = (j.get("currency") or "").strip()
            per = (j.get("salaryPeriod") or "").strip()
            salary = f"{cur} {lo or ''}-{hi or ''}".strip()
            salary = f"{salary} / {per}" if per else salary
        out.append({
            "source": "himalayas",
            "company": j.get("companyName", ""),
            "title": j.get("title", ""),
            "location": ", ".join(locs) if locs else "Worldwide",
            "url": j.get("applicationLink") or j.get("guid", ""),
            "description": re.sub(r"<[^>]+>", " ", j.get("description") or ""),
            "posted_at": j.get("pubDate"),
            "salary": salary,
            "remote": True,
        })
    return out


def fetch_weworkremotely(category: str = "programming") -> list:
    """We Work Remotely publishes a public RSS feed per category."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    xml = _get_text("https://weworkremotely.com/categories/"
                    f"remote-{urllib.parse.quote(category)}-jobs.rss")
    root = ET.fromstring(xml)
    out = []
    for item in root.findall("./channel/item"):
        raw_title = (item.findtext("title") or "").strip()
        company, _, title = raw_title.partition(": ")
        if not title:
            company, title = "", raw_title
        desc = item.findtext("description") or ""
        posted = None
        pub = item.findtext("pubDate")
        if pub:
            try:
                posted = parsedate_to_datetime(pub).timestamp()
            except (TypeError, ValueError):
                posted = None
        out.append({
            "source": "weworkremotely",
            "company": company.strip(),
            "title": title.strip(),
            "location": (item.findtext("region") or "Anywhere in the World").strip(),
            "url": (item.findtext("link") or "").strip(),
            "description": re.sub(r"<[^>]+>", " ", desc),
            "posted_at": posted,
            "salary": "",
            "remote": True,
        })
    return out


def fetch_remotive(keyword: str, limit: int = 30) -> list:
    q = urllib.parse.urlencode({"search": keyword, "limit": limit})
    data = _get_json(f"https://remotive.com/api/remote-jobs?{q}")
    out = []
    for j in (data.get("jobs") or [])[:limit]:
        out.append({
            "source": "remotive",
            "company": j.get("company_name", ""),
            "title": j.get("title", ""),
            "location": j.get("candidate_required_location", "Remote"),
            "url": j.get("url", ""),
            "description": j.get("description", ""),
            "posted_at": _epoch(j.get("publication_date")),
            "salary": j.get("salary") or "",
            "remote": True,
        })
    return out


def fetch_adzuna(keyword: str, country: str, app_id: str, app_key: str,
                 results: int = 30) -> list:
    q = urllib.parse.urlencode({
        "app_id": app_id, "app_key": app_key, "results_per_page": results,
        "what": keyword, "content-type": "application/json",
    })
    data = _get_json(f"https://api.adzuna.com/v1/api/jobs/"
                     f"{urllib.parse.quote(country)}/search/1?{q}")
    out = []
    for j in (data.get("results") or [])[:results]:
        lo, hi = j.get("salary_min"), j.get("salary_max")
        salary = ""
        if lo and hi:
            salary = f"₹{lo / 100000:.1f}–{hi / 100000:.1f} LPA" \
                if str(country).lower() == "in" else f"{lo:.0f}–{hi:.0f}"
        desc = re.sub(r"<[^>]+>", " ", j.get("description") or "")
        out.append({
            "source": "adzuna",
            "company": (j.get("company") or {}).get("display_name", ""),
            "title": j.get("title", ""),
            "location": (j.get("location") or {}).get("display_name", ""),
            "url": j.get("redirect_url", ""),
            "description": desc,
            "posted_at": _epoch(j.get("created")),
            "salary": salary,
            "remote": bool(re.search(r"\bremote\b", desc, re.I)),
        })
    return out


# ------------------------------------------------------ orchestration ---

_SENIOR_TITLE_RE = re.compile(
    r"\b(manager|director|head of|principal|staff|vp|vice president|lead)\b",
    re.I)
_ENTRY_TITLE_RE = re.compile(
    r"\b(intern|internship|trainee|fresher|new grad|graduate)\b", re.I)
_OFF_FIELD_RE = re.compile(
    r"\b(designer|design|recruiter|sales|marketing|accountant|writer|"
    r"customer support|support engineer)\b", re.I)

# Remote eligibility signals in the posting body (location labels lie:
# "Anywhere in the World" often hides a closed country list).
_US_ONLY_RE = re.compile(
    r"(authori[sz]ed to work in the (?:u\.?s\.?|united states)"
    r"|u\.?s\.? work authori[sz]ation"
    r"|must (?:be )?(?:based|reside|live) in the (?:u\.?s\.?|united states)"
    r"|us[- ]based only"
    r"|open only to (?:candidates in )?(?:the )?(?:u\.?s\.?|united states))",
    re.I)
_COUNTRY_LIST_RE = re.compile(
    r"(?:countries across|selected countries in|eligible countries"
    r"|open to (?:candidates )?(?:in|from)|hiring (?:in|only in)"
    r"|must (?:be )?(?:based|reside|live) in)[^.]{0,400}", re.I)
_COUNTRY_NAMES = (
    "europe", "north america", "latin america", "south america", "u.s.",
    "united states", "canada", "united kingdom", "emea", "australia",
    "japan", "singapore", "south korea", "philippines", "indonesia",
    "malaysia", "vietnam", "thailand", "israel", "germany", "france",
    "brazil", "mexico", "argentina", "africa", "middle east",
)
_WORLD_RE = re.compile(r"\b(worldwide|anywhere|global(?:ly)?)\b", re.I)


def remote_scope(job: dict, prof: dict) -> str:
    """Classify remote eligibility: worldwide | country | unknown | blocked
    (plus 'onsite' for non-remote roles). Conservative: an unstated region
    is 'unknown', never assumed eligible."""
    if not match.is_remote(job):
        return "onsite"
    loc = (job.get("location") or "").strip().lower()
    desc = job.get("description") or ""
    country = (prof.get("country") or "").strip().lower()

    if _US_ONLY_RE.search(desc):
        return "blocked"

    # closed country lists ("countries across ... selected countries in Asia
    # (Japan, Singapore, ...)") -- eligible only when India is named.
    for window in _COUNTRY_LIST_RE.findall(desc):
        w = window.lower()
        if "india" in w:
            return "country"
        if _WORLD_RE.search(w) and not any(c in w for c in _COUNTRY_NAMES):
            return "worldwide"
        if any(c in w for c in _COUNTRY_NAMES):
            return "blocked"

    if any(k in loc for k in ("worldwide", "anywhere", "global", "asia",
                              "apac")) or (country and country in loc):
        return "worldwide" if country and country in loc else "worldwide"
    if country and country in desc.lower() and "india" in desc.lower():
        return "country"
    if loc and loc not in ("remote", "n/a", "anywhere", "anywhere in the world"):
        return "blocked"          # explicit different place
    if _WORLD_RE.search(desc):
        return "worldwide"
    return "unknown"
def _want(job: dict, prof: dict) -> bool:
    title = (job.get("title") or "").lower()
    mine = int(prof.get("min_years") or 0)

    # seniority sanity: don't send a 2-year dev manager/principal roles,
    # nor entry-level roles to someone with experience
    if mine < 5 and _SENIOR_TITLE_RE.search(title):
        return False
    if mine >= 1 and _ENTRY_TITLE_RE.search(title):
        return False
    off = _OFF_FIELD_RE.search(title)
    if off and not any(off.group(1).lower() in str(r).lower()
                       for r in prof.get("roles") or []):
        return False
    need_years = match.required_years(f"{title} {job.get('description') or ''}")
    if need_years and need_years > mine + 6:
        return False
    scope = remote_scope(job, prof)
    if scope == "blocked":
        return False
    if scope == "unknown" and config.get_bool("JARVIS_JOBS_HIDE_UNCLEAR_REMOTE"):
        return False
    job["remote_scope"] = scope

    # title relevance: roles/skills as whole-word tokens
    tokens = set()
    for role in prof.get("roles") or []:
        for word in re.split(r"[^a-z0-9+.#]+", str(role).lower()):
            if len(word) >= 2:
                tokens.add(word)
    tokens |= {str(s).lower() for s in (prof.get("skills") or [])[:8]}
    if any(("frontend" in r or "front end" in r)
           for r in (prof.get("roles") or [])):
        tokens |= {"web", "ui"}      # "Web Developer", "UI Engineer" titles
    tokens -= _GENERIC_WORDS
    if tokens and not any(
            re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", title)
            for t in tokens):
        return False

    excluded = [c.lower() for c in prof.get("exclude_companies") or []]
    if excluded and any(c in (job.get("company") or "").lower() for c in excluded):
        return False

    days = int(config.get_float("JARVIS_JOBS_MAX_AGE_DAYS") or 21)
    if job.get("posted_at"):
        if (time.time() - float(job["posted_at"])) / 86400 > days:
            return False

    # target locations: Bangalore (etc.) or remote
    targets = [str(t).strip().lower()
               for t in (prof.get("job_locations") or []) if str(t).strip()]
    if not targets and prof.get("location"):
        targets = [str(prof["location"]).strip().lower()]
    loc = (job.get("location") or "").strip().lower()
    remote = scope != "onsite"

    if remote:
        if not prof.get("remote_ok", True):
            return False
    elif targets:
        if loc and loc not in ("n/a", "anywhere in the world"):
            if not match.loc_matches(loc, targets):
                return False
    return True


def _enrich(job: dict) -> None:
    """Greenhouse list rows have no description; fetch the matched posts."""
    detail = job.pop("_detail", None)
    if not detail or job.get("description"):
        return
    try:
        data = _get_json(detail)
        job["description"] = data.get("content", "") or ""
        job["posted_at"] = _epoch(data.get("first_published")) or job.get("posted_at")
    except Exception as e:
        log.debug("greenhouse detail failed: %s", e)


def search_all(prof: dict) -> list:
    """Query every configured source and return filtered, normalised jobs."""
    terms = [r for r in (prof.get("roles") or []) if r][:4]
    jobs, errors = [], []

    for provider, slug in boards():
        fn = {"greenhouse": fetch_greenhouse, "lever": fetch_lever,
              "ashby": fetch_ashby}[provider]
        try:
            jobs += fn(slug)
        except Exception as e:
            errors.append(f"{provider}:{slug}: {e}")

    for kw in terms[:2]:
        short = kw.split()[0] if kw.split() else kw
        try:
            jobs += fetch_remotive(short)
        except Exception as e:
            errors.append(f"remotive: {e}")

    try:
        jobs += fetch_himalayas()
    except Exception as e:
        errors.append(f"himalayas: {e}")

    try:
        jobs += fetch_weworkremotely()
    except Exception as e:
        errors.append(f"weworkremotely: {e}")

    app_id = config.get("JARVIS_ADZUNA_APP_ID")
    app_key = config.get("JARVIS_ADZUNA_APP_KEY")
    if app_id and app_key:
        country = config.get("JARVIS_JOBS_COUNTRY") or "in"
        for kw in terms[:2]:
            try:
                jobs += fetch_adzuna(kw, country, app_id, app_key)
            except Exception as e:
                errors.append(f"adzuna: {e}")

    picked, seen = [], set()
    per_source = {}
    for job in jobs:
        if not _want(job, prof):
            continue
        url = (job.get("url") or "").strip()
        if not url or url in seen:
            continue
        provider = (job.get("source") or "").split(":")[0]
        cap = _MAX_PER_BOARD if provider in ("greenhouse", "lever", "ashby") \
            else _MAX_PER_FEED
        key = job.get("source") or provider
        if per_source.get(key, 0) >= cap:
            continue
        seen.add(url)
        per_source[key] = per_source.get(key, 0) + 1
        picked.append(job)

    # fetch descriptions only for the shortlist we will actually store
    for job in picked[:_MAX_TOTAL]:
        _enrich(job)

    if errors:
        log.warning("discovery errors: %s", "; ".join(errors[:4]))
    return picked[:_MAX_TOTAL]

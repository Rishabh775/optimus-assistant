"""Ingest job-alert emails instead of scraping the job boards.

Indeed / LinkedIn / Naukri explicitly prohibit automated access, so we do
not touch their sites. What is allowed: you create job alerts on those
sites (a normal user feature), their emails land in your inbox, and we read
your own mail over IMAP. That gives broad coverage -- including Indeed --
without scraping or account risk.

Flow: recent alert emails -> extract job links + titles from the HTML ->
canonicalise/dedupe -> score against the profile+resume -> tracker.
Location is often missing in alert emails, so those rows are flagged
"location not stated" rather than silently trusted.
"""
import email as emaillib
import html
import re
import time
from datetime import datetime, timedelta

from .. import config
from ..log import log as _mk_log
from . import match, profile, resume, tracker

log = _mk_log("jobs.email")

_SENDERS = ("indeed", "linkedin", "naukri", "glassdoor", "instahyre",
            "wellfound", "hirist", "cutshort", "foundit", "iimjobs")
_JOB_HOSTS = ("indeed.", "linkedin.com/jobs", "naukri.com", "glassdoor.",
              "instahyre.com", "wellfound.com", "hirist.com",
              "cutshort.io", "foundit.in", "iimjobs.com")
_ANCHOR = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
                     re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_NOISE = re.compile(r"^(new|view job|view|apply|apply now|job alert|"
                    r"recommended|see more|manage alerts?|see all jobs?|"
                    r"view all|your job alert|jobs in|unsubscribe|settings|"
                    r"save search|edit alert|create alert|alert preferences|"
                    r"similar jobs|company page|back to results)\b", re.I)
_AT_COMPANY = re.compile(r"^(.{3,90}?)\s+(?:at|@)\s+(.{2,60})$", re.I)
_COMPANY_LOC = re.compile(
    r"^([A-Z][\w&.,'’()\- ]{1,48}?)\s*[·•|]\s*([A-Za-z .(),\-]{2,48})")

_CACHE = {"ts": 0.0, "jobs": []}
_ENTRY_TITLE = re.compile(r"\b(intern|internship|trainee|fresher|new grad|"
                          r"graduate)\b", re.I)


def _clean(text: str) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", text or "")).split())


def canonical_url(url: str) -> str:
    """Normalise tracking links so the same job dedupes across emails."""
    url = html.unescape(url or "").strip()
    if "indeed." in url:
        m = re.search(r"[?&]jk=([A-Za-z0-9]+)", url)
        host = re.match(r"https?://([^/]+)", url)
        if m and host:
            return f"https://{host.group(1)}/viewjob?jk={m.group(1)}"
    if "linkedin.com" in url:
        m = re.search(r"/jobs/view/(\d+)", url)
        if m:
            return f"https://www.linkedin.com/jobs/view/{m.group(1)}/"
    return url.split("?utm_")[0]


def _is_job_link(url: str) -> bool:
    """Only real job pages, never alert/search/navigation links."""
    u = (url or "").lower()
    if not u.startswith("http"):
        return False
    if "indeed." in u:
        return "viewjob" in u or "jk=" in u
    if "linkedin.com" in u:
        return bool(re.search(r"/jobs/view/\d+", u))
    if "naukri.com" in u:
        return "job-listings" in u or "/jobs/" in u
    if "glassdoor." in u:
        return "job-listing" in u or "/job/" in u
    if any(k in u for k in ("instahyre", "wellfound", "hirist", "cutshort",
                            "foundit", "iimjobs")):
        return any(k in u for k in ("/job/", "/jobs/", "/job-detail", "/apply"))
    return "/jobs/" in u or "/job/" in u


def _sender_label(host_url: str) -> str:
    if "indeed" in host_url:
        return "Indeed"
    if "linkedin" in host_url:
        return "LinkedIn"
    if "naukri" in host_url:
        return "Naukri"
    if "glassdoor" in host_url:
        return "Glassdoor"
    return "Email alert"


def parse_html(html_body: str) -> list:
    """Extract {title, company, location, url} candidates from an alert email."""
    out, seen = [], set()
    for m in _ANCHOR.finditer(html_body or ""):
        href, inner = m.group(1), m.group(2)
        url = canonical_url(href)
        if not _is_job_link(url) or url in seen:
            continue
        title = _NOISE.sub("", _clean(inner)).strip(" -–—|·:")
        if not (3 <= len(title) <= 120) or _NOISE.match(title):
            continue
        company, location = "", ""
        at = _AT_COMPANY.match(title)
        if at:
            title, company = at.group(1).strip(), at.group(2).strip()
        else:
            # alert emails usually print "Company · City" right after the link
            near = _clean(html_body[m.end():m.end() + 320])
            cm = _COMPANY_LOC.match(near)
            if cm:
                company, location = cm.group(1).strip(), cm.group(2).strip()
        seen.add(url)
        out.append({"title": title,
                    "company": company or f"{_sender_label(url)} alert",
                    "location": location,
                    "url": url})
    return out


def _recent_alert_messages(days: int = 2, limit: int = 40):
    """IMAP: newest alert emails from known senders (PEEK, never mark read)."""
    from ..tools.email import _imap
    conn, user, err = _imap()
    if err:
        return [], err
    found, seen_ids = [], set()
    try:
        conn.select("INBOX")
        since = (datetime.now() - timedelta(days=max(1, days))) \
            .strftime("%d-%b-%Y")
        for sender in ("indeed", "linkedin", "naukri", "glassdoor"):
            try:
                typ, data = conn.search(None, f'(SINCE {since} FROM "{sender}")')
                ids = data[0].split()[-limit:]
            except Exception:
                continue
            for mid in ids:
                typ, md = conn.fetch(mid, "(BODY.PEEK[])")
                raw = md[0][1]
                msg = emaillib.message_from_bytes(raw)
                mid_key = msg.get("Message-ID") or mid
                if mid_key in seen_ids:
                    continue
                seen_ids.add(mid_key)
                body = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        body += part.get_payload(decode=True).decode(
                            "utf-8", "replace")
                if body:
                    found.append({"id": mid.decode(),
                                  "subject": msg.get("Subject", ""),
                                  "html": body})
        conn.logout()
    except Exception as e:
        try:
            conn.logout()
        except Exception:
            pass
        return [], f"IMAP error: {e}"
    return found, None


def ingest(days: int = 2) -> dict:
    """Read alert emails, add unseen jobs. Returns found/inserted/new rows."""
    if not config.get_bool("JARVIS_JOBS_INGEST_EMAIL"):
        return {"ok": False, "error": "email ingestion disabled",
                "found": 0, "inserted": 0, "new": []}
    messages, err = _recent_alert_messages(days=days)
    if err:
        return {"ok": False, "error": err, "found": 0, "inserted": 0,
                "new": []}

    prof = profile.load()
    resume_kw = resume.keywords()
    candidates, seen = [], set()
    for msg in messages:
        for job in parse_html(msg["html"]):
            if job["url"] in seen:
                continue
            seen.add(job["url"])
            candidates.append(job)

    inserted = []
    experienced = int(prof.get("min_years") or 0) >= 1
    for job in candidates[:60]:
        if experienced and _ENTRY_TITLE.search(job["title"]):
            continue                       # no intern/fresher roles
        row = {
            "source": "email",
            "company": job["company"],
            "title": job["title"],
            "location": job.get("location") or "",
            "url": job["url"],
            "description": "",
            "posted_at": time.time(),
            "salary": "",
            # don't assume remote from the domain: alert emails omit location
            "remote": "remote" in (job.get("location") or "").lower(),
        }
        score, notes = match.score_job(row, prof, resume_kw)
        row["score"], row["score_notes"] = score, notes
        jid = tracker.upsert(row)
        if jid:
            inserted.append(jid)

    new_rows = [tracker.get(j) for j in inserted]
    new_rows = [j for j in new_rows if j]
    if new_rows:
        log.info("email alerts: %d messages, %d new jobs",
                 len(messages), len(new_rows))
    return {"ok": True, "error": "", "found": len(candidates),
            "inserted": len(new_rows), "new": new_rows}

"""Fast job alerts: poll the boards every few minutes and push genuinely
new roles the moment they appear (ntfy + in-app note).

Why polling: the public ATS APIs are pull-only -- Greenhouse/Lever/Ashby
don't offer webhooks to candidates. A 15-20 minute cadence is the sweet
spot before rate limits and noise; configurable down to 5 minutes.

Exactly-once across processes: the interval window is claimed in SQLite
(scheduler.interval_runs), so the CLI and the phone-app server never both
alert for the same window. Dedupe is by URL + company/title in the tracker,
so a role is announced once, ever.

Only "new" rows above JARVIS_JOBS_ALERT_MIN_SCORE are pushed; the rest wait
in the Jobs panel. A per-run cap prevents notification storms.
"""
from .. import config, push, scheduler
from ..log import log as _mk_log
from . import digest, discovery, email_ingest

import time

log = _mk_log("jobs.alerts")

_NOTIFIER = None
_LAST_EMAIL = 0.0


def set_notifier(fn) -> None:
    """Optional in-app broadcast hook (the server registers a WS sender)."""
    global _NOTIFIER
    _NOTIFIER = fn


def _notify(title: str, text: str) -> None:
    push.send(title, text)
    if _NOTIFIER:
        try:
            _NOTIFIER(text)
        except Exception:
            log.exception("job alert notifier failed")


def sweep() -> dict:
    if not config.get_bool("JARVIS_JOBS_ALERTS"):
        return {"ok": False, "reason": "alerts disabled", "alerted": 0}
    floor = config.get_float("JARVIS_JOBS_ALERT_MIN_SCORE")
    cap = max(1, config.get_int("JARVIS_JOBS_ALERT_MAX_PER_RUN"))

    result = digest.sweep(notify=False)
    if not result.get("ok"):
        return {"ok": False, "reason": result.get("reason", ""), "alerted": 0}
    st = discovery.stats()
    log.info("alert scan: %d requests, %d unchanged, %.2f MB",
             st["requests"], st["not_modified"], st["kb"] / 1024)

    # job-alert emails (Indeed/LinkedIn/...) on their own slower cadence
    global _LAST_EMAIL
    if config.get_bool("JARVIS_JOBS_INGEST_EMAIL"):
        every = max(5, config.get_int("JARVIS_JOBS_EMAIL_INTERVAL_MIN"))
        if time.time() - _LAST_EMAIL >= every * 60:
            _LAST_EMAIL = time.time()
            try:
                er = email_ingest.ingest()
                if er.get("ok"):
                    result["new"] = (result.get("new") or []) + er["new"]
                    if er.get("inserted"):
                        log.info("email alerts: %d new", er["inserted"])
                else:
                    log.debug("email ingest skipped: %s", er.get("error"))
            except Exception:
                log.exception("email ingest failed")

    fresh = [j for j in result.get("new") or [] if j["score"] >= floor]
    fresh.sort(key=lambda j: -j["score"])
    alerted = 0
    for job in fresh[:cap]:
        warn = "⚠ check region" if "⚠" in (job.get("score_notes") or "") else ""
        text = (f"{job['company']} — {job['title']}\n"
                f"{job['location'] or 'location n/a'}  ·  "
                f"score {job['score']:.0f}"
                f"{'  ·  ' + warn if warn else ''}\n"
                f"{job['score_notes']}\n{job['url']}")
        _notify(f"🚨 New role: {job['company']}", text.strip())
        alerted += 1
    if len(fresh) > cap:
        _notify("More new roles",
                f"{len(fresh) - cap} more matching roles are waiting in the "
                "Jobs panel.")

    if alerted:
        log.info("job alert: %d pushed (%d new, %d found)",
                 alerted, len(fresh), result.get("found", 0))
    return {"ok": True, "found": result.get("found", 0),
            "inserted": result.get("inserted", 0), "alerted": alerted}


def install() -> bool:
    """Register the polling job when JARVIS_JOBS_ALERTS=1. Idempotent."""
    if not config.get_bool("JARVIS_JOBS_ALERTS"):
        return False
    minutes = max(5, config.get_int("JARVIS_JOBS_ALERT_INTERVAL_MIN"))
    try:
        scheduler.add_interval("jobs_alerts", minutes * 60, sweep)
    except ValueError as e:
        log.warning("job alerts not scheduled: %s", e)
        return False
    log.info("job alerts every %s min (min score %.0f, max %s/run)",
             minutes, config.get_float("JARVIS_JOBS_ALERT_MIN_SCORE"),
             config.get_int("JARVIS_JOBS_ALERT_MAX_PER_RUN"))
    return True

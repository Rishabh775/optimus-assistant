"""Jobs skill: profile, tracker, matching, discovery, sweep, tools."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from jarvis import jobs
from jarvis.jobs import digest, discovery, match, profile, resume, tracker


def _iso(days_ago: float = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)) \
        .isoformat().replace("+00:00", "Z")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_JOBS_DB", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("JARVIS_JOBS_PROFILE", str(tmp_path / "profile.json"))
    monkeypatch.setenv("JARVIS_JOBS_ENABLED", "0")
    monkeypatch.setenv("JARVIS_JOBS_BOARDS", "greenhouse:acme,ashby:acme2,lever:acme3")
    monkeypatch.setenv("JARVIS_ADZUNA_APP_ID", "")
    monkeypatch.setenv("JARVIS_ADZUNA_APP_KEY", "")


# ------------------------------------------------------------- profile ---

def test_profile_ensure_creates_file():
    p = profile.ensure()
    assert profile.path().exists()
    assert p["remote_ok"] is True and p["ai_disclosure"] is True


def test_profile_update_normalizes_and_validates():
    p = profile.update({"roles": ["Frontend Engineer", "frontend engineer", "React"],
                        "skills": ["React", "TypeScript"], "min_years": "2",
                        "location": "Hyderabad", "email": "a@b.com"})
    assert p["roles"] == ["Frontend Engineer", "React"]      # case-deduped
    assert p["min_years"] == 2
    with pytest.raises(ValueError):
        profile.update({"nope": 1})
    with pytest.raises(ValueError):
        profile.update({"email": "not-an-email"})
    with pytest.raises(ValueError):
        profile.update({"min_years": "lots"})


def test_profile_summary_mentions_state():
    profile.update({"roles": ["frontend"], "location": "Hyderabad"})
    s = profile.summary()
    assert "frontend" in s and "Hyderabad" in s


# ------------------------------------------------------------- tracker ---

def _job(**kw):
    base = {"source": "test", "company": "Acme", "title": "Frontend Engineer",
            "location": "Remote", "url": "https://x/1", "description": "react",
            "score": 80, "score_notes": "title match"}
    base.update(kw)
    return base


def test_tracker_upsert_dedupes_url_and_company_title():
    jid = tracker.upsert(_job())
    assert jid and tracker.get(jid)["status"] == "new"
    assert tracker.upsert(_job()) is None                     # same url
    assert tracker.upsert(_job(url="https://x/2")) is None    # same company+title
    assert len(tracker.list_jobs()) == 1


def test_tracker_status_lifecycle_and_stats():
    jid = tracker.upsert(_job())
    assert tracker.update_status(jid, "applied", "via email") is True
    row = tracker.get(jid)
    assert row["status"] == "applied" and row["applied_at"]
    assert tracker.stats()["by_status"]["applied"] == 1
    with pytest.raises(ValueError):
        tracker.update_status(jid, "banana")
    assert tracker.update_status("nope", "applied") is False
    kinds = [e["kind"] for e in tracker.events(jid)]
    assert "discovered" in kinds and "status:applied" in kinds


# ------------------------------------------------------------- matching --

def _prof(**kw):
    p = {"roles": ["frontend engineer", "react developer"],
         "skills": ["react", "typescript", "javascript", "css"],
         "location": "Hyderabad", "remote_ok": True, "min_years": 2}
    p.update(kw)
    return p


def test_score_strong_vs_weak():
    strong, notes = match.score_job(
        {"title": "Frontend Engineer", "location": "Remote",
         "description": "React TypeScript CSS, remote team", "posted_at": None},
        _prof())
    weak, _ = match.score_job(
        {"title": "Senior Staff Rust Engineer", "location": "Berlin",
         "description": "linux kernel", "posted_at": None},
        _prof())
    assert strong >= 70 and "stack" in notes
    assert weak < 30


def test_score_penalizes_over_senior_and_required_years():
    score, notes = match.score_job(
        {"title": "Senior Frontend Engineer",
         "description": "requires 7+ years react typescript",
         "posted_at": None}, _prof())
    assert "asks 7+ yrs" in notes and score < 80


def test_is_remote_detection():
    assert match.is_remote({"location": "Remote - India"})
    assert match.is_remote({"description": "work from home"})
    assert not match.is_remote({"location": "Hyderabad"})


# ------------------------------------------------------------ discovery --

def _fake_json(url, timeout=None):
    if "boards-api.greenhouse.io" in url and "/jobs/11" in url:
        return {"content": "<p>React TypeScript role</p>"}
    if "boards-api.greenhouse.io" in url:
        return {"jobs": [
            {"id": 11, "title": "Frontend Engineer",
             "location": {"name": "Remote"}, "absolute_url": "https://gh/1",
             "updated_at": _iso(1)},
            {"id": 12, "title": "Backend Engineer",
             "location": {"name": "NYC"}, "absolute_url": "https://gh/2",
             "updated_at": _iso(1)},
        ]}
    if "ashbyhq.com" in url:
        return {"jobs": [
            {"title": "Senior Frontend Engineer", "location": "Remote",
             "jobUrl": "https://ashby/1", "descriptionPlain": "react",
             "publishedAt": _iso(2), "isRemote": True},
            {"title": "Recruiter", "location": "Remote",
             "jobUrl": "https://ashby/2", "descriptionPlain": "",
             "publishedAt": _iso(2), "isRemote": True},
        ]}
    if "api.lever.co" in url:
        return [{"text": "Frontend Developer", "hostedUrl": "https://lever/1",
                 "descriptionPlain": "typescript", "categories":
                 {"location": "Hyderabad"},
                 "createdAt": 1760000000000}]
    if "remotive" in url:
        return {"jobs": [{"title": "Frontend Engineer", "company_name": "RemoteCo",
                          "candidate_required_location": "Worldwide",
                          "url": "https://remotive/1", "description": "react",
                          "publication_date": _iso(1), "salary": ""}]}
    if "adzuna" in url:
        return {"results": [{"title": "Frontend Engineer",
                             "company": {"display_name": "AdzunaCo"},
                             "location": {"display_name": "Hyderabad"},
                             "redirect_url": "https://adzuna/1",
                             "description": "react remote",
                             "created": _iso(3),
                             "salary_min": 1200000, "salary_max": 1800000}]}
    raise AssertionError(f"unexpected URL {url}")


def test_discovery_normalizes_filters_and_dedupes(monkeypatch):
    monkeypatch.setattr(discovery, "_get_json", _fake_json)
    prof = _prof()
    rows = discovery.search_all(prof)
    titles = {j["title"] for j in rows}
    assert "Frontend Engineer" in titles
    assert "Backend Engineer" not in titles        # title filter
    assert "Recruiter" not in titles
    urls = [j["url"] for j in rows]
    assert len(urls) == len(set(urls))
    gh = next(j for j in rows if j["url"] == "https://gh/1")
    assert "React TypeScript" in gh["description"]   # detail enrichment


def test_discovery_filters_excluded_and_stale(monkeypatch):
    monkeypatch.setattr(discovery, "_get_json", _fake_json)
    prof = _prof(exclude_companies=["acme"], min_years=2)
    rows = discovery.search_all(prof)
    assert all("acme" not in j["company"].lower() for j in rows)

    # stale remotive posting is dropped when max age is small
    monkeypatch.setenv("JARVIS_JOBS_MAX_AGE_DAYS", "1")
    old = _iso(10)

    def stale(url, timeout=None):
        if "remotive" in url:
            return {"jobs": [{"title": "Frontend Engineer", "company_name": "Old",
                              "url": "https://old/1", "description": "react",
                              "publication_date": old,
                              "candidate_required_location": "Remote"}]}
        return _fake_json(url)
    monkeypatch.setattr(discovery, "_get_json", stale)
    rows = discovery.search_all(_prof())
    assert all(j["url"] != "https://old/1" for j in rows)


def test_adzuna_salary_formatting(monkeypatch):
    monkeypatch.setattr(discovery, "_get_json",
                        lambda url, timeout=None: {"results": [{
                            "title": "Frontend Engineer",
                            "company": {"display_name": "AdzunaCo"},
                            "location": {"display_name": "Hyderabad"},
                            "redirect_url": "https://a/1",
                            "description": "react",
                            "created": _iso(1),
                            "salary_min": 1200000, "salary_max": 1800000}]})
    rows = discovery.fetch_adzuna("frontend", "in", "id", "key")
    assert rows[0]["salary"] == "₹12.0–18.0 LPA"


# --------------------------------------------------------------- sweep ---

def test_sweep_stores_and_pushes(monkeypatch):
    profile.update({"roles": ["frontend engineer"], "skills": ["react"],
                    "location": "Hyderabad"})
    pushed = []
    monkeypatch.setattr(digest.push, "send",
                        lambda t, m: pushed.append((t, m)) or True)
    monkeypatch.setattr(discovery, "search_all", lambda prof: [
        _job(url="https://x/10", score=0, description="react remote"),
        _job(url="https://x/11", title="Frontend Developer", score=0),
    ])
    res = digest.sweep()
    assert res["ok"] and res["found"] == 2 and res["inserted"] == 2
    assert pushed and "Acme" in pushed[0][1]
    assert len(tracker.list_jobs()) == 2

    # second sweep: nothing new -> no duplicate push
    pushed.clear()
    res2 = digest.sweep()
    assert res2["inserted"] == 0 and pushed == []


def test_sweep_without_roles_is_helpful(monkeypatch):
    res = digest.sweep()
    assert res["ok"] is False and "profile" in res["reason"]


# --------------------------------------------------------------- tools ---

def test_tools_registered_and_work(monkeypatch):
    from jarvis.builtin_tools import TOOL_FUNCS, TOOL_SCHEMAS
    names = {s["name"] for s in TOOL_SCHEMAS}
    for n in ("jobs_profile_get", "jobs_profile_update", "jobs_sweep",
              "jobs_digest", "jobs_list", "job_update"):
        assert n in names and n in TOOL_FUNCS

    assert "error:" in TOOL_FUNCS["jobs_profile_update"](patch={"bogus": 1})
    out = TOOL_FUNCS["jobs_profile_update"](
        patch={"roles": ["frontend"], "skills": ["react"]})
    assert "profile updated" in out
    assert "frontend" in TOOL_FUNCS["jobs_profile_get"]()

    jid = tracker.upsert(_job())
    assert "updated" in TOOL_FUNCS["job_update"](id=jid, status="shortlisted")
    assert "shortlisted" in TOOL_FUNCS["jobs_list"]()
    assert "error:" in TOOL_FUNCS["job_update"](id="nope", status="applied")
    assert "no new roles yet" in TOOL_FUNCS["jobs_digest"]()


# ------------------------------------------------ Bangalore / remote ------


def test_target_locations_bangalore_or_remote():
    p = _prof(job_locations=["Bengaluru"], location="Hyderabad",
              country="India")
    base = {"title": "Frontend Engineer", "company": "X", "description": "react",
            "source": "t", "url": "https://u", "posted_at": None}
    assert discovery._want({**base, "location": "Bangalore, India"}, p)
    assert discovery._want({**base, "location": "Bengaluru, India"}, p)  # alias
    assert discovery._want({**base, "location": "Remote - India"}, p)
    assert not discovery._want({**base, "location": "Pune, India"}, p)
    assert not discovery._want({**base, "location": "Remote - United States"}, p)


def test_profile_job_locations_normalized():
    p = profile.update({"job_locations": ["Bengaluru", "bengaluru", "Remote"]})
    assert p["job_locations"] == ["Bengaluru", "Remote"]


# ------------------------------------------------------------- resume -----


def test_resume_text_ready_and_keywords(tmp_path):
    f = tmp_path / "cv.txt"
    f.write_text("Rishabh Singh\nReact TypeScript GraphQL Playwright\n")
    profile.update({"resume_path": str(f)})
    assert "Rishabh" in resume.text()
    st = resume.ready()
    assert st["exists"] and st["chars"] > 0
    assert {"react", "typescript", "graphql"} <= resume.keywords()


def test_resume_missing_is_reported(tmp_path):
    profile.update({"resume_path": str(tmp_path / "nope.pdf")})
    st = resume.ready()
    assert st["exists"] is False and "not found" in st["error"]
    out = jobs.TOOL_FUNCS["jobs_resume_get"]()
    assert out.startswith("error:")


def test_resume_bonus_affects_score():
    job = {"title": "Frontend Engineer", "location": "Remote",
           "description": "react typescript graphql playwright jest",
           "posted_at": None}
    base, _ = match.score_job(job, _prof())
    boosted, notes = match.score_job(
        job, _prof(), {"react", "typescript", "graphql", "playwright", "jest"})
    assert boosted > base and "resume fit" in notes


def test_agent_prompt_mentions_resume(tmp_path):
    f = tmp_path / "cv.txt"
    f.write_text("react typescript")
    profile.update({"resume_path": str(f)})
    from jarvis.agent import _build_system
    assert "Resume on file" in _build_system("Test", 0)


def test_jobs_job_get_includes_description_and_history():
    jid = tracker.upsert(_job(description="We build React design systems."))
    tracker.update_status(jid, "shortlisted")
    out = jobs.TOOL_FUNCS["jobs_job_get"](id=jid)
    assert "Frontend Engineer" in out and "React design systems" in out
    assert "status: shortlisted" in out and "status:shortlisted" in out
    assert jobs.TOOL_FUNCS["jobs_job_get"](id="nope").startswith("error:")

"""Resume access: the single source the agent compares jobs against.

Points at profile.resume_path (a stable copy lives in data/jobs/resume.pdf).
Text extraction is local and dependency-light:
  .pdf  -> pdftotext (poppler)
  .docx -> python-docx when installed
  .txt/.md -> read directly
Nothing is uploaded; the text is only fed to the local matcher and to the
agent when it explicitly reads the resume (jobs_resume_get).
"""
import re
import subprocess
from pathlib import Path

from .. import config
from . import profile

_CACHE = {"key": None, "text": ""}
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "using", "into",
    "across", "over", "your", "you", "our", "their", "are", "was", "were",
    "will", "can", "has", "have", "had", "not", "but", "all", "any", "per",
    "about", "also", "who", "what", "when", "where", "which", "while",
    "working", "work", "built", "build", "team", "teams", "role", "roles",
    "experience", "experienced", "years", "year", "including", "within",
}


def path() -> Path | None:
    raw = (profile.load().get("resume_path") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / raw
    return p


def _extract(p: Path) -> str:
    suffix = p.suffix.lower()
    if suffix in (".txt", ".md"):
        return p.read_text(errors="replace")
    if suffix == ".pdf":
        out = subprocess.run(["pdftotext", str(p), "-"],
                             capture_output=True, timeout=30)
        if out.returncode != 0:
            raise RuntimeError("pdftotext failed (is poppler installed?)")
        return out.stdout.decode("utf-8", "replace")
    if suffix in (".docx", ".doc"):
        try:
            import docx
        except ImportError:
            raise RuntimeError("python-docx not installed (pip install python-docx)")
        d = docx.Document(str(p))
        return "\n".join(par.text for par in d.paragraphs)
    raise RuntimeError(f"unsupported resume type {suffix!r}")


def text(max_chars: int = 12000) -> str:
    """Return cleaned resume text. Raises RuntimeError when unavailable."""
    p = path()
    if not p:
        raise RuntimeError("no resume_path set in the jobs profile")
    if not p.is_file():
        raise RuntimeError(f"resume not found at {p}")
    key = f"{p}:{p.stat().st_mtime_ns}"
    if _CACHE["key"] != key:
        cleaned = re.sub(r"[ \t]+", " ", _extract(p))
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        _CACHE.update(key=key, text=cleaned)
    return _CACHE["text"][:max_chars]


def text_or_empty() -> str:
    try:
        return text()
    except RuntimeError:
        return ""


def ready() -> dict:
    """Small status dict for tools/UI: {path, exists, chars, error}."""
    p = path()
    info = {"path": str(p) if p else "", "exists": False, "chars": 0,
            "error": ""}
    if not p:
        info["error"] = "no resume set"
        return info
    try:
        info["chars"] = len(text())
        info["exists"] = True
    except RuntimeError as e:
        info["error"] = str(e)
    return info


def keywords(limit: int = 40) -> set:
    """Distinctive resume terms, used for job-vs-resume overlap scoring."""
    body = text_or_empty().lower()
    if not body:
        return set()
    counts = {}
    for tok in re.findall(r"[a-z][a-z0-9+#.]{2,}", body):
        if tok in _STOP:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    return set(ranked[:limit])

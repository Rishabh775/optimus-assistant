"""Image understanding for photos/screenshots uploaded from the phone.

The text models Soundwave uses cannot see images, so this module sends the
picture to a vision-capable model on the SAME OpenCode gateway (cheap tier
by default) and returns a description + extracted text.

Design notes:
- local-first files: the image never leaves the machine except for this
  explicit analysis call, and only when the agent/user asks
- model fallback chain: configured model -> claude-haiku-5-5 -> gemini-4-argon
- hard limits: allowed extensions, size cap, path must be a readable file
"""
import base64
import os
import uuid
from pathlib import Path

from . import config
from .log import log as _mk_log

log = _mk_log("vision")

_EXT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
}
_CLIENT = None
_SESSION = str(uuid.uuid4())
_CANDIDATES = ("gpt-6-luna", "claude-haiku-5-5", "gemini-4-argon")


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        key = os.getenv("OPENCODE_API_KEY")
        if not key:
            raise RuntimeError("OPENCODE_API_KEY is not set")
        _CLIENT = OpenAI(
            base_url="https://opencode.ai/zen/go/v1", api_key=key,
            default_headers={
                "User-Agent": "jarvis-personal-assistant/0.1",
                "x-opencode-session": _SESSION,
            },
            timeout=90, max_retries=0)
    return _CLIENT


def _resolve(path: str) -> Path:
    p = Path(str(path or "")).expanduser()
    if p.is_absolute():
        return p
    repo = Path(__file__).resolve().parents[1]
    for base in (Path.cwd(), repo, Path.home() / "jarvis" / "workspace"):
        cand = base / p
        if cand.is_file():
            return cand
    return Path.home() / "jarvis" / "workspace" / p


def validate(path: str) -> Path:
    p = _resolve(path)
    if not p.is_file():
        raise ValueError(f"image not found: {p}")
    mime = _EXT_MIME.get(p.suffix.lower())
    if not mime:
        raise ValueError(f"unsupported image type {p.suffix!r} "
                         "(png/jpg/jpeg/webp/gif)")
    mb = config.get_float("JARVIS_VISION_MAX_MB") or 12
    if p.stat().st_size > mb * 1024 * 1024:
        raise ValueError(f"image is larger than {mb:.0f} MB")
    return p


def data_uri(path: str) -> str:
    p = validate(path)
    raw = p.read_bytes()
    mime = _EXT_MIME[p.suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _ask(model: str, uri: str, prompt: str) -> str:
    r = _client().chat.completions.create(
        model=model, max_tokens=700,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}])
    return (r.choices[0].message.content or "").strip()


def analyze(path: str, prompt: str = "") -> str:
    """Describe an image and extract its text. Raises ValueError on bad input."""
    p = validate(path)
    uri = data_uri(str(p))
    question = (prompt or "").strip() or (
        "Describe this image concisely (what it shows, important details). "
        "If it contains text, probably a screenshot or document, transcribe "
        "the text faithfully and keep the layout intelligible.")
    configured = config.get("JARVIS_VISION_MODEL") or _CANDIDATES[0]
    models = [configured] + [m for m in _CANDIDATES if m != configured]
    errors = []
    for model in models:
        try:
            answer = _ask(model, uri, question)
            if answer:
                return (f"image: {p.name} ({p.stat().st_size // 1024} KB, "
                        f"model {model})\n\n{answer}")
            errors.append(f"{model}: empty response")
        except Exception as e:
            errors.append(f"{model}: {str(e)[:120]}")
            log.warning("vision model %s failed: %s", model, e)
    raise RuntimeError("vision analysis failed — " + "; ".join(errors[:3]))

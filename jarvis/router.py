"""Hybrid router: Ollama (local) + opencode (cloud).

OpenCode requires:
  - custom user agent
  - x-opencode-session header (stable per conversation)
  - chat/completions endpoint (only this endpoint is implemented here)

Routing is decided on the ORIGINAL user prompt (agent passes route_text),
never on mid-loop tool results. Keyword matching uses word boundaries so
"capital" no longer routes as "api".
"""
import re
import socket
import time
import uuid
from dataclasses import dataclass
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CLOUD_MODEL = "longcat-2.5-preview-free"
DEFAULT_CODE_MODEL = "kimi-k2.7-code"
DEFAULT_LOCAL_MODEL = "llama3.2:3b"  # 7B is too slow on 4-core CPUs
# If the primary cloud model fails (free previews can vanish), retry once
# with the cheapest solid paid model instead of erroring the conversation.
FALLBACK_CLOUD_MODEL = os.getenv("JARVIS_CLOUD_FALLBACK", "mimo-v2.6-flash")

# Circuit breaker: when the primary cloud model fails, route straight to the
# fallback for this many seconds instead of re-hitting a dead model (which
# otherwise costs a full timeout on every single call).
CLOUD_COOLDOWN = float(os.getenv("JARVIS_CLOUD_COOLDOWN", "90"))
_cloud_down_until = 0.0

TIMEOUT_S = float(os.getenv("JARVIS_LLM_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("JARVIS_LLM_RETRIES", "2"))

# Only chat/completions models are supported by this client. If a configured
# model isn't in this set we warn once and fall back (responses/anthropic
# endpoints are NOT implemented here).
CHAT_COMPLETIONS_MODELS = {
    "glm-5.3-flash", "glm-5.3", "glm-5.2", "glm-5.1", "kimi-k3", "kimi-k2.7-code",
    "kimi-k2.6", "longcat-2.0", "longcat-2.5-preview-free", "space-bunny-free",
    "deepseek-v4.1-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-flash",
    "deepseek-v4-flash-vision-exp", "mimo-v2.6-flash", "mimo-v2.6-pro",
    "mimo-v2.5", "mimo-v2.5-pro", "qwen3.8-max", "qwen3.8-flash",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "hy4-preview", "hy3",
}


@dataclass
class Route:
    provider: str
    model: str
    reason: str


CODE_KEYWORDS = {"code", "function", "class", "bug", "error", "compile",
                 "refactor", "implement", "script", "regex", "api"}
HARD_KEYWORDS = {"plan", "analyze", "compare", "design", "architecture",
                 "tradeoff", "explain", "reasoning"}
HARD_PHRASES = ("explain why", "think through")
ACTION_KEYWORDS = {"list", "show", "read", "open", "run", "execute", "search",
                    "find", "create", "delete", "move", "copy", "install",
                    "check", "get", "fetch", "download", "upload", "send",
                    "write", "edit", "rename", "git", "docker", "npm", "pip"}

_WARNED_MODELS = set()


def _has_keyword(text: str, keywords: set) -> bool:
    """Word-boundary keyword match (substring match mis-routed badly)."""
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def needs_tools(prompt: str) -> bool:
    p = prompt.lower()
    if _has_keyword(p, ACTION_KEYWORDS):
        return True
    return "?" in p and any(v in p for v in ("how do i", "can you", "please"))


def _supported(model: str) -> str:
    if model in CHAT_COMPLETIONS_MODELS:
        return model
    if model not in _WARNED_MODELS:
        print(f"[router] model '{model}' not on the chat/completions endpoint;"
              f" falling back to {DEFAULT_CLOUD_MODEL}")
        _WARNED_MODELS.add(model)
    return DEFAULT_CLOUD_MODEL


def pick_route(prompt: str) -> Route:
    p = prompt.lower()
    # Local Ollama is OFF by default (set JARVIS_LOCAL_OK=1 to enable)
    prefer_local = os.getenv("JARVIS_LOCAL_OK", "0") == "1"
    # Code intent is checked BEFORE generic action verbs so "write a
    # function" goes to the code model, not the generic cloud model.
    if _has_keyword(p, CODE_KEYWORDS):
        return Route("cloud",
                     _supported(os.getenv("JARVIS_CODE_MODEL", DEFAULT_CODE_MODEL)),
                     "code task -> cloud code model")
    if needs_tools(prompt):
        return Route("cloud",
                     _supported(os.getenv("JARVIS_CLOUD_MODEL", DEFAULT_CLOUD_MODEL)),
                     "action/tool task -> cloud")
    if _has_keyword(p, HARD_KEYWORDS) or any(s in p for s in HARD_PHRASES) \
            or len(prompt) > 800:
        return Route("cloud",
                     _supported(os.getenv("JARVIS_CLOUD_MODEL", DEFAULT_CLOUD_MODEL)),
                     "complex task -> cloud smart model")
    if prefer_local:
        return Route("local",
                     os.getenv("JARVIS_LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
                     "simple task -> local ollama")
    return Route("cloud",
                 _supported(os.getenv("JARVIS_CLOUD_MODEL", DEFAULT_CLOUD_MODEL)),
                 "default -> cloud")


def cloud_status() -> str:
    """Human-readable state of the primary cloud model (for self_status)."""
    if not os.getenv("OPENCODE_API_KEY"):
        return "no api key"
    if time.time() < _cloud_down_until:
        return f"cooling down {int(_cloud_down_until - time.time())}s (using fallback)"
    return "ok"


def _ollama_up(timeout: float = 1.0) -> bool:
    """Cheap reachability check for the Ollama daemon (cached)."""
    global _OLLAMA_UP
    try:
        return _OLLAMA_UP
    except NameError:
        pass
    try:
        with socket.create_connection(("localhost", 11434), timeout=timeout):
            _OLLAMA_UP = True
    except OSError:
        _OLLAMA_UP = False
    return _OLLAMA_UP


class LLM:
    def __init__(self):
        self.ollama = OpenAI(base_url="http://localhost:11434/v1",
                             api_key="ollama",
                             timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
        self.session_id = str(uuid.uuid4())
        self.cloud = None

    def chat(self, messages, tools=None, force: Route = None,
             route_text: str = None):
        """route_text: the original user prompt -- routing decision input."""
        global _cloud_down_until
        text = route_text if route_text is not None else (
            messages[-1]["content"] if messages else "")
        route = force or pick_route(text)
        if route.provider == "local" and not _ollama_up():
            route = Route("local", "", "ollama daemon not running")
            raise RuntimeError(
                "Local route selected but Ollama is not running on "
                "localhost:11434. Start it with: ollama serve")
        if route.provider == "local":
            client = self.ollama
        else:
            client = self._cloud_client()
            if client is None:
                route = Route("local",
                             os.getenv("JARVIS_LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
                             "no cloud key -> falling back to local")
                if not _ollama_up():
                    raise RuntimeError(
                        "No OPENCODE_API_KEY and Ollama is not running "
                        "(localhost:11434). Configure .env or run: ollama serve")
                client = self.ollama
        kwargs = {"model": route.model, "messages": messages}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if (route.provider == "cloud" and time.time() < _cloud_down_until
                and route.model != FALLBACK_CLOUD_MODEL):
            fb = _supported(FALLBACK_CLOUD_MODEL)
            route = Route("cloud", fb, "primary cooling down -> fallback")
            kwargs["model"] = fb
        try:
            resp = client.chat.completions.create(**kwargs)
            if route.provider == "cloud" and route.model != FALLBACK_CLOUD_MODEL:
                _cloud_down_until = 0.0   # primary healthy again
        except Exception as e:
            if (route.provider == "cloud"
                    and route.model != FALLBACK_CLOUD_MODEL):
                _cloud_down_until = time.time() + CLOUD_COOLDOWN
                fb = _supported(FALLBACK_CLOUD_MODEL)
                print(f"[router] '{route.model}' failed ({type(e).__name__}); "
                      f"cooling down {int(CLOUD_COOLDOWN)}s, retrying with '{fb}'")
                route = Route("cloud", fb, f"fallback from {route.model}")
                kwargs["model"] = fb
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        return resp, route

    def _cloud_client(self):
        if self.cloud is None:
            key = os.getenv("OPENCODE_API_KEY")
            if not key:
                return None
            headers = {
                "User-Agent": "jarvis-personal-assistant/0.1",
                "x-opencode-session": self.session_id,
            }
            self.cloud = OpenAI(base_url="https://opencode.ai/zen/go/v1",
                                api_key=key, default_headers=headers,
                                timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
        return self.cloud

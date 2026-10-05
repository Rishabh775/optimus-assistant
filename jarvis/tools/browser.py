"""Browser control via Playwright (Chromium)."""
import os
from pathlib import Path

CACHE = Path.home() / "jarvis" / "data" / "browser_cache"
CACHE.mkdir(parents=True, exist_ok=True)

_BROWSER = None
_PAGE = None


def _ensure_browser():
    global _BROWSER, _PAGE
    if _BROWSER is None:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        _BROWSER = pw.chromium.launch(headless=True)
        _PAGE = _BROWSER.new_page()
    return _PAGE


def web_open(url: str) -> str:
    page = _ensure_browser()
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        title = page.title()
        return f"OPENED: {title} ({url})"
    except Exception as e:
        return f"ERROR opening {url}: {e}"


def web_extract(url: str = None, max_chars: int = 4000) -> str:
    """Get text content from current page or a new URL."""
    page = _ensure_browser()
    if url:
        try:
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
        except Exception as e:
            return f"ERROR: {e}"
    text = page.inner_text("body")
    return text[:max_chars]


def web_screenshot(path: str = None) -> str:
    page = _ensure_browser()
    if not path:
        path = str(CACHE / f"shot-{os.getpid()}.png")
    page.screenshot(path=path, full_page=False)
    return f"SCREENSHOT: {path}"


def web_search(query: str, n: int = 5) -> str:
    """DuckDuckGo HTML search, returns result snippets via browser."""
    from duckduckgo_search import DDGS
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n))
        return "\n".join(f"- {r['title']}\n  {r['href']}\n  {r['body'][:150]}"
                         for r in results)
    except Exception as e:
        return f"ERROR: {e}"
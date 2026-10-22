"""Browser control via Playwright (Chromium).

Thread-safety: Playwright's sync API requires every call to happen on the
thread that created the browser -- but Optimus invokes tools from different
threads (hotkey mode spawns a fresh thread per key press). So the whole browser
lives in ONE dedicated worker thread and the public web_* functions post a
command to its queue and wait for the result. Callers can be on any thread.
"""
import os
import queue
import threading
from pathlib import Path

CACHE = Path.home() / "jarvis" / "data" / "browser_cache"
CACHE.mkdir(parents=True, exist_ok=True)

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_COMMAND_TIMEOUT = 60          # seconds; a stuck command never hangs forever


def _find(page, target: str):
    """Resolve a human/LLM target to a locator, trying in order:
    CSS selector, link/button by accessible name, placeholder/label (inputs),
    then any element containing the visible text. Returns a Locator or None."""
    if not target:
        return None
    probes = [
        lambda: page.locator(target),
        lambda: page.get_by_role("link", name=target, exact=False),
        lambda: page.get_by_role("button", name=target, exact=False),
        lambda: page.get_by_placeholder(target, exact=False),
        lambda: page.get_by_label(target, exact=False),
        lambda: page.get_by_text(target, exact=False),
    ]
    for probe in probes:
        try:
            loc = probe()
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


def _page_desc(page) -> str:
    try:
        return f"{page.title()} ({page.url})"
    except Exception:
        return page.url or "(unknown page)"


class _BrowserWorker:
    """Owns the Playwright objects; every command runs on its own thread."""

    def __init__(self):
        self._q = queue.Queue()
        self._thread = None
        self._pw = None
        self._browser = None
        self._page = None

    # ---- worker-thread only ----
    def _start(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page(
            user_agent=_UA, viewport={"width": 1280, "height": 900})

    def _page_or_start(self):
        if self._page is None:
            self._start()
        return self._page

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:            # shutdown sentinel
                break
            fn, args, kwargs, ev, box = item
            try:
                box["result"] = fn(*args, **kwargs)
            except Exception as e:
                box["result"] = f"ERROR: {e}"
            finally:
                ev.set()
        self._shutdown()

    def _shutdown(self):
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._browser = self._pw = self._page = None

    # ---- dispatch (callable from any thread) ----
    def submit(self, fn, *args, **kwargs):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="playwright-worker")
            self._thread.start()
        ev, box = threading.Event(), {}
        self._q.put((fn, args, kwargs, ev, box))
        if not ev.wait(timeout=_COMMAND_TIMEOUT):
            return "ERROR: browser worker timed out"
        return box.get("result", "ERROR: no result")

    # ---- commands (run on the worker thread via submit) ----
    def _open(self, url):
        page = self._page_or_start()
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        return f"OPENED: {page.title()} ({url})"

    def _extract(self, url=None, max_chars=4000):
        page = self._page_or_start()
        if url:
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
        return page.inner_text("body")[:max_chars]

    def _screenshot(self, path=None):
        page = self._page_or_start()
        if not path:
            path = str(CACHE / f"shot-{os.getpid()}.png")
        page.screenshot(path=path, full_page=False)
        return f"SCREENSHOT: {path}"

    def _click(self, target):
        page = self._page_or_start()
        loc = _find(page, target)
        if loc is None:
            return f"ERROR: no element matching {target!r} (try browser_links)"
        loc.scroll_into_view_if_needed(timeout=3000)
        loc.click(timeout=8000)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        return f"CLICKED {target!r} -> {_page_desc(page)}"

    def _type(self, selector, text, submit=False):
        page = self._page_or_start()
        loc = _find(page, selector)
        if loc is None:
            return f"ERROR: no input matching {selector!r}"
        loc.click(timeout=5000)
        try:
            loc.fill(text, timeout=5000)
        except Exception:                 # contenteditable / rich text
            loc.type(text, timeout=5000)
        if submit:
            loc.press("Enter")
            try:
                page.wait_for_load_state("domcontentloaded", timeout=8000)
            except Exception:
                pass
        return (f"TYPED into {selector!r}" + (" and submitted" if submit else "")
                + f" -> {_page_desc(page)}")

    def _press(self, key):
        page = self._page_or_start()
        page.keyboard.press(key)
        return f"PRESSED {key} -> {_page_desc(page)}"

    def _back(self):
        page = self._page_or_start()
        page.go_back(timeout=8000)
        return f"BACK -> {_page_desc(page)}"

    def _links(self, max_items=40):
        page = self._page_or_start()
        items = []
        for a in page.query_selector_all("a[href]"):
            t = (a.inner_text() or "").strip().replace("\n", " ")
            h = a.get_attribute("href") or ""
            if t and h and not h.lower().startswith("javascript"):
                items.append(f"- {t[:70]} -> {h[:120]}")
            if len(items) >= max_items:
                break
        return "\n".join(items) if items else "(no links found)"

    def _wait(self, selector="", ms=1000):
        page = self._page_or_start()
        if selector:
            try:
                page.wait_for_selector(selector, timeout=max(ms, 1000))
                return f"READY: {selector}"
            except Exception as e:
                return f"TIMEOUT waiting for {selector}: {e}"
        page.wait_for_timeout(min(max(ms, 0), 15000))
        return f"WAITED {ms}ms"

    def _scroll(self, direction="down", amount=800):
        page = self._page_or_start()
        dy = amount if str(direction).lower().startswith("d") else -amount
        page.mouse.wheel(0, dy)
        return f"SCROLLED {direction} {amount}px"

    def _close(self):
        self._shutdown()
        return "BROWSER CLOSED"


_WORKER = _BrowserWorker()


def web_open(url: str) -> str:
    return _WORKER.submit(_WORKER._open, url)


def web_extract(url: str = None, max_chars: int = 4000) -> str:
    """Get text content from current page or a new URL."""
    return _WORKER.submit(_WORKER._extract, url, max_chars)


def web_screenshot(path: str = None) -> str:
    return _WORKER.submit(_WORKER._screenshot, path)


def web_click(target: str) -> str:
    """Click a link/button/element by CSS selector, visible text, or name."""
    return _WORKER.submit(_WORKER._click, target)


def web_type(selector: str, text: str, submit: bool = False) -> str:
    """Fill an input (by selector, placeholder, or label). Optionally submit."""
    return _WORKER.submit(_WORKER._type, selector, text, submit)


def web_press(key: str) -> str:
    """Send a keyboard key to the page (Enter, Tab, Escape, PageDown, ...)."""
    return _WORKER.submit(_WORKER._press, key)


def web_back() -> str:
    return _WORKER.submit(_WORKER._back)


def web_links(max_items: int = 40) -> str:
    """List clickable links (text + href) so the agent knows what it can click."""
    return _WORKER.submit(_WORKER._links, max_items)


def web_wait(selector: str = "", ms: int = 1000) -> str:
    """Wait for a selector to appear, or just wait `ms` (for slow pages)."""
    return _WORKER.submit(_WORKER._wait, selector, ms)


def web_scroll(direction: str = "down", amount: int = 800) -> str:
    return _WORKER.submit(_WORKER._scroll, direction, amount)


def web_close() -> str:
    """Close the browser (frees memory; next call relaunches)."""
    return _WORKER.submit(_WORKER._close)


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

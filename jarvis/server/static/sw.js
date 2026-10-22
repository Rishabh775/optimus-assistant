// Soundwave service worker.
// Strategy:
//   - vendor assets (marked/purify/highlight + icons): cache-first (big, static)
//   - HTML shell ("/"): network-first, fall back to cache offline  -> updates
//     always arrive immediately, avoiding the old stale-login bug
//   - API + WebSocket: never intercepted (always live)
const STATIC_CACHE = "soundwave-static-v1";
const SHELL_CACHE = "soundwave-shell-v1";

const STATIC_ASSETS = [
  "/static/vendor/marked.min.js",
  "/static/vendor/purify.min.js",
  "/static/vendor/highlight.min.js",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/manifest.webmanifest"
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE).then((c) => c.addAll(STATIC_ASSETS)),
      caches.open(SHELL_CACHE).then((c) => c.add("/"))
    ]).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== STATIC_CACHE && k !== SHELL_CACHE)
            .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;                          // skip POST etc.
  if (url.pathname === "/ws" || url.pathname.startsWith("/api/")) return; // always live

  if (url.pathname.startsWith("/static/vendor/") ||
      url.pathname.startsWith("/static/icons/") ||
      url.pathname === "/static/manifest.webmanifest") {
    // cache-first for immutable-ish static assets
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
        const copy = res.clone();
        if (res.ok) caches.open(STATIC_CACHE).then((c) => c.put(e.request, copy));
        return res;
      }))
    );
    return;
  }

  // HTML shell (and anything else GET): network-first, offline fallback
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request).then((hit) => hit || caches.match("/")))
  );
});

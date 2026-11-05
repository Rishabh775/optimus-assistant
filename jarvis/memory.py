"""Persistent memory with Chroma vector store.

Improvements over naive version:
- query() only returns memories within a similarity distance threshold
  (before: top-4 came back no matter how irrelevant -> context pollution)
- add() prunes to a retention cap (oldest dropped)
- add() skips trivial/empty exchanges

When Chroma / sentence-transformers are unavailable (or
JARVIS_MEMORY_BACKEND=simple), a small JSONL backend keeps conversation
memory working instead of failing the whole agent.
"""
import json
import os
import re
import time
from pathlib import Path

from . import config

DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", Path.home() / "jarvis" / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_DIST = config.get_float("JARVIS_MEMORY_MAX_DIST")   # cosine distance
RETENTION = config.get_int("JARVIS_MEMORY_MAX_ENTRIES")
MIN_CHARS = int(os.getenv("JARVIS_MEMORY_MIN_CHARS", "20"))
BACKEND = os.getenv("JARVIS_MEMORY_BACKEND", "auto").strip().lower()

_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when",
    "where", "which", "who", "how", "why", "are", "was", "were", "you",
    "your", "our", "their", "his", "her", "its", "have", "has", "had",
    "can", "could", "would", "should", "will", "just", "about", "into",
    "over", "under", "please", "tell", "give", "show", "get", "got",
}


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in _STOP}


class _SimpleMemory:
    """JSONL fallback: keyword-overlap retrieval, newest first."""

    def __init__(self, name: str):
        self.path = DATA_DIR / f"memory-{name}.jsonl"

    def _rows(self) -> list:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("text"):
                rows.append(row)
        return rows

    def _prune(self, rows: list) -> None:
        if len(rows) > RETENTION:
            del rows[:-RETENTION]

    def add(self, text: str, kind: str = "note") -> str:
        text = (text or "").strip()
        if len(text) < MIN_CHARS:
            return ""
        rows = self._rows()
        rows.append({"text": text, "kind": kind, "ts": time.time()})
        self._prune(rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return f"simple-{int(time.time() * 1000)}"

    def query(self, text: str, n: int = 4) -> str:
        wanted = _tokens(text)
        if not wanted:
            return ""
        scored = []
        for row in self._rows():
            got = _tokens(row.get("text", ""))
            if not got:
                continue
            score = len(wanted & got) / len(wanted)
            if score > 0:
                scored.append((score, row.get("ts", 0), row["text"]))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return "\n".join(f"- {t}" for _s, _ts, t in scored[:n])

    def count(self) -> int:
        return len(self._rows())


class Memory:
    def __init__(self, name: str = "jarvis"):
        self.col = None
        self.simple = None
        if BACKEND != "simple":
            try:
                import chromadb
                from chromadb.utils import embedding_functions
                self.client = chromadb.PersistentClient(
                    path=str(DATA_DIR / "chroma"))
                self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                self.col = self.client.get_or_create_collection(
                    name=name, embedding_function=self.ef,
                    metadata={"hnsw:space": "cosine"})
            except Exception:
                self.col = None
        if self.col is None:
            self.simple = _SimpleMemory(name)

    def add(self, text: str, kind: str = "note") -> str:
        if self.simple is not None:
            return self.simple.add(text, kind)
        text = (text or "").strip()
        if len(text) < MIN_CHARS:
            return ""
        mid = f"{kind}-{int(time.time() * 1000)}"
        self.col.add(documents=[text],
                     metadatas=[{"kind": kind, "ts": time.time()}],
                     ids=[mid])
        self._prune()
        return mid

    def _prune(self):
        extra = self.col.count() - RETENTION
        if extra <= 0:
            return
        got = self.col.get(include=["metadatas"], limit=10_000)
        items = sorted(zip(got["ids"],
                            [m.get("ts", 0) for m in got["metadatas"]],
                            range(len(got["ids"]))),
                       key=lambda x: x[1])  # oldest first
        self.col.delete(ids=[i for i, _ts, _r in items[:extra]])

    def query(self, text: str, n: int = 4) -> str:
        if self.simple is not None:
            return self.simple.query(text, n)
        if self.col.count() == 0:
            return ""
        res = self.col.query(query_texts=[text],
                             n_results=min(n, self.col.count()),
                             include=["documents", "distances"])
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0] or [1.0] * len(docs)
        keep = [d for d, dist in zip(docs, dists) if dist <= MAX_DIST]
        return "\n".join(f"- {d}" for d in keep)

    def count(self) -> int:
        if self.simple is not None:
            return self.simple.count()
        return self.col.count()

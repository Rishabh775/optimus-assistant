"""Persistent memory with Chroma vector store.

Improvements over naive version:
- query() only returns memories within a similarity distance threshold
  (before: top-4 came back no matter how irrelevant -> context pollution)
- add() prunes to a retention cap (oldest dropped)
- add() skips trivial/empty exchanges
"""
import os
import time
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions


DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", Path.home() / "jarvis" / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_DIST = float(os.getenv("JARVIS_MEMORY_MAX_DIST", "0.55"))   # cosine distance
RETENTION = int(os.getenv("JARVIS_MEMORY_MAX_ENTRIES", "500"))
MIN_CHARS = int(os.getenv("JARVIS_MEMORY_MIN_CHARS", "20"))


class Memory:
    def __init__(self, name: str = "jarvis"):
        self.client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.col = self.client.get_or_create_collection(
            name=name, embedding_function=self.ef, metadata={"hnsw:space": "cosine"}
        )

    def add(self, text: str, kind: str = "note") -> str:
        text = (text or "").strip()
        if len(text) < MIN_CHARS:
            return ""
        mid = f"{kind}-{int(time.time() * 1000)}"
        self.col.add(documents=[text], metadatas=[{"kind": kind, "ts": time.time()}],
                     ids=[mid])
        self._prune()
        return mid

    def _prune(self):
        extra = self.col.count() - RETENTION
        if extra <= 0:
            return
        got = self.col.get(include=["metadatas"], limit=10_000)
        items = sorted(zip(got["ids"], [m.get("ts", 0) for m in got["metadatas"]],
                            range(len(got["ids"]))),
                       key=lambda x: x[1])  # oldest first
        self.col.delete(ids=[i for i, _ts, _r in items[:extra]])

    def query(self, text: str, n: int = 4) -> str:
        if self.col.count() == 0:
            return ""
        res = self.col.query(query_texts=[text], n_results=min(n, self.col.count()),
                             include=["documents", "distances"])
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0] or [1.0] * len(docs)
        keep = [d for d, dist in zip(docs, dists) if dist <= MAX_DIST]
        return "\n".join(f"- {d}" for d in keep)

    def count(self) -> int:
        return self.col.count()

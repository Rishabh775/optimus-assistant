"""Durable user facts (identity, preferences) that persist across sessions.

Stored as JSONL in JARVIS_DATA_DIR/facts.jsonl. Injected into the agent's
context every turn and writable through the remember_fact tool. This is the
"durable layer" of memory (layered-memory pattern from Leon / Oleksii's
assistant): small, explicit, always-considered facts, separate from the
similarity-searched conversation memory in memory.py.
"""
import json
import os
import time
from pathlib import Path

DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", Path.home() / "jarvis" / "data"))
FACTS = DATA_DIR / "facts.jsonl"
MAX = int(os.getenv("JARVIS_FACTS_MAX", "100"))


def _load() -> list:
    if not FACTS.exists():
        return []
    out = []
    for line in FACTS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line)["fact"])
        except Exception:
            pass
    return out


def add(fact: str) -> str:
    fact = " ".join((fact or "").split())
    if not fact:
        return "nothing to remember"
    if any(fact.lower() == e.lower() for e in _load()):
        return f"already known: {fact}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FACTS.open("a") as f:
        f.write(json.dumps({"t": time.time(), "fact": fact}) + "\n")
    return f"remembered: {fact}"


def forget(fact: str) -> str:
    """Remove facts containing the given substring (case-insensitive)."""
    if not FACTS.exists():
        return "no facts stored"
    keep, removed = [], 0
    for line in FACTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            f = json.loads(line)["fact"]
        except Exception:
            continue
        if fact and fact.lower() in f.lower():
            removed += 1
        else:
            keep.append(line)
    FACTS.write_text("\n".join(keep) + ("\n" if keep else ""))
    return f"forgot {removed} fact(s)"


def all_facts(limit: int = None) -> list:
    facts = _load()
    return facts[-limit:] if limit else facts


def text(limit: int = None) -> str:
    return "\n".join(f"- {f}" for f in all_facts(limit))

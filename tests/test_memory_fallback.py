"""JSONL fallback memory used when Chroma is unavailable."""
import json

from jarvis import memory as mem


def test_simple_memory_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mem, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mem, "BACKEND", "simple")
    m = mem.Memory("test")
    assert m.simple is not None
    m.add("my favourite colour is blue and I like tea")
    m.add("the deployment runs on a small linux server")
    assert m.count() == 2
    assert "blue" in m.query("what colour do I like")
    assert "linux" in m.query("where is the deployment")


def test_simple_memory_prunes(tmp_path, monkeypatch):
    monkeypatch.setattr(mem, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mem, "RETENTION", 2)
    sm = mem._SimpleMemory("prune")
    sm.add("first memory about apples and oranges")
    sm.add("second memory about cars and trains")
    sm.add("third memory about boats and planes")
    rows = [json.loads(line) for line in sm.path.read_text().splitlines()]
    assert len(rows) == 2
    assert "third" in rows[-1]["text"]

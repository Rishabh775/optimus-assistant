"""Tests for the phone-app chat store (jarvis/server/store.py)."""
import pytest

from jarvis.server import store


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_APP_DB", str(tmp_path / "chats.db"))


def test_new_and_list_sessions():
    s1 = store.new_session("first")
    s2 = store.new_session()
    assert s1["id"] and s2["id"] and s1["id"] != s2["id"]
    rows = store.list_sessions()  # newest updated_at first
    assert [r["id"] for r in rows] == [s2["id"], s1["id"]]
    assert rows[1]["title"] == "first"


def test_messages_roundtrip_and_latest_order():
    s = store.new_session("chat")
    store.add_message(s["id"], "user", "hi there")
    store.add_message(s["id"], "assistant", "hello!", meta={"route": "cloud"})
    msgs = store.get_messages(s["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hi there"
    # listSessions preview = newest message
    assert store.list_sessions()[0]["last"] == "hello!"


def test_touch_session_updates_updated_at():
    s = store.new_session("t")
    store.add_message(s["id"], "user", "x")
    row = store.list_sessions()[0]
    assert row["updated_at"] >= s["updated_at"]


def test_rename_and_update_fields():
    s = store.new_session()
    store.rename_session(s["id"], "renamed")
    store.update_session(s["id"], pinned=True)
    row = store.list_sessions()[0]
    assert row["title"] == "renamed"
    assert row["pinned"] is True


def test_update_session_rejects_bad_fields():
    s = store.new_session()
    with pytest.raises(ValueError):
        store.update_session(s["id"], nonexistent=1)


def test_delete_session_cascades():
    s = store.new_session()
    store.add_message(s["id"], "user", "hi")
    assert store.delete_session(s["id"]) is True
    assert store.session_exists(s["id"]) is False
    assert store.get_messages(s["id"]) == []
    assert store.delete_session(s["id"]) is False


def test_get_messages_limit():
    s = store.new_session()
    for i in range(5):
        store.add_message(s["id"], "user", f"m{i}")
    msgs = store.get_messages(s["id"], limit=2)
    assert [m["content"] for m in msgs] == ["m3", "m4"]


def test_prune_empty_sessions_only_drops_unused_old_ones():
    used = store.new_session("used")
    store.add_message(used["id"], "user", "hello")
    old_empty = store.new_session()
    recent_empty = store.new_session()

    import sqlite3
    from jarvis.server.store import db_path
    conn = sqlite3.connect(db_path())
    with conn:
        conn.execute("UPDATE sessions SET created_at=created_at-7200 WHERE id=?",
                     (old_empty["id"],))
    conn.close()

    assert store.prune_empty_sessions() == 1
    ids = [s["id"] for s in store.list_sessions()]
    assert used["id"] in ids and recent_empty["id"] in ids
    assert old_empty["id"] not in ids

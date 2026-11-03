"""Vision tool: image uploads -> description/text via a vision model."""
import base64
import io

import pytest
from PIL import Image

from jarvis import vision


@pytest.fixture
def png(tmp_path, monkeypatch):
    img = Image.new("RGB", (120, 60), "white")
    p = tmp_path / "shot.png"
    img.save(p)
    monkeypatch.setenv("JARVIS_VISION_MAX_MB", "12")
    return p


def test_validate_rejects_missing_and_non_image(tmp_path):
    with pytest.raises(ValueError):
        vision.validate(str(tmp_path / "nope.png"))
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(ValueError):
        vision.validate(str(f))


def test_data_uri_prefix(png):
    uri = vision.data_uri(str(png))
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_falls_back_between_models(png, monkeypatch):
    calls = []

    def fake_ask(model, uri, prompt):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("model down")
        return "A screenshot with the text HELLO."

    monkeypatch.setattr(vision, "_ask", fake_ask)
    monkeypatch.setenv("JARVIS_VISION_MODEL", "gpt-6-luna")
    out = vision.analyze(str(png))
    assert "HELLO" in out and "claude-haiku-5-5" in out   # fallback model
    assert len(calls) == 2                    # first failed, fallback worked


def test_analyze_all_models_fail(png, monkeypatch):
    monkeypatch.setattr(vision, "_ask",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        vision.analyze(str(png))


def test_tool_registered_and_errors_are_strings(tmp_path):
    from jarvis.builtin_tools import TOOL_FUNCS, TOOL_SCHEMAS
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert "analyze_image" in names and "analyze_image" in TOOL_FUNCS
    out = TOOL_FUNCS["analyze_image"](path=str(tmp_path / "missing.png"))
    assert out.startswith("error:")

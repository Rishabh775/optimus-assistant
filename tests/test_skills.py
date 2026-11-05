"""On-demand skills (shared with ~/.config/opencode/skills)."""
import pytest

from jarvis import builtin_tools as bt


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "SKILLS_DIR", tmp_path)
    return tmp_path


def _make_skill(d, skill_id, name, desc, body="1. Do the thing.\n2. Check it."):
    base = d / skill_id
    base.mkdir()
    (base / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {desc}\n"
        "---\n\n" + body,
        encoding="utf-8",
    )
    return base


def test_discover_finds_dir_skills(skills_dir):
    _make_skill(skills_dir, "frontend",
                "Frontend", "React/TS component building guidance.")
    skills = bt.discover_skills()
    assert [s["id"] for s in skills] == ["frontend"]
    assert skills[0]["name"] == "Frontend"
    assert "component" in skills[0]["description"]


def test_discover_finds_flat_md_and_skips_readme(skills_dir):
    (skills_dir / "testing.md").write_text(
        "---\nname: Testing\ndescription: write better tests\n---\nbody",
        encoding="utf-8")
    (skills_dir / "README.md").write_text("not a skill", encoding="utf-8")
    ids = [s["id"] for s in bt.discover_skills()]
    assert ids == ["testing"]


def test_skill_load_returns_body_and_strips_frontmatter(skills_dir):
    _make_skill(skills_dir, "design", "Design", "make it pretty",
                body="INTENDED:\n- contrast first")
    out = bt.skill_load("design")
    assert out.startswith("[skill design]")
    assert "INTENDED:" in out
    assert "---" not in out.split("]", 1)[1]      # frontmatter stripped
    assert "make it pretty" not in out


def test_skill_load_unknown_lists_available(skills_dir):
    _make_skill(skills_dir, "api-endpoints", "API", "endpoint design")
    out = bt.skill_load("nope")
    assert "Unknown skill 'nope'" in out
    assert "api-endpoints" in out                 # helpfully lists menu


def test_skills_list_and_registry_parity(skills_dir):
    assert "no skills installed" in bt.skills_list().lower()
    _make_skill(skills_dir, "ui-review", "UI", "audit a running UI")
    assert "ui-review" in bt.skills_list()
    # registry stays in sync (schema for every callable, and vice versa)
    assert "skill_load" in bt.TOOL_FUNCS and "skills_list" in bt.TOOL_FUNCS
    assert len(bt.TOOL_FUNCS) == len(bt.TOOL_SCHEMAS)
    names = {t["name"] for t in bt.TOOL_SCHEMAS}
    assert "skill_load" in names and "skills_list" in names


def test_system_prompt_includes_skills_menu(skills_dir, monkeypatch):
    _make_skill(skills_dir, "frontend", "Frontend", "build React UIs")
    import jarvis.agent as agent_mod
    monkeypatch.setattr(bt, "SKILLS_DIR", skills_dir)
    system = agent_mod._build_system("Optimus", 0)
    assert "skill_load" in system
    assert "frontend" in system

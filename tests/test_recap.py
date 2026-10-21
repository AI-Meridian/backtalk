"""The "where were we?" Tier 0 shortcut: a template built purely from
the vault's own latest daily note and Active Priorities file, no model
call. Covers markdown-stripping for speech, picking the latest note by
its YYYY-MM-DD filename (not mtime), and graceful handling when the
vault or its files are missing."""
from backtalk import main


def _make_vault(tmp_path, index_bullets=None, open_tasks=None):
    notes_dir = tmp_path / "01 - Daily Notes" / "10 - October 2026"
    notes_dir.mkdir(parents=True)
    if index_bullets is not None:
        for name, bullets in index_bullets.items():
            body = "## Index\n" + "\n".join(f"- {b}" for b in bullets) + "\n## Session 1\n"
            (notes_dir / name).write_text(body)
    if open_tasks is not None:
        body = ("### Open Tasks\n"
                + "\n".join(f"- [ ] {t}" for t in open_tasks)
                + "\n### Completed Tasks\n")
        (tmp_path / "Active Priorities.md").write_text(body)
    return tmp_path


def test_speakable_strips_wikilinks_and_bold():
    raw = "Fixed **[[Hugo Face Visualizer|the visualizer]]** for real"
    assert main._speakable(raw) == "Fixed the visualizer for real"


def test_speakable_bare_wikilink_uses_note_name():
    assert main._speakable("See [[Active Priorities]] for detail") == \
        "See Active Priorities for detail"


def test_latest_daily_note_picks_newest_filename(tmp_path):
    _make_vault(tmp_path, index_bullets={
        "2026-10-19.md": ["Old session"],
        "2026-10-21.md": ["Newest session"],
        "2026-10-20.md": ["Middle session"],
    })
    note = main._latest_daily_note(str(tmp_path))
    assert note.endswith("2026-10-21.md")


def test_recap_combines_last_session_and_open_tasks(tmp_path):
    _make_vault(tmp_path,
                index_bullets={"2026-10-21.md": [
                    "**Status check shipped** — built and pushed."]},
                open_tasks=["**Thing A** — do this next",
                           "Thing B", "Thing C", "Thing D (not spoken)"])
    spoken = main._spoken_recap(str(tmp_path))
    assert spoken.startswith("Last session: Status check shipped.")
    assert "Top open items: Thing A; Thing B; Thing C." in spoken
    assert "Thing D" not in spoken


def test_recap_missing_vault_dir_is_handled(tmp_path):
    missing = tmp_path / "nope"
    assert main._spoken_recap(str(missing)) == \
        "I can't find the vault to recap from."


def test_recap_no_vault_configured():
    assert main._spoken_recap(None) == "I can't find the vault to recap from."


def test_recap_handles_missing_active_priorities(tmp_path):
    _make_vault(tmp_path, index_bullets={"2026-10-21.md": ["Only a daily note."]})
    spoken = main._spoken_recap(str(tmp_path))
    assert spoken == "Last session: Only a daily note.."


def test_recap_phrases_recognized():
    for phrase in ("where were we", "where did we leave off", "catch me up",
                   "recall our last session", "what's the recap"):
        assert main.console_match(phrase) == "recap"

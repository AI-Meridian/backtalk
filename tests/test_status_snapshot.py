"""The Tier 0 "any update?" shortcut: a zero-token voice-console verb
that answers from the existing task/subagent/last-done state instead of
ever reaching the model. Covers the exact-phrase recognition, the
snapshot reader in signals.py, and the spoken formatting in main.py."""
from backtalk import main, signals


def test_status_phrases_recognized():
    for phrase in ("any update", "any updates", "status check",
                   "are you there", "you still there",
                   "how's it going", "what's happening"):
        assert main.console_match(phrase) == "status"


def test_ordinary_sentence_does_not_match_status():
    assert main.console_match("any update on the Meridian website build")\
        is None


def test_snapshot_reports_nothing_running():
    signals._tasks.clear()
    with signals._subagent_lock:
        signals._subagent_tasks.clear()
    signals._last_done = None
    snap = signals.status_snapshot()
    assert snap == {"background": [], "tasks": [], "last_done": None}
    assert main._spoken_status(snap) == "Nothing running right now."


def test_snapshot_reports_running_background_agent():
    signals._tasks.clear()
    with signals._subagent_lock:
        signals._subagent_tasks.clear()
        signals._subagent_tasks["a1"] = {
            "id": "a1", "ts": signals.time.time() - 90,
            "hb": signals.time.time(), "label": "Researching X", "eta": None}
    snap = signals.status_snapshot()
    assert len(snap["background"]) == 1
    spoken = main._spoken_status(snap)
    assert spoken.startswith("Still running — Researching X, going on")
    with signals._subagent_lock:
        signals._subagent_tasks.clear()


def test_snapshot_reports_active_tool_call():
    signals._tasks.clear()
    signals._tasks["t1"] = {"id": "t1", "ts": signals.time.time(),
                            "label": "Reading a file", "eta": None}
    snap = signals.status_snapshot()
    assert main._spoken_status(snap) == "Working on it — Reading a file."
    signals._tasks.clear()


def test_snapshot_reports_last_done_when_idle():
    signals._tasks.clear()
    with signals._subagent_lock:
        signals._subagent_tasks.clear()
    signals._last_done = {"id": "t1", "label": "Fixed the bug",
                          "completed_ts": signals.time.time() - 5}
    snap = signals.status_snapshot()
    spoken = main._spoken_status(snap)
    assert spoken.startswith("All done. Last thing I finished was "
                             "Fixed the bug, about")
    signals._last_done = None


def test_background_takes_priority_over_tasks_and_last_done():
    signals._tasks.clear()
    with signals._subagent_lock:
        signals._subagent_tasks.clear()
        signals._subagent_tasks["a1"] = {
            "id": "a1", "ts": signals.time.time(),
            "hb": signals.time.time(), "label": "Background job", "eta": None}
    signals._tasks["t1"] = {"id": "t1", "ts": signals.time.time(),
                            "label": "Foreground tool", "eta": None}
    signals._last_done = {"id": "t2", "label": "Old thing",
                          "completed_ts": signals.time.time() - 1}
    snap = signals.status_snapshot()
    assert main._spoken_status(snap).startswith("Still running — Background job")
    signals._tasks.clear()
    with signals._subagent_lock:
        signals._subagent_tasks.clear()
    signals._last_done = None

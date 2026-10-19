"""Shared test fixtures.

Autouse, session-wide: signals.py writes the task panel to real files
in signals_dir (the repo root by default) -- the exact files Hugo's
own live voice line reads from. Tests exercising task-tracking logic
(start_task/end_task/clear_all_tasks, exercised indirectly through
brain.py's writing-block tracking) must never touch that real file,
whether or not a real session happens to be running at the same time.
"""
import pytest

from backtalk import signals

# The genuine file writers, captured BEFORE the autouse fixtures below
# replace them, so a test that wants to exercise the real write path
# against a temp file can put the original back for itself.
REAL_WRITERS = {"last_reply": signals._write_last_reply,
                "alert": signals._write_alert}


@pytest.fixture
def real_writers():
    return REAL_WRITERS


@pytest.fixture(autouse=True)
def isolate_signal_files(monkeypatch):
    monkeypatch.setattr(signals, "_write_tasks", lambda: None)
    monkeypatch.setattr(signals, "_write_last_done", lambda: None)
    monkeypatch.setattr(signals, "_write_last_reply", lambda: None)
    # .voice_alert is read by a live face the moment it exists; a test
    # must never raise a real alert on Hugo's actual screen.
    monkeypatch.setattr(signals, "_write_alert", lambda on: None)


@pytest.fixture(autouse=True)
def reset_reply_state():
    signals._reply["turn"] = 0
    signals._reply["text"] = ""
    yield
    signals._reply["turn"] = 0
    signals._reply["text"] = ""


@pytest.fixture(autouse=True)
def reset_signal_state():
    """_tasks/_subagent_tasks are module-level globals mutated in
    place by start_task/end_task/clear_all_tasks -- without a reset,
    one test's leftover task row would leak into the next test's
    assertions."""
    signals._tasks.clear()
    with signals._subagent_lock:
        signals._subagent_tasks.clear()
    yield
    signals._tasks.clear()
    with signals._subagent_lock:
        signals._subagent_tasks.clear()

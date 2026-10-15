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


@pytest.fixture(autouse=True)
def isolate_signal_files(monkeypatch):
    monkeypatch.setattr(signals, "_write_tasks", lambda: None)
    monkeypatch.setattr(signals, "_write_last_done", lambda: None)


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

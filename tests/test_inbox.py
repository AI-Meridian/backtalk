"""Tests for the inbound half of the signal bus (inbox.py) -- the one
directory a face writes INTO rather than reads from. poll_once() is
the load-bearing primitive: claim-by-delete, chronological ordering,
and never crashing on a bad file are all real correctness properties,
not incidental behavior.
"""
import json
import os
import time

from backtalk import inbox


def _write_cmd(d, name, obj):
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


def test_poll_once_reads_and_deletes_a_valid_command_file(tmp_path, monkeypatch):
    monkeypatch.setitem(inbox.CFG, "signals_dir", str(tmp_path))
    d = inbox.inbox_dir()
    os.makedirs(d)
    _write_cmd(d, "1.json", {"type": "effort", "value": "low"})

    out = inbox.poll_once()

    assert out == [{"type": "effort", "value": "low"}]
    assert os.listdir(d) == []


def test_poll_once_processes_in_filename_order(tmp_path, monkeypatch):
    monkeypatch.setitem(inbox.CFG, "signals_dir", str(tmp_path))
    d = inbox.inbox_dir()
    os.makedirs(d)
    # Deliberately created out of lexicographic order to prove poll_once
    # sorts by name rather than relying on listdir's own order.
    _write_cmd(d, "00000002-bbbb.json", {"type": "message", "text": "second"})
    _write_cmd(d, "00000001-aaaa.json", {"type": "message", "text": "first"})
    _write_cmd(d, "00000003-cccc.json", {"type": "message", "text": "third"})

    out = inbox.poll_once()

    assert [c["text"] for c in out] == ["first", "second", "third"]


def test_poll_once_drops_malformed_json_without_raising(tmp_path, monkeypatch):
    monkeypatch.setitem(inbox.CFG, "signals_dir", str(tmp_path))
    d = inbox.inbox_dir()
    os.makedirs(d)
    bad = os.path.join(d, "1.json")
    with open(bad, "w") as f:
        f.write("{not valid json")
    _write_cmd(d, "2.json", {"type": "effort", "value": "high"})

    out = inbox.poll_once()

    assert out == [{"type": "effort", "value": "high"}]
    assert os.listdir(d) == []   # the bad file is gone too, never retried


def test_poll_once_ignores_non_json_files_in_the_directory(tmp_path, monkeypatch):
    monkeypatch.setitem(inbox.CFG, "signals_dir", str(tmp_path))
    d = inbox.inbox_dir()
    os.makedirs(d)
    with open(os.path.join(d, "readme.txt"), "w") as f:
        f.write("not a command")
    _write_cmd(d, "1.json", {"type": "effort", "value": "max"})

    out = inbox.poll_once()

    assert out == [{"type": "effort", "value": "max"}]
    assert os.path.exists(os.path.join(d, "readme.txt"))   # untouched


def test_poll_once_returns_empty_when_directory_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(inbox.CFG, "signals_dir", str(tmp_path / "nope"))
    assert inbox.poll_once() == []


def test_poll_once_drops_non_object_json(tmp_path, monkeypatch):
    monkeypatch.setitem(inbox.CFG, "signals_dir", str(tmp_path))
    d = inbox.inbox_dir()
    os.makedirs(d)
    _write_cmd(d, "1.json", ["not", "an", "object"])
    _write_cmd(d, "2.json", {"type": "effort", "value": "medium"})

    out = inbox.poll_once()

    assert out == [{"type": "effort", "value": "medium"}]
    assert os.listdir(d) == []


def test_sweep_uploads_deletes_only_files_older_than_seven_days(tmp_path, monkeypatch):
    monkeypatch.setitem(inbox.CFG, "uploads_dir", str(tmp_path))
    old = tmp_path / "old.png"
    new = tmp_path / "new.png"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    eight_days_ago = time.time() - 8 * 24 * 3600
    os.utime(old, (eight_days_ago, eight_days_ago))

    inbox.sweep_uploads()

    assert not old.exists()
    assert new.exists()


def test_sweep_uploads_never_raises_with_no_uploads_dir_configured(monkeypatch):
    monkeypatch.setitem(inbox.CFG, "uploads_dir", "")
    inbox.sweep_uploads()   # must not raise

"""set_session_info -- the live model/effort readout a face's pill
renders. Uses the `real_writers` fixture (see conftest.py) to opt out
of the autouse no-op patch for the one test that needs to see a real
file land.
"""
import json

from backtalk import signals


def test_set_session_info_writes_expected_json_shape(tmp_path, monkeypatch, real_writers):
    session_file = tmp_path / ".voice_session"
    monkeypatch.setattr(signals, "_SESSION_FILE", str(session_file))
    monkeypatch.setattr(signals, "_write_session_info",
                        real_writers["session_info"])

    signals.set_session_info("fable", "high")

    data = json.loads(session_file.read_text())
    assert data["model"] == "fable"
    assert data["effort"] == "high"
    assert "ts" in data


def test_set_session_info_never_raises_on_unwritable_dir(monkeypatch, real_writers):
    monkeypatch.setattr(signals, "_SESSION_FILE",
                        "/nonexistent-dir-xyz/.voice_session")
    monkeypatch.setattr(signals, "_write_session_info",
                        real_writers["session_info"])

    signals.set_session_info("fast", "")   # must not raise

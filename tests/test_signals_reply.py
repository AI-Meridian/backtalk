"""The reply feed behind the floating reply panel (.voice_last_reply):
text grows chunk by chunk, a new turn starts it fresh WITHOUT wiping
the screen until the new reply actually begins, and the published
payload has the shape the visualizer reads."""
from backtalk import signals


def test_chunks_accumulate_in_order():
    signals.reply_new()
    signals.reply_chunk_started("First sentence.")
    signals.reply_chunk_started("  Second   one. ")
    assert signals._reply["text"] == "First sentence. Second one."


def test_new_reply_resets_text_and_bumps_turn():
    signals.reply_new()
    signals.reply_chunk_started("Old reply.")
    t0 = signals._reply["turn"]
    signals.reply_new()
    assert signals._reply["turn"] == t0 + 1
    assert signals._reply["text"] == ""
    signals.reply_chunk_started("New reply.")
    assert signals._reply["text"] == "New reply."


def test_new_reply_does_not_publish_until_first_chunk(monkeypatch):
    writes = []
    monkeypatch.setattr(signals, "_write_last_reply",
                        lambda: writes.append(dict(signals._reply)))
    signals.reply_new()
    assert writes == []                 # the old text stays on screen
    signals.reply_chunk_started("Now.")
    assert len(writes) == 1 and writes[0]["text"] == "Now."


def test_empty_chunk_is_ignored(monkeypatch):
    writes = []
    monkeypatch.setattr(signals, "_write_last_reply",
                        lambda: writes.append(1))
    signals.reply_chunk_started("   ")
    signals.reply_chunk_started("")
    assert writes == []


def test_payload_shape(tmp_path, monkeypatch, real_writers):
    import json
    target = tmp_path / ".voice_last_reply"
    monkeypatch.setattr(signals, "_LAST_REPLY_FILE", str(target))
    # the REAL writer, against a temp file, not conftest's no-op
    monkeypatch.setattr(signals, "_write_last_reply", real_writers["last_reply"])
    signals.reply_new()
    signals.reply_chunk_started("Hello there.")
    data = json.loads(target.read_text())
    assert set(data) == {"ts", "turn", "text"}
    assert data["text"] == "Hello there."
    assert data["turn"] == signals._reply["turn"]
    assert isinstance(data["ts"], float)


def test_set_alert_writes_and_clears(tmp_path, monkeypatch, real_writers):
    target = tmp_path / ".voice_alert"
    monkeypatch.setattr(signals, "_ALERT_FILE", str(target))
    monkeypatch.setattr(signals, "_write_alert", real_writers["alert"])
    signals.set_alert(True)
    assert target.exists() and target.stat().st_size > 0
    signals.set_alert(False)
    assert not target.exists()
    signals.set_alert(False)            # clearing twice never raises

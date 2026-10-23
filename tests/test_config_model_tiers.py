"""Tests for the new model-tier/upload config keys added for the
dropdown + chat-input feature: fable_model and uploads_dir.
"""
import json

from backtalk import config as backtalk_config


def test_fable_model_default_is_claude_fable_5():
    assert backtalk_config.DEFAULTS["fable_model"] == "claude-fable-5"


def test_uploads_dir_defaults_to_signals_dir_slash_voice_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(backtalk_config, "CONFIG_PATH", tmp_path / "backtalk.json")

    cfg = backtalk_config.load()

    from pathlib import Path
    assert cfg["uploads_dir"] == str(Path(cfg["signals_dir"]) / ".voice_uploads")


def test_uploads_dir_honors_explicit_override(tmp_path, monkeypatch):
    cfg_path = tmp_path / "backtalk.json"
    cfg_path.write_text(json.dumps({"uploads_dir": str(tmp_path / "custom-uploads")}))
    monkeypatch.setattr(backtalk_config, "CONFIG_PATH", cfg_path)

    cfg = backtalk_config.load()

    assert cfg["uploads_dir"] == str(tmp_path / "custom-uploads")

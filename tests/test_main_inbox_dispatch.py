"""Tests for handle_inbox_command -- the dispatcher that turns one
inbox.py command dict into an actual action. Module-level and taking
brain/handle_fn explicitly (not a closure inside amain()) specifically
so it's testable with fakes, no real asyncio main loop or SDK
connection required.
"""
import pytest

from backtalk import main as backtalk_main


class FakeBrain:
    def __init__(self):
        self.commands = []
        self.reset_calls = 0

    async def reset_turn(self):
        self.reset_calls += 1

    async def command(self, cmd):
        self.commands.append(cmd)
        return ""


@pytest.fixture(autouse=True)
def reset_session_state():
    backtalk_main._SESSION_STATE["model_tier"] = "fast"
    backtalk_main._SESSION_STATE["effort"] = ""
    yield
    backtalk_main._SESSION_STATE["model_tier"] = "fast"
    backtalk_main._SESSION_STATE["effort"] = ""


@pytest.fixture(autouse=True)
def no_real_config_write(monkeypatch):
    # _write_config_key really touches backtalk.json on disk; every test
    # here must stay purely in-memory.
    monkeypatch.setattr(backtalk_main, "_write_config_key",
                        lambda k, v: True)


async def _fake_handle_true(text):
    return True


@pytest.mark.asyncio
async def test_message_command_calls_handle_with_literal_text():
    brain = FakeBrain()
    seen = []

    async def fake_handle(text):
        seen.append(text)
        return True

    ok = await backtalk_main.handle_inbox_command(
        {"type": "message", "text": "look at /tmp/x.png"},
        brain, fake_handle)

    assert ok is True
    assert seen == ["look at /tmp/x.png"]


@pytest.mark.asyncio
async def test_message_command_propagates_quit_signal():
    brain = FakeBrain()

    async def fake_handle(text):
        return False   # the quit-phrase path

    ok = await backtalk_main.handle_inbox_command(
        {"type": "message", "text": "goodbye"}, brain, fake_handle)

    assert ok is False


@pytest.mark.asyncio
async def test_message_command_with_empty_text_never_calls_handle():
    brain = FakeBrain()
    calls = []

    async def fake_handle(text):
        calls.append(text)
        return True

    ok = await backtalk_main.handle_inbox_command(
        {"type": "message", "text": "   "}, brain, fake_handle)

    assert ok is True
    assert calls == []


@pytest.mark.asyncio
async def test_model_command_fable_calls_brain_command_with_fable_model_id():
    brain = FakeBrain()

    await backtalk_main.handle_inbox_command(
        {"type": "model", "value": "fable"}, brain, _fake_handle_true)

    assert brain.commands == [f"/model {backtalk_main.CFG['fable_model']}"]
    assert backtalk_main._SESSION_STATE["model_tier"] == "fable"


@pytest.mark.asyncio
async def test_model_command_deep_calls_brain_command_with_deep_model_id():
    brain = FakeBrain()

    await backtalk_main.handle_inbox_command(
        {"type": "model", "value": "deep"}, brain, _fake_handle_true)

    assert brain.commands == [f"/model {backtalk_main.CFG['deep_model']}"]


@pytest.mark.asyncio
async def test_model_command_unknown_tier_is_logged_and_ignored():
    brain = FakeBrain()

    ok = await backtalk_main.handle_inbox_command(
        {"type": "model", "value": "haiku"}, brain, _fake_handle_true)

    assert ok is True
    assert brain.commands == []
    assert backtalk_main._SESSION_STATE["model_tier"] == "fast"   # unchanged


@pytest.mark.asyncio
async def test_effort_command_xhigh_persists_via_write_config_key(monkeypatch):
    brain = FakeBrain()
    calls = []
    monkeypatch.setattr(backtalk_main, "_write_config_key",
                        lambda k, v: calls.append((k, v)) or True)

    await backtalk_main.handle_inbox_command(
        {"type": "effort", "value": "xhigh"}, brain, _fake_handle_true)

    assert brain.commands == ["/effort xhigh"]
    assert calls == [("effort", "xhigh")]
    assert backtalk_main._SESSION_STATE["effort"] == "xhigh"


@pytest.mark.asyncio
async def test_effort_command_unknown_value_is_ignored():
    brain = FakeBrain()

    ok = await backtalk_main.handle_inbox_command(
        {"type": "effort", "value": "ludicrous"}, brain, _fake_handle_true)

    assert ok is True
    assert brain.commands == []
    assert backtalk_main._SESSION_STATE["effort"] == ""


@pytest.mark.asyncio
async def test_unknown_command_type_never_raises():
    brain = FakeBrain()

    ok = await backtalk_main.handle_inbox_command(
        {"type": "self_destruct"}, brain, _fake_handle_true)

    assert ok is True
    assert brain.commands == []


@pytest.mark.asyncio
async def test_brain_exception_is_swallowed_not_raised():
    class ExplodingBrain(FakeBrain):
        async def reset_turn(self):
            raise RuntimeError("boom")

    ok = await backtalk_main.handle_inbox_command(
        {"type": "model", "value": "fast"}, ExplodingBrain(), _fake_handle_true)

    assert ok is True   # swallowed, never crashes the voice line

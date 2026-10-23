"""Real tests for reset_turn's background-agent-aware drain timeout --
the 2026-10-22 fix for "stream desynced beyond repair" firing on a
perfectly healthy session that was simply still running real
background work (an Agent/Task tool call) when the user talked over
it. Before this fix, reset_turn's drain always used the same short
`timeout` (8s default) no matter what, so a turn that needed longer
than that to actually finish its background work got misread as a
dead pipe and rebuilt, wiping conversation memory for no reason. Field
case: two sessions on 2026-10-22, both dead on the default's 8-second
mark, both mid a turn with real background work still running.

No real Claude Agent SDK connection anywhere here, same convention as
test_warmbrain_unsolicited.py -- fake message objects duck-typed by
class name only.
"""
import asyncio

import pytest

from backtalk import brain as brain_module
from backtalk import signals
from backtalk.brain import WarmBrain


class ResultMessage:
    def __init__(self, session_id="fake-session"):
        self.usage = {}
        self.total_cost_usd = None
        self.session_id = session_id
        self.subtype = "success"
        self.is_error = False


class FakeClient:
    """Enough of ClaudeSDKClient's surface for reset_turn to run
    without ever touching a real subprocess."""
    def __init__(self):
        self.interrupted = 0

    async def interrupt(self):
        self.interrupted += 1


def make_brain():
    b = WarmBrain()
    b._client = FakeClient()
    return b


async def _deliver_after(inbox: asyncio.Queue, delay: float, *messages):
    await asyncio.sleep(delay)
    for m in messages:
        await inbox.put(m)


def test_has_active_subagents_false_when_empty():
    assert signals.has_active_subagents() is False


def test_has_active_subagents_true_with_one_or_many():
    signals.start_subagent_task("a1", "Background agent: researcher")
    assert signals.has_active_subagents() is True
    signals.start_subagent_task("a2", "Background agent: writer")
    assert signals.has_active_subagents() is True
    signals.end_subagent_task("a1")
    assert signals.has_active_subagents() is True   # a2 still running
    signals.end_subagent_task("a2")
    assert signals.has_active_subagents() is False


@pytest.mark.asyncio
async def test_reset_turn_rebuilds_on_the_short_default_when_nothing_is_running():
    """No background agent active: a turn that doesn't produce its
    ResultMessage within the short default window is correctly read as
    a dead pipe, exactly as before this fix -- the common case must
    stay just as fast as it always was."""
    b = make_brain()
    b._dirty = True
    rebuilt = []

    async def fake_rebuild():
        rebuilt.append(True)
        b._dirty = False

    b._rebuild_session = fake_rebuild
    # Nothing is ever put in the inbox -- the drain can only time out.
    await b.reset_turn(timeout=0.05)
    assert rebuilt == [True]
    assert b._client.interrupted == 1


@pytest.mark.asyncio
async def test_reset_turn_waits_longer_and_recovers_when_a_subagent_is_active(
        monkeypatch):
    """The actual fix: a background agent genuinely still running buys
    the drain real extra time, so a turn that would have been killed
    at the short default instead gets found and realigned cleanly."""
    monkeypatch.setattr(brain_module, "_DESYNC_GRACE_WITH_ACTIVE_SUBAGENTS",
                         0.2)   # keep the test fast
    b = make_brain()
    b._dirty = True
    signals.start_subagent_task("a1", "Background agent: report writer")
    rebuilt = []

    async def fake_rebuild():
        rebuilt.append(True)

    b._rebuild_session = fake_rebuild

    # Arrives AFTER the short default (0.05s) but WITHIN the subagent
    # grace window (0.2s) -- only recoverable because the fix is live.
    asyncio.create_task(_deliver_after(b._inbox, 0.1, ResultMessage()))
    await b.reset_turn(timeout=0.05)

    assert rebuilt == []          # never rebuilt
    assert b._dirty is False      # realigned cleanly instead


@pytest.mark.asyncio
async def test_reset_turn_still_rebuilds_if_subagent_active_but_truly_stuck(
        monkeypatch):
    """The grace period is longer, not infinite -- a pipe that stays
    silent even past the extended subagent window still rebuilds."""
    monkeypatch.setattr(brain_module, "_DESYNC_GRACE_WITH_ACTIVE_SUBAGENTS",
                         0.05)
    b = make_brain()
    b._dirty = True
    signals.start_subagent_task("a1", "Background agent: stuck")
    rebuilt = []

    async def fake_rebuild():
        rebuilt.append(True)
        b._dirty = False

    b._rebuild_session = fake_rebuild
    await b.reset_turn(timeout=0.01)
    assert rebuilt == [True]

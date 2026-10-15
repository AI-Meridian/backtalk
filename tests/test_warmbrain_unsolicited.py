"""Real, permanent tests for WarmBrain's unsolicited-reply mechanism --
the single most bug-prone piece of this stack. It's been the site of
two real, shipped bugs in one week: the original "a background task's
follow-up turn sits silent until the next real input" bug (fixed
2026-10-09, see brain.py's own __init__ comment), and a second gap in
HOW that fix gets re-armed after a barge-in (fixed 2026-10-15, in
main.py's dispatch loop, not tested here -- this file covers the
WarmBrain-level primitive that fix depends on).

Both prior fixes were only ever proven with throwaway scripts, written
once, run once, deleted. These are the same tests made permanent, so a
future refactor of _pump_sentences/_inbox_lock/next_unsolicited can't
silently reintroduce either bug.

No real Claude Agent SDK connection anywhere here -- WarmBrain.start()
is never called. Fake message objects are duck-typed by class name
only (type(msg).__name__), exactly what brain.py itself switches on,
so these fakes don't need to subclass or import the real SDK types.
"""
import asyncio

import pytest

from backtalk.brain import WarmBrain


class StreamEvent:
    def __init__(self, event):
        self.event = event


class ResultMessage:
    def __init__(self, session_id="fake-session"):
        self.usage = {}
        self.total_cost_usd = None
        self.session_id = session_id
        self.subtype = "success"
        self.is_error = False


class FakeClient:
    """Enough of ClaudeSDKClient's surface for ask_stream to run
    without ever touching a real subprocess."""
    def __init__(self):
        self.queries = []

    async def query(self, utterance):
        self.queries.append(utterance)


def text_delta(index, text):
    return StreamEvent({"type": "content_block_delta", "index": index,
                             "delta": {"type": "text_delta", "text": text}})


def block_stop(index):
    return StreamEvent({"type": "content_block_stop", "index": index})


def make_brain():
    b = WarmBrain()
    b._client = FakeClient()
    return b


async def drain(gen):
    return [s async for s in gen]


@pytest.mark.asyncio
async def test_unsolicited_pending_false_when_inbox_empty():
    b = make_brain()
    assert b.unsolicited_pending() is False


@pytest.mark.asyncio
async def test_unsolicited_pending_true_once_something_arrives_unread():
    b = make_brain()
    await b._inbox.put(text_delta(0, "hello. "))
    await b._inbox.put(ResultMessage())
    assert b.unsolicited_pending() is True


@pytest.mark.asyncio
async def test_unsolicited_pending_false_while_lock_held():
    """The exact guarantee next_unsolicited()'s docstring promises: a
    real ask_stream/command already in flight is never double-consumed,
    because unsolicited_pending() refuses to fire while the lock is held."""
    b = make_brain()
    await b._inbox.put(text_delta(0, "hello. "))
    await b._inbox.put(ResultMessage())
    async with b._inbox_lock:
        assert b.unsolicited_pending() is False
    # released again afterward
    assert b.unsolicited_pending() is True


@pytest.mark.asyncio
async def test_ask_stream_yields_sentences_and_stops_at_result():
    b = make_brain()
    await b._inbox.put(text_delta(0, "Hello there. "))
    await b._inbox.put(text_delta(0, "How are you?"))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage())

    sentences = await drain(b.ask_stream("hi"))
    assert sentences == ["Hello there.", "How are you?"]
    assert b._client.queries == ["hi"]
    # turn fully consumed: pipe realigned, lock released
    assert b._dirty is False
    assert not b._inbox_lock.locked()


@pytest.mark.asyncio
async def test_next_unsolicited_catches_a_genuine_follow_up_turn():
    """Formalizes the original 2026-10-09 fix: a follow-up turn nobody
    asked for (e.g. a background agent's own completion) must be
    caught and spoken the instant nothing else is reading, not
    silently dropped or left for the next real question to absorb."""
    b = make_brain()
    await b._inbox.put(text_delta(0, "By the way, that background job finished."))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage())

    assert b.unsolicited_pending() is True
    sentences = await drain(b.next_unsolicited())
    assert sentences == ["By the way, that background job finished."]
    assert b.unsolicited_pending() is False
    assert not b._inbox_lock.locked()


@pytest.mark.asyncio
async def test_second_claimant_blocked_while_first_holds_the_lock():
    """Two readers racing for the same follow-up turn must never both
    get it. Deliberately doesn't let a second next_unsolicited() run
    to completion on an empty queue -- _pump_sentences blocks on
    self._inbox.get() for the real _ASK_IDLE_TIMEOUT (240s) when
    nothing arrives, which would make this test itself hang for real
    minutes. Instead: start the first claimant, let it acquire the
    lock and consume one sentence, then prove directly that the lock
    is held and a second acquire attempt can't jump in."""
    b = make_brain()
    await b._inbox.put(text_delta(0, "Only one of you gets this. "))
    await b._inbox.put(ResultMessage())

    gen = b.next_unsolicited()
    first_sentence = await gen.__anext__()
    assert first_sentence == "Only one of you gets this."
    assert b._inbox_lock.locked()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(b._inbox_lock.acquire(), timeout=0.05)

    await drain(gen)   # let the first claimant finish normally
    assert not b._inbox_lock.locked()


@pytest.mark.asyncio
async def test_content_arriving_after_ask_stream_releases_is_unsolicited():
    """The real-world shape of the original bug: a normal question gets
    answered and the turn completes, THEN more content arrives on the
    same pipe with nobody asking for it (a background task's deferred
    follow-up). It must show up as unsolicited, not get lost."""
    b = make_brain()
    await b._inbox.put(text_delta(0, "Here's your answer."))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage())
    await drain(b.ask_stream("what's the answer?"))
    assert b.unsolicited_pending() is False

    # Time passes; a background task's own follow-up lands with nobody
    # asking for it.
    await b._inbox.put(text_delta(0, "Also, that other thing finished."))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage())
    assert b.unsolicited_pending() is True
    sentences = await drain(b.next_unsolicited())
    assert sentences == ["Also, that other thing finished."]


@pytest.mark.asyncio
async def test_writing_response_task_panel_row_starts_and_ends():
    """2026-10-09 Session 3's fix: the task panel must show SOMETHING
    while the model is purely composing text with no tool call running,
    not just while tool calls are active. Checks signals._tasks
    directly (file writes are stubbed by conftest's isolate fixture)."""
    from backtalk import signals
    b = make_brain()
    await b._inbox.put(text_delta(0, "Thinking out loud. "))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage())

    gen = b.next_unsolicited()
    await gen.__anext__()   # pull the first sentence
    # a writing-0 row must exist while the text block is still open
    assert "writing-0" in signals._tasks

    await drain(gen)        # exhaust the rest (block_stop, ResultMessage)
    # clear_all_tasks() in the finally block is the safety net that
    # must leave no leftover writing-task row behind.
    assert signals._tasks == {}

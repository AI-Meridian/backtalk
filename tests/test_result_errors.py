"""THE SIGNED-OUT FIX, made permanent. Field case 2026-10-19: Claude
Code had been signed out; every turn's ResultMessage came back
is_error with zero sentences before it, and _pump_sentences treated
that as a turn that simply ended — the voice line went quiet, and the
boot even logged "brain warm". These pin down the new contract: an
error result is SPOKEN; an auth-shaped one raises the face's alert
and sets auth_failed; a clean result afterwards clears the alert.

Same fake-message, no-SDK harness as test_warmbrain_unsolicited.py.
"""
import asyncio

import pytest

from backtalk import auth, signals
from backtalk.brain import WarmBrain


class StreamEvent:
    def __init__(self, event):
        self.event = event


class ResultMessage:
    def __init__(self, is_error=False, result=None, subtype="success",
                 errors=None):
        self.usage = {}
        self.total_cost_usd = None
        self.session_id = "fake-session"
        self.subtype = subtype
        self.is_error = is_error
        self.result = result
        self.errors = errors


class FakeClient:
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


async def collect(gen):
    out = []
    async for s in gen:
        out.append(s)
    return out


@pytest.fixture
def alert_log(monkeypatch):
    calls = []
    monkeypatch.setattr(signals, "_write_alert", lambda on: calls.append(on))
    return calls


@pytest.mark.asyncio
async def test_signed_out_result_is_spoken_and_flagged(alert_log):
    b = make_brain()
    await b._inbox.put(ResultMessage(
        is_error=True, result="Not logged in · Please run /login"))
    got = await collect(b.ask_stream("hello?"))
    assert got == [auth.SPOKEN_SIGNED_OUT]
    assert b.auth_failed is True
    assert "Not logged in" in b.last_error
    assert alert_log == [True]


@pytest.mark.asyncio
async def test_generic_error_result_is_spoken_with_its_text(alert_log):
    b = make_brain()
    await b._inbox.put(ResultMessage(
        is_error=True,
        result="You've hit your usage limit · resets at 3pm "
               "https://claude.ai/settings"))
    got = await collect(b.ask_stream("hello?"))
    assert len(got) == 1
    assert got[0].startswith("My Claude session returned an error.")
    assert "usage limit" in got[0]
    assert "https://" not in got[0]          # URLs never spoken
    assert "·" not in got[0]            # middle dots never spoken
    assert b.auth_failed is False
    assert alert_log == []                   # not an auth fault: no alert


@pytest.mark.asyncio
async def test_error_with_no_text_still_says_something(alert_log):
    b = make_brain()
    await b._inbox.put(ResultMessage(is_error=True, result="",
                                     subtype="error_during_execution"))
    got = await collect(b.ask_stream("hello?"))
    assert len(got) == 1 and "error" in got[0].lower()
    assert b.last_error == "error_during_execution"


@pytest.mark.asyncio
async def test_errors_list_is_used_when_result_is_empty(alert_log):
    b = make_brain()
    await b._inbox.put(ResultMessage(is_error=True, result=None,
                                     errors=["OAuth token has expired"]))
    got = await collect(b.ask_stream("hello?"))
    assert got == [auth.SPOKEN_SIGNED_OUT]
    assert b.auth_failed is True


@pytest.mark.asyncio
async def test_clean_turn_after_auth_failure_clears_the_alert(alert_log):
    b = make_brain()
    await b._inbox.put(ResultMessage(is_error=True, result="Not logged in"))
    await collect(b.ask_stream("one"))
    assert alert_log == [True]
    await b._inbox.put(text_delta(0, "Back with you. "))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage())
    got = await collect(b.ask_stream("two"))
    assert got == ["Back with you."]
    assert b.last_error is None
    assert alert_log == [True, False]


@pytest.mark.asyncio
async def test_clean_turn_never_touches_an_alert_it_did_not_raise(alert_log):
    b = make_brain()
    await b._inbox.put(text_delta(0, "Fine. "))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage())
    await collect(b.ask_stream("hi"))
    assert alert_log == []


@pytest.mark.asyncio
async def test_sentences_before_an_error_result_are_kept(alert_log):
    """A turn that spoke a sentence and THEN died still speaks the
    sentence, then the error — nothing is swallowed either way."""
    b = make_brain()
    await b._inbox.put(text_delta(0, "On it. "))
    await b._inbox.put(block_stop(0))
    await b._inbox.put(ResultMessage(is_error=True, result="Internal error"))
    got = await collect(b.ask_stream("go"))
    assert got[0] == "On it."
    assert got[1].startswith("My Claude session returned an error.")

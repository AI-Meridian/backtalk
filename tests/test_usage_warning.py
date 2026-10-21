"""The proactive "last 10% of your usage window" warning: instead of
staying silent all the way to the real limit and then going dark,
_pull_rate_limits speaks a one-time heads-up the instant a window
crosses 90% utilization. Same fake-message style as
test_warmbrain_unsolicited.py -- no real SDK connection."""
import pytest

from backtalk import signals
from backtalk.brain import WarmBrain
from backtalk.config import CFG


class FakeQuery:
    def __init__(self, usage):
        self.usage = usage

    async def _send_control_request(self, payload):
        return self.usage


class FakeClient:
    def __init__(self, usage):
        self._query = FakeQuery(usage)


def make_brain(usage, monkeypatch):
    monkeypatch.setitem(CFG, "show_usage", True)
    monkeypatch.setattr(signals, "set_rate_limit", lambda *a, **k: None)
    b = WarmBrain()
    b._client = FakeClient(usage)
    return b


def usage_payload(five_hour_pct=None, seven_day_pct=None, resets_at=1000):
    rate_limits = {}
    if five_hour_pct is not None:
        rate_limits["five_hour"] = {"utilization": five_hour_pct,
                                    "resets_at": resets_at}
    if seven_day_pct is not None:
        rate_limits["seven_day"] = {"utilization": seven_day_pct,
                                    "resets_at": resets_at}
    return {"rate_limits": rate_limits}


@pytest.mark.asyncio
async def test_no_warning_below_90_percent(monkeypatch):
    b = make_brain(usage_payload(five_hour_pct=50), monkeypatch)
    assert await b._pull_rate_limits() is None


@pytest.mark.asyncio
async def test_warns_once_crossing_90_percent(monkeypatch):
    b = make_brain(usage_payload(five_hour_pct=92), monkeypatch)
    warning = await b._pull_rate_limits()
    assert warning is not None
    assert "five-hour" in warning
    assert "ten percent" in warning


@pytest.mark.asyncio
async def test_does_not_repeat_within_the_same_window(monkeypatch):
    b = make_brain(usage_payload(five_hour_pct=95, resets_at=1000), monkeypatch)
    first = await b._pull_rate_limits()
    second = await b._pull_rate_limits()
    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_warns_again_after_the_window_rolls_over(monkeypatch):
    b = make_brain(usage_payload(five_hour_pct=95, resets_at=1000), monkeypatch)
    first = await b._pull_rate_limits()
    assert first is not None
    # a new window: later resets_at, utilization still high
    b._client = FakeClient(usage_payload(five_hour_pct=93, resets_at=2000))
    second = await b._pull_rate_limits()
    assert second is not None


@pytest.mark.asyncio
async def test_seven_day_window_warns_independently_of_five_hour(monkeypatch):
    b = make_brain(usage_payload(five_hour_pct=50, seven_day_pct=91,
                                 resets_at=1000), monkeypatch)
    warning = await b._pull_rate_limits()
    assert warning is not None
    assert "weekly" in warning


@pytest.mark.asyncio
async def test_show_usage_off_never_polls_or_warns(monkeypatch):
    monkeypatch.setitem(CFG, "show_usage", False)
    b = WarmBrain()
    b._client = FakeClient(usage_payload(five_hour_pct=99))
    assert await b._pull_rate_limits() is None

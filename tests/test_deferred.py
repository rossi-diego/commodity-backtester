"""Held contracts: which ones, their Yahoo symbols, and a fetch that never trusts a bad print.

Every fetch goes through a fake ``yf.download``; nothing touches the network.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import src.deferred as deferred
from src.constants import BEANS, DIESEL, MEAL, OIL
from src.deferred import held_session
from src.rolls import BEAN_MONTHS, PRODUCT_MONTHS, held_contract, month_name, yahoo_ticker

# The real held-contract closes on 6 Nov 2026, with the =F row of that session.
DATE = pd.Timestamp("2026-11-06")
QUOTES = {"ZSF27.CBT": 1193.75, "ZMZ26.CBT": 319.2, "ZLZ26.CBT": 62.26}
FRONT = {BEANS: 1189.0, MEAL: 319.2, OIL: 62.26, DIESEL: 4.33}


def _prices(row: dict[str, float] = FRONT, date: pd.Timestamp = DATE) -> pd.DataFrame:
    earlier = {leg: value * 1.01 for leg, value in row.items()}
    return pd.DataFrame([earlier, row], index=pd.DatetimeIndex([date - pd.Timedelta(days=1), date]))


def _response(quotes: dict[str, float], date: pd.Timestamp = DATE) -> pd.DataFrame:
    """A yfinance multi-ticker frame: (field, ticker) columns."""
    columns = pd.MultiIndex.from_product([["Close", "Open"], list(quotes)])
    values = [list(quotes.values()) * 2]
    return pd.DataFrame(values, index=pd.DatetimeIndex([date], name="Date"), columns=columns)


class FakeYahoo:
    """Answers each call from ``replies`` in turn; records the tickers asked."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.asked: list[list[str]] = []

    def __call__(self, tickers, **_kwargs):
        self.asked.append(list(tickers))
        reply = self.replies.pop(0) if self.replies else _response({})
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize(
    ("date", "beans", "products"),
    [
        ("2026-10-29", (2026, 11), (2026, 12)),  # the day before first notice: Nov is held
        ("2026-10-30", (2027, 1), (2026, 12)),  # first notice itself: already in Jan
        ("2026-11-13", (2027, 1), (2026, 12)),  # Nov's last trade
        ("2026-11-16", (2027, 1), (2026, 12)),  # the switch day on =F
        ("2026-11-30", (2027, 1), (2027, 1)),  # Dec products go to delivery
        ("2026-12-31", (2027, 3), (2027, 3)),  # Jan's first notice, across the year
        ("2026-08-31", (2026, 11), (2026, 10)),  # Sep beans in delivery; Oct products held
        ("2026-09-30", (2026, 11), (2026, 12)),  # Oct products in delivery
    ],
)
def test_held_contract_at_each_boundary(date, beans, products) -> None:
    day = pd.Timestamp(date)
    assert held_contract(day, BEAN_MONTHS) == beans
    assert held_contract(day, PRODUCT_MONTHS) == products


def test_yahoo_symbols_and_month_names() -> None:
    assert yahoo_ticker("ZS", 2027, 1) == "ZSF27.CBT"
    assert yahoo_ticker("ZM", 2026, 12) == "ZMZ26.CBT"
    assert yahoo_ticker("ZL", 2030, 10) == "ZLV30.CBT"
    assert [month_name(m) for m in (1, 12)] == ["Jan", "Dec"]


class TestHeldSession:
    def test_the_held_closes_replace_the_expiring_contract(self) -> None:
        yahoo = FakeYahoo(_response(QUOTES))
        held = held_session(_prices(), fetcher=yahoo)
        assert held is not None
        assert held.date == DATE
        assert held.pair == "Jan beans / Dec products"
        assert held.prices.to_dict() == {BEANS: 1193.75, MEAL: 319.2, OIL: 62.26, DIESEL: 4.33}
        assert yahoo.asked == [list(QUOTES)]

    def test_missing_symbols_are_asked_again(self) -> None:
        first = _response({"ZSF27.CBT": 1193.75})
        yahoo = FakeYahoo(first, _response({"ZMZ26.CBT": 319.2, "ZLZ26.CBT": 62.26}))
        assert held_session(_prices(), fetcher=yahoo) is not None
        assert yahoo.asked[1] == ["ZMZ26.CBT", "ZLZ26.CBT"]

    def test_three_failed_calls_give_none(self, caplog) -> None:
        yahoo = FakeYahoo(RuntimeError("429"), _response({}), _response({}))
        with caplog.at_level(logging.WARNING, logger="src.deferred"):
            assert held_session(_prices(), fetcher=yahoo) is None
        assert len(yahoo.asked) == deferred.FETCH_ATTEMPTS
        assert "event=held_fetch_failed" in caplog.text

    def test_a_session_on_another_day_is_not_used(self) -> None:
        stale = _response(QUOTES, date=DATE - pd.Timedelta(days=1))
        yahoo = FakeYahoo(stale, stale, stale)
        assert held_session(_prices(), fetcher=yahoo) is None

    def test_a_non_positive_close_is_not_used(self) -> None:
        bad = _response(QUOTES | {"ZLZ26.CBT": 0.0})
        assert held_session(_prices(), fetcher=FakeYahoo(bad, bad, bad)) is None

    def test_an_unreadable_reply_is_not_used(self) -> None:
        assert held_session(_prices(), fetcher=FakeYahoo("junk", "junk", "junk")) is None

    def test_a_held_front_month_must_match_the_continuous_series(self, caplog) -> None:
        """Dec meal is both held and the =F front: 1% off means a wrong symbol or unit."""
        yahoo = FakeYahoo(_response(QUOTES | {"ZMZ26.CBT": 319.2 * 1.01}))
        with caplog.at_level(logging.WARNING, logger="src.deferred"):
            assert held_session(_prices(), fetcher=yahoo) is None
        assert "event=held_rejected ticker=ZMZ26.CBT" in caplog.text

    def test_a_deferred_contract_must_sit_within_a_calendar_spread(self) -> None:
        yahoo = FakeYahoo(_response(QUOTES | {"ZSF27.CBT": 1189.0 * 1.2}))
        assert held_session(_prices(), fetcher=yahoo) is None

    def test_a_series_reply_for_one_symbol_is_read(self, monkeypatch) -> None:
        reply = pd.DataFrame({"Close": [1193.75]}, index=pd.DatetimeIndex([DATE]))["Close"]
        closes = deferred._closes_once(["ZSF27.CBT"], DATE, lambda *a, **k: {"Close": reply})
        assert closes == {"ZSF27.CBT": 1193.75}

    def test_a_timezone_aware_index_is_matched_by_day(self) -> None:
        reply = _response(QUOTES)
        reply.index = reply.index.tz_localize("America/Chicago")
        assert held_session(_prices(), fetcher=FakeYahoo(reply)) is not None

    def test_missing_legs_give_none_without_a_call(self) -> None:
        yahoo = FakeYahoo()
        assert held_session(_prices().drop(columns=[DIESEL]), fetcher=yahoo) is None
        assert held_session(_prices().iloc[:0], fetcher=yahoo) is None
        assert yahoo.asked == []

    def test_the_default_fetcher_is_looked_up_when_called(self, monkeypatch) -> None:
        yahoo = FakeYahoo(_response(QUOTES))
        monkeypatch.setattr(deferred.yf, "download", yahoo)
        assert held_session(_prices()) is not None
        assert yahoo.asked

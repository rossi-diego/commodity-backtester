"""The latest session on the contracts a crusher actually holds, when ``=F`` is in delivery.

Yahoo's continuous series stays in the expiring contract through its delivery period, so
for about two weeks around each delivery month the latest crush is a thin, expiring
contract and is flagged. A trader who never takes delivery is already in the next one.
This module fetches that session's closes for the held contracts (ZSF27.CBT and so on),
so the page can show today's level instead of the last clean day's.

Only the latest session is fetched: Yahoo drops expired contracts, so the history stays
on ``=F``. Every failure returns None and is logged; the page then keeps the last clean
session.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd
import yfinance as yf

from .constants import BEANS, DIESEL, MEAL, OIL
from .rolls import (
    BEAN_MONTHS,
    PRODUCT_MONTHS,
    front_contracts,
    grain_last_trade,
    held_contract,
    month_name,
    yahoo_ticker,
)

__all__ = ["Held", "held_session"]

logger = logging.getLogger(__name__)

# Yahoo's root symbol and calendar for each CBOT leg.
LEGS: Final[dict[str, tuple[str, tuple[int, ...]]]] = {
    BEANS: ("ZS", BEAN_MONTHS),
    MEAL: ("ZM", PRODUCT_MONTHS),
    OIL: ("ZL", PRODUCT_MONTHS),
}
# A held contract that is the =F front month must close where =F does: a wrong symbol or
# unit shows here first.
SAME_CONTRACT_TOLERANCE: Final = 0.005
# A deferred contract sits a calendar spread from the expiring one; past this the print
# is treated as bad.
CALENDAR_SPREAD_LIMIT: Final = 0.08
# Calls to Yahoo for the held contracts, each asking only for the closes still missing.
FETCH_ATTEMPTS: Final = 3

Fetcher = Callable[..., Any]


@dataclass(frozen=True)
class Held:
    """Closes on the held contracts for one session, in exchange units like ``=F``.

    ``prices`` has the three CBOT legs from the held contracts and heating oil from
    ``=F``. ``beans`` and ``products`` name the held months, as in "Jan beans".
    """

    date: pd.Timestamp
    prices: pd.Series
    beans: str
    products: str

    @property
    def pair(self) -> str:
        return f"{self.beans} beans / {self.products} products"


def held_session(prices: pd.DataFrame, fetcher: Fetcher | None = None) -> Held | None:
    """Closes of the held contracts on the last ``=F`` session, or None if unusable.

    ``fetcher`` defaults to ``yf.download``, looked up at call time so tests can patch it.
    """
    if prices.empty or any(leg not in prices for leg in (*LEGS, DIESEL)):
        return None
    date = pd.Timestamp(prices.index[-1])
    front_row = prices.iloc[-1]
    held = {leg: held_contract(date, months) for leg, (_, months) in LEGS.items()}
    tickers = {leg: yahoo_ticker(LEGS[leg][0], *held[leg]) for leg in LEGS}
    closes = _session_closes(list(tickers.values()), date, fetcher or yf.download)
    if closes is None:
        return None
    index = pd.DatetimeIndex([date])
    out = {}
    for leg, (_, months) in LEGS.items():
        front = front_contracts(index, months, grain_last_trade).iloc[0]
        same = (int(front["year"]), int(front["month"])) == held[leg]
        close, reference = closes[tickers[leg]], float(front_row[leg])
        limit = SAME_CONTRACT_TOLERANCE if same else CALENDAR_SPREAD_LIMIT
        if not abs(close / reference - 1) <= limit:
            logger.warning(
                "event=held_rejected ticker=%s close=%s front=%s limit=%s",
                tickers[leg],
                close,
                reference,
                limit,
            )
            return None
        out[leg] = close
    out[DIESEL] = float(front_row[DIESEL])
    logger.info("event=held_session date=%s tickers=%s", date.date(), ",".join(tickers.values()))
    return Held(
        date=date,
        prices=pd.Series(out, name=date),
        beans=month_name(held[BEANS][1]),
        products=month_name(held[MEAL][1]),
    )


def _session_closes(
    tickers: list[str], date: pd.Timestamp, fetcher: Fetcher
) -> dict[str, float] | None:
    """Each ticker's close on ``date``, or None unless every one has a positive close.

    Yahoo often drops some symbols from a call on these contracts, so the missing ones are
    asked again, up to ``FETCH_ATTEMPTS`` calls in all.
    """
    closes: dict[str, float] = {}
    for _ in range(FETCH_ATTEMPTS):
        missing = [ticker for ticker in tickers if ticker not in closes]
        if not missing:
            break
        closes |= _closes_once(missing, date, fetcher)
    absent = [ticker for ticker in tickers if ticker not in closes]
    if absent:
        logger.warning("event=held_fetch_failed missing=%s date=%s", ",".join(absent), date.date())
        return None
    return closes


def _closes_once(tickers: list[str], date: pd.Timestamp, fetcher: Fetcher) -> dict[str, float]:
    """The positive closes on ``date`` that one call returns; any failure returns none."""
    try:
        raw = fetcher(
            tickers,
            start=str((date - pd.Timedelta(days=10)).date()),
            end=str((date + pd.Timedelta(days=1)).date()),
            auto_adjust=True,
            progress=False,
        )["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        index = pd.DatetimeIndex(raw.index)
        raw.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
        day = date.normalize()
        if day not in raw.index:
            return {}
        row = raw.loc[day]
        values = {ticker: float(row[ticker]) for ticker in tickers if ticker in row.index}
    except Exception as exc:  # noqa: BLE001
        logger.warning("event=held_fetch_error reason=%r", str(exc))
        return {}
    return {ticker: v for ticker, v in values.items() if math.isfinite(v) and v > 0}

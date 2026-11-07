"""Copy length limits: the rails sit beside 400 px charts, the method notes share one row."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.cards import backtest_line
from src.risk import Backtest
from src.text import (
    BACKTEST_LINE,
    METHOD,
    READING_BY_MONTH,
    READING_HISTORY,
    READING_RIN,
    RISK_NOTE,
    WHAT,
)

# The gap rail appends this when the RIN is on and the view is by month.
RIN_SUFFIX = " RIN months use EPA weeks only."


def _words(text: str) -> int:
    return len(re.sub(r"<[^>]+>", "", text).split())


@pytest.mark.parametrize("key", ["crush", "share", "gap"])
def test_what_it_is_fits_the_rail(key) -> None:
    assert _words(WHAT[key]) <= 35


@pytest.mark.parametrize(
    "text", [READING_HISTORY, READING_BY_MONTH + RIN_SUFFIX, READING_RIN + RIN_SUFFIX]
)
def test_reading_fits_the_rail(text) -> None:
    assert _words(text) <= 35


def test_risk_note_fits_under_the_strip() -> None:
    assert _words(RISK_NOTE) <= 30


def test_backtest_line_fits_under_the_strip() -> None:
    worst = Backtest(
        start=pd.Timestamp("1990-01-02"),
        end=pd.Timestamp("2026-10-29"),
        days=12_345,
        exceptions=1_234,
        hs_exceptions=2_345,
        kupiec_p=0.0001,
        christoffersen_p=0.0001,
    )
    line = backtest_line(worst, BACKTEST_LINE)
    assert line.count("< 0.01") == 2
    assert "12,345" in line
    assert _words(line) <= 30


def test_method_cards_have_equal_budgets() -> None:
    assert len(METHOD) == 3
    for card in METHOD:
        assert len(card.entries) == 3
        words = sum(_words(lead) + _words(text) for lead, text in card.entries)
        assert 55 <= words + _words(card.closing) <= 80, card.title


def test_gap_copy_never_calls_it_the_biodiesel_price() -> None:
    assert "not the biodiesel price" in WHAT["gap"]

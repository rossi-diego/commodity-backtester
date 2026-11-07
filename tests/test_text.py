"""Copy length limits: the rails sit beside 400 px charts, the method notes share one row."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.cards import backtest_line, es_backtest_line, worst_line
from src.risk import Backtest
from src.text import (
    BACKTEST_LINE,
    ES_BACKTEST_LINE,
    METHOD,
    READING_BY_MONTH,
    READING_HISTORY,
    READING_RIN,
    RISK_EFFECT,
    RISK_NOTE,
    WHAT,
    WORST_LINE,
)

WORST_CASE = Backtest(
    start=pd.Timestamp("1990-01-02"),
    end=pd.Timestamp("2026-10-29"),
    days=12_345,
    exceptions=1_234,
    hs_exceptions=2_345,
    kupiec_p=0.0001,
    christoffersen_p=0.0001,
    es_exceptions=1_234,
    es_kupiec_p=0.0001,
    es_christoffersen_p=0.0001,
    es_ratio=12.345,
    z2=-12.3,
    z2_p=0.0,
)

# The gap rail appends this when the RIN is on and the view is by month.
RIN_SUFFIX = " RIN months use EPA weeks only."


def _words(text: str) -> int:
    return len(re.sub(r"<[^>]+>", "", text).split())


@pytest.mark.parametrize("key", ["crush", "share", "gap"])
def test_what_it_is_fits_the_rail(key) -> None:
    assert _words(WHAT[key]) <= 35


@pytest.mark.parametrize(
    "text",
    [*READING_HISTORY.values(), READING_BY_MONTH + RIN_SUFFIX, READING_RIN + RIN_SUFFIX],
)
def test_reading_fits_the_rail(text) -> None:
    assert _words(text) <= 35


def test_history_reading_is_one_per_section() -> None:
    assert set(READING_HISTORY) == {"crush", "share", "gap"}
    assert len(set(READING_HISTORY.values())) == 3


@pytest.mark.parametrize(
    ("key", "colour"), [("crush", "Amber"), ("share", "Blue"), ("gap", "White")]
)
def test_history_reading_names_the_line_colour(key, colour) -> None:
    """Amber is CRUSH, blue is OIL and white is INK_SERIES, the lines app.py draws."""
    assert READING_HISTORY[key].startswith(f"{colour} line: ")
    assert f"where the {colour.lower()} breaks" in READING_HISTORY[key]


@pytest.mark.parametrize("effect", ["lower", "raise", "none"])
def test_risk_note_fits_under_the_strip(effect) -> None:
    note = RISK_NOTE.format(effect=RISK_EFFECT[effect])
    assert "{" not in note
    assert _words(note) <= 30


def test_backtest_line_fits_under_the_strip() -> None:
    line = backtest_line(WORST_CASE, BACKTEST_LINE)
    assert line.count("p\N{NO-BREAK SPACE}<\N{NO-BREAK SPACE}0.01") == 2
    assert "12,345" in line
    assert _words(line) <= 30


def test_es_backtest_line_fits_under_the_strip() -> None:
    line = es_backtest_line(WORST_CASE, ES_BACKTEST_LINE)
    assert "1,234 of 12,345 changes beyond the 97.5% VaR (10.0%, 2.5% expected)" in line
    assert "mean loss 12.35× ES" in line
    assert line.endswith(
        "Acerbi–\N{WORD JOINER}Szekely\N{NO-BREAK SPACE}p\N{NO-BREAK SPACE}<\N{NO-BREAK SPACE}0.01, "
        "independence\N{NO-BREAK SPACE}p\N{NO-BREAK SPACE}<\N{NO-BREAK SPACE}0.01."
    )
    assert _words(line) <= 30


def test_worst_line_fits_under_the_strip() -> None:
    days = tuple((pd.Timestamp(f"2026-12-{day}"), -123_456.0) for day in (28, 29, 30))
    line = worst_line(days, WORST_LINE)
    assert line.count("−\N{WORD JOINER}$123,456\N{NO-BREAK SPACE}(") == 3
    assert "(30\N{NO-BREAK SPACE}Dec\N{NO-BREAK SPACE}2026)" in line
    assert _words(line) <= 30


def test_method_cards_have_equal_budgets() -> None:
    assert len(METHOD) == 3
    for card in METHOD:
        assert len(card.entries) == 3
        words = sum(_words(lead) + _words(text) for lead, text in card.entries)
        assert 55 <= words + _words(card.closing) <= 80, card.title


def test_gap_copy_never_calls_it_the_biodiesel_price() -> None:
    assert "not the biodiesel price" in WHAT["gap"]

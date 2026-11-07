"""Copy length limits: the rails sit beside 400 px charts, the method notes share one row."""

from __future__ import annotations

import re

import pytest

from src.text import (
    METHOD,
    READING_BY_MONTH,
    READING_HISTORY,
    READING_RIN,
    SCOPE,
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


def test_the_scope_line_is_short() -> None:
    assert _words(SCOPE) <= 40


def test_method_cards_have_equal_budgets() -> None:
    assert len(METHOD) == 3
    for card in METHOD:
        assert len(card.entries) == 3
        words = sum(_words(lead) + _words(text) for lead, text in card.entries)
        assert 55 <= words + _words(card.closing) <= 80, card.title


def test_gap_copy_never_calls_it_the_biodiesel_price() -> None:
    assert "not the biodiesel price" in WHAT["gap"]

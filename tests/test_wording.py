"""Tests for src.wording: the page's sentences and numbers as plain text."""

from __future__ import annotations

import pandas as pd
import pytest

from src.rins import RinResult
from src.seasonal import Reading
from src.wording import (
    after_rin_words,
    day,
    delivery_tag,
    month_sentence,
    previous_years,
    rank_clause,
    rin_note,
    usd,
)

LATEST = pd.Timestamp("2026-11-05")


def _reading(date: str = "2026-10-29", percentile: float | None = 75.0) -> Reading:
    return Reading(83.3, pd.Timestamp(date), percentile, LATEST, 1200)


@pytest.mark.parametrize(
    ("args", "shown"),
    [
        ((1234.4, "t"), "$1,234/t"),
        ((-72.2, "m³"), "−$72/m³"),
        ((72.2, "m³", True), "+$72/m³"),
        ((0.2, "t", True), "$0/t"),
        ((-0.004, "gal", True, 2), "$0.00/gal"),
        ((0.27, "gal", True, 2), "+$0.27/gal"),
    ],
)
def test_usd(args, shown) -> None:
    assert usd(*args) == shown


def test_day_has_no_leading_zero() -> None:
    assert day(pd.Timestamp("2026-11-05")) == "5 Nov 2026"


def test_rank_clause() -> None:
    assert rank_clause(_reading()) == "higher than 75% of clean days in the past 5 years"
    assert rank_clause(_reading(percentile=None)) == "with too little clean history to rank"


def test_previous_years() -> None:
    assert previous_years(2026) == "2021–2025"


def test_after_rin_words_both_signs() -> None:
    assert after_rin_words(-757.0) == (
        "the oil costs $757/m³ ($2.87/gal) less than heating oil plus the RIN"
    )
    assert after_rin_words(40.0) == "the oil still costs $40/m³ more than heating oil plus the RIN"


class TestRinNote:
    def test_week_and_age(self) -> None:
        prices = pd.Series([2.09], index=[pd.Timestamp("2026-09-21")])
        note = rin_note(RinResult(prices, "epa"), pd.Timestamp("2026-10-30"))
        assert note == "D4 week of 21 Sep (EPA, 5 weeks old). "

    def test_saved_copy_and_no_negative_age(self) -> None:
        prices = pd.Series([2.09], index=[pd.Timestamp("2026-09-21")])
        note = rin_note(RinResult(prices, "seed"), pd.Timestamp("2026-09-01"))
        assert note == "D4 week of 21 Sep (EPA, saved copy, 0 weeks old). "

    def test_no_prices_no_note(self) -> None:
        assert rin_note(RinResult(None, "none"), LATEST) == ""


def test_delivery_tag_names_the_first_session_in_delivery() -> None:
    assert delivery_tag(pd.Timestamp("2026-10-30")) == "In delivery since 30 Oct"


class TestMonthSentence:
    @staticmethod
    def _means(current: float | None) -> pd.DataFrame:
        # Months down, years across, as monthly_means returns them.
        means = pd.DataFrame({year: [70.0 + 2 * (year - 2021)] * 12 for year in range(2021, 2026)})
        means[2026] = float("nan") if current is None else current
        return means

    def test_month_so_far(self) -> None:
        sentence = month_sentence(
            self._means(90.0), pd.Timestamp("2026-10-29"), LATEST, lambda v: usd(v, "t")
        )
        assert sentence == "Oct 2026 averages $90/t, against $74/t for Oct in 2021–2025."

    def test_month_in_progress_says_so_far(self) -> None:
        latest = pd.Timestamp("2026-10-30")
        sentence = month_sentence(
            self._means(90.0), pd.Timestamp("2026-10-29"), latest, lambda v: f"{v:.0f}"
        )
        assert "averages 90 so far, against 74" in sentence

    def test_too_few_sessions(self) -> None:
        sentence = month_sentence(
            self._means(None), pd.Timestamp("2026-10-29"), LATEST, lambda v: f"{v:.0f}"
        )
        assert sentence == (
            "Monthly averages of clean sessions. Oct 2026 has too few so far to compare."
        )

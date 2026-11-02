"""Figure content and colour discipline, no rendering."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.margins import crush_table, gap_table
from src.seasonal import REFERENCE_YEAR, monthly_band
from src.visualization import (
    AXIS,
    CRUSH,
    FLAT,
    GRID,
    INK_SERIES,
    MEAL,
    MUTED,
    OIL,
    SURFACE,
    feedstock_figure,
    history_figure,
    oil_share_figure,
    seasonal_figure,
    value_figure,
)

TOKENS = {SURFACE, INK_SERIES, MUTED, GRID, AXIS, CRUSH, OIL, MEAL, FLAT}
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


@pytest.fixture(scope="module")
def tables(synthetic_prices) -> tuple[pd.DataFrame, pd.DataFrame]:
    return crush_table(synthetic_prices), gap_table(synthetic_prices)


def _trace(fig, name: str):
    return next(trace for trace in fig.data if trace.name == name)


def _hex_colors(fig) -> set[str]:
    spec = fig.to_plotly_json()
    spec["layout"].pop("template", None)
    return set(HEX.findall(str(spec)))


def _figures(tables) -> list:
    crush, gap = tables
    year = crush.index[-1].year
    band = monthly_band(crush["crush"], crush["flagged"], year)
    return [
        seasonal_figure(crush["crush"], crush["flagged"], band, year),
        history_figure(crush["crush"], crush["flagged"], "USD"),
        value_figure(crush),
        oil_share_figure(crush),
        feedstock_figure(gap),
        history_figure(gap["gap"], gap["flagged"], "Gap", color=INK_SERIES),
    ]


class TestSeasonal:
    def test_band_this_year_and_flagged_underlay(self, tables) -> None:
        crush, _ = tables
        year = crush.index[-1].year
        band = monthly_band(crush["crush"], crush["flagged"], year)
        fig = seasonal_figure(crush["crush"], crush["flagged"], band, year)
        assert _trace(fig, f"Range {year - 5} to {year - 1}").fill == "tonexty"
        this_year = _trace(fig, str(year))
        assert this_year.line.color == CRUSH
        assert pd.Timestamp(this_year.x[0]).year == REFERENCE_YEAR
        flags = crush.loc[str(year), "flagged"].to_numpy()
        assert np.isnan(np.asarray(this_year.y, dtype=float)[flags]).all()
        assert _trace(fig, "Roll or delivery").line.color == FLAT


class TestHistory:
    def test_clean_line_has_gaps_where_flagged(self, tables) -> None:
        crush, _ = tables
        fig = history_figure(crush["crush"], crush["flagged"], "USD")
        clean = np.asarray(_trace(fig, "Clean sessions").y, dtype=float)
        assert np.isnan(clean[crush["flagged"].to_numpy()]).all()
        assert list(fig.layout.xaxis.range) == [crush.index[0], crush.index[-1]]

    def test_last_dot_is_the_last_clean_value(self, tables) -> None:
        crush, _ = tables
        fig = history_figure(crush["crush"], crush["flagged"], "USD")
        dot = _trace(fig, "Latest clean session")
        clean = crush.loc[~crush["flagged"], "crush"]
        assert dot.y[0] == pytest.approx(clean.iloc[-1])


class TestColour:
    def test_no_hex_outside_tokens(self, tables) -> None:
        for fig in _figures(tables):
            assert _hex_colors(fig) <= TOKENS

    def test_amber_only_marks_the_crush(self, tables) -> None:
        crush, gap = tables
        for fig in (value_figure(crush), oil_share_figure(crush), feedstock_figure(gap)):
            assert CRUSH not in _hex_colors(fig)

    def test_one_y_axis_everywhere(self, tables) -> None:
        for fig in _figures(tables):
            axes = [key for key in fig.to_plotly_json()["layout"] if key.startswith("yaxis")]
            assert axes == ["yaxis"]

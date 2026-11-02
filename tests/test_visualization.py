"""Unit tests for src/visualization.py: figure content, no rendering."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import visualization
from src.visualization import (
    AXIS,
    ENTRY,
    EXIT,
    FLAT,
    GRID,
    INK_SERIES,
    LONG_BAND,
    MAX_VISIBLE_MARKERS,
    MUTED,
    REFERENCE,
    SURFACE,
    price_signals_figure,
    ratio_figure,
    unadjusted_pnl_figure,
    zscore_figure,
)

TOKENS = {SURFACE, INK_SERIES, MUTED, GRID, AXIS, ENTRY, EXIT, FLAT, LONG_BAND, REFERENCE}
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def _window(sessions: int = 500) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=sessions, freq="B")
    return pd.Series(np.linspace(100.0, 200.0, sessions), index=dates, name="Soybean")


def _trades(window: pd.Series, fills: int) -> pd.DataFrame:
    """Alternating fills on evenly spaced sessions. An odd count leaves a position open."""
    rows = np.linspace(1, len(window) - 2, fills, dtype=int)
    pnl = [0.0 if i % 2 == 0 else 10.0 for i in range(fills)]
    trades = pd.DataFrame(
        {
            "trade_price": window.iloc[rows].to_numpy(),
            "position": ["buy" if i % 2 == 0 else "sell" for i in range(fills)],
            "pnl_usd": pnl,
            "pnl_usd_cumsum": np.cumsum(pnl),
        },
        index=window.index[rows],
    )
    trades.index.name = "date"
    return trades


def _trace(fig, name: str):
    return next(trace for trace in fig.data if trace.name == name)


def _y_axes(fig) -> list[str]:
    return [key for key in fig.to_plotly_json()["layout"] if key.startswith("yaxis")]


def _hex_colors(fig) -> set[str]:
    """Hex colors the figure itself sets, leaving out the template."""
    spec = fig.to_plotly_json()
    spec["layout"].pop("template", None)
    return set(HEX.findall(str(spec)))


@pytest.fixture()
def stats() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    return pd.DataFrame(
        {"mean_1y": np.linspace(2.0, 2.5, 300), "z": np.linspace(-3.0, 3.0, 300)}, index=dates
    )


class TestPriceSignals:
    def test_markers_at_fill_dates(self) -> None:
        window = _window()
        trades = _trades(window, 6)
        fig = price_signals_figure(window, trades, "cents/bushel")
        entries = trades[trades["position"] == "buy"]
        exits = trades[trades["position"] == "sell"]
        assert list(_trace(fig, "Entry").x) == list(entries.index)
        assert list(_trace(fig, "Exit").x) == list(exits.index)
        assert list(_trace(fig, "Entry").y) == list(entries["trade_price"])
        assert _trace(fig, "Entry").marker.symbol == "triangle-up"
        assert _trace(fig, "Exit").marker.symbol == "triangle-down"

    def test_unit_on_axis_and_right_side(self) -> None:
        window = _window()
        fig = price_signals_figure(window, _trades(window, 2), "USD/barrel")
        assert fig.layout.yaxis.title.text == "USD/barrel"
        assert fig.layout.yaxis.side == "right"
        assert fig.layout.height == 420

    def test_x_range_is_the_window(self) -> None:
        """Autorange would pad for the markers and leave a blank strip before the window."""
        window = _window()
        fig = price_signals_figure(window, _trades(window, 2), "u")
        assert list(fig.layout.xaxis.range) == [window.index[0], window.index[-1]]

    def test_one_y_axis(self) -> None:
        window = _window()
        fig = price_signals_figure(window, _trades(window, 2), "USD/barrel")
        assert _y_axes(fig) == ["yaxis"]

    def test_no_fills_shows_price_only(self) -> None:
        empty = pd.DataFrame(columns=["trade_price", "position", "quantity", "VaR_95"])
        fig = price_signals_figure(_window(), empty, "cents/bushel")
        assert [trace.name for trace in fig.data] == ["Soybean", "Last close"]
        assert fig.data[0].line.color == FLAT
        assert len(fig.layout.shapes) == 0

    def test_long_and_flat_cover_the_window_and_join(self) -> None:
        window = _window()
        trades = _trades(window, 4)
        fig = price_signals_figure(window, trades, "u")
        long_y = pd.Series(_trace(fig, "Long").y, index=window.index)
        flat_y = pd.Series(_trace(fig, "Flat").y, index=window.index)
        assert (long_y.notna() | flat_y.notna()).all()
        for fill in trades.index:
            assert pd.notna(long_y[fill]) and pd.notna(flat_y[fill])

    def test_line_follows_the_position_between_fills(self) -> None:
        window = _window()
        trades = _trades(window, 2)
        entry, leave = trades.index
        fig = price_signals_figure(window, trades, "u")
        long_y = pd.Series(_trace(fig, "Long").y, index=window.index)
        flat_y = pd.Series(_trace(fig, "Flat").y, index=window.index)
        assert pd.isna(long_y.iloc[0])
        inside = window.index[(window.index > entry) & (window.index < leave)]
        assert long_y[inside].notna().all()
        assert flat_y[inside].isna().all()

    def test_open_position_stays_long_to_the_end(self) -> None:
        window = _window()
        fig = price_signals_figure(window, _trades(window, 3), "u")
        assert pd.notna(_trace(fig, "Long").y[-1])
        assert _trace(fig, "Last close").marker.color == ENTRY

    def test_last_close_is_flat_after_an_exit(self) -> None:
        window = _window()
        fig = price_signals_figure(window, _trades(window, 2), "u")
        last = _trace(fig, "Last close")
        assert last.marker.color == FLAT
        assert last.y[0] == window.iloc[-1]

    def test_markers_visible_at_limit(self) -> None:
        window = _window()
        fig = price_signals_figure(window, _trades(window, MAX_VISIBLE_MARKERS), "u")
        assert _trace(fig, "Entry").visible is True
        assert _trace(fig, "Exit").visible is True

    def test_markers_hidden_above_limit(self) -> None:
        window = _window()
        fig = price_signals_figure(window, _trades(window, MAX_VISIBLE_MARKERS + 1), "u")
        assert _trace(fig, "Entry").visible == "legendonly"
        assert _trace(fig, "Exit").visible == "legendonly"

    def test_limit_is_sixty(self) -> None:
        assert MAX_VISIBLE_MARKERS == 60

    def test_bands_are_layout_shapes(self) -> None:
        window = _window()
        trades = _trades(window, 4)
        fig = price_signals_figure(window, trades, "u")
        shapes = fig.layout.shapes
        assert len(shapes) == 2
        assert all(shape.type == "rect" and shape.layer == "below" for shape in shapes)
        assert all(shape.yref == "paper" and shape.line.width == 0 for shape in shapes)
        assert pd.Timestamp(shapes[0].x0) == trades.index[0]
        assert pd.Timestamp(shapes[0].x1) == trades.index[1]

    def test_open_position_band_reaches_window_end(self) -> None:
        window = _window()
        trades = _trades(window, 3)
        shapes = price_signals_figure(window, trades, "u").layout.shapes
        assert len(shapes) == 2
        assert pd.Timestamp(shapes[1].x1) == window.index[-1]

    def test_price_figure_fast_with_many_bands(self) -> None:
        window = _window(3000)
        trades = _trades(window, 300)
        started = time.perf_counter()
        fig = price_signals_figure(window, trades, "u")
        elapsed = time.perf_counter() - started
        assert len(fig.layout.shapes) == 150
        assert elapsed < 0.5, f"Building 150 bands took {elapsed:.2f} s."

    def test_no_add_vrect_in_source(self) -> None:
        code = "\n".join(
            line
            for line in Path(visualization.__file__).read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "add_vrect" not in code


class TestPnl:
    def test_realised_is_a_step_line_to_window_end(self) -> None:
        window = _window()
        trades = _trades(window, 4)
        fig = unadjusted_pnl_figure(trades, window, None)
        realised = _trace(fig, "Realised")
        assert realised.line.shape == "hv"
        assert realised.x[0] == window.index[0]
        assert realised.x[-1] == window.index[-1]
        assert realised.y[0] == 0.0
        assert realised.y[-1] == trades["pnl_usd_cumsum"].iloc[-1]
        assert realised.fill is None

    def test_axis_title(self) -> None:
        window = _window()
        fig = unadjusted_pnl_figure(_trades(window, 2), window, None)
        assert fig.layout.yaxis.title.text == "USD, per contract, unadjusted"

    def test_open_position_is_dotted_to_a_marker(self) -> None:
        window = _window()
        trades = _trades(window, 3)
        mtm = {"date": window.index[-1], "pnl_usd": 25.0}
        fig = unadjusted_pnl_figure(trades, window, mtm)
        open_trace = _trace(fig, "With open position")
        assert open_trace.line.dash == "dot"
        assert open_trace.line.color == ENTRY
        assert open_trace.x[0] == trades.index[-1]
        assert open_trace.x[-1] == window.index[-1]
        assert open_trace.y[-1] == trades["pnl_usd_cumsum"].iloc[-1] + 25.0

    def test_without_open_position_has_one_trace(self) -> None:
        window = _window()
        assert len(unadjusted_pnl_figure(_trades(window, 2), window, None).data) == 1

    def test_no_fills_is_a_zero_line(self) -> None:
        empty = pd.DataFrame(columns=["trade_price", "position", "quantity", "VaR_95"])
        fig = unadjusted_pnl_figure(empty, _window(), None)
        assert list(fig.data[0].y) == [0.0, 0.0]


class TestMonitorFigures:
    def test_ratio_has_dotted_mean_and_axis_title(self, stats: pd.DataFrame) -> None:
        ratio = stats["mean_1y"] * 1.01
        fig = ratio_figure(ratio, stats)
        assert _trace(fig, "1y mean").line.dash == "dot"
        assert _trace(fig, "1y mean").line.color == ENTRY
        assert fig.layout.yaxis.title.text == "Ratio, USD/MT basis"
        assert _y_axes(fig) == ["yaxis"]
        assert fig.layout.height == 300

    def test_zscore_has_reference_lines_and_axis_title(self, stats: pd.DataFrame) -> None:
        fig = zscore_figure(stats)
        assert fig.layout.yaxis.title.text == "Z-score (1y)"
        assert sorted(shape.y0 for shape in fig.layout.shapes) == [-2, 2]
        assert all(shape.line.color == REFERENCE for shape in fig.layout.shapes)
        assert fig.layout.height == 180

    def test_beyond_trace_only_outside_two_and_its_neighbours(self, stats: pd.DataFrame) -> None:
        fig = zscore_figure(stats)
        beyond = pd.Series(_trace(fig, "Beyond +/-2").y, index=stats.index)
        z = stats["z"]
        assert beyond[z.abs() > 2].notna().all()
        neighbour = (z.abs().shift(1) > 2) | (z.abs().shift(-1) > 2)
        assert beyond[(z.abs() <= 2) & ~neighbour].isna().all()

    def test_both_charts_share_the_x_range(self, stats: pd.DataFrame) -> None:
        ratio = stats["mean_1y"] * 1.01
        assert list(ratio_figure(ratio, stats).layout.xaxis.range) == list(
            zscore_figure(stats).layout.xaxis.range
        )


class TestMonitorAlignment:
    def test_stacked_charts_have_the_same_fixed_margin(self, stats: pd.DataFrame) -> None:
        """Automargin would size each plot by its tick labels and misalign the dates."""
        figures = [ratio_figure(stats["mean_1y"], stats), zscore_figure(stats)]
        assert figures[0].layout.margin.r == figures[1].layout.margin.r
        assert all(fig.layout.yaxis.automargin is False for fig in figures)


class TestPalette:
    def test_no_hex_outside_tokens(self, stats: pd.DataFrame) -> None:
        window = _window()
        trades = _trades(window, 5)
        mtm = {"date": window.index[-1], "pnl_usd": 25.0}
        figures = [
            price_signals_figure(window, trades, "u"),
            unadjusted_pnl_figure(trades, window, mtm),
            ratio_figure(stats["mean_1y"], stats),
            zscore_figure(stats),
        ]
        for fig in figures:
            assert _hex_colors(fig) <= TOKENS

"""Figure content and colour discipline, no rendering."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.margins import add_rins, crush_table, gap_table
from src.seasonal import monthly_means
from src.visualization import (
    AXIS,
    CRUSH,
    CURRENT_LEGEND_PX,
    FLAT,
    GRID,
    INK_SERIES,
    LEGEND_ENTRY_PX,
    MEAL,
    MONTHS,
    MUTED,
    OIL,
    RIN,
    ROLL,
    SURFACE,
    YEAR_COLORS,
    ZERO_LABEL_CHAR_PX,
    ZERO_LABEL_PAD_PX,
    ZERO_LABEL_PLOT_PX,
    by_month_figure,
    history_figure,
)

TOKENS = {SURFACE, INK_SERIES, MUTED, GRID, AXIS, CRUSH, OIL, MEAL, FLAT, RIN, *YEAR_COLORS}
# The line colours the RIN cyan must stay apart from.
LINES = {"CRUSH": CRUSH, "OIL": OIL, "MEAL": MEAL, "INK_SERIES": INK_SERIES, "FLAT": FLAT} | {
    f"YEAR_COLORS[{i}]": color for i, color in enumerate(YEAR_COLORS)
}
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


@pytest.fixture(scope="module")
def tables(synthetic_prices) -> tuple[pd.DataFrame, pd.DataFrame]:
    # A fixed end, mid-month and outside delivery, so the month to date has sessions.
    prices = synthetic_prices.loc[:"2025-10-31"]
    gap = gap_table(prices)
    # EPA weeks stop five weeks before the prices, as they do in production.
    d4 = pd.Series(1.5, index=pd.date_range("2010-07-05", gap.index[-36], freq="W-MON"))
    return crush_table(prices), add_rins(gap, d4)


def _linear(color: str) -> tuple[float, float, float]:
    """sRGB ``#rrggbb`` as linear RGB."""
    channels = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return r, g, b


def _luminance(color: str) -> float:
    r, g, b = _linear(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(first: str, second: str) -> float:
    """WCAG 2 contrast ratio."""
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _lab(color: str) -> tuple[float, float, float]:
    """CIELAB under D65."""
    r, g, b = _linear(color)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _delta_e(first: str, second: str) -> float:
    """CIE76 colour difference."""
    return float(np.linalg.norm(np.subtract(_lab(first), _lab(second))))


def _trace(fig, name: str):
    return next(trace for trace in fig.data if trace.name == name)


def _names(fig) -> list[str]:
    return [trace.name for trace in fig.data]


def _hex_colors(fig) -> set[str]:
    spec = fig.to_plotly_json()
    spec["layout"].pop("template", None)
    return set(HEX.findall(str(spec)))


def _by_month(crush: pd.DataFrame, **kwargs):
    means, counts = monthly_means(crush["crush"], crush["flagged"])
    year = crush.index[-1].year
    options = {
        "years": [year - 1, year - 2, year - 3],
        "current": year,
        "color": CRUSH,
        "y_title": "USD",
        "month_to_date": crush.index[-1].month,
    }
    return by_month_figure(means, counts, **(options | kwargs)), means


def _figures(tables) -> list:
    crush, gap = tables
    return [
        history_figure(crush["crush"], crush["flagged"], name="Crush", color=CRUSH, y_title="USD"),
        history_figure(
            gap["after_rin"],
            gap["flagged"],
            name="After",
            color=OIL,
            y_title="USD",
            reference=("Before", gap["gap"]),
            held=("Held", gap["rin_held"]),
            zero_labels=("above", "below"),
            hover={"Oil": gap["feedstock"]},
        ),
        _by_month(crush)[0],
        _by_month(crush, band=False, percent=True)[0],
    ]


class TestHistory:
    def test_clean_line_has_gaps_where_flagged(self, tables) -> None:
        crush, _ = tables
        fig = history_figure(
            crush["crush"], crush["flagged"], name="Crush", color=CRUSH, y_title="USD"
        )
        clean = np.asarray(_trace(fig, "Crush").y, dtype=float)
        assert np.isnan(clean[crush["flagged"].to_numpy()]).all()
        assert list(fig.layout.xaxis.range) == [crush.index[0], crush.index[-1]]
        roll = _trace(fig, "Roll or delivery")
        assert roll.line.color == ROLL
        assert roll.line.width <= 0.8

    def test_last_dot_is_the_last_clean_value(self, tables) -> None:
        crush, _ = tables
        fig = history_figure(
            crush["crush"], crush["flagged"], name="Crush", color=CRUSH, y_title="USD"
        )
        dot = _trace(fig, "Latest clean session")
        clean = crush.loc[~crush["flagged"], "crush"]
        assert dot.y[0] == pytest.approx(clean.iloc[-1])

    def test_held_tail_is_dashed_and_joined(self, tables) -> None:
        _, gap = tables
        fig = _figures(tables)[1]
        main = np.asarray(_trace(fig, "After").y, dtype=float)
        assert np.isnan(main[gap["rin_held"].to_numpy()]).all()
        tail = _trace(fig, "Held")
        assert tail.line.dash == "dash"
        first_held = gap.index[gap["rin_held"].to_numpy()][0]
        assert pd.Timestamp(tail.x[0]) < first_held

    def test_reference_is_dotted_and_not_the_roll_grey(self, tables) -> None:
        fig = _figures(tables)[1]
        before = _trace(fig, "Before")
        assert before.line.dash == "dot"
        assert before.line.color != _trace(fig, "Roll or delivery").line.color

    def test_tick_step_only_when_asked(self, tables) -> None:
        crush, _ = tables
        args = (crush["crush"], crush["flagged"])
        fig = history_figure(*args, name="C", color=CRUSH, y_title="", y_dtick=250)
        assert fig.layout.yaxis.dtick == 250
        fig = history_figure(*args, name="C", color=CRUSH, y_title="")
        assert fig.layout.yaxis.dtick is None

    def test_hover_carries_the_components(self, tables) -> None:
        fig = _figures(tables)[1]
        assert "Oil: %{customdata[0]" in _trace(fig, "After").hovertemplate

    def test_zero_labels_only_when_zero_is_drawn(self, tables) -> None:
        crush, _ = tables
        assert len(_figures(tables)[1].layout.annotations) == 2
        positive = crush["crush"].abs() + 1
        fig = history_figure(
            positive, crush["flagged"], name="P", color=CRUSH, y_title="", zero_labels=("a", "b")
        )
        assert not fig.layout.annotations

    def test_zero_labels_sit_where_the_line_is_away_from_zero(self) -> None:
        """On a V, 'above' goes over the trough and 'below' under an arm, off the line."""
        dates = pd.bdate_range("2024-01-01", periods=200)
        values = pd.Series(np.abs(np.linspace(-100.0, 100.0, 200)) - 40.0, index=dates)
        fig = _zero_labelled(values)
        windows = _label_windows(fig, values)
        assert set(windows) == {"above", "below"}
        for note in fig.layout.annotations:
            assert note.xref == "x"
            assert note.bgcolor == "rgba(20,27,35,0.85)"
        assert values.iloc[windows["above"]].max() < 0
        assert values.iloc[windows["below"]].min() > 0

    def test_zero_labels_find_room_when_every_tenth_crosses_zero(self) -> None:
        """Each of 10 bins crosses zero, so the labels move to finer bins, still clear."""
        dates = pd.bdate_range("2020-01-01", periods=800)
        values = pd.Series(100.0 * np.sin(np.arange(800) * 2 * np.pi / 140), index=dates)
        for chunk in np.array_split(np.arange(800), 10):
            assert values.iloc[chunk].min() < 0 < values.iloc[chunk].max()
        fig = _zero_labelled(values)
        windows = _label_windows(fig, values)
        assert set(windows) == {"above", "below"}
        assert values.iloc[windows["above"]].max() < 0
        assert values.iloc[windows["below"]].min() > 0

    def test_zero_labels_are_dropped_when_no_place_is_clear(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=400)
        values = pd.Series(np.where(np.arange(400) % 2, 50.0, -50.0), index=dates)
        assert not _zero_labelled(values).layout.annotations

    def test_zero_labels_also_clear_the_reference_line(self) -> None:
        """A reference just over zero fills the 'above' box everywhere, so only 'below' stays."""
        dates = pd.bdate_range("2024-01-01", periods=200)
        values = pd.Series(np.abs(np.linspace(-100.0, 100.0, 200)) - 40.0, index=dates)
        reference = pd.Series(1.0, index=dates)
        fig = _zero_labelled(values, reference)
        assert [note.text for note in fig.layout.annotations] == ["below"]

    def test_an_all_nan_reference_raises_no_warning(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=200)
        values = pd.Series(np.abs(np.linspace(-100.0, 100.0, 200)) - 40.0, index=dates)
        reference = pd.Series(np.nan, index=dates)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = _zero_labelled(values, reference)
        assert {note.text for note in fig.layout.annotations} == {"above", "below"}


class TestByMonth:
    def test_current_year_bold_past_years_in_their_own_colours(self, tables) -> None:
        crush, _ = tables
        fig, _ = _by_month(crush)
        year = crush.index[-1].year
        current = _trace(fig, f"{year} (to date)")
        assert current.line.color == CRUSH
        assert current.line.width > _trace(fig, str(year - 1)).line.width
        assert _trace(fig, str(year - 1)).line.color == YEAR_COLORS[0]
        assert _trace(fig, str(year - 3)).line.color == YEAR_COLORS[2]
        past = {_trace(fig, str(year - back)).line.color for back in (1, 2, 3)}
        assert len(past) == 3
        assert CRUSH not in past
        assert list(current.x) == MONTHS
        assert fig.layout.legend.traceorder == "reversed"
        lines = [name for name in _names(fig) if name and name[:4].isdigit()]
        assert lines == [str(year - 3), str(year - 2), str(year - 1), f"{year} (to date)"]

    def test_current_year_is_named_to_date_only_with_a_month_in_progress(self, tables) -> None:
        crush, _ = tables
        year = crush.index[-1].year
        fig, _ = _by_month(crush, month_to_date=None)
        assert str(year) in _names(fig)
        assert f"{year} (to date)" not in _names(fig)
        assert "To date" not in _names(fig)

    def test_past_years_draw_in_year_colours_solid(self, tables) -> None:
        """Picked years one to five back are solid, in YEAR_COLORS, never the grey of the band."""
        crush, _ = tables
        year = crush.index[-1].year
        fig, _ = _by_month(crush, years=[year - n for n in range(1, 6)])
        for n in range(1, 6):
            line = _trace(fig, str(year - n)).line
            assert line.color == YEAR_COLORS[n - 1]
            assert line.dash is None
            assert line.color != FLAT

    def test_a_year_keeps_its_colour_whichever_others_are_picked(self, tables) -> None:
        crush, _ = tables
        year = crush.index[-1].year
        alone, _ = _by_month(crush, years=[year - 3])
        assert _trace(alone, str(year - 3)).line.color == YEAR_COLORS[2]
        far, _ = _by_month(crush, years=[year - 1, year - 6])
        assert _trace(far, str(year - 6)).line.color == YEAR_COLORS[0]
        assert _trace(far, str(year - 6)).line.dash == "dash"
        assert _trace(far, str(year - 1)).line.dash is None

    def test_month_to_date_is_a_hollow_marker_off_the_line(self, tables) -> None:
        crush = tables[0]
        fig, means = _by_month(crush)
        month, year = crush.index[-1].month, crush.index[-1].year
        line = np.asarray(_trace(fig, f"{year} (to date)").y, dtype=float)
        assert np.isnan(line[month - 1])
        marker = _trace(fig, "To date")
        # The year's "(to date)" entry names the marker, which shares its legend group.
        assert marker.showlegend is False
        assert marker.legendgroup == _trace(fig, f"{year} (to date)").legendgroup
        assert f"<extra>{MONTHS[month - 1]} to date</extra>" in marker.hovertemplate
        assert marker.y[0] == pytest.approx(means.loc[month, year])
        assert marker.marker.color == SURFACE

    def test_band_is_optional(self, tables) -> None:
        crush, _ = tables
        year = crush.index[-1].year
        with_band, _ = _by_month(crush)
        assert {"Range", "Avg"} <= set(_names(with_band))
        assert f"Avg {year - 5}–{year - 1}" in _trace(with_band, "Avg").hovertemplate
        without, _ = _by_month(crush, band=False)
        assert not any(name and name.startswith("Range") for name in _names(without))

    def test_five_past_years_have_five_colours_and_none_dotted(self, tables) -> None:
        crush, _ = tables
        year = crush.index[-1].year
        fig, _ = _by_month(crush, years=[year - n for n in range(1, 6)])
        lines = [_trace(fig, str(year - n)).line for n in range(1, 6)]
        assert len({line.color for line in lines}) == 5
        assert "dot" not in {line.dash for line in lines}
        assert _trace(fig, str(year - 1)).mode == "lines+markers"
        assert _trace(fig, "Avg").line.dash == "dot"

    def test_legend_sits_below_and_months_are_level(self, tables) -> None:
        fig, _ = _by_month(tables[0])
        assert fig.layout.legend.y < 0
        assert fig.layout.legend.yanchor == "top"
        # Three entries to a row on a phone: a fixed entry width that every name fits.
        assert fig.layout.legend.entrywidth == 36
        assert fig.layout.legend.entrywidthmode == "pixels"
        shown = [t for t in fig.data if t.showlegend is not False]
        year = tables[0].index[-1].year
        current = _trace(fig, f"{year} (to date)")
        assert current.legendwidth == CURRENT_LEGEND_PX
        others = [t.name for t in shown if t is not current]
        assert max(len(name) for name in others) <= 7
        # 7 characters fit 36 px at 10.5 px, so 14 fit the current year's own width.
        assert len(current.name) / 7 * LEGEND_ENTRY_PX <= CURRENT_LEGEND_PX
        assert fig.layout.xaxis.tickangle == 0
        history = _figures(tables)[0]
        assert history.layout.legend.y > 1

    def test_percent_axis(self, tables) -> None:
        fig = _figures(tables)[3]
        assert fig.layout.yaxis.ticksuffix == "%"


class TestColour:
    def test_no_hex_outside_tokens(self, tables) -> None:
        for fig in _figures(tables):
            assert _hex_colors(fig) <= TOKENS

    def test_rin_cyan_is_its_own_colour(self) -> None:
        assert RIN not in LINES.values()
        assert YEAR_COLORS == ("#2bb39b", "#a98bf0", "#86b83a", "#e5664c", "#d86fb0")

    @pytest.mark.parametrize("name", list(LINES))
    def test_rin_cyan_is_far_from_every_other_line(self, name) -> None:
        assert _delta_e(RIN, LINES[name]) >= 30, name

    def test_rin_cyan_reads_on_the_surface(self) -> None:
        assert _contrast(RIN, SURFACE) >= 4.5

    def test_delta_e_helper_matches_known_values(self) -> None:
        assert _delta_e("#000000", "#ffffff") == pytest.approx(100, abs=0.05)
        assert _delta_e(RIN, RIN) == 0

    def test_one_y_axis_everywhere(self, tables) -> None:
        for fig in _figures(tables):
            axes = [key for key in fig.to_plotly_json()["layout"] if key.startswith("yaxis")]
            assert axes == ["yaxis"]


def _zero_labelled(values: pd.Series, reference: pd.Series | None = None) -> go.Figure:
    return history_figure(
        values,
        pd.Series(False, index=values.index),
        name="Z",
        color=OIL,
        y_title="",
        reference=None if reference is None else ("Ref", reference),
        zero_labels=("above", "below"),
    )


def _label_windows(fig: go.Figure, values: pd.Series) -> dict[str, slice]:
    """The sessions each zero label covers, from its left edge and its budgeted width."""
    windows = {}
    for note in fig.layout.annotations:
        assert note.xanchor == "left"
        first = values.index.get_loc(pd.Timestamp(note.x))
        width_px = len(note.text) * ZERO_LABEL_CHAR_PX + ZERO_LABEL_PAD_PX
        width = int(np.ceil(len(values) * width_px / ZERO_LABEL_PLOT_PX[0]))
        assert first + width <= len(values)
        windows[note.text] = slice(first, first + width)
    return windows

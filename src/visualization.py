"""Plotly figure builders for the crush margin page. Pure, no UI imports.

Every section draws the same two figures, so the three charts read alike:
``history_figure`` (a metric over time) and ``by_month_figure`` (monthly
averages, one line per year).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .seasonal import monthly_band

__all__ = ["by_month_figure", "history_figure"]

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

SURFACE: Final[str] = "#141b23"
INK_SERIES: Final[str] = "#eef2f6"
MUTED: Final[str] = "#a3afbd"
GRID: Final[str] = "#232c37"
AXIS: Final[str] = "#3b4653"
# Amber is the crush margin and nothing else.
CRUSH: Final[str] = "#c58512"
OIL: Final[str] = "#4d8fe8"
MEAL: Final[str] = "#c76fae"
FLAT: Final[str] = "#7d8796"
# FLAT at 0.35 opacity for roll and delivery sessions, deliberately quiet: context, not a mark.
ROLL: Final[str] = "rgba(125,135,150,0.35)"
BAND_FILL: Final[str] = "rgba(125, 135, 150, 0.22)"
# The fill behind the zero labels: SURFACE, nearly opaque, so the grid does not cross the text.
LABEL_FILL: Final[str] = "rgba(20,27,35,0.85)"
# Past years in the by-month view, by how many years back: teal is last year. Hues kept
# clear of amber, blue and the near-white ink, which the current year draws in. Each year
# keeps its colour whichever others are picked; six to ten years back repeat the five
# colours, dashed.
YEAR_COLORS: Final[tuple[str, ...]] = ("#2bb39b", "#a98bf0", "#86b83a", "#e5664c", "#d86fb0")
# Zero labels: the bin counts tried in turn, and the label box in pixels. The plot size
# is not known when the figure is built, so the box is sized against a conservative
# plot: narrower than any desktop chart, so the label is never wider than budgeted there.
ZERO_LABEL_BINS: Final[tuple[int, ...]] = (10, 20, 40)
ZERO_LABEL_PLOT_PX: Final[tuple[int, int]] = (640, 300)
ZERO_LABEL_CHAR_PX: Final[float] = 5.5
ZERO_LABEL_PAD_PX: Final[int] = 8
ZERO_LABEL_HEIGHT_PX: Final[int] = 20

HEIGHT: Final[int] = 400
MONTHS: Final[list[str]] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]  # fmt: skip


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def history_figure(
    values: pd.Series,
    flagged: pd.Series,
    *,
    name: str,
    color: str,
    y_title: str,
    percent: bool = False,
    reference: tuple[str, pd.Series] | None = None,
    held: tuple[str, pd.Series] | None = None,
    zero_labels: tuple[str, str] | None = None,
    hover: dict[str, pd.Series] | None = None,
    y_dtick: float | None = None,
) -> go.Figure:
    """A metric over time, with roll and delivery sessions drawn as a thin grey line.

    ``reference`` adds a dotted comparison line in faded ink, apart from the roll grey.
    ``held`` marks the sessions whose value rests on a carried-forward input; they are
    drawn dashed and named. ``hover`` adds component levels to the tooltip of the main
    line. ``y_dtick`` fixes the step between y ticks; by default Plotly picks it.
    """
    flags = flagged.reindex(values.index, fill_value=True).astype(bool)
    held_mask = (
        held[1].reindex(values.index, fill_value=False).astype(bool)
        if held is not None
        else pd.Series(False, index=values.index)
    )
    fig = go.Figure()
    if reference is not None:
        ref_name, ref_values = reference
        faded = _fade(INK_SERIES, 0.6)
        reference_values = ref_values.where(~flags)
        fig.add_trace(
            _line(values.index, reference_values, ref_name, faded, 1.25, False, dash="dot")
        )
    fig.add_trace(_line(values.index, values, "Roll or delivery", ROLL, 0.75, hover=False))
    main = _line(values.index, values.where(~flags & ~held_mask), name, color, 1.75)
    _add_hover(main, values.index, hover, percent)
    fig.add_trace(main)
    if held is not None and held_mask.any():
        # Join the dashed tail to the last real point so the line does not break.
        start = int(np.flatnonzero(held_mask.to_numpy())[0])
        tail = values.iloc[max(start - 1, 0) :].where(~flags)
        fig.add_trace(_line(tail.index, tail, held[0], color, 1.75, dash="dash"))
    clean = values[~flags]
    if not clean.empty:
        fig.add_trace(_dot(clean.index[-1:], clean.iloc[-1:], color))
    fig.update_layout(**_layout())
    fig.update_xaxes(range=[values.index[0], values.index[-1]])
    _y_axis(fig, y_title, percent)
    if y_dtick is not None:
        fig.update_yaxes(dtick=y_dtick)
    if zero_labels is not None:
        others = [] if reference is None else [reference[1].where(~flags)]
        _zero_labels(fig, values, zero_labels, others)
    return fig


def by_month_figure(
    means: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    years: list[int],
    current: int,
    color: str,
    y_title: str,
    band: bool = True,
    percent: bool = False,
    month_to_date: int | None = None,
) -> go.Figure:
    """Monthly averages of clean sessions, one line per year, on a January-to-December axis.

    ``current`` is drawn bold in the metric colour; ``years`` are earlier years, each
    in its own colour. ``month_to_date`` is the month in progress, drawn as a hollow marker
    and not joined to the line. ``band`` shades the range of the five years before
    ``current``, with their average dotted. The legend sits under
    the plot, where it has the full width for the picked years.
    """
    fmt = _value_format(percent)
    fig = go.Figure()
    if band:
        stats = monthly_band(means, current)
        previous = f"{current - 5}–{current - 1}"
        fig.add_trace(_band_edge(stats["low"], "Range low", legend=False))
        # Short legend names keep three entries to a row on a phone; the hover and the
        # reading above the chart name the years.
        fig.add_trace(_band_edge(stats["high"], "Range", fill=True))
        average = _month_line(stats["average"], "Avg", FLAT, 1.25, fmt, "dot", markers=False)
        average.update(hovertemplate=f"{fmt}<extra>Avg {previous}</extra>")
        fig.add_trace(average)
    # Oldest first, so the reversed legend reads from the current year back.
    past = sorted(year for year in years if year != current and year in means)
    for year in past:
        back, cycle = divmod(current - year - 1, len(YEAR_COLORS))
        fig.add_trace(
            _month_line(
                means[year],
                str(year),
                YEAR_COLORS[cycle],
                1.5,
                fmt,
                "dash" if back else None,
                counts=counts[year],
            )
        )
    if current in means:
        line = means[current]
        if month_to_date is not None:
            line = line.where(line.index != month_to_date)
        fig.add_trace(_month_line(line, str(current), color, 2.5, fmt, counts=counts[current]))
        to_date = None if month_to_date is None else means[current].iloc[month_to_date - 1]
        if month_to_date is not None and pd.notna(to_date):
            fig.add_trace(
                go.Scatter(
                    x=[MONTHS[month_to_date - 1]],
                    y=[float(to_date)],
                    mode="markers",
                    name="To date",
                    customdata=[int(counts[current].iloc[month_to_date - 1])],
                    hovertemplate=fmt + " (%{customdata} sessions so far)"
                    f"<extra>{MONTHS[month_to_date - 1]} to date</extra>",
                    marker={"size": 10, "color": SURFACE, "line": {"color": color, "width": 2}},
                    cliponaxis=False,
                )
            )
    fig.update_layout(**_layout())
    fig.update_layout(
        legend={
            "traceorder": "reversed",
            "orientation": "h",
            "y": -0.12,
            "yanchor": "top",
            "x": 0,
            "xanchor": "left",
            # A wrapped legend gives every entry the widest one's width. Names of at most
            # 7 characters in a fixed 36 px of text (the symbol comes on top) fit three
            # to a row on a phone.
            "font": {"size": 10.5},
            "entrywidth": 36,
            "entrywidthmode": "pixels",
        },
        margin={"b": 56},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=MONTHS, range=[-0.3, 11.3], tickangle=0)
    _y_axis(fig, y_title, percent)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layout() -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "height": HEIGHT,
        "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
        "font": {"family": "IBM Plex Sans, sans-serif", "size": 12, "color": MUTED},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.02, "x": 0, "xanchor": "left", "yanchor": "bottom"},
        "xaxis": {"linecolor": AXIS, "showgrid": False, "ticks": "outside", "tickcolor": AXIS},
        "yaxis": {
            "gridcolor": GRID,
            "zeroline": False,
            "side": "right",
            "tickformat": ",.0f",
            "ticklabelstandoff": 8,
            "title_font": {"size": 11},
        },
    }


def _y_axis(fig: go.Figure, title: str, percent: bool) -> None:
    if percent:
        fig.update_yaxes(title=title, ticksuffix="%", tickformat=".0f")
    else:
        fig.update_yaxes(title=title, zeroline=True, zerolinecolor=AXIS)


def _value_format(percent: bool) -> str:
    return "%{y:.1f}%" if percent else "%{y:,.1f}"


def _line(
    x: Any,
    y: Any,
    name: str,
    color: str,
    width: float,
    hover: bool = True,
    dash: str | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        hovertemplate="%{y:,.1f}" if hover else None,
        hoverinfo=None if hover else "skip",
        line={"color": color, "width": width, "dash": dash},
    )


def _add_hover(
    trace: go.Scatter, index: pd.Index, hover: dict[str, pd.Series] | None, percent: bool
) -> None:
    template = _value_format(percent)
    if hover:
        names = list(hover)
        data = pd.concat([hover[name].reindex(index) for name in names], axis=1)
        trace.update(customdata=data.to_numpy())
        template += "".join(
            f"<br>{name}: %{{customdata[{i}]:,.0f}}" for i, name in enumerate(names)
        )
    trace.update(hovertemplate=template)


def _month_line(
    values: pd.Series,
    name: str,
    color: str,
    width: float,
    fmt: str,
    dash: str | None = None,
    counts: pd.Series | None = None,
    markers: bool = True,
) -> go.Scatter:
    trace = go.Scatter(
        x=MONTHS,
        y=values.to_numpy(),
        mode="lines+markers" if markers else "lines",
        name=name,
        connectgaps=False,
        hovertemplate=fmt,
        line={"color": color, "width": width, "dash": dash},
        marker={"size": 5, "color": color},
    )
    if counts is not None:
        trace.update(customdata=counts.to_numpy(), hovertemplate=fmt + " (%{customdata} sessions)")
    return trace


def _band_edge(values: pd.Series, name: str, legend: bool = True, fill: bool = False) -> go.Scatter:
    return go.Scatter(
        x=MONTHS,
        y=values.to_numpy(),
        mode="lines",
        name=name,
        showlegend=legend,
        fill="tonexty" if fill else None,
        fillcolor=BAND_FILL if fill else None,
        hoverinfo="skip",
        line={"color": FLAT, "width": 0},
    )


def _zero_labels(
    fig: go.Figure, values: pd.Series, labels: tuple[str, str], others: list[pd.Series]
) -> None:
    """Name both sides of zero, when zero is inside the drawn range.

    "Above zero" sits just over the zero line, "below zero" just under it, each where no
    drawn line (``values`` and ``others``) passes through the label's box. A label with
    no clear place is left out rather than drawn over a line.
    """
    drawn = values.dropna()
    if drawn.empty or not drawn.min() < 0 < drawn.max():
        return
    lines = [drawn.to_numpy(dtype=float)]
    for other in others:
        s = other.reindex(drawn.index)
        # An all-NaN line has nothing to clear, and nanmax would warn on it.
        if not s.notna().any():
            continue
        lines.append(s.to_numpy(dtype=float))
    span = float(np.nanmax([np.nanmax(line) for line in lines])) - float(
        np.nanmin([np.nanmin(line) for line in lines])
    )
    height = ZERO_LABEL_HEIGHT_PX * span / ZERO_LABEL_PLOT_PX[1]
    for text, side in ((labels[0], 1), (labels[1], -1)):
        start = _zero_label_start(lines, text, side, height)
        if start is None:
            continue
        fig.add_annotation(
            x=drawn.index[start],
            xref="x",
            y=0,
            yshift=10 * side,
            text=text,
            showarrow=False,
            xanchor="left",
            bgcolor=LABEL_FILL,
            borderpad=2,
            font={"size": 11, "color": MUTED},
        )


def _zero_label_start(lines: list[np.ndarray], text: str, side: int, height: float) -> int | None:
    """The first session of a clear place for a zero label, or None when there is none.

    The label box spans ``(0, height)`` above zero (``side`` 1) or ``(-height, 0)``
    below, and the label's width in sessions from its left edge. Bin starts are tried
    at 10, then 20, then 40 bins. A place where the main line stays on the other side of
    zero is preferred; failing that, any place where every line clears the box. Among
    the places at the first bin count that has any, the one with the widest clearance wins.
    """
    count = len(lines[0])
    width_px = len(text) * ZERO_LABEL_CHAR_PX + ZERO_LABEL_PAD_PX
    width = max(int(np.ceil(count * width_px / ZERO_LABEL_PLOT_PX[0])), 2)
    low, high = (0.0, height) if side > 0 else (-height, 0.0)
    # Each drawn segment between two sessions, as its lowest and highest value.
    segments = []
    for line in lines:
        lo, hi = np.fmin(line[:-1], line[1:]), np.fmax(line[:-1], line[1:])
        # Distance from the box; zero or less means the segment crosses it.
        distance = np.maximum(low - hi, lo - high)
        segments.append(np.where(np.isnan(distance), np.inf, distance))
    clearance = np.minimum.reduce(segments)
    opposite = -side * lines[0] > 0
    for strict in (True, False):
        for bins in ZERO_LABEL_BINS:
            best: tuple[float, int] | None = None
            for chunk in np.array_split(np.arange(count), bins):
                if not len(chunk) or chunk[0] + width > count:
                    continue
                first = int(chunk[0])
                room = float(clearance[first : first + width - 1].min())
                if room <= 0 or (strict and not opposite[first : first + width].all()):
                    continue
                if best is None or room > best[0]:
                    best = (room, first)
            if best is not None:
                return best[1]
    return None


def _fade(color: str, alpha: float) -> str:
    """A ``#rrggbb`` token as ``rgba`` at ``alpha`` opacity."""
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def _dot(x: Any, y: pd.Series, color: str) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=[float(value) for value in y],
        mode="markers",
        name="Latest clean session",
        showlegend=False,
        hoverinfo="skip",
        cliponaxis=False,
        marker={"size": 9, "color": color, "line": {"color": SURFACE, "width": 2}},
    )

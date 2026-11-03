"""Soybean Crush Margin: Streamlit UI. All computation lives in ``src``."""

from __future__ import annotations

import html
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cards import (
    Kpi,
    kpi_row_html,
    method_row_html,
    rail_html,
    section_head_html,
    split_bar_html,
)
from src.constants import BUSHELS_PER_TONNE
from src.data_loader import LoadResult, load_prices
from src.margins import add_rins, crush_table, gap_table, per_gallon
from src.rins import RinResult, load_d4
from src.seasonal import (
    LOOKBACK_YEARS,
    Reading,
    latest_reading,
    month_vs_average,
    monthly_means,
    range_52w,
)
from src.text import (
    FULL_METHOD,
    LINKEDIN_URL,
    METHOD,
    READING_BY_MONTH,
    READING_HISTORY,
    READING_RIN,
    REPOSITORY_URL,
    SCOPE,
    WHAT,
)
from src.visualization import CRUSH, INK_SERIES, MONTHS, OIL, by_month_figure, history_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

VIEWS = ["History", "By month"]
WINDOWS = ["5Y", "10Y", "All"]
DEFAULT_WINDOW = "10Y"
YEAR_CHOICES = 10
MAX_YEARS = 5

# Amber means "the crush margin" and nothing else: controls and the other cards stay neutral.
# The rules target Streamlit test ids and key classes, so redo the screenshots when upgrading.
# The top padding clears the 56 px bar Streamlit Cloud draws over the app from its parent page.
APP_CSS = """<style>
:root{--page:#0b1016;--card:#141b23;--raised:#1e2732;--line:#2a3440;--ink:#eef2f6;--mut:#a3afbd;--head:linear-gradient(115deg,#1a2430 0%,#16212c 55%,#3a2c10 100%);--crush:#c58512;--oil:#4d8fe8;--meal:#c76fae;--step:rgba(255,255,255,.07)}
[data-testid='stMainBlockContainer']{max-width:1480px;margin:0 auto;padding:3.75rem 2rem 2rem}
[data-testid='stHeaderActionElements']{display:none}
:focus-visible{outline:2px solid var(--ink)!important;outline-offset:2px}
.cm-hero{background:var(--head);border-radius:12px;padding:1.1rem 1.5rem;color:#fff;display:grid;grid-template-columns:minmax(0,13fr) minmax(0,11fr);gap:1.5rem;align-items:center}
.cm-hero h1{color:#fff;font-size:2.25rem;font-weight:600;letter-spacing:-.02em;line-height:1.1;margin:0 0 .4rem;padding:0}
.cm-q{font-size:1.0625rem;line-height:1.35;margin:0 0 .4rem;color:#fff}
.cm-meta{font-size:.875rem;line-height:1.45;margin:0;color:#dbe6f2}
.cm-meta a{color:#fff;font-weight:500}
.cm-meta strong{color:#fff;font-weight:600}
.cm-scope{margin-top:.4rem;font-size:.8125rem}
.cm-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:0;padding:0;list-style:none}
.cm-steps a{display:block;height:100%;box-sizing:border-box;background:var(--step);border:1px solid rgba(255,255,255,.18);border-radius:10px;padding:.6rem .75rem;font-size:.8125rem;line-height:1.3;color:#e8eff7;text-decoration:none}
.cm-steps a:hover{border-color:rgba(255,255,255,.45)}
.cm-steps b{color:#fff;font-size:.9375rem;font-weight:600}
.cm-steps span{display:block;margin-top:.3rem}
.cm-steps i,.cm-label i{font-style:normal;display:inline-block;width:1.35rem;height:1.35rem;line-height:1.35rem;text-align:center;border-radius:50%;background:#fff;color:#0b1016;font-weight:600;font-size:.75rem;margin-right:.35rem;flex:none}
.cm-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(5,auto);gap:0 12px}
.cm-kpi{grid-row:span 5;display:grid;grid-template-rows:subgrid;row-gap:8px;background:var(--card);border:1px solid var(--line);border-top:3px solid var(--acc);border-radius:10px;padding:14px 18px 16px;color:var(--ink);text-decoration:none}
.cm-kpi:hover{border-color:#3b4653;border-top-color:var(--acc)}
.cm-kpi-share{--acc:var(--oil)}
.cm-kpi-gap{--acc:var(--ink)}
.cm-kpi-crush{--acc:var(--crush)}
.cm-kpi-crush .cm-value{color:var(--crush)}
.cm-label{margin:0;display:flex;align-items:center;flex-wrap:wrap;gap:.2rem;font-size:.875rem;font-weight:600;color:var(--ink)}
.cm-tag{font-style:normal;font-weight:400;font-size:.6875rem;border:1px solid var(--line);border-radius:4px;padding:0 .3rem;margin-left:.3rem;color:var(--mut)}
.cm-value{margin:0;font-size:2rem;font-weight:600;line-height:1.1;letter-spacing:-.01em;white-space:nowrap;font-variant-numeric:tabular-nums}
.cm-value small{font-size:.8125rem;font-weight:400;letter-spacing:0;color:var(--mut);margin-left:.5rem;white-space:normal}
.cm-detail{font-size:.875rem;line-height:1.4;color:var(--ink);align-self:center}
.cm-rank{display:flex;flex-direction:column;gap:5px;font-size:.75rem;color:var(--mut)}
.cm-track{position:relative;height:6px;border-radius:3px;background:repeating-linear-gradient(90deg,var(--raised) 0 calc(20% - 2px),transparent calc(20% - 2px) 20%)}
.cm-mark{position:absolute;top:-4px;width:4px;height:14px;border-radius:2px;background:var(--acc);transform:translateX(-50%)}
.cm-caption{margin:0;font-size:.78rem;line-height:1.35;color:var(--mut)}
.cm-split{display:flex;gap:2px;height:10px;border-radius:5px;overflow:hidden}
.cm-split-oil{background:var(--oil)}
.cm-split-meal{flex:1;background:var(--meal)}
.cm-legend{margin:5px 0 0;font-size:.8125rem;font-variant-numeric:tabular-nums}
.cm-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:.3rem}
.cm-key-oil{background:var(--oil)}
.cm-key-meal{background:var(--meal);margin-left:.75rem}
[class*='st-key-card-sec']{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem 1rem;gap:.6rem}
.cm-head{scroll-margin-top:72px}
.cm-head p{margin:0}
.cm-head .cm-label{font-size:1.0625rem}
.cm-head p+p{margin-top:.3rem;font-size:.9375rem;line-height:1.4;color:var(--ink)}
[class*='st-key-bar-']{min-height:40px;align-items:center;flex-wrap:wrap}
.st-key-rin-gap{margin-left:auto}
[class*='st-key-view-'] [data-testid='stButtonGroup']{justify-content:flex-end}
[class*='st-key-body-'] [data-testid='stHorizontalBlock']{align-items:stretch}
.cm-rail{height:400px;box-sizing:border-box;display:flex;flex-direction:column;gap:.75rem;border-left:1px solid var(--line);padding-left:1rem;font-size:.8125rem;line-height:1.45;color:var(--ink);overflow:hidden}
.cm-rail h3{font-size:.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 .2rem;padding:0}
.cm-rail p{margin:0}
.cm-today{margin-top:auto}
.cm-today dl{display:grid;grid-template-columns:1fr auto;gap:.25rem .75rem;margin:0;padding:0}
.cm-today dt{margin:0;padding:0;color:var(--mut)}
.cm-today dd{margin:0;padding:0;text-align:right;font-variant-numeric:tabular-nums}
.cm-method{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.cm-method section{display:flex;flex-direction:column;background:var(--card);border:1px solid var(--line);border-top:3px solid var(--line);border-radius:10px;padding:.9rem 1rem}
.cm-method h2{font-size:.9375rem;font-weight:600;margin:0 0 .5rem;padding:0;color:var(--ink)}
.cm-method p{font-size:.875rem;line-height:1.45;color:var(--ink);margin:0 0 .5rem}
.cm-method a{color:var(--ink)}
.cm-method .cm-closing{margin:auto 0 0;padding-top:.25rem;font-size:.8125rem;color:var(--mut)}
[data-testid='stCaptionContainer']{opacity:1}
[data-testid='stCaptionContainer'] p{color:var(--mut)}
[data-testid='stCaptionContainer'] a{color:var(--ink)}
@media (max-width:1279px){
 [class*='st-key-body-'] [data-testid='stColumn']{min-width:100%}
 .cm-rail{height:auto;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;border-left:0;border-top:1px solid var(--line);padding:.75rem 0 0}
 .cm-today{margin-top:0}
}
@media (max-width:1024px){
 .cm-hero{grid-template-columns:1fr;gap:.75rem}
 .cm-steps span{display:none}
 [class*='st-key-card-sec'] [data-testid='stColumn']{min-width:100%}
 .cm-value{font-size:1.625rem}
}
@media (max-width:760px){
 .cm-kpis{grid-template-columns:1fr;grid-template-rows:none;gap:12px}
 .cm-kpi{grid-row:auto;grid-template-rows:none}
 .cm-method{grid-template-columns:1fr}
 .cm-rail{grid-template-columns:1fr}
}
@media (max-width:640px){
 [data-testid='stMainBlockContainer']{padding:3.75rem .75rem 2rem}
 .cm-hero{padding:1rem;border-radius:10px}
 .cm-hero h1{font-size:1.75rem}
 .cm-q{font-size:1rem}
 .cm-steps{display:none}
 [class*='st-key-card-sec']{padding:.7rem .8rem}
 [data-testid='stPlotlyChart'] .nsewdrag{pointer-events:none}
 [data-testid='stPlotlyChart'] .modebar-container{display:none}
}
</style>"""

# Static markup: the runtime values are the data line and the links.
HERO_HTML = """<div class="cm-hero"><div>
<h1>Soybean Crush Margin</h1>
<p class="cm-q">What is a soybean crusher earning today, and how much of it is biodiesel?</p>
<p class="cm-meta">{data_line}</p>
<p class="cm-meta">Built by <strong>Diego Rossi</strong>. <a href="{linkedin}" target="_blank" rel="noopener">LinkedIn</a> &middot; <a href="{repository}" target="_blank" rel="noopener">GitHub</a></p>
<p class="cm-meta cm-scope">{scope}</p>
</div>
<ol class="cm-steps">
<li><a href="#crush"><i>1</i><b>Crush margin</b><span>What crushing a tonne of soybeans earns, against the past 5 years.</span></a></li>
<li><a href="#share"><i>2</i><b>Oil share</b><span>How much of the value is oil, now pulled by biodiesel.</span></a></li>
<li><a href="#gap"><i>3</i><b>Oil cost vs diesel</b><span>What the oil in biodiesel costs next to diesel, with and without the D4 RIN.</span></a></li>
</ol></div>"""
DATA_SOURCE = (
    "CBOT soybeans, meal and oil, and NYMEX heating oil. Daily futures closes from Yahoo Finance"
)
LOAD_ERROR = (
    "Prices could not be loaded from Yahoo Finance and no saved copy is available. "
    "Try again in a minute."
)


@st.cache_data(ttl=3600, show_spinner="Loading prices")
def _load_prices() -> LoadResult:
    return load_prices()


@st.cache_data(ttl=6 * 3600, show_spinner="Loading RIN prices")
def _load_rins() -> RinResult:
    return load_d4()


@st.cache_data(ttl=3600, show_spinner=False)
def _tables(prices: pd.DataFrame, d4: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return crush_table(prices), add_rins(gap_table(prices), d4)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _usd(value: float, unit: str, signed: bool = False, decimals: int = 0) -> str:
    # A typographic minus, as on the chart axes, so a negative value never reads as a dash.
    shown = round(value, decimals)
    sign = "−" if shown < 0 else ("+" if signed and shown > 0 else "")
    return f"{sign}${abs(value):,.{decimals}f}/{unit}"


def _day(date: pd.Timestamp) -> str:
    return f"{date.day} {date:%b %Y}"


def _rank_clause(reading: Reading) -> str:
    if reading.percentile is None:
        return "with too little clean history to rank"
    return f"higher than {reading.percentile:.0f}% of clean days in the past 5 years"


def _previous(year: int) -> str:
    return f"{year - LOOKBACK_YEARS}–{year - 1}"


def _window(table: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "All" or table.empty:
        return table
    start = pd.Timestamp(table.index[-1]) - pd.DateOffset(years=int(window[:-1]))
    return table.loc[start:]


def _median_5y(values: pd.Series, flagged: pd.Series, date: pd.Timestamp) -> float:
    clean = values[~flagged.astype(bool)].loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    return float(clean.median())


def _after_words(value: float) -> str:
    if value < 0:
        return f"fuel value covers the oil by {_usd(-value, 'm³')} ({_usd(-per_gallon(value), 'gal', decimals=2)})"
    return f"the oil still costs {_usd(value, 'm³')} more than fuel value"


def _rin_note(rins: RinResult, latest: pd.Timestamp) -> str:
    if rins.prices is None or rins.as_of is None:
        return "D4 RIN prices could not be loaded"
    week = pd.Timestamp(rins.as_of)
    weeks = max((latest - week).days // 7, 0)
    saved = ", saved copy" if rins.source == "seed" else ""
    return f"D4 ${float(rins.prices.iloc[-1]):.2f}, week of {week.day} {week:%b} (EPA{saved}, {weeks} weeks old)"


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def _kpis(
    crush: pd.DataFrame,
    gap: pd.DataFrame,
    readings: dict[str, Reading | None],
    rins: RinResult,
) -> None:
    reading = readings["crush"]
    assert reading is not None
    means, _ = monthly_means(crush["crush"], crush["flagged"])
    versus = month_vs_average(means, reading.date)
    month = MONTHS[reading.date.month - 1]
    crush_caption = "Gross margin, before plant costs."
    if versus is not None:
        crush_caption = (
            f"{_usd(versus[0] - versus[1], 't', signed=True)} against the {month} average of "
            f"{_previous(reading.date.year)}. Gross, before plant costs."
        )
    cards = [
        Kpi(
            key="crush",
            number=1,
            label="Crush margin",
            value=_usd(reading.value, "t"),
            note="per tonne",
            detail=f"${reading.value / BUSHELS_PER_TONNE:,.2f}/bu board crush · {_day(reading.date)}",
            percentile=reading.percentile,
            caption=crush_caption,
            tag="" if reading.date == reading.latest_date else "last close is a roll day",
        )
    ]
    share = readings["share"]
    if share is None:
        cards.append(Kpi("share", 2, "Oil share", "n/a", "", "No clean session yet.", None, ""))
    else:
        row = cast(pd.Series, crush.loc[share.date])
        median = _median_5y(crush["oil_share"], crush["flagged"], share.date)
        cards.append(
            Kpi(
                key="share",
                number=2,
                label="Oil share",
                value=f"{share.value:.0%}",
                note="of product value",
                detail="",
                percentile=share.percentile,
                caption=f"5-year median {median:.0%}. Oil's part of what a tonne of soybeans sells for.",
                visual=split_bar_html(
                    share.value,
                    _usd(row["oil_value"], "t"),
                    _usd(row["meal_value"], "t"),
                ),
            )
        )
    before = readings["gap"]
    if before is None:
        cards.append(Kpi("gap", 3, "Oil cost vs diesel", "n/a", "", "Prices missing.", None, ""))
    else:
        after = gap["after_rin"].where(~gap["flagged"]).dropna()
        detail = (
            "D4 RIN prices could not be loaded."
            if after.empty
            else f"With the D4 RIN: {_after_words(float(after.iloc[-1]))}."
        )
        side = "dearer" if before.value >= 0 else "cheaper"
        cards.append(
            Kpi(
                key="gap",
                number=3,
                label="Oil cost vs diesel",
                value=_usd(before.value, "m³", signed=True),
                note=f"oil {side} · {_usd(per_gallon(before.value), 'gal', signed=True, decimals=2)}",
                detail=detail,
                percentile=before.percentile,
                caption=f"{_rin_note(rins, before.latest_date)}. Feedstock only: before conversion costs, 45Z and LCFS.",
            )
        )
    st.html(kpi_row_html(cards))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Controls:
    view: str
    window: str
    years: list[int]
    band: bool
    rin: bool


def _controls(view: str, key: str, years: list[int], rin: bool | None = None) -> Controls:
    """The options of the chosen view, in one row of fixed height under the title.

    ``rin`` adds the D4 toggle: None leaves it out, False shows it disabled.
    """
    window, picked, band, rin_on = DEFAULT_WINDOW, [], True, False
    with st.container(key=f"bar-{key}", horizontal=True, vertical_alignment="center"):
        if view == "History":
            window = st.segmented_control(
                "History window",
                WINDOWS,
                default=DEFAULT_WINDOW,
                required=True,
                key=f"win-{key}",
                label_visibility="collapsed",
            )
        else:
            picked = st.pills(
                "Compare with",
                years,
                selection_mode="multi",
                default=years[:2],
                key=f"years-{key}",
                label_visibility="collapsed",
            )
            if len(picked) > MAX_YEARS:
                st.caption(f"Up to {MAX_YEARS} years at a time: showing the most recent.")
            picked = sorted(picked, reverse=True)[:MAX_YEARS]
            band = st.checkbox("5-year range", value=True, key=f"band-{key}")
        if rin is not None:
            rin_on = st.toggle(
                "Include D4 RIN credit",
                value=False,
                key=f"rin-{key}",
                disabled=not rin,
                help=None if rin else "D4 RIN prices could not be loaded.",
            )
    return Controls(view, window, picked, band, rin_on)


def _chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=f"chart-{key}", theme=None)


def _past_years(index: pd.Index, current: int) -> list[int]:
    years = sorted({int(year) for year in pd.DatetimeIndex(index).year if year < current})
    return years[::-1][:YEAR_CHOICES]


def _month_sentence(
    means: pd.DataFrame, date: pd.Timestamp, latest: pd.Timestamp, fmt: Callable[[float], str]
) -> str:
    month = MONTHS[date.month - 1]
    versus = month_vs_average(means, date)
    if versus is None:
        return f"Monthly averages of clean sessions. {month} {date.year} has too few so far to compare."
    so_far = " so far" if (date.year, date.month) == (latest.year, latest.month) else ""
    return (
        f"{month} {date.year} averages {fmt(versus[0])}{so_far}, against {fmt(versus[1])} "
        f"for {month} in {_previous(date.year)}."
    )


def _section(
    key: str,
    number: int,
    title: str,
    body: Callable[[Controls], tuple[str, go.Figure, str, list[tuple[str, str]]]],
    years: list[int],
    rin: bool | None = None,
) -> None:
    """One chart card: title and sentence, view switch, options, chart and its rail."""
    with st.container(key=f"card-sec-{key}"):
        head_col, view_col = st.columns([5, 2], vertical_alignment="center")
        head = head_col.empty()
        with view_col:
            view = st.segmented_control(
                "Chart view",
                VIEWS,
                default=VIEWS[0],
                required=True,
                key=f"view-{key}",
                label_visibility="collapsed",
            )
        controls = _controls(view, key, years, rin)
        sentence, fig, reading, today = body(controls)
        # Filled after the widgets are read, so the sentence never lags one click behind.
        head.html(section_head_html(key, number, title, sentence))
        chart_col, rail_col = st.container(key=f"body-{key}").columns([3, 1], gap="medium")
        with chart_col:
            _chart(fig, key)
        with rail_col:
            st.html(rail_html(WHAT[key], reading, today))


@st.fragment
def _crush_section(table: pd.DataFrame, reading: Reading) -> None:
    def body(c: Controls) -> tuple[str, go.Figure, str, list[tuple[str, str]]]:
        means, counts = monthly_means(table["crush"], table["flagged"])
        versus = month_vs_average(means, reading.date)
        low, high = range_52w(table["crush"], table["flagged"], reading.date)
        month = MONTHS[reading.date.month - 1]
        today = [
            ("Board crush", f"${reading.value / BUSHELS_PER_TONNE:,.2f}/bu"),
            (
                f"{month} vs {_previous(reading.date.year)}",
                "n/a" if versus is None else _usd(versus[0] - versus[1], "t", signed=True),
            ),
            ("52-week range", f"{_usd(low, 't')} to {_usd(high, 't')}"),
        ]
        if c.view == "History":
            shown = _window(table, c.window)
            fig = history_figure(
                shown["crush"],
                shown["flagged"],
                name="Crush margin",
                color=CRUSH,
                y_title="USD per tonne of soybeans",
                hover={
                    "Meal value": shown["meal_value"],
                    "Oil value": shown["oil_value"],
                    "Soybean cost": shown["beans"],
                },
            )
            sentence = (
                f"{_usd(reading.value, 't')} on {_day(reading.date)}, {_rank_clause(reading)}."
            )
            return sentence, fig, READING_HISTORY, today
        fig = by_month_figure(
            means,
            counts,
            years=c.years,
            current=reading.latest_date.year,
            color=CRUSH,
            y_title="USD per tonne of soybeans",
            band=c.band,
            month_to_date=reading.latest_date.month,
        )
        sentence = _month_sentence(means, reading.date, reading.latest_date, lambda v: _usd(v, "t"))
        return sentence, fig, READING_BY_MONTH, today

    _section("crush", 1, "Crush margin", body, _past_years(table.index, reading.latest_date.year))


@st.fragment
def _share_section(table: pd.DataFrame, reading: Reading) -> None:
    percent = table["oil_share"] * 100

    def body(c: Controls) -> tuple[str, go.Figure, str, list[tuple[str, str]]]:
        row = cast(pd.Series, table.loc[reading.date])
        today = [
            ("Oil value", _usd(row["oil_value"], "t")),
            ("Meal value", _usd(row["meal_value"], "t")),
            ("Soybean cost", _usd(row["beans"], "t")),
        ]
        if c.view == "History":
            shown = _window(table, c.window)
            fig = history_figure(
                shown["oil_share"] * 100,
                shown["flagged"],
                name="Oil share",
                color=OIL,
                y_title="Oil share of product value, %",
                percent=True,
                hover={"Oil value": shown["oil_value"], "Meal value": shown["meal_value"]},
            )
            median = _median_5y(table["oil_share"], table["flagged"], reading.date)
            sentence = (
                f"Oil is {reading.value:.0%} of the product value on {_day(reading.date)}, "
                f"against a 5-year median of {median:.0%}."
            )
            return sentence, fig, READING_HISTORY, today
        means, counts = monthly_means(percent, table["flagged"])
        fig = by_month_figure(
            means,
            counts,
            years=c.years,
            current=reading.latest_date.year,
            color=OIL,
            y_title="Oil share of product value, %",
            band=c.band,
            percent=True,
            month_to_date=reading.latest_date.month,
        )
        sentence = _month_sentence(means, reading.date, reading.latest_date, lambda v: f"{v:.1f}%")
        return sentence, fig, READING_BY_MONTH, today

    _section("share", 2, "Oil share", body, _past_years(table.index, reading.latest_date.year))


@st.fragment
def _gap_section(table: pd.DataFrame, reading: Reading, rins: RinResult) -> None:
    has_rins = bool(table["rin_value"].notna().any())

    def body(c: Controls) -> tuple[str, go.Figure, str, list[tuple[str, str]]]:
        row = cast(pd.Series, table.loc[reading.date])
        rin_row = "n/a"
        if has_rins and pd.notna(row["rin_shown"]):
            rin_row = _usd(row["rin_shown"], "m³")
        today = [
            ("Oil feedstock", _usd(row["feedstock"], "m³")),
            ("Heating oil (ULSD)", _usd(row["diesel"], "m³")),
            ("D4 RIN value", rin_row),
        ]
        rin_on = c.rin and has_rins
        column = "after_rin" if rin_on else "gap"
        y_title = (
            "Oil minus heating oil and D4 RIN, USD per m³"
            if rin_on
            else "Oil minus heating oil, USD per m³"
        )
        color = OIL if rin_on else INK_SERIES
        flagged = table["flagged"] | (table["rin_held"] if rin_on else False)
        if c.view == "History":
            shown = _window(table, c.window)
            held = None
            reference = None
            if rin_on:
                last = float(rins.prices.iloc[-1]) if rins.prices is not None else 0.0
                week = pd.Timestamp(rins.as_of) if rins.as_of is not None else reading.date
                held = (f"D4 held at ${last:.2f} (week of {week.day} {week:%b})", shown["rin_held"])
                reference = ("Before the RIN", shown["gap"])
            fig = history_figure(
                shown[column],
                shown["flagged"],
                name="After the D4 RIN" if rin_on else "Oil minus heating oil",
                color=color,
                y_title=y_title,
                reference=reference,
                held=held,
                zero_labels=("oil dearer", "fuel covers the oil" if rin_on else "oil cheaper"),
                hover={"Oil feedstock": shown["feedstock"], "Heating oil": shown["diesel"]},
            )
            if rin_on:
                ranked = latest_reading(table[column], table["flagged"], unranked=table["rin_held"])
                assert ranked is not None
                sentence = f"With the D4 RIN, {_after_words(ranked.value)}, {_rank_clause(ranked)}."
                return sentence, fig, READING_RIN, today
            side = "more" if reading.value >= 0 else "less"
            sentence = (
                f"The oil for 1 m³ of biodiesel costs {_usd(abs(reading.value), 'm³')} {side} "
                f"than 1 m³ of heating oil, {_rank_clause(reading)}."
            )
            return sentence, fig, READING_HISTORY, today
        means, counts = monthly_means(table[column], flagged)
        fig = by_month_figure(
            means,
            counts,
            years=c.years,
            current=reading.latest_date.year,
            color=color,
            y_title=y_title,
            band=c.band,
            month_to_date=reading.latest_date.month,
        )
        # With the RIN on, the last month with EPA weeks, not the month of the latest close.
        dated = pd.Timestamp(table[column].where(~flagged).dropna().index[-1])
        sentence = _month_sentence(
            means, dated, reading.latest_date, lambda v: _usd(v, "m³", signed=True)
        )
        reading_text = READING_BY_MONTH
        if rin_on:
            reading_text += " RIN months use EPA weeks only."
        return sentence, fig, reading_text, today

    _section(
        "gap",
        3,
        "Oil cost vs diesel",
        body,
        _past_years(table.index, reading.latest_date.year),
        rin=has_rins,
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _data_line(loaded: LoadResult) -> str:
    if loaded.prices is None:
        return f"{DATA_SOURCE}."
    freshness = " (may be out of date)" if loaded.is_stale else ""
    return f"{DATA_SOURCE}, 2000 to {loaded.as_of}{freshness}."


def _footer(loaded: LoadResult, rins: RinResult | None = None) -> None:
    data = ""
    if loaded.prices is not None:
        freshness = " (may be out of date)" if loaded.is_stale else ""
        data = f"Data to {loaded.as_of}{freshness}. "
    if rins is not None and rins.prices is not None:
        saved = ", saved copy" if rins.source == "seed" else ""
        data += f"D4 RIN prices from EPA to the week of {rins.as_of}{saved}. "
    st.caption(
        f"{data}Personal project, not affiliated with any employer. Not investment advice. "
        f"[Source code]({REPOSITORY_URL})"
    )


def main() -> None:
    st.set_page_config(
        page_title="Soybean Crush Margin",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(APP_CSS)
    loaded = _load_prices()
    st.html(
        HERO_HTML.format(
            data_line=html.escape(_data_line(loaded)),
            linkedin=LINKEDIN_URL,
            repository=REPOSITORY_URL,
            scope=html.escape(SCOPE),
        )
    )

    if loaded.prices is None:
        # A failure is a value, so st.cache_data would keep it for an hour. Clear it.
        _load_prices.clear()
        st.error(LOAD_ERROR, icon=":material/warning:")
        if st.button("Retry"):
            st.rerun()
        _footer(loaded)
        return

    rins = _load_rins()
    crush, gap = _tables(loaded.prices, rins.prices)
    readings = {
        "crush": latest_reading(crush["crush"], crush["flagged"]),
        "share": latest_reading(crush["oil_share"], crush["flagged"]),
        "gap": latest_reading(gap["gap"], gap["flagged"]),
    }
    if readings["crush"] is None:
        st.warning(
            "Soybean, meal or oil prices are missing, so the crush margin cannot be computed.",
            icon=":material/warning:",
        )
        _footer(loaded)
        return

    _kpis(crush, gap, readings, rins)
    _crush_section(crush, readings["crush"])
    if readings["share"] is not None:
        _share_section(crush, readings["share"])
    if readings["gap"] is None:
        st.warning(
            "Soybean oil or heating oil prices are missing, so the feedstock gap is not shown.",
            icon=":material/warning:",
        )
    else:
        _gap_section(gap, readings["gap"], rins)
    st.html(method_row_html(METHOD))
    with st.expander("Full methodology and sources"):
        st.markdown(FULL_METHOD.replace("$", "\\$"))
    _footer(loaded, rins)


main()

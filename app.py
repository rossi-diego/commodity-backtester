"""Soybean Crush Margin: Streamlit UI. All computation lives in ``src``."""

from __future__ import annotations

import datetime
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
    risk_strip_html,
    section_head_html,
    split_bar_html,
)
from src.constants import BUSHELS_PER_TONNE
from src.data_loader import LoadResult, load_prices
from src.margins import add_rins, crush_table, gap_table, per_gallon
from src.rins import RinResult, load_d4
from src.risk import RiskSummary, risk_summary
from src.seasonal import (
    LOOKBACK_YEARS,
    Reading,
    latest_reading,
    month_vs_average,
    monthly_means,
    range_52w,
)
from src.styles import APP_CSS
from src.text import (
    FULL_METHOD,
    LINKEDIN_URL,
    METHOD,
    READING_BY_MONTH,
    READING_HISTORY,
    READING_RIN,
    REPOSITORY_URL,
    RISK_STRIP,
    SCOPE,
    WHAT,
)
from src.visualization import CRUSH, INK_SERIES, MONTHS, OIL, RIN, by_month_figure, history_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

VIEWS = ["History", "By month"]
WINDOWS = ["5Y", "10Y", "All"]
DEFAULT_WINDOW = "10Y"
YEAR_CHOICES = 10
MAX_YEARS = 5

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


@st.cache_data(ttl=3600, show_spinner=False)
def _risk(crush: pd.DataFrame) -> RiskSummary | None:
    return risk_summary(crush)


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
        return (
            f"the oil costs {_usd(-value, 'm³')} ({_usd(-per_gallon(value), 'gal', decimals=2)}) "
            "less than heating oil plus the RIN"
        )
    return f"the oil still costs {_usd(value, 'm³')} more than heating oil plus the RIN"


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
        detail = "D4 RIN prices could not be loaded."
        if not after.empty:
            detail = f"With the D4 RIN: {_after_words(float(after.iloc[-1]))}."
            if bool(gap.loc[after.index[-1], "rin_held"]):
                detail += " D4 price held."
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
                rank_of="before the RIN",
                caption=f"{_rin_note(rins, before.latest_date)}. Feedstock only: before conversion costs, BTC/45Z and LCFS.",
            )
        )
    st.html(kpi_row_html(cards))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionBody:
    """What a section's body function returns for the chosen controls."""

    sentence: str
    figure: go.Figure
    reading: str
    today: list[tuple[str, str]]


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
    body: Callable[[Controls], SectionBody],
    years: list[int],
    rin: bool | None = None,
    footer: str = "",
) -> None:
    """One chart card: title and sentence, view switch, options, chart and its rail.

    ``footer`` is markup from ``src.cards``, drawn across the card under the chart.
    """
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
        shown = body(controls)
        # Filled after the widgets are read, so the sentence never lags one click behind.
        head.html(section_head_html(key, number, title, shown.sentence))
        chart_col, rail_col = st.container(key=f"body-{key}").columns([3, 1], gap="medium")
        with chart_col:
            _chart(shown.figure, key)
        with rail_col:
            st.html(rail_html(WHAT[key], shown.reading, shown.today))
        if footer:
            st.html(footer)


@st.fragment
def _crush_section(table: pd.DataFrame, reading: Reading, risk: RiskSummary | None) -> None:
    def body(c: Controls) -> SectionBody:
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
                y_title="USD/t of soybeans",
                hover={
                    "Meal value": shown["meal_value"],
                    "Oil value": shown["oil_value"],
                    "Soybean cost": shown["beans"],
                },
            )
            sentence = (
                f"{_usd(reading.value, 't')} on {_day(reading.date)}, {_rank_clause(reading)}."
            )
            return SectionBody(sentence, fig, READING_HISTORY, today)
        fig = by_month_figure(
            means,
            counts,
            years=c.years,
            current=reading.latest_date.year,
            color=CRUSH,
            y_title="USD/t of soybeans",
            band=c.band,
            month_to_date=reading.latest_date.month,
        )
        sentence = _month_sentence(means, reading.date, reading.latest_date, lambda v: _usd(v, "t"))
        return SectionBody(sentence, fig, READING_BY_MONTH, today)

    _section(
        "crush",
        1,
        "Crush margin",
        body,
        _past_years(table.index, reading.latest_date.year),
        footer=risk_strip_html(risk, RISK_STRIP),
    )


@st.fragment
def _share_section(table: pd.DataFrame, reading: Reading) -> None:
    percent = table["oil_share"] * 100

    def body(c: Controls) -> SectionBody:
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
                y_title="Oil share, %",
                percent=True,
                hover={"Oil value": shown["oil_value"], "Meal value": shown["meal_value"]},
            )
            median = _median_5y(table["oil_share"], table["flagged"], reading.date)
            sentence = (
                f"Oil is {reading.value:.0%} of the product value on {_day(reading.date)}, "
                f"against a 5-year median of {median:.0%}."
            )
            return SectionBody(sentence, fig, READING_HISTORY, today)
        means, counts = monthly_means(percent, table["flagged"])
        fig = by_month_figure(
            means,
            counts,
            years=c.years,
            current=reading.latest_date.year,
            color=OIL,
            y_title="Oil share, %",
            band=c.band,
            percent=True,
            month_to_date=reading.latest_date.month,
        )
        sentence = _month_sentence(means, reading.date, reading.latest_date, lambda v: f"{v:.1f}%")
        return SectionBody(sentence, fig, READING_BY_MONTH, today)

    _section("share", 2, "Oil share", body, _past_years(table.index, reading.latest_date.year))


@st.fragment
def _gap_section(table: pd.DataFrame, reading: Reading, rins: RinResult) -> None:
    has_rins = bool(table["rin_value"].notna().any())

    def body(c: Controls) -> SectionBody:
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
        y_title = "Oil − heating oil − RIN, USD/m³" if rin_on else "Oil − heating oil, USD/m³"
        color = RIN if rin_on else INK_SERIES
        flagged = table["flagged"] | (table["rin_held"] if rin_on else False)
        if c.view == "History":
            shown = _window(table, c.window)
            held = None
            reference = None
            week = pd.Timestamp(rins.as_of) if rins.as_of is not None else reading.date
            if rin_on:
                last = float(rins.prices.iloc[-1]) if rins.prices is not None else 0.0
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
                zero_labels=("oil dearer", "oil cheaper after RIN" if rin_on else "oil cheaper"),
                hover={"Oil feedstock": shown["feedstock"], "Heating oil": shown["diesel"]},
                # The RIN widens the range: fixed ticks keep zero and the gap legible.
                y_dtick=250 if rin_on else None,
            )
            if rin_on:
                ranked = latest_reading(table[column], table["flagged"], unranked=table["rin_held"])
                assert ranked is not None
                sentence = f"With the D4 RIN, {_after_words(ranked.value)}, {_rank_clause(ranked)}."
                if bool(table.loc[ranked.date, "rin_held"]):
                    sentence = (
                        f"With the D4 RIN held at the EPA week of {week.day} {week:%b}, "
                        f"{_after_words(ranked.value)}. Not ranked while the price is held."
                    )
                return SectionBody(sentence, fig, READING_RIN, today)
            side = "more" if reading.value >= 0 else "less"
            sentence = (
                f"The oil for 1 m³ of biodiesel costs {_usd(abs(reading.value), 'm³')} {side} "
                f"than 1 m³ of heating oil, {_rank_clause(reading)}."
            )
            return SectionBody(sentence, fig, READING_HISTORY, today)
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
        return SectionBody(sentence, fig, reading_text, today)

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
    """Hero markup: the source and the data date, kept on one line."""
    if loaded.prices is None or loaded.as_of is None:
        return f"{html.escape(DATA_SOURCE)}."
    freshness = " (may be out of date)" if loaded.is_stale else ""
    day = f'<span class="cm-nowrap">{_day(pd.Timestamp(loaded.as_of))}</span>'
    return f"{html.escape(DATA_SOURCE)}, 2000 to {day}{freshness}."


def _footer_day(date: datetime.date) -> str:
    # The footer is markdown, which takes no span, so no-break spaces keep the date whole.
    return _day(pd.Timestamp(date)).replace(" ", " ")


def _footer(loaded: LoadResult, rins: RinResult | None = None) -> None:
    data = ""
    if loaded.prices is not None and loaded.as_of is not None:
        freshness = " (may be out of date)" if loaded.is_stale else ""
        data = f"Data to {_footer_day(loaded.as_of)}{freshness}. "
    if rins is not None and rins.prices is not None and rins.as_of is not None:
        saved = ", saved copy" if rins.source == "seed" else ""
        data += f"D4 RIN prices from EPA to the week of {_footer_day(rins.as_of)}{saved}. "
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
            data_line=_data_line(loaded),
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
    risk = _risk(crush)
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
    _crush_section(crush, readings["crush"], risk)
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

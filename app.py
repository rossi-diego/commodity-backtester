"""Soybean Crush Margin: Streamlit UI. All computation lives in ``src``."""

from __future__ import annotations

import datetime
import html
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import cast


def _drop_stale_src() -> None:
    """Forget every ``src`` module when one of its files changed since it was imported.

    Streamlit Cloud pulls a push and reruns this script, but keeps the ``src`` modules of the
    running process, so a new ``app.py`` can meet an old ``src``. Run before the imports below.
    """
    loaded = _src_modules()
    if any(_changed_since_import(module) for module in loaded.values()):
        for name in loaded:
            del sys.modules[name]
        logging.getLogger(__name__).info("event=src_reloaded modules=%d", len(loaded))


def _src_modules() -> dict[str, ModuleType]:
    return {n: m for n, m in sys.modules.items() if n == "src" or n.startswith("src.")}


def _changed_since_import(module: ModuleType) -> bool:
    """Whether the module's file is newer than its import stamp, or than its bytecode.

    A process started before the stamps existed has none: Python wrote each ``.pyc`` when it
    first compiled the source, so a pulled file is newer than it.
    """
    path = getattr(module, "__file__", None)
    if not path or not os.path.exists(path):
        return False
    imported = module.__dict__.get("_imported_at")
    cached = getattr(module, "__cached__", None)
    if imported is None and cached and os.path.exists(cached):
        imported = os.path.getmtime(cached)
    return imported is not None and os.path.getmtime(path) > imported


_drop_stale_src()

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
from src.deferred import Held, held_session
from src.margins import add_rins, break_even_d4, crush_table, gap_table, per_gallon
from src.rins import RinResult, load_d4
from src.seasonal import (
    LOOKBACK_YEARS,
    Reading,
    latest_reading,
    month_vs_average,
    monthly_band,
    monthly_means,
    range_52w,
    reading_at,
)
from src.styles import APP_CSS
from src.text import (
    FULL_METHOD,
    GAP_CARD,
    LINKEDIN_URL,
    METHOD,
    READING_BY_MONTH,
    READING_HISTORY,
    READING_RIN,
    REPOSITORY_URL,
    SCOPE,
    WHAT,
)
from src.visualization import CRUSH, INK_SERIES, MONTHS, OIL, RIN, by_month_figure, history_figure
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

# Stamp each src module with its import time, for the next run's _drop_stale_src.
for _module in _src_modules().values():
    _module.__dict__.setdefault("_imported_at", time.time())

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
def _held(prices: pd.DataFrame) -> Held | None:
    return held_session(prices)


@st.cache_data(ttl=3600, show_spinner=False)
def _tables(prices: pd.DataFrame, d4: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return crush_table(prices), add_rins(gap_table(prices), d4)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _window(table: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "All" or table.empty:
        return table
    start = pd.Timestamp(table.index[-1]) - pd.DateOffset(years=int(window[:-1]))
    return table.loc[start:]


def _median_5y(values: pd.Series, flagged: pd.Series, date: pd.Timestamp) -> float:
    clean = values[~flagged.astype(bool)].loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    return float(clean.median())


@dataclass(frozen=True)
class Now:
    """What a card and its rail show: the latest clean session, or the held contracts.

    ``row`` is the table row of ``reading.date``. When the ``=F`` series is in delivery and
    the held contracts could be fetched, it comes from them and ``label`` names them; else
    ``label`` is empty and ``since`` is the first session in delivery, if any.
    """

    reading: Reading
    row: pd.Series
    label: str = ""
    since: pd.Timestamp | None = None

    @property
    def tag(self) -> str:
        if self.label:
            return self.label
        return "" if self.since is None else delivery_tag(self.since)


def _now(table: pd.DataFrame, column: str, held_row: pd.Series | None, label: str) -> Now | None:
    """The reading of ``column``: on the held contracts when ``held_row`` is given."""
    if held_row is not None:
        date = pd.Timestamp(str(held_row.name))
        reading = reading_at(table[column], table["flagged"], date, float(held_row[column]))
        return None if reading is None else Now(reading, held_row, label)
    reading = latest_reading(table[column], table["flagged"])
    if reading is None:
        return None
    later = table.index[table.index > reading.date]
    since = pd.Timestamp(later[0]) if len(later) else None
    return Now(reading, cast(pd.Series, table.loc[reading.date]), since=since)


def _versus(means: pd.DataFrame, now: Now) -> float | None:
    """This month against the same month of the 5 years before: the month to date, or the
    held level when there is one, against that month's average."""
    date = now.reading.date
    if not now.label:
        versus = month_vs_average(means, date)
        return None if versus is None else versus[0] - versus[1]
    average = monthly_band(means, date.year)["average"].iloc[date.month - 1]
    return None if pd.isna(average) else now.reading.value - float(average)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def _kpis(
    crush: pd.DataFrame,
    nows: dict[str, Now | None],
    rins: RinResult,
) -> None:
    now = nows["crush"]
    assert now is not None
    reading = now.reading
    means, _ = monthly_means(crush["crush"], crush["flagged"])
    versus = _versus(means, now)
    month = MONTHS[reading.date.month - 1]
    crush_caption = "Gross margin, before plant costs."
    if versus is not None:
        crush_caption = (
            f"{usd(versus, 't', signed=True)} against the {month} average of "
            f"{previous_years(reading.date.year)}. Gross, before plant costs."
        )
    cards = [
        Kpi(
            key="crush",
            number=1,
            label="Crush margin",
            value=usd(reading.value, "t"),
            note="per tonne",
            detail=f"${reading.value / BUSHELS_PER_TONNE:,.2f}/bu board crush · {day(reading.date)}",
            percentile=reading.percentile,
            caption=crush_caption,
            tag=now.tag,
        )
    ]
    share_now = nows["share"]
    if share_now is None:
        cards.append(Kpi("share", 2, "Oil share", "n/a", "", "No clean session yet.", None, ""))
    else:
        share, row = share_now.reading, share_now.row
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
                    usd(row["oil_value"], "t"),
                    usd(row["meal_value"], "t"),
                ),
                tag=share_now.tag,
            )
        )
    gap_now = nows["gap"]
    if gap_now is None:
        cards.append(Kpi("gap", 3, "Oil cost vs diesel", "n/a", "", "Prices missing.", None, ""))
    else:
        before = gap_now.reading
        detail = GAP_CARD["no_rin"] if before.value <= 0 else GAP_CARD["no_d4"]
        if before.value > 0 and rins.prices is not None:
            held = " (held)" if bool(gap_now.row["rin_held"]) else ""
            d4 = float(rins.prices.iloc[-1])
            detail = GAP_CARD["break_even"].format(be=break_even_d4(before.value), d4=d4, held=held)
        side = "dearer" if before.value >= 0 else "cheaper"
        gallon = usd(per_gallon(before.value), "gal", signed=True, decimals=2)
        # No-break spaces keep the date whole when the note wraps beside the value.
        gap_day = day(before.date).replace(" ", " ")
        cards.append(
            Kpi(
                key="gap",
                number=3,
                label="Oil cost vs diesel",
                value=usd(before.value, "m³", signed=True),
                note=f"oil {side} · {gallon} · {gap_day}",
                detail=detail,
                percentile=before.percentile,
                rank_of="before the RIN",
                caption=f"{rin_note(rins, before.latest_date)}{GAP_CARD['scope']}",
                tag=gap_now.tag,
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


def _section(
    key: str,
    number: int,
    title: str,
    body: Callable[[Controls], SectionBody],
    years: list[int],
    as_of: pd.Timestamp,
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
        shown = body(controls)
        # Filled after the widgets are read, so the sentence never lags one click behind.
        head.html(section_head_html(key, number, title, shown.sentence))
        chart_col, rail_col = st.container(key=f"body-{key}").columns([3, 1], gap="medium")
        with chart_col:
            _chart(shown.figure, key)
        with rail_col:
            title = f"Levels on {as_of.day} {as_of:%b}"
            st.html(rail_html(WHAT[key], shown.reading, shown.today, title))


def _marker(now: Now, scale: float = 1.0) -> tuple[str, pd.Timestamp, float] | None:
    """The held level as a hollow marker on the history chart, when there is one."""
    if not now.label:
        return None
    return now.label, now.reading.date, now.reading.value * scale


def _on(now: Now) -> str:
    """Names the held contracts in a section sentence."""
    return f" ({now.label})" if now.label else ""


@st.fragment
def _crush_section(table: pd.DataFrame, now: Now) -> None:
    reading = now.reading

    def body(c: Controls) -> SectionBody:
        means, counts = monthly_means(table["crush"], table["flagged"])
        versus = _versus(means, now)
        low, high = range_52w(table["crush"], table["flagged"], reading.date)
        month = MONTHS[reading.date.month - 1]
        today = [
            ("Per bushel (CME quote)", f"${reading.value / BUSHELS_PER_TONNE:,.2f}/bu"),
            (
                f"Against a usual {month}",
                "n/a" if versus is None else usd(versus, "t", signed=True),
            ),
            ("Past year, low to high", f"{usd(low, 't')} to {usd(high, 't')}"),
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
                latest=_marker(now),
            )
            sentence = f"{usd(reading.value, 't')} on {day(reading.date)}{_on(now)}, {rank_clause(reading)}."
            return SectionBody(sentence, fig, READING_HISTORY["crush"], today)
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
        sentence = month_sentence(means, reading.date, reading.latest_date, lambda v: usd(v, "t"))
        return SectionBody(sentence, fig, READING_BY_MONTH, today)

    _section(
        "crush",
        1,
        "Crush margin",
        body,
        _past_years(table.index, reading.latest_date.year),
        reading.date,
    )


@st.fragment
def _share_section(table: pd.DataFrame, now: Now) -> None:
    percent = table["oil_share"] * 100
    reading = now.reading

    def body(c: Controls) -> SectionBody:
        row = now.row
        today = [
            ("Oil value", usd(row["oil_value"], "t")),
            ("Meal value", usd(row["meal_value"], "t")),
            ("Soybean cost", usd(row["beans"], "t")),
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
                latest=_marker(now, scale=100),
            )
            median = _median_5y(table["oil_share"], table["flagged"], reading.date)
            sentence = (
                f"Oil is {reading.value:.0%} of the product value on {day(reading.date)}{_on(now)}, "
                f"against a 5-year median of {median:.0%}."
            )
            return SectionBody(sentence, fig, READING_HISTORY["share"], today)
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
        sentence = month_sentence(means, reading.date, reading.latest_date, lambda v: f"{v:.1f}%")
        return SectionBody(sentence, fig, READING_BY_MONTH, today)

    _section(
        "share",
        2,
        "Oil share",
        body,
        _past_years(table.index, reading.latest_date.year),
        reading.date,
    )


@st.fragment
def _gap_section(table: pd.DataFrame, now: Now, rins: RinResult) -> None:
    has_rins = bool(table["rin_value"].notna().any())
    reading = now.reading

    def body(c: Controls) -> SectionBody:
        row = now.row
        rin_row = "n/a"
        if has_rins and pd.notna(row["rin_shown"]):
            rin_row = usd(row["rin_shown"], "m³")
        today = [
            ("Oil feedstock", usd(row["feedstock"], "m³")),
            ("Heating oil (ULSD)", usd(row["diesel"], "m³")),
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
                latest=None if rin_on else _marker(now),
            )
            if rin_on:
                ranked = latest_reading(table[column], table["flagged"], unranked=table["rin_held"])
                assert ranked is not None
                sentence = (
                    f"With the D4 RIN, {after_rin_words(ranked.value)}, {rank_clause(ranked)}."
                )
                if bool(table.loc[ranked.date, "rin_held"]):
                    sentence = (
                        f"With the D4 RIN held at the EPA week of {week.day} {week:%b}, "
                        f"{after_rin_words(ranked.value)}. Not ranked while the price is held."
                    )
                return SectionBody(sentence, fig, READING_RIN, today)
            side = "more" if reading.value >= 0 else "less"
            sentence = (
                f"The oil for 1 m³ of biodiesel costs {usd(abs(reading.value), 'm³')} {side} "
                f"than 1 m³ of heating oil{_on(now)}, {rank_clause(reading)}."
            )
            return SectionBody(sentence, fig, READING_HISTORY["gap"], today)
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
        sentence = month_sentence(
            means, dated, reading.latest_date, lambda v: usd(v, "m³", signed=True)
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
        reading.date,
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
    shown = f'<span class="cm-nowrap">{day(pd.Timestamp(loaded.as_of))}</span>'
    return f"{html.escape(DATA_SOURCE)}, 2000 to {shown}{freshness}."


def _footer_day(date: datetime.date) -> str:
    # The footer is markdown, which takes no span, so no-break spaces keep the date whole.
    return day(pd.Timestamp(date)).replace(" ", " ")


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


def _nows(
    prices: pd.DataFrame, crush: pd.DataFrame, gap: pd.DataFrame, rins: RinResult
) -> dict[str, Now | None]:
    """Each section's reading, on the held contracts where the last ``=F`` session is in
    delivery and they could be fetched, else on the last clean session."""
    in_delivery = {
        "crush": not crush.empty and bool(crush["flagged"].iloc[-1]),
        "gap": not gap.empty and bool(gap["flagged"].iloc[-1]),
    }
    held = _held(prices) if any(in_delivery.values()) else None
    crush_row = gap_row = None
    if held is not None:
        frame = pd.DataFrame([held.prices], index=pd.DatetimeIndex([held.date]))
        if in_delivery["crush"]:
            crush_row = crush_table(frame).iloc[0]
        if in_delivery["gap"]:
            gap_row = add_rins(gap_table(frame), rins.prices).iloc[0]
    pair = held.pair if held is not None else ""
    oil = f"{held.products} oil" if held is not None else ""
    return {
        "crush": _now(crush, "crush", crush_row, pair),
        "share": _now(crush, "oil_share", crush_row, pair),
        "gap": _now(gap, "gap", gap_row, oil),
    }


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
    nows = _nows(loaded.prices, crush, gap, rins)
    if nows["crush"] is None:
        st.warning(
            "Soybean, meal or oil prices are missing, so the crush margin cannot be computed.",
            icon=":material/warning:",
        )
        _footer(loaded)
        return

    _kpis(crush, nows, rins)
    _crush_section(crush, nows["crush"])
    if nows["share"] is not None:
        _share_section(crush, nows["share"])
    if nows["gap"] is None:
        st.warning(
            "Soybean oil or heating oil prices are missing, so the feedstock gap is not shown.",
            icon=":material/warning:",
        )
    else:
        _gap_section(gap, nows["gap"], rins)
    st.html(method_row_html(METHOD))
    with st.expander("Full methodology and sources"):
        st.markdown(FULL_METHOD.replace("$", "\\$"))
    _footer(loaded, rins)


main()

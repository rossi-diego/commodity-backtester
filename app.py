"""Soybean Crush Margin: Streamlit UI. All computation lives in ``src``."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.constants import BUSHELS_PER_TONNE
from src.data_loader import LoadResult, load_prices
from src.margins import add_rins, crush_table, gap_table, rin_per_m3
from src.rins import EPA_PAGE, RinResult, load_d4
from src.seasonal import Reading, latest_reading, monthly_band
from src.visualization import (
    feedstock_figure,
    gap_figure,
    history_figure,
    oil_share_figure,
    seasonal_figure,
    value_figure,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

REPOSITORY_URL = "https://github.com/rossi-diego/soybean-crush-margin"
LINKEDIN_URL = "https://www.linkedin.com/in/rossi-diego/"
CME_GUIDE_URL = "https://www.cmegroup.com/education/files/soybean-crush-reference-guide.pdf"
FARMDOC_URL = "https://farmdocdaily.illinois.edu/wp-content/uploads/2022/02/fdd021622.pdf"
AFDC_URL = "https://afdc.energy.gov/fuels/properties"
EQUIVALENCE_URL = (
    "https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-80/subpart-M/section-80.1415"
)
WINDOWS = ["5Y", "10Y", "All"]
DEFAULT_WINDOW = "10Y"

METRIC_STYLE = {"delta_color": "off", "delta_arrow": "off"}

# Amber means "the crush margin" and nothing else: controls, badges and the warning stay neutral.
# The rules target Streamlit test ids, so redo the screenshots when upgrading Streamlit.
# The top padding clears the 56 px bar Streamlit Cloud draws over the app from its parent page.
APP_CSS = """<style>
:root{--page:#0b1016;--card:#141b23;--raised:#1e2732;--line:#2a3440;--ink:#eef2f6;--mut:#a3afbd;--head:linear-gradient(115deg,#1a2430 0%,#16212c 55%,#3a2c10 100%);--crush:#c58512;--oncrush:#0b1016;--step:rgba(255,255,255,.07)}
[data-testid='stMainBlockContainer']{max-width:none;padding:3.75rem 2rem 2rem}
.cm-hero{background:var(--head);border-radius:12px;padding:1.1rem 1.5rem;color:#fff;display:grid;grid-template-columns:minmax(0,13fr) minmax(0,11fr);gap:1.5rem;align-items:center}
.cm-hero h1{color:#fff;font-size:2.25rem;font-weight:600;letter-spacing:-.02em;line-height:1.1;margin:0 0 .4rem;padding:0}
.cm-q{font-size:1.0625rem;line-height:1.35;margin:0 0 .4rem;color:#fff}
.cm-meta{font-size:.875rem;line-height:1.45;margin:0;color:#dbe6f2}
.cm-meta a{color:#fff;font-weight:500}
.cm-meta strong{color:#fff;font-weight:600}
.cm-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:0;padding:0;list-style:none}
.cm-steps li{background:var(--step);border:1px solid rgba(255,255,255,.18);border-radius:10px;padding:.6rem .75rem;font-size:.8125rem;line-height:1.3;color:#e8eff7}
.cm-steps b{color:#fff;font-size:.9375rem;font-weight:600}
.cm-steps span{display:block;margin-top:.3rem}
.cm-steps i{font-style:normal;display:inline-block;width:1.35rem;height:1.35rem;line-height:1.35rem;text-align:center;border-radius:50%;background:#fff;color:#0b1016;font-weight:600;font-size:.75rem;margin-right:.35rem}
[class*='st-key-card']{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.85rem 1rem}
[data-testid='stMarkdownContainer']{margin-bottom:0}
[data-testid='stMetricDelta']{background:none!important;padding:0!important;color:var(--mut)!important;font-size:.8125rem!important}
[data-testid='stMetricDelta'] *{white-space:normal!important}
[data-testid='stMetricLabel'] p{color:var(--mut);font-size:.8125rem;font-weight:500}
[data-testid='stMetricValue']{white-space:nowrap}
[class*='st-key-card-kpi']>div{flex-shrink:0}
[class*='st-key-card-kpi'] [data-testid='stMarkdown'],[class*='st-key-card-kpi'] [data-testid='stMarkdown']>div{display:block;height:auto!important}
@media (min-width:1101px){[class*='st-key-card-kpi'] [data-testid='stElementContainer']:has([data-testid='stCaptionContainer']){min-height:3.1rem}}
[class*='st-key-card-kpi'] [data-testid='stCaptionContainer'] p{font-size:.78rem;line-height:1.3;margin:0;color:var(--mut)}
[class*='st-key-card-kpi']{gap:.2rem;flex:1 1 0;min-width:0;height:auto!important;padding:.7rem .85rem}
.st-key-kpis>div{display:flex;flex:1 1 0;min-width:0;width:auto!important}
[data-testid='stCaptionContainer']{opacity:1}
[data-testid='stCaptionContainer'] p{color:var(--mut)}
.st-key-kpis{align-items:stretch;height:auto!important;flex:0 0 auto}
.st-key-readrow [data-testid='stHorizontalBlock'],[class*='st-key-row'] [data-testid='stHorizontalBlock']{align-items:stretch}
.st-key-readrow [data-testid='stColumn'] [data-testid='stVerticalBlock']:has(>[data-testid='stLayoutWrapper']>.st-key-kpis){flex:1 1 auto;height:auto}
.st-key-readrow [data-testid='stColumn'] [data-testid='stLayoutWrapper']:has(>.st-key-kpis){flex:0 0 auto}
[class*='st-key-row'] [data-testid='stColumn']>div{height:100%}
.st-key-readrow [data-testid='stColumn']:last-child>div,.st-key-readrow [data-testid='stElementContainer']:has([data-testid='stAlert']){height:100%}
.st-key-readrow [data-testid='stAlert'],[class*='st-key-card-rail']{height:100%}
[class*='st-key-row'] [data-testid='stColumn']:last-child [data-testid='stVerticalBlock'],[class*='st-key-row'] [data-testid='stColumn']:last-child [data-testid='stLayoutWrapper']{height:100%}
.st-key-method>div{flex:1 1 0;min-width:0;width:auto!important}
.st-key-card-kpi-crush{background:var(--crush);border-color:var(--crush)}
.st-key-card-kpi-crush *{color:var(--oncrush)!important}
.st-key-card-kpi-crush [data-testid='stMetricValue']{font-size:2.25rem;line-height:1.15;letter-spacing:-.02em}
.st-key-readrow [data-testid='stAlertContainer']{background:var(--raised);color:var(--ink);border-left:4px solid var(--ink);height:100%}
.st-key-readrow [data-testid='stAlertContainer'] p,.st-key-readrow [data-testid='stAlertContainer'] [data-testid='stAlertDynamicIcon']{font-size:.875rem;line-height:1.45;margin:0;color:var(--ink)}
[class*='st-key-row'] [data-testid='stMarkdownContainer'] p{margin:0 0 .5rem}
[class*='st-key-card-rail'] h2,.st-key-method h2{font-size:.9375rem;font-weight:600;margin:0 0 .35rem;padding:0;color:var(--ink)}
[class*='st-key-card-rail'] p,.st-key-method p{font-size:.875rem;line-height:1.45;color:var(--ink)}
[data-testid='stHeaderActionElements']{display:none}
[class*='st-key-head'] p{font-size:.875rem;color:var(--mut);margin:0}
[class*='st-key-head'] strong{color:var(--ink);font-weight:600;font-size:1rem}
:focus-visible{outline:2px solid var(--ink)!important;outline-offset:2px}
@media (max-width:1100px){[class*='st-key-row'] [data-testid='stColumn']{min-width:100%} .cm-hero{grid-template-columns:1fr;gap:.75rem} .cm-steps span{display:none} [class*='st-key-card-kpi'] [data-testid='stCaptionContainer']{display:none} .st-key-kpis [data-testid='stMetricValue']{font-size:1.25rem} .st-key-card-kpi-crush [data-testid='stMetricValue']{font-size:2rem}}
@media (max-width:640px){
 [data-testid='stMainBlockContainer']{padding:3.75rem .75rem 2rem}
 .cm-hero{padding:1rem;gap:.75rem;border-radius:10px}
 .cm-hero h1{font-size:1.75rem}
 .cm-q{font-size:1rem}
 .cm-steps{display:none}
 [class*='st-key-card']{padding:.7rem .8rem}
 .st-key-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.5rem}
 .st-key-kpis>div{width:auto!important}
 .st-key-kpis>div:first-child{grid-column:1/-1}
 [class*='st-key-card-kpi']{padding:.55rem .6rem}
 [data-testid='stColumn']{min-width:100%}
 .st-key-method{flex-direction:column}
 [data-testid='stPlotlyChart'] .nsewdrag{pointer-events:none}
 [data-testid='stPlotlyChart'] .modebar-container{display:none}
}
</style>"""

# Static markup: the only runtime values are the data line and the author link.
HERO_HTML = """<div class="cm-hero"><div>
<h1>Soybean Crush Margin</h1>
<p class="cm-q">What is a soybean crusher earning today, and how much of it is biodiesel?</p>
<p class="cm-meta">{data_line}</p>
<p class="cm-meta">Built by <strong>Diego Rossi</strong>. <a href="{linkedin}" target="_blank" rel="noopener">LinkedIn</a> &middot; <a href="{repository}" target="_blank" rel="noopener">GitHub</a></p>
</div>
<ol class="cm-steps">
<li><i>1</i><b>Today's margin</b><span>What crushing a tonne of soybeans earns, against the past 5 years.</span></li>
<li><i>2</i><b>Where it comes from</b><span>How much of the value is oil, now pulled by biodiesel.</span></li>
<li><i>3</i><b>Oil against diesel</b><span>What the oil in biodiesel costs next to diesel.</span></li>
</ol></div>"""
DATA_SOURCE = (
    "CBOT soybeans, meal and oil, and NYMEX heating oil. Daily futures closes from Yahoo Finance"
)

WARNING = (
    "**Gross margins, before costs.** Processing, energy, freight and local basis are not "
    "deducted. Of the biofuel credits, only the D4 RIN is shown, beside the oil cost; 45Z and "
    "LCFS are not, though they can decide real profits. Sessions in a roll or delivery window "
    "are drawn grey and left out of every statistic."
)
LOAD_ERROR = (
    "Prices could not be loaded from Yahoo Finance and no saved copy is available. "
    "Try again in a minute."
)
MARGIN_RAIL = (
    "## How to read it\n\n"
    "A crusher buys soybeans and sells the meal and oil they yield. The margin is what is left "
    "per tonne, before running the plant.\n\n"
    "**Amber line:** this year. **Grey band:** the range of the five previous years, month by "
    "month; the dotted line is their median. **Thin grey line:** roll and delivery sessions.\n\n"
    "Higher than most days means crushing pays well now. It says nothing about where it goes "
    "next."
)
VALUE_RAIL = (
    "## Why oil matters more now\n\n"
    "A tonne of soybeans yields about 0.73 t of meal and 0.18 t of oil. Oil is the smaller "
    "volume, but demand from biodiesel and renewable diesel can make it half of the value.\n\n"
    "**Blue:** oil value. **Pink:** meal value. **White:** soybean cost. The margin is blue plus "
    "pink minus white."
)
FEEDSTOCK_RAIL = (
    "## Oil as a fuel feedstock\n\n"
    "About 0.90 t of soybean oil makes 1 m³ of biodiesel. The chart compares what that oil "
    "costs with 1 m³ of heating oil, which stands in for diesel.\n\n"
    "**It is not the biodiesel price.** Methanol, energy and the plant come on top, and "
    f"biodiesel holds about 7% less energy than diesel ([AFDC]({AFDC_URL})), so biodiesel costs "
    "more to make than this gap shows. Blending mandates and credits pay the difference.\n\n"
    "A higher gap means dearer feedstock. A lower one means soybean oil is cheap as a fuel, "
    "which pulls its demand toward biodiesel.\n\n"
    "**D4 RIN:** the US credit biodiesel earns under the Renewable Fuel Standard, 1.5 per "
    "gallon. Refiners must buy them to comply, so the credit adds to what biodiesel is worth. "
    "**Blue line:** the gap after that credit. Prices are EPA weekly averages, published "
    "monthly, so they lag the futures by weeks."
)
METHOD_FORMULA = (
    "## How it is calculated\n\n"
    "**Crush margin** = 0.733 × meal + 0.183 × oil − soybeans, in USD per tonne of soybeans. "
    "The yields are the CME board crush convention, 44 lb of meal and 11 lb of oil per 60 lb "
    "bushel; in CME units it is 0.022 × meal + 0.11 × oil − soybeans, in USD per bushel "
    f"([CME guide]({CME_GUIDE_URL})). **Feedstock cost** = 0.90 t of oil per m³ of biodiesel, "
    f"from EIA plant data averaging 7.5 lb per gallon ([farmdoc daily]({FARMDOC_URL})). "
    "**RIN value** = 1.5 D4 RINs per gallon "
    f"([40 CFR 80.1415]({EQUIVALENCE_URL})) × 264.17 gallons per m³ × the weekly D4 price "
    f"([EPA]({EPA_PAGE})), used for the sessions of its week and the next."
)
METHOD_ROLLS = (
    "## Rolls and delivery\n\n"
    "A roll is the switch to the next contract. These series stay in the nearest contract until "
    "its last trading day, so they trade through the delivery period and then jump. A trader "
    "who does not take delivery would roll earlier. Sessions in a delivery period, or on the day "
    "of a switch, are drawn grey and left out of every range and percentile: about 4 in 10 "
    "sessions for the crush. From mid-September to mid-December the front months are not the "
    "standard board pair; those sessions are kept, because they move like the rest, but their "
    "level includes a product calendar spread."
)
METHOD_LIMITS = (
    "## What it leaves out\n\n"
    "Plant running costs, energy, freight and local basis; the 45Z and LCFS credits; who keeps "
    "the RIN, which goes to whoever blends the fuel; later contracts on the curve. Prices are "
    "unadjusted futures closes, and the "
    "contract calendars ignore exchange holidays, so a flag can be one session off. "
    f"[Source and known issues]({REPOSITORY_URL})."
)


@st.cache_data(ttl=3600, show_spinner="Loading prices")
def _load_prices() -> LoadResult:
    return load_prices()


@st.cache_data(ttl=6 * 3600, show_spinner="Loading RIN prices")
def _load_rins() -> RinResult:
    return load_d4()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _usd(value: float, unit: str, signed: bool = False) -> str:
    # A typographic minus, as on the chart axes, so a negative value never reads as a dash.
    sign = "−" if round(value) < 0 else ("+" if signed and round(value) > 0 else "")
    return f"{sign}${abs(value):,.0f}/{unit}"


def _markdown(text: str) -> None:
    """Render markdown with dollar signs escaped: st.markdown reads a pair of them as LaTeX."""
    st.markdown(text.replace("$", "\\$"))


def _rank(reading: Reading) -> str:
    return "n/a" if reading.percentile is None else f"{reading.percentile:.0f}%"


def _rank_clause(reading: Reading) -> str:
    if reading.percentile is None:
        return "with too little clean history to rank"
    return f"higher than {reading.percentile:.0f}% of clean days in the past 5 years"


def _older_note(reading: Reading) -> str:
    if reading.date == reading.latest_date:
        return ""
    return (
        f" The latest close, {reading.latest_date.date()}, falls in a roll or delivery window, "
        f"so the reading uses {reading.date.date()}."
    )


def _window(series: pd.Series | pd.DataFrame, window: str) -> pd.Series | pd.DataFrame:
    if window == "All" or series.empty:
        return series
    start = pd.Timestamp(series.index[-1]) - pd.DateOffset(years=int(window[:-1]))
    return series.loc[start:]


def _share_years_before(table: pd.DataFrame, date: pd.Timestamp, years: int) -> float | None:
    clean = table.loc[~table["flagged"].astype(bool), "oil_share"]
    earlier = clean.loc[: date - pd.DateOffset(years=years)]
    return None if earlier.empty else float(earlier.iloc[-1])


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _kpi(key: str, label: str, value: str, delta: str, caption: str) -> None:
    with st.container(key=f"card-kpi-{key}"):
        st.metric(label, value, delta=delta, **METRIC_STYLE)
        st.caption(caption.replace("$", "\\$"))


def _after_rin(gap: Reading | None, rins: RinResult) -> float | None:
    """Gap at the latest clean session minus the latest published D4 RIN, per m3."""
    if gap is None or rins.prices is None:
        return None
    return gap.value - float(rin_per_m3(float(rins.prices.iloc[-1])))


def _summary(crush: Reading, share: Reading | None, gap: Reading | None, rins: RinResult) -> None:
    after = _after_rin(gap, rins)
    gap_caption = (
        "Not the biodiesel price: processing comes on top."
        if after is None
        else f"After the D4 RIN credit: {_usd(after, 'm³', signed=True)}."
    )
    left, right = st.container(key="readrow").columns([3, 2], gap="small", vertical_alignment="top")
    with left, st.container(key="kpis", horizontal=True, gap="small"):
        _kpi(
            "crush",
            "Crush margin",
            _usd(crush.value, "t"),
            f"on {crush.date.date()}",
            "Meal and oil value minus soybean cost, per tonne.",
        )
        _kpi(
            "rank",
            "Against 5 years",
            _rank(crush),
            "of clean days were lower",
            "Roll and delivery sessions left out.",
        )
        _kpi(
            "oil",
            "Oil share",
            "n/a" if share is None else f"{share.value:.0%}",
            "of the product value",
            "Oil's part of what a tonne of soybeans yields.",
        )
        _kpi(
            "gap",
            "Oil cost vs diesel",
            "n/a" if gap is None else _usd(gap.value, "m³", signed=True),
            "per m³ of biodiesel, oil only",
            gap_caption,
        )
    with right:
        st.warning(WARNING, icon=":material/info:")


def _margin_section(table: pd.DataFrame, crush: Reading, window: str) -> None:
    head = (
        f"**Soybean crush margin: {_usd(crush.value, 't')} "
        f"(${crush.value / BUSHELS_PER_TONNE:,.2f}/bu) on {crush.date.date()}, {_rank_clause(crush)}.**"
        f"{_older_note(crush)}"
    )
    year = crush.date.year
    band = monthly_band(table["crush"], table["flagged"], year)
    shown = _window(table, window)
    with st.container(key="row-margin"):
        main, rail = st.columns([3, 1], gap="small", vertical_alignment="top")
        with main, st.container(key="card-margin"):
            with st.container(key="head-margin"):
                _markdown(head)
            seasonal_tab, history_tab = st.tabs([f"{year} against the past 5 years", "History"])
            with seasonal_tab:
                st.plotly_chart(
                    seasonal_figure(table["crush"], table["flagged"], band, year),
                    width="stretch",
                    key="seasonal_chart",
                    theme=None,
                )
            with history_tab:
                st.plotly_chart(
                    history_figure(shown["crush"], shown["flagged"], "USD per tonne of soybeans"),
                    width="stretch",
                    key="history_chart",
                    theme=None,
                )
        with rail, st.container(key="card-rail-margin"):
            st.markdown(MARGIN_RAIL)


def _value_section(table: pd.DataFrame, share: Reading, window: str) -> None:
    before = _share_years_before(table, share.date, 5)
    against = "" if before is None else f", against {before:.0%} five years earlier"
    head = (
        f"**Oil is {share.value:.0%} of the product value on {share.date.date()}{against}.** "
        "Meal and oil value per tonne crushed, against the soybean cost."
    )
    shown = _window(table, window)
    with st.container(key="row-value"):
        main, rail = st.columns([3, 1], gap="small", vertical_alignment="top")
        with main, st.container(key="card-value"):
            with st.container(key="head-value"):
                _markdown(head)
            st.plotly_chart(value_figure(shown), width="stretch", key="value_chart", theme=None)
            st.plotly_chart(oil_share_figure(shown), width="stretch", key="share_chart", theme=None)
        with rail, st.container(key="card-rail-value"):
            st.markdown(VALUE_RAIL)


def _gap_reading(gap: Reading) -> str:
    if gap.percentile is None or 20 < gap.percentile < 80:
        return ""
    cheap = gap.percentile <= 20
    return f", so soybean oil is unusually {'cheap' if cheap else 'dear'} against diesel"


def _rin_clause(gap: Reading, rins: RinResult) -> str:
    after = _after_rin(gap, rins)
    if after is None or rins.prices is None:
        return " D4 RIN prices could not be loaded, so the credit is not shown."
    price = float(rins.prices.iloc[-1])
    side = "more" if after >= 0 else "less"
    saved = ", saved copy" if rins.source == "seed" else ""
    return (
        f" The latest D4 RIN, ${price:,.2f} in the week of {rins.as_of} (EPA{saved}), is worth "
        f"{_usd(float(rin_per_m3(price)), 'm³')} of biodiesel; after it, the oil costs "
        f"{_usd(abs(after), 'm³')} {side} than diesel."
    )


def _feedstock_section(table: pd.DataFrame, gap: Reading, rins: RinResult, window: str) -> None:
    row = table.loc[gap.date]
    side = "more" if gap.value >= 0 else "less"
    head = (
        f"**The soybean oil in 1 m³ of biodiesel costs {_usd(row['feedstock'], 'm³')} on "
        f"{gap.date.date()}, {_usd(abs(gap.value), 'm³')} {side} than 1 m³ of heating oil.** "
        f"That is the feedstock alone, not the biodiesel price. The gap is {_rank_clause(gap)}"
        f"{_gap_reading(gap)}.{_rin_clause(gap, rins)}{_older_note(gap)}"
    )
    shown = _window(table, window)
    with st.container(key="row-feedstock"):
        main, rail = st.columns([3, 1], gap="small", vertical_alignment="top")
        with main, st.container(key="card-feedstock"):
            with st.container(key="head-feedstock"):
                _markdown(head)
            st.plotly_chart(
                feedstock_figure(shown), width="stretch", key="feedstock_chart", theme=None
            )
            st.plotly_chart(gap_figure(shown), width="stretch", key="gap_chart", theme=None)
        with rail, st.container(key="card-rail-feedstock"):
            st.markdown(FEEDSTOCK_RAIL)


def _method() -> None:
    with st.container(key="method", horizontal=True, gap="small"):
        for name, body in (("m1", METHOD_FORMULA), ("m2", METHOD_ROLLS), ("m3", METHOD_LIMITS)):
            with st.container(key=f"card-{name}"):
                st.markdown(body)


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
            data_line=_data_line(loaded), linkedin=LINKEDIN_URL, repository=REPOSITORY_URL
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

    crush = crush_table(loaded.prices)
    crush_reading = latest_reading(crush["crush"], crush["flagged"])
    if crush_reading is None:
        st.warning(
            "Soybean, meal or oil prices are missing, so the crush margin cannot be computed.",
            icon=":material/warning:",
        )
        _footer(loaded)
        return

    rins = _load_rins()
    gap = add_rins(gap_table(loaded.prices), rins.prices)
    share_reading = latest_reading(crush["oil_share"], crush["flagged"])
    gap_reading = latest_reading(gap["gap"], gap["flagged"])

    _summary(crush_reading, share_reading, gap_reading, rins)
    window = st.segmented_control(
        "History window for the charts below",
        WINDOWS,
        default=DEFAULT_WINDOW,
        key="window",
        required=True,
    )
    _margin_section(crush, crush_reading, window)
    if share_reading is not None:
        _value_section(crush, share_reading, window)
    if gap_reading is None:
        st.warning(
            "Soybean oil or heating oil prices are missing, so the feedstock gap is not shown.",
            icon=":material/warning:",
        )
    else:
        _feedstock_section(gap, gap_reading, rins, window)
    _method()
    _footer(loaded, rins)


main()

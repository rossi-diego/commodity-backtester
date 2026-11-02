"""Commodity Backtester: Streamlit UI. All computation lives in ``src``."""

from __future__ import annotations

import datetime
import logging

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.catalog import RULES
from src.constants import COMMODITY_UNITS, RATIO_PAIRS, commodities_dict, tons_conversion
from src.data_loader import LoadResult, load_prices
from src.monitor import ratio_series, ratio_snapshot, rolling_stats
from src.runner import BacktestResult, run_backtest
from src.strategy import BacktestError
from src.visualization import (
    markers_hidden,
    price_signals_figure,
    ratio_figure,
    unadjusted_pnl_figure,
    zscore_figure,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

VIEWS = ["Backtest", "Ratio monitor"]
MARKETS = list(commodities_dict.values())
RULE_KEY_BY_LABEL = {rule.label: rule.key for rule in RULES}
RULE_HELP = "\n\n".join(f"**{rule.label}:** {rule.description}" for rule in RULES)
PERIODS = ["1Y", "3Y", "5Y", "10Y", "All", "Custom"]
DEFAULT_PERIOD = "10Y"
DEFAULT_CUSTOM_START = datetime.date(2008, 1, 2)
DEFAULT_CUSTOM_END = datetime.date(2012, 12, 31)
FEW_TRADES_BELOW = 30
REPOSITORY_URL = "https://github.com/rossi-diego/commodity-backtester"

METRIC_STYLE = {"delta_color": "off", "delta_arrow": "off"}

# Amber means "holds the contract" and nothing else, so the warning beside it is not amber.
# Other alerts keep the theme colours: they replace the results, so they must look different.
# The rules target Streamlit test ids, so redo the screenshots when upgrading Streamlit.
# The top padding clears the 56 px bar Streamlit Cloud draws over the app from its parent page.
APP_CSS = """<style>
:root{--page:#0b1016;--card:#141b23;--raised:#1e2732;--line:#2a3440;--ink:#eef2f6;--mut:#a3afbd;--head:linear-gradient(115deg,#1a2430 0%,#16212c 55%,#3a2c10 100%);--posl:#c58512;--onpos:#0b1016;--posf:#3d4957;--step:rgba(255,255,255,.07)}
[data-testid='stMainBlockContainer']{max-width:none;padding:3.75rem 2rem 2rem}
.bt-hero{background:var(--head);border-radius:12px;padding:1.1rem 1.5rem;color:#fff;display:grid;grid-template-columns:minmax(0,13fr) minmax(0,11fr);gap:1.5rem;align-items:center}
.bt-hero h1{color:#fff;font-size:2.25rem;font-weight:600;letter-spacing:-.02em;line-height:1.1;margin:0 0 .4rem;padding:0}
.bt-q{font-size:1.0625rem;line-height:1.35;margin:0 0 .4rem;color:#fff}
.bt-meta{font-size:.875rem;line-height:1.45;margin:0;color:#dbe6f2}
.bt-meta a{color:#fff;font-weight:500}
.bt-meta strong{color:#fff;font-weight:600}
.bt-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:0;padding:0;list-style:none}
.bt-steps li{background:var(--step);border:1px solid rgba(255,255,255,.18);border-radius:10px;padding:.6rem .75rem;font-size:.8125rem;line-height:1.3;color:#e8eff7}
.bt-steps b{color:#fff;font-size:.9375rem;font-weight:600}
.bt-steps span{display:block;margin-top:.3rem}
.bt-steps i{font-style:normal;display:inline-block;width:1.35rem;height:1.35rem;line-height:1.35rem;text-align:center;border-radius:50%;background:#fff;color:#0b1016;font-weight:600;font-size:.75rem;margin-right:.35rem}
[class*='st-key-card']{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.85rem 1rem}
.st-key-card-controls{border-top:3px solid var(--posl);gap:.5rem}
.st-key-card-controls [data-testid='stMarkdownContainer'] p{margin:0;font-size:.9375rem}
.st-key-controls>div{flex:1 1 9rem;min-width:0;width:auto!important}
.st-key-controls>div:has([data-testid='stButtonGroup']){flex:0 0 auto}
.st-key-controls>div:has([data-testid='stPopover']){flex:0 0 auto}
.st-key-controls [data-testid='stButtonGroup'] button{min-height:2.5rem}
[data-testid='stMarkdownContainer']{margin-bottom:0}
[data-testid='stMetricDelta']{background:none!important;padding:0!important;color:var(--mut)!important;font-size:.8125rem!important}
[data-testid='stMetricDelta'] *{white-space:normal!important}
[data-testid='stMetricLabel'] p{color:var(--mut);font-size:.8125rem;font-weight:500}
[class*='st-key-card-kpi']>div{flex-shrink:0}
[class*='st-key-card-kpi'] [data-testid='stMarkdown'],[class*='st-key-card-kpi'] [data-testid='stMarkdown']>div{display:block;height:auto!important}
@media (min-width:1101px){[class*='st-key-card-kpi'] [data-testid='stElementContainer']:has([data-testid='stCaptionContainer']){min-height:2.1rem}}
[class*='st-key-card-kpi'] [data-testid='stCaptionContainer'] p{font-size:.78rem;line-height:1.3;margin:0;color:var(--mut);opacity:1}
[class*='st-key-card-kpi']{gap:.2rem;flex:1 1 auto;height:auto!important;padding:.7rem .85rem}
.st-key-kpis>div{display:flex}
[data-testid='stCaptionContainer']{opacity:1}
[data-testid='stCaptionContainer'] p{color:var(--mut)}
.st-key-kpis{align-items:stretch}
.st-key-readrow [data-testid='stHorizontalBlock'],.st-key-chartrow [data-testid='stHorizontalBlock']{align-items:stretch}
.st-key-kpis{height:auto!important;flex:0 0 auto}
.st-key-readrow [data-testid='stColumn'] [data-testid='stVerticalBlock']:has(>[data-testid='stLayoutWrapper']>.st-key-kpis){flex:1 1 auto;height:auto}
.st-key-readrow [data-testid='stColumn'] [data-testid='stLayoutWrapper']:has(>.st-key-kpis){flex:0 0 auto}
.st-key-chartrow [data-testid='stColumn']>div{height:100%}
.st-key-readrow [data-testid='stColumn']:last-child>div,.st-key-readrow [data-testid='stElementContainer']:has([data-testid='stAlert']){height:100%}
.st-key-readrow [data-testid='stAlert'],[class*='st-key-card-rail']{height:100%}
.st-key-chartrow [data-testid='stColumn']:last-child [data-testid='stVerticalBlock'],.st-key-chartrow [data-testid='stColumn']:last-child [data-testid='stLayoutWrapper']{height:100%}
[data-testid='stExpander']{background:var(--card);border-radius:10px}
.st-key-method>div{flex:1 1 0;min-width:0;width:auto!important}
.st-key-charthead{margin-bottom:-.5rem}
.st-key-kpis>div{flex:1 1 0;min-width:0;width:auto!important}
.st-key-card-kpi-long{background:var(--posl);border-color:var(--posl)}
.st-key-card-kpi-flat{background:var(--posf);border-color:var(--posf)}
.st-key-card-kpi-hero{background:var(--raised);border-color:var(--mut)}
.st-key-card-kpi-long *{color:var(--onpos)!important}
.st-key-card-kpi-flat *,.st-key-card-kpi-hero *{color:#fff!important}
.st-key-card-kpi-long [data-testid='stMetricValue'],.st-key-card-kpi-flat [data-testid='stMetricValue'],.st-key-card-kpi-hero [data-testid='stMetricValue']{font-size:2.25rem;line-height:1.15;letter-spacing:-.02em}
.st-key-card-kpi-pnl [data-testid='stMetricValue']{color:var(--mut);font-weight:500}
.st-key-readrow [data-testid='stAlertContainer']{background:var(--raised);color:var(--ink);border-left:4px solid var(--ink);height:100%}
.st-key-readrow [data-testid='stAlertContainer'] p,.st-key-readrow [data-testid='stAlertContainer'] [data-testid='stAlertDynamicIcon']{font-size:.875rem;line-height:1.45;margin:0;color:var(--ink)}
.st-key-readrow [data-testid='stAlert']{height:100%}
.st-key-chartrow [data-testid='stMarkdownContainer'] p{margin:0 0 .5rem}
[class*='st-key-card-rail'] h4,.st-key-method h4{font-size:.9375rem;font-weight:600;margin:0 0 .35rem;padding:0;color:var(--ink)}
[class*='st-key-card-rail'] p,.st-key-method p{font-size:.875rem;line-height:1.45;color:var(--ink)}
[class*='st-key-charthead'] p{font-size:.875rem;color:var(--mut);margin:0}
[class*='st-key-charthead'] strong{color:var(--ink);font-weight:600}
@media (max-width:1100px){.st-key-chartrow [data-testid='stColumn']{min-width:100%} .bt-hero{grid-template-columns:1fr;gap:.75rem} .bt-steps span{display:none} [class*='st-key-card-kpi'] [data-testid='stCaptionContainer']{display:none}}
@media (max-width:640px){
 [data-testid='stMainBlockContainer']{padding:3.75rem .75rem 2rem}
 .bt-hero{padding:1rem;gap:.75rem;border-radius:10px}
 .bt-hero h1{font-size:1.75rem}
 .st-key-card-controls [data-testid='stElementContainer']:has(>[data-testid='stMarkdown']){display:none}
 .st-key-controls label[data-testid='stWidgetLabel']{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
 .bt-q{font-size:1rem}
 .bt-steps{gap:.4rem}
 .bt-steps li{padding:.5rem;font-size:.75rem}
 .bt-steps li span{display:none}
 .bt-steps b{font-size:.75rem;display:block;margin-top:.2rem}
 .bt-steps i{margin:0}
 [class*='st-key-card']{padding:.7rem .8rem}
 .st-key-controls{column-gap:.5rem;row-gap:.4rem}
 .st-key-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.5rem}
 .st-key-kpis>div{width:auto!important}
 .st-key-kpis>div:first-child{grid-column:1/-1}
 [class*='st-key-card-kpi'] [data-testid='stCaptionContainer']{display:none}
 .st-key-kpis [data-testid='stMetricValue']{font-size:1.25rem}
 .st-key-card-kpi-long [data-testid='stMetricValue'],.st-key-card-kpi-flat [data-testid='stMetricValue'],.st-key-card-kpi-hero [data-testid='stMetricValue']{font-size:2rem}
 [class*='st-key-card-kpi']{padding:.55rem .6rem}
 [data-testid='stColumn']{min-width:100%}
 .st-key-method{flex-direction:column}
}
</style>"""

# Static markup: the only runtime value is the data line, built from the date index.
HERO_HTML = """<div class="bt-hero"><div>
<h1>Commodity Backtester</h1>
<p class="bt-q">What does a simple rule say today in grain and energy futures, and how far can you trust it?</p>
<p class="bt-meta">{data_line}</p>
<p class="bt-meta">Built by <strong>Diego Rossi</strong>, Market Risk Coordinator at Grupo Oleoplan, Porto Alegre, Brazil. <a href="https://www.linkedin.com/in/diego-rossi-santanna/" target="_blank" rel="noopener">LinkedIn</a> &middot; <a href="https://github.com/rossi-diego/commodity-backtester" target="_blank" rel="noopener">GitHub</a></p>
</div>
<ol class="bt-steps">
<li><i>1</i><b>Pick a market</b><span>Soybean, corn, wheat, soy oil, soy meal, heating oil or crude.</span></li>
<li><i>2</i><b>Pick a rule</b><span>Three plain rules, explained beside the chart.</span></li>
<li><i>3</i><b>Read the result</b><span>Position first, then the warning. It updates by itself.</span></li>
</ol></div>"""
DATA_SOURCE = "Seven CBOT and NYMEX front-month futures. Daily closes from Yahoo Finance"
START_HERE = (
    "**Start here:** pick a market and a rule. "
    "Results update by themselves, there is nothing to run."
)

# Dollar signs are escaped because st.markdown reads a pair of them as LaTeX. The example is
# dated because the engine's number moves with the data, and its three claims stay together:
# without the last one, "+$63k" would suggest an edge.
BACKTEST_WARNING = (
    "**P&L is indicative: contract rolls book as P&L, so the sign can flip.** "
    "Soybean Meal, 12M momentum, full history to 2026-10-27: about -\\$12.5k here, "
    "about +\\$63k with roll days removed. "
    "Neither rule has been shown to beat holding the front month."
)
MONITOR_WARNING = (
    "**A monitor, not a strategy: no trades and no P&L.** Ratios use unadjusted front-month "
    "closes, so contract rolls cause jumps."
)
CHART_KEY_HELP = (
    "The line turns amber and the background shades while the rule holds the contract. "
    "Triangles mark each entry and exit. The dot is the last close."
)
PNL_HEAD = (
    "**Unadjusted P&L, one contract.** Steps are realised at each exit. "
    "The dotted line marks an open position at the window end."
)
RATIO_HELP = (
    "#### How to read a ratio\n\n"
    "A ratio compares two markets. A high z-score says the first leg is expensive against the "
    "second compared with the past year, not that it will fall.\n\n"
    "**Z-score beyond +/-2** is unusual. **Percentile** puts the level in a 5-year context.\n\n"
    "Rolls move ratios, so a jump may be a contract change, not news."
)
METHOD_NUMBERS = (
    "#### What the numbers mean\n\n"
    "**Position:** long means the rule holds one contract, flat means none. "
    "**Round trips:** completed buy and sell pairs; under 30, treat with caution. "
    "**Time in market:** share of sessions held long. "
    "**Unadjusted P&L:** USD on one contract, rolls included."
)
METHOD_TEST = (
    "#### How the test runs\n\n"
    "Signals use the close of day T and fill at the close of T+1, so a rule cannot trade on the "
    "price that fired it. One contract, long or flat, starting flat. Commission and slippage are "
    "on by default; change them under Costs."
)
METHOD_LIMITS = (
    "#### What it does not show\n\n"
    "No edge: every result is in-sample, and neither rule has been shown to beat holding the "
    "front month from 2000 to 2026. Risk metrics stay off until the engine has a daily ledger. "
    f"[Known issues]({REPOSITORY_URL}/milestone/1)."
)


@st.cache_data(ttl=3600, show_spinner="Loading prices")
def _load_prices() -> LoadResult:
    return load_prices()


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _period_control(bounds: tuple[datetime.date, datetime.date]) -> tuple[str, object, object]:
    """Render Period, plus Start and End when it is Custom. Call inside the control bar."""
    period = st.selectbox(
        "Period",
        PERIODS,
        index=PERIODS.index(DEFAULT_PERIOD),
        key="period",
        width="stretch",
        persist_state="session",
    )
    if period != "Custom":
        return period, None, None

    first, last = bounds
    start = st.date_input(
        "Start",
        value=min(max(DEFAULT_CUSTOM_START, first), last),
        min_value=first,
        max_value=last,
        key="custom_start",
        width="stretch",
        persist_state="session",
    )
    end = st.date_input(
        "End",
        value=min(max(DEFAULT_CUSTOM_END, first), last),
        min_value=first,
        max_value=last,
        key="custom_end",
        width="stretch",
        persist_state="session",
    )
    return period, start, end


def _resolve_window(
    period: str,
    custom_start: object,
    custom_end: object,
    series: pd.Series | pd.DataFrame,
) -> tuple[datetime.date, datetime.date] | None:
    """Return the window for a period, or None after warning about an invalid custom one."""
    if period == "Custom":
        if not isinstance(custom_start, datetime.date) or not isinstance(custom_end, datetime.date):
            st.warning("Pick a start and an end date.", icon=":material/warning:")
            return None
        if custom_start >= custom_end:
            st.warning("Start must be before End.", icon=":material/warning:")
            return None
        return custom_start, custom_end

    last = pd.Timestamp(series.index[-1])
    if period == "All":
        return pd.Timestamp(series.index[0]).date(), last.date()
    return (last - pd.DateOffset(years=int(period[:-1]))).date(), last.date()


def _kpi(
    key: str, label: str, value: str, delta: str, caption: str, help: str | None = None
) -> None:
    with st.container(key=f"card-kpi-{key}"):
        st.metric(label, value, delta=delta, help=help, **METRIC_STYLE)
        st.caption(caption)


def _period_label(period: str, window: tuple[datetime.date, datetime.date]) -> str:
    return f"{window[0]} to {window[1]}" if period == "Custom" else period


# ---------------------------------------------------------------------------
# Backtest view
# ---------------------------------------------------------------------------


def _render_backtest(
    prices: pd.DataFrame,
    bounds: tuple[datetime.date, datetime.date],
    controls: DeltaGenerator,
) -> None:
    with controls:
        market = st.selectbox(
            "Market", MARKETS, key="market", width="stretch", persist_state="session"
        )
        rule_label = st.selectbox(
            "Rule",
            list(RULE_KEY_BY_LABEL),
            key="rule",
            width="stretch",
            help=RULE_HELP,
            persist_state="session",
        )
        period, custom_start, custom_end = _period_control(bounds)
        with st.popover("Costs", width="content"):
            commission = st.number_input(
                "Commission per fill (USD)",
                min_value=0.0,
                max_value=100.0,
                value=2.5,
                step=0.5,
                key="commission",
                persist_state="session",
            )
            slippage = st.number_input(
                "Slippage per fill (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                key="slippage",
                persist_state="session",
            )

    rule = RULE_KEY_BY_LABEL[rule_label]
    series = prices[market].dropna() if market in prices else pd.Series(dtype=float)
    if series.empty:
        st.warning(f"No {market} prices are available.", icon=":material/warning:")
        return

    window = _resolve_window(period, custom_start, custom_end, series)
    if window is None:
        return

    try:
        result = run_backtest(rule, market, window[0], window[1], prices, commission, slippage)
    except BacktestError:
        st.warning(f"No {market} prices in this window.", icon=":material/warning:")
        return

    at_latest_close = result.window.index[-1] == series.index[-1]
    left, right = st.container(key="readrow").columns([3, 2], gap="small", vertical_alignment="top")
    with left:
        _backtest_metrics(result, rule, at_latest_close)
    with right:
        st.warning(BACKTEST_WARNING, icon=":material/info:")

    unit = COMMODITY_UNITS[market]
    head = _price_head(result, market, rule_label, _period_label(period, window), unit)
    with st.container(key="chartrow"):
        main, rail = st.columns([3, 1], gap="small", vertical_alignment="top")
        with main, st.container(key="card-chart"):
            price_tab, pnl_tab, log_tab = st.tabs(
                ["Price and signals", "Unadjusted P&L", "Trade log"]
            )
            with price_tab:
                with st.container(key="charthead"):
                    st.markdown(head)
                st.plotly_chart(
                    price_signals_figure(result.window, result.trades, unit),
                    width="stretch",
                    key="price_chart",
                    theme=None,
                )
                if markers_hidden(result.trades):
                    st.caption(
                        f"{len(result.trades)} fills: markers hidden, "
                        "toggle them in the legend or zoom in."
                    )
            with pnl_tab:
                with st.container(key="charthead-pnl"):
                    st.markdown(PNL_HEAD)
                st.plotly_chart(
                    unadjusted_pnl_figure(result.trades, result.window, result.mtm),
                    width="stretch",
                    key="pnl_chart",
                    theme=None,
                )
            with log_tab:
                _trade_log(result.trades, unit, f"{market}_{rule}_{window[0]}_{window[1]}")
        with rail, st.container(key="card-rail-rules"):
            st.markdown(_rules_rail(rule_label))

    with st.container(key="method", horizontal=True, gap="small"):
        for name, body in (("m1", METHOD_NUMBERS), ("m2", METHOD_TEST), ("m3", METHOD_LIMITS)):
            with st.container(key=f"card-{name}"):
                st.markdown(body)

    with st.expander("Indicative statistics"):
        if result.sharpe is not None:
            st.metric(
                "Sharpe",
                f"{result.sharpe:.2f}",
                delta="unadjusted, incl. rolls",
                **METRIC_STYLE,
            )
        else:
            st.caption(
                f"Sharpe needs the daily ledger for a rule with {result.summary.round_trips} "
                "round trips in this window, so it is not shown."
            )


def _backtest_metrics(result: BacktestResult, rule: str, at_latest_close: bool) -> None:
    summary = result.summary
    show_few_trades = rule != "buy_hold" and summary.round_trips < FEW_TRADES_BELOW
    with st.container(key="kpis", horizontal=True, gap="small"):
        _kpi(
            summary.position.lower(),
            "Position now" if at_latest_close else "Position at end",
            summary.position,
            f"since {summary.since}" if summary.since else "no fills",
            "Does the rule hold a contract at the "
            + ("last close?" if at_latest_close else "window end?"),
        )
        _kpi(
            "rt",
            "Round trips",
            str(summary.round_trips),
            f"{summary.fills} fill" + ("" if summary.fills == 1 else "s"),
            "Completed buy and sell pairs.",
            help="Few trades: treat with caution." if show_few_trades else None,
        )
        _kpi(
            "tim",
            "Time in market",
            f"{summary.time_in_market * 100:.0f}%",
            "of the window",
            "Share of sessions the rule was long.",
        )
        _kpi(
            "pnl",
            "Unadjusted P&L",
            _usd(summary.unadjusted_pnl),
            "per contract, incl. rolls",
            "Profit or loss in USD, with no roll adjustment.",
        )


def _price_head(
    result: BacktestResult, market: str, rule_label: str, period_label: str, unit: str
) -> str:
    title = f"**{market}, {period_label}.**"
    action = (
        f"No fills: {rule_label} stayed flat."
        if result.trades.empty
        else f"Amber while {rule_label} is long, grey while flat."
    )
    close = float(result.window.iloc[-1])
    price = f"{close:,.0f}" if abs(close) >= 100 else f"{close:,.2f}"
    return f"{title} {action} Close {price} {unit} on {result.window.index[-1].date()}."


def _rules_rail(selected: str) -> str:
    rules = "\n\n".join(
        f"**{rule.label}**{' :orange-badge[Selected]' if rule.label == selected else ''}  \n"
        f"{rule.description}"
        for rule in RULES
    )
    return f"#### The three rules\n\n{rules}\n\n#### Reading the chart\n\n{CHART_KEY_HELP}"


def _trade_log(trades: pd.DataFrame, unit: str, file_stem: str) -> None:
    if trades.empty:
        st.caption("No fills in this window.")
        return

    price_label = f"Price ({unit})"
    log = trades.reset_index()[["date", "position", "trade_price", "pnl_usd", "pnl_usd_cumsum"]]
    log.columns = ["Date", "Position", price_label, "P&L (USD)", "Cumulative (USD)"]
    st.dataframe(
        log,
        hide_index=True,
        width="stretch",
        height=380,
        column_config={
            "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            price_label: st.column_config.NumberColumn(format="%.2f"),
            "P&L (USD)": st.column_config.NumberColumn(format="%.0f"),
            "Cumulative (USD)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.download_button(
        "Download trades (CSV)",
        data=lambda: log.to_csv(index=False),
        file_name=f"trades_{file_stem}.csv".replace(" ", "_"),
        mime="text/csv",
        on_click="ignore",
        key="download_trades",
    )


def _usd(value: float) -> str:
    sign = "-" if round(value) < 0 else ""
    return f"{sign}${abs(value):,.0f}"


# ---------------------------------------------------------------------------
# Ratio monitor view
# ---------------------------------------------------------------------------


def _render_monitor(
    prices: pd.DataFrame,
    bounds: tuple[datetime.date, datetime.date],
    controls: DeltaGenerator,
) -> None:
    with controls:
        pair = st.selectbox(
            "Pair", list(RATIO_PAIRS), key="pair", width="stretch", persist_state="session"
        )
        period, custom_start, custom_end = _period_control(bounds)

    first_leg, second_leg = RATIO_PAIRS[pair]
    has_legs = first_leg in prices and second_leg in prices
    ratio = (
        ratio_series(prices, first_leg, second_leg, tons_conversion)
        if has_legs
        else pd.Series(dtype=float)
    )
    if ratio.empty:
        st.warning(f"No {pair} data is available.", icon=":material/warning:")
        return

    window = _resolve_window(period, custom_start, custom_end, ratio)
    if window is None:
        return

    start, end = str(window[0]), str(window[1])
    history = ratio.loc[:end]
    shown = history.loc[start:]
    if shown.empty:
        st.warning(f"No {pair} data in this window.", icon=":material/warning:")
        return

    snapshot = ratio_snapshot(history)
    left, right = st.container(key="readrow").columns([3, 2], gap="small", vertical_alignment="top")
    with left, st.container(key="kpis", horizontal=True, gap="small"):
        _kpi(
            "hero",
            "Z-score",
            "n/a" if snapshot.z is None else f"{snapshot.z:+.2f}",
            "trailing 1y",
            "Distance from the 1-year mean, in standard deviations.",
        )
        _kpi(
            "p",
            "Percentile",
            "n/a" if snapshot.percentile is None else f"{snapshot.percentile:.0f}",
            "trailing 5y",
            "Share of the past 5 years below this level.",
        )
        _kpi(
            "l",
            "Level",
            f"{snapshot.level:.3f}",
            "USD/MT basis",
            "First leg over second, both in USD per metric ton.",
            help=f"Ratio at the close of {snapshot.as_of}.",
        )
    with right:
        st.info(MONITOR_WARNING, icon=":material/info:")

    stats = rolling_stats(history).loc[start:]
    head = (
        f"**{pair}, {_period_label(period, window)}.** Last {snapshot.level:.3f} on "
        f"{snapshot.as_of}. Below, the z-score: brighter beyond +/-2."
    )
    with st.container(key="chartrow"):
        main, rail = st.columns([3, 1], gap="small", vertical_alignment="top")
        with main, st.container(key="card-chart"):
            with st.container(key="charthead"):
                st.markdown(head)
            st.plotly_chart(
                ratio_figure(shown, stats), width="stretch", key="ratio_chart", theme=None
            )
            st.plotly_chart(zscore_figure(stats), width="stretch", key="zscore_chart", theme=None)
        with rail, st.container(key="card-rail-mon"):
            st.markdown(RATIO_HELP)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _data_line(loaded: LoadResult) -> str:
    if loaded.prices is None:
        return f"{DATA_SOURCE}."
    freshness = " (may be out of date)" if loaded.is_stale else ""
    return f"{DATA_SOURCE}, 2000 to {loaded.as_of}{freshness}."


def main() -> None:
    st.set_page_config(
        page_title="Commodity Backtester",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(APP_CSS)
    loaded = _load_prices()
    st.html(HERO_HTML.format(data_line=_data_line(loaded)))

    if loaded.prices is None:
        st.error(f"Prices could not be loaded. {loaded.error}", icon=":material/warning:")
        if st.button("Retry"):
            _load_prices.clear()
            st.rerun()
        st.caption(f"[Source code]({REPOSITORY_URL}) · educational use")
        return

    with st.container(key="card-controls"):
        st.markdown(START_HERE)
        controls = st.container(
            key="controls", horizontal=True, vertical_alignment="bottom", gap="small"
        )
        with controls:
            view = st.segmented_control(
                "View",
                VIEWS,
                default=VIEWS[0],
                key="view",
                required=True,
                label_visibility="collapsed",
            )
    bounds = (loaded.prices.index.min().date(), loaded.prices.index.max().date())
    render = _render_backtest if view == VIEWS[0] else _render_monitor
    render(loaded.prices, bounds, controls)

    freshness = " (may be out of date)" if loaded.is_stale else ""
    st.caption(
        "Daily closes, fills at next close, one contract. "
        f"Data to {loaded.as_of}{freshness}. [Source code]({REPOSITORY_URL}) · educational use"
    )


main()

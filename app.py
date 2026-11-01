"""Commodity Backtester: Streamlit UI. All computation lives in ``src``."""

from __future__ import annotations

import datetime
import logging

import pandas as pd
import streamlit as st

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

METRIC_STYLE = {"width": 160, "delta_color": "off", "delta_arrow": "off"}

MAX_WIDTH_CSS = (
    "<style>[data-testid='stMainBlockContainer']{padding-bottom:2rem;max-width:1320px}</style>"
)

BACKTEST_WARNING = (
    "P&L is indicative: unadjusted front-month prices, contract rolls included, "
    "and the sign can flip."
)
MONITOR_WARNING = (
    "A monitor, not a strategy: no trades and no P&L. Ratios use unadjusted front-month "
    "closes, so contract rolls cause jumps."
)
# Dollar signs are escaped because st.markdown reads a pair of them as LaTeX.
HOW_TO_READ = (
    "Signals use the close of day T and fill at the close of T+1. One contract, long or flat, "
    "starts flat at the window start. Neither rule has been shown to beat holding the front "
    "month over 2000 to 2026; the difference is within noise. In an in-sample study of the "
    "seven markets, equal notional with roll days removed, drawdown was 64% for 12M momentum, "
    "26% for Donchian 20/10 and 79% for buy & hold, with lower returns for the rules. Example "
    "of the roll problem: Soybean Meal with 12M momentum, full history, shows about -\\$12.5k "
    "here and about +\\$63k with roll days removed. Known issues: "
    "[v0.3 milestone](https://github.com/rossi-diego/commodity-backtester/milestone/1)."
)
FOOTER = "[GitHub](https://github.com/rossi-diego/commodity-backtester) · educational use"


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
        width=100,
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
        width=130,
        persist_state="session",
    )
    end = st.date_input(
        "End",
        value=min(max(DEFAULT_CUSTOM_END, first), last),
        min_value=first,
        max_value=last,
        key="custom_end",
        width=130,
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


# ---------------------------------------------------------------------------
# Backtest view
# ---------------------------------------------------------------------------


def _render_backtest(prices: pd.DataFrame, bounds: tuple[datetime.date, datetime.date]) -> None:
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        market = st.selectbox("Market", MARKETS, key="market", width=170, persist_state="session")
        rule_label = st.selectbox(
            "Rule",
            list(RULE_KEY_BY_LABEL),
            key="rule",
            width=170,
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
    _backtest_metrics(result, rule, at_latest_close)
    st.warning(BACKTEST_WARNING, icon=":material/info:")

    unit = COMMODITY_UNITS[market]
    price_tab, pnl_tab, log_tab = st.tabs(["Price and signals", "Unadjusted P&L", "Trade log"])
    with price_tab:
        st.plotly_chart(
            price_signals_figure(result.window, result.trades, unit),
            width="stretch",
            key="price_chart",
        )
        if markers_hidden(result.trades):
            st.caption(
                f"{len(result.trades)} fills: markers hidden, toggle them in the legend or zoom in."
            )
    with pnl_tab:
        st.plotly_chart(
            unadjusted_pnl_figure(result.trades, result.window, result.mtm),
            width="stretch",
            key="pnl_chart",
        )
    with log_tab:
        _trade_log(result.trades, unit, f"{market}_{rule}_{window[0]}_{window[1]}")

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
    with st.expander("How to read this"):
        st.markdown(HOW_TO_READ)


def _backtest_metrics(result: BacktestResult, rule: str, at_latest_close: bool) -> None:
    summary = result.summary
    show_few_trades = rule != "buy_hold" and summary.round_trips < FEW_TRADES_BELOW
    with st.container(horizontal=True, gap="medium"):
        st.metric(
            "Position now" if at_latest_close else "Position at end",
            summary.position,
            delta=f"since {summary.since}" if summary.since else "no fills",
            **METRIC_STYLE,
        )
        st.metric(
            "Round trips",
            str(summary.round_trips),
            delta=f"{summary.fills} fill" + ("" if summary.fills == 1 else "s"),
            help="Few trades: treat with caution." if show_few_trades else None,
            **METRIC_STYLE,
        )
        st.metric(
            "Time in market",
            f"{summary.time_in_market * 100:.0f}%",
            delta="of the window",
            **METRIC_STYLE,
        )
        st.metric(
            "Unadjusted P&L",
            _usd(summary.unadjusted_pnl),
            delta="per contract, incl. rolls",
            **METRIC_STYLE,
        )


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


def _render_monitor(prices: pd.DataFrame, bounds: tuple[datetime.date, datetime.date]) -> None:
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        pair = st.selectbox(
            "Pair", list(RATIO_PAIRS), key="pair", width=170, persist_state="session"
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
    with st.container(horizontal=True, gap="medium"):
        st.metric(
            "Level",
            f"{snapshot.level:.3f}",
            delta="USD/MT basis",
            help=f"Ratio at the close of {snapshot.as_of}.",
            **METRIC_STYLE,
        )
        st.metric(
            "Z-score",
            "n/a" if snapshot.z is None else f"{snapshot.z:+.2f}",
            delta="trailing 1y",
            **METRIC_STYLE,
        )
        st.metric(
            "Percentile",
            "n/a" if snapshot.percentile is None else f"{snapshot.percentile:.0f}",
            delta="trailing 5y",
            **METRIC_STYLE,
        )
    st.info(MONITOR_WARNING, icon=":material/info:")

    stats = rolling_stats(history).loc[start:]
    st.plotly_chart(ratio_figure(shown, stats), width="stretch", key="ratio_chart")
    st.plotly_chart(zscore_figure(stats), width="stretch", key="zscore_chart")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Commodity Backtester",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(MAX_WIDTH_CSS)
    st.markdown("### Commodity Backtester")

    loaded = _load_prices()
    if loaded.prices is None:
        st.error(f"Prices could not be loaded. {loaded.error}", icon=":material/warning:")
        if st.button("Retry"):
            _load_prices.clear()
            st.rerun()
    else:
        freshness = " (may be out of date)" if loaded.is_stale else ""
        st.caption(
            "Rules on CBOT and NYMEX front-month futures. Daily closes, fills at next close, "
            f"one contract. Data to {loaded.as_of}{freshness}."
        )
        view = st.segmented_control(
            "View",
            VIEWS,
            default=VIEWS[0],
            key="view",
            required=True,
            label_visibility="collapsed",
        )
        bounds = (loaded.prices.index.min().date(), loaded.prices.index.max().date())
        if view == VIEWS[0]:
            _render_backtest(loaded.prices, bounds)
        else:
            _render_monitor(loaded.prices, bounds)

    st.caption(FOOTER)


main()

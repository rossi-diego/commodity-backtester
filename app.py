"""
Commodity Backtester - Streamlit Application

An interactive web application for backtesting commodity trading strategies.
Supports multiple strategies including ratio trading, mean reversion, momentum, and breakout.

Usage:
    streamlit run app.py

Author: Diego Rossi
Version: 2.0.0
"""

import datetime
import io
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.constants import commodities_dict, contract_sizes, tons_conversion, COMMODITIES
from src.data_loader import yahoo_quotes, DataLoaderError
from src.strategy import backtest, BacktestError, StrategyType, get_available_strategies
from src.utils import pnl_trades, backtest_performance, backtest_performance_extended
from src.visualization import backtest_charts, create_strategy_comparison_chart


# ---------------------------------------------------------------------------
# Cached data loader — avoids re-downloading on every widget interaction
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _load_market_data(
    start_date_str: str, end_date_str: str
) -> tuple[pd.DataFrame, datetime.date | None]:
    """Thin wrapper around yahoo_quotes that Streamlit can cache."""
    return yahoo_quotes(start_date_str, end_date_str)

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Commodity Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .strategy-card {
        background-color: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: border-color 0.3s;
    }
    .strategy-card:hover {
        border-color: #1e3a5f;
    }
    .positive-value {
        color: #28a745;
        font-weight: bold;
    }
    .negative-value {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196f3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Application Header
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">📈 Commodity Backtester</h1>
    <p style="margin:5px 0 0 0; opacity:0.9;">Backtest commodity spread strategies using real futures data</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar Configuration
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Transaction costs settings
    st.markdown("#### Transaction Costs")
    enable_costs = st.checkbox("Enable Transaction Costs", value=False)

    if enable_costs:
        commission_per_trade = st.number_input(
            "Commission per Trade ($)",
            min_value=0.0,
            max_value=100.0,
            value=2.50,
            step=0.50,
            help="Fixed commission charged per trade"
        )
        slippage_pct = st.slider(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Expected slippage as percentage of trade price"
        )
    else:
        commission_per_trade = 0.0
        slippage_pct = 0.0

    st.markdown("---")

    # Risk management settings
    st.markdown("#### Risk Management")
    enable_stop_loss = st.checkbox("Enable Stop Loss", value=False)

    if enable_stop_loss:
        stop_loss_pct = st.slider(
            "Stop Loss (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Close position if loss exceeds this percentage"
        )
    else:
        stop_loss_pct = None

    enable_take_profit = st.checkbox("Enable Take Profit", value=False)

    if enable_take_profit:
        take_profit_pct = st.slider(
            "Take Profit (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Close position if profit exceeds this percentage"
        )
    else:
        take_profit_pct = None

    st.markdown("---")

    # Display settings
    st.markdown("#### Display")
    show_advanced_metrics = st.checkbox("Show Advanced Metrics", value=True)
    show_trade_log = st.checkbox("Show Trade Log", value=True)

# =============================================================================
# Strategy Selection with Tabs
# =============================================================================

st.markdown("### 📊 Select Trading Strategy")

# Strategy descriptions
STRATEGY_INFO = {
    "ratio": {
        "name": "Ratio Strategy",
        "icon": "📐",
        "description": "Trade based on price ratios between two commodities in metric ton terms. Ideal for spread trading.",
        "best_for": "Spread trading, mean-reverting pairs",
        "details": """
Calculates the metric-ton price ratio between two commodities (e.g., Soybean/Corn). A buy signal triggers when the ratio falls below the entry threshold — indicating the primary commodity is cheap relative to the reference. A sell triggers when the ratio recovers above the exit threshold.

The entry and exit levels are auto-populated from historical mean ± 1 std dev, but can be adjusted manually. The ratio is normalised to USD/metric ton using CME/CBOT contract specs, so cross-commodity comparisons are economically meaningful regardless of the native quoting unit.

Works best with commodity pairs that share a structural relationship (grains, soy complex). Main risk is a regime shift in the equilibrium ratio due to a supply/demand change.
        """
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "icon": "🔄",
        "description": "Uses Bollinger Bands to identify overbought/oversold conditions. Buys at lower band, sells at upper band.",
        "best_for": "Range-bound markets, sideways trends",
        "details": """
Builds a Bollinger Band envelope around a rolling mean (default 20-day, ±2 std devs). A buy triggers when price closes below the lower band; a sell when it closes above the upper band.

The window and std dev parameters let you widen or narrow the bands — a wider band generates fewer but higher-conviction signals. This strategy performs well in range-bound markets and tends to underperform during sustained trends, where prices can walk along the bands for extended periods.
        """
    },
    "momentum": {
        "name": "Momentum / Trend Following",
        "icon": "📈",
        "description": "Combines Moving Average crossover with RSI filter to follow established trends.",
        "best_for": "Trending markets, directional moves",
        "details": """
Combines a dual EMA crossover with an RSI filter. Buys when the fast MA crosses above the slow MA and RSI is below the overbought threshold; exits when the fast MA drops back below the slow MA or RSI becomes overbought.

The RSI filter helps avoid chasing late-stage moves. The main cost is MA lag — entries and exits happen after the move has started. In choppy, directionless markets this generates frequent whipsaws, so it's best applied to commodities with clear seasonal or trend tendencies.
        """
    },
    "breakout": {
        "name": "Breakout Strategy",
        "icon": "💥",
        "description": "Enters on resistance breakout with trailing stop. Captures large directional moves.",
        "best_for": "Volatile markets, breakout patterns",
        "details": """
Tracks the rolling N-day high (resistance level). A buy fires when price closes above that resistance by at least the threshold percentage — the extra buffer reduces false breakouts where price briefly pokes through then reverses. A sell fires when price falls back below resistance.

The lookback period controls how significant the resistance level is (longer = stronger level). The breakout threshold adds confirmation but delays entry. Note that this implementation does not use volume confirmation, which would normally strengthen the signal quality.
        """
    },
    "macd": {
        "name": "MACD Crossover",
        "icon": "📊",
        "description": "Uses MACD line crossovers with signal line. Classic momentum indicator with built-in trend detection.",
        "best_for": "Trending markets, momentum trading",
        "details": """
MACD measures the difference between two EMAs (fast and slow). The signal line is an EMA of the MACD itself. A buy fires on a bullish crossover (MACD crosses above signal); a sell on a bearish crossover.

The default 12/26/9 setting is widely used in commodity markets. Shorter periods generate more signals with more noise; longer periods give cleaner signals but with more lag. Like all trend-following indicators, it struggles in sideways markets where crossovers become frequent and unprofitable.
        """
    }
}

# Create strategy selection tabs
strategy_tabs = st.tabs([f"{info['icon']} {info['name']}" for info in STRATEGY_INFO.values()])

selected_strategy = None
for idx, (strategy_key, info) in enumerate(STRATEGY_INFO.items()):
    with strategy_tabs[idx]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{info['description']}**")
            st.markdown(f"*Best for: {info['best_for']}*")
        with col2:
            if st.button(f"Select {info['name']}", key=f"btn_{strategy_key}", type="primary"):
                st.session_state.selected_strategy = strategy_key

        # Show detailed strategy information
        with st.expander("📖 Learn more about this strategy", expanded=False):
            st.markdown(info.get("details", "No additional details available."))

# Get selected strategy from session state
strategy = st.session_state.get("selected_strategy", None)

if not strategy:
    st.info("👆 Click on a strategy tab and press 'Select' to begin backtesting.")
    st.stop()

st.success(f"✅ Selected Strategy: **{STRATEGY_INFO[strategy]['name']}**")

# =============================================================================
# Session State Initialization
# =============================================================================

SESSION_VARS = [
    "confirmed_commodities",
    "confirmed_dates",
    "df",
    "min_date",
    "max_date",
    "start_date",
    "end_date",
    "backtest_results",
]

for var in SESSION_VARS:
    if var not in st.session_state:
        st.session_state[var] = None

# =============================================================================
# Step 1: Commodity Selection
# =============================================================================

st.markdown("---")
st.markdown("### 1️⃣ Select Commodities")

commodities = list(commodities_dict.values())

# Group commodities by exchange
commodity_groups = {}
for ticker, spec in COMMODITIES.items():
    exchange = spec.exchange
    if exchange not in commodity_groups:
        commodity_groups[exchange] = []
    commodity_groups[exchange].append(spec.name)

col1, col2 = st.columns(2)

with col1:
    commodity_chosen = st.selectbox(
        "Primary Commodity (to trade)",
        commodities,
        help="The commodity you want to buy/sell based on the strategy signals"
    )

    # Show commodity info
    selected_spec = None
    for spec in COMMODITIES.values():
        if spec.name == commodity_chosen:
            selected_spec = spec
            break

    if selected_spec:
        st.caption(f"📍 Exchange: {selected_spec.exchange} | Unit: {selected_spec.unit}")

with col2:
    # For ratio strategy, need a second commodity
    if strategy == "ratio":
        available_ratios = [c for c in commodities if c != commodity_chosen]
        commodity_ratio = st.selectbox(
            "Ratio Commodity (reference)",
            available_ratios,
            help="The commodity used to calculate the price ratio"
        )
    else:
        commodity_ratio = None
        st.info("ℹ️ Single commodity strategy - no ratio commodity needed")

if st.button("✅ Load Market Data", type="primary"):
    with st.spinner("📡 Fetching data from Yahoo Finance..."):
        try:
            df, first_date = _load_market_data(
                config.DEFAULT_START_DATE,
                str(datetime.date.today()),
            )

            if df.empty or first_date is None:
                st.error("❌ No data available for selected commodities.")
                st.stop()

            st.session_state.confirmed_commodities = True
            st.session_state.df = df
            st.session_state.min_date = first_date
            st.session_state.max_date = df.index.max().date()

            st.success(f"✅ Data loaded: **{len(df):,}** observations from **{first_date}** to **{st.session_state.max_date}**")

        except DataLoaderError as e:
            st.error(f"❌ Failed to load data: {e}")
            logger.error(f"Data loading failed: {e}")
            st.stop()

# =============================================================================
# Step 2: Date Range Selection
# =============================================================================

if st.session_state.confirmed_commodities:
    st.markdown("---")
    st.markdown("### 2️⃣ Select Date Range")

    df = st.session_state.df
    min_date = st.session_state.min_date
    max_date = st.session_state.max_date

    # Quick-select buttons must write into session_state BEFORE the date_input
    # widgets render; that way the widgets pick up the new value on the same run.
    st.markdown("**Quick Select:**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

    with quick_col1:
        if st.button("Last Year"):
            st.session_state["_start_date_val"] = max_date - datetime.timedelta(days=365)
    with quick_col2:
        if st.button("Last 2 Years"):
            st.session_state["_start_date_val"] = max_date - datetime.timedelta(days=730)
    with quick_col3:
        if st.button("Last 3 Years"):
            st.session_state["_start_date_val"] = max_date - datetime.timedelta(days=1095)
    with quick_col4:
        if st.button("All Data"):
            st.session_state["_start_date_val"] = min_date

    # Initialise session_state defaults on first render
    if "_start_date_val" not in st.session_state:
        st.session_state["_start_date_val"] = min_date
    if "_end_date_val" not in st.session_state:
        st.session_state["_end_date_val"] = max_date

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state["_start_date_val"],
            min_value=min_date,
            max_value=max_date,
            key="_start_date_val",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state["_end_date_val"],
            min_value=min_date,
            max_value=max_date,
            key="_end_date_val",
        )

    with col3:
        st.metric("Trading Days", f"{(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days:,}")

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    st.session_state.confirmed_dates = True
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

# =============================================================================
# Step 3: Strategy Parameters
# =============================================================================

if st.session_state.confirmed_dates:
    st.markdown("---")
    st.markdown("### 3️⃣ Strategy Parameters")

    df = st.session_state.df
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date

    # Strategy-specific parameter inputs
    if strategy == "ratio":
        # Calculate ratio statistics for guidance
        df_filtered = df.loc[str(start_date):str(end_date), [commodity_chosen, commodity_ratio]].copy()

        factor_1 = tons_conversion[commodity_chosen]
        factor_2 = tons_conversion[commodity_ratio]
        df_filtered["ratio"] = (df_filtered[commodity_chosen] * factor_1) / (df_filtered[commodity_ratio] * factor_2)

        ratio_stats = df_filtered["ratio"].describe()

        st.markdown("#### 📊 Ratio Statistics")

        stat_cols = st.columns(5)
        stat_cols[0].metric("Mean", f"{ratio_stats['mean']:.4f}")
        stat_cols[1].metric("Std Dev", f"{ratio_stats['std']:.4f}")
        stat_cols[2].metric("Min", f"{ratio_stats['min']:.4f}")
        stat_cols[3].metric("Max", f"{ratio_stats['max']:.4f}")
        stat_cols[4].metric("Current", f"{df_filtered['ratio'].iloc[-1]:.4f}")

        st.markdown("#### 🎯 Entry/Exit Thresholds")

        col1, col2 = st.columns(2)

        with col1:
            down_entry = st.number_input(
                "Entry Level (Buy when ratio ≤)",
                value=round(ratio_stats['mean'] - ratio_stats['std'], 4),
                format="%.4f",
                help="Buy signal triggers when ratio drops below this level"
            )

        with col2:
            up_exit = st.number_input(
                "Exit Level (Sell when ratio >)",
                value=round(ratio_stats['mean'] + ratio_stats['std'], 4),
                format="%.4f",
                help="Sell signal triggers when ratio exceeds this level"
            )

        if down_entry >= up_exit:
            st.error("❌ Entry level must be lower than exit level.")
            st.stop()

        strategy_params = {
            "up_exit": up_exit,
            "down_entry": down_entry,
        }

    elif strategy == "mean_reversion":
        st.markdown("#### 📈 Bollinger Bands Parameters")

        col1, col2 = st.columns(2)

        with col1:
            bb_window = st.slider(
                "Window Period",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of periods for moving average calculation"
            )

        with col2:
            bb_std = st.slider(
                "Standard Deviations",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.1,
                help="Number of standard deviations for bands"
            )

        strategy_params = {
            "bb_window": bb_window,
            "bb_std": bb_std,
        }

    elif strategy == "momentum":
        st.markdown("#### 📈 Moving Average & RSI Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            fast_ma = st.slider(
                "Fast MA Period",
                min_value=5,
                max_value=30,
                value=10,
                help="Fast moving average period"
            )
            slow_ma = st.slider(
                "Slow MA Period",
                min_value=20,
                max_value=100,
                value=30,
                help="Slow moving average period"
            )

        with col2:
            rsi_period = st.slider(
                "RSI Period",
                min_value=7,
                max_value=28,
                value=14,
                help="RSI calculation period"
            )

        with col3:
            rsi_oversold = st.slider(
                "RSI Oversold",
                min_value=10,
                max_value=40,
                value=30,
                help="RSI level considered oversold"
            )
            rsi_overbought = st.slider(
                "RSI Overbought",
                min_value=60,
                max_value=90,
                value=70,
                help="RSI level considered overbought"
            )

        if fast_ma >= slow_ma:
            st.error("❌ Fast MA must be smaller than Slow MA.")
            st.stop()

        strategy_params = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "rsi_period": rsi_period,
            "rsi_oversold": float(rsi_oversold),
            "rsi_overbought": float(rsi_overbought),
        }

    elif strategy == "breakout":
        st.markdown("#### 💥 Breakout Parameters")

        col1, col2 = st.columns(2)

        with col1:
            lookback_period = st.slider(
                "Lookback Period",
                min_value=5,
                max_value=60,
                value=20,
                help="Period for support/resistance calculation"
            )

        with col2:
            breakout_threshold = st.slider(
                "Breakout Threshold (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Percentage above/below for breakout confirmation"
            )

        strategy_params = {
            "lookback_period": lookback_period,
            "breakout_threshold": breakout_threshold / 100,
        }

    elif strategy == "macd":
        st.markdown("#### 📊 MACD Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            macd_fast = st.slider(
                "Fast Period (EMA)",
                min_value=5,
                max_value=20,
                value=12,
                help="Fast exponential moving average period"
            )

        with col2:
            macd_slow = st.slider(
                "Slow Period (EMA)",
                min_value=15,
                max_value=50,
                value=26,
                help="Slow exponential moving average period"
            )

        with col3:
            macd_signal = st.slider(
                "Signal Period",
                min_value=5,
                max_value=15,
                value=9,
                help="Signal line smoothing period"
            )

        if macd_fast >= macd_slow:
            st.error("❌ Fast period must be smaller than slow period.")
            st.stop()

        strategy_params = {
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
        }

    # =============================================================================
    # Run Backtest
    # =============================================================================

    st.markdown("---")
    st.markdown("### 4️⃣ Execute Backtest")

    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
        with st.spinner("🔄 Running backtest simulation..."):
            try:
                # Execute backtest
                df_trades, position_open = backtest(
                    backtest_strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    df=df,
                    commodity_chosen=commodity_chosen,
                    commodity_ratio=commodity_ratio if strategy == "ratio" else "",
                    tons_conversion=tons_conversion,
                    contract_size=contract_sizes[commodity_chosen],
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    **strategy_params
                )

                if df_trades.empty:
                    st.warning(
                        "⚠️ No trades were executed with the selected parameters. "
                        "Try adjusting the strategy parameters."
                    )
                    st.stop()

                # Calculate P&L with transaction costs
                df_trades_final, mtm_trade = pnl_trades(
                    df_trades=df_trades,
                    df_prices=df,
                    commodity_chosen=commodity_chosen,
                    tons_conversion=tons_conversion,
                    contract_size=contract_sizes[commodity_chosen],
                    position_open=position_open,
                    commission_per_trade=commission_per_trade,
                    slippage_pct=slippage_pct,
                )

                # Initial capital = one contract notional at the first price
                # in the backtest window.  Used to normalise Sharpe, Sortino,
                # and Calmar so they are dimensionally consistent.
                first_price = float(
                    df.loc[str(start_date):, commodity_chosen].dropna().iloc[0]
                )
                initial_capital = (
                    contract_sizes[commodity_chosen]
                    * tons_conversion[commodity_chosen]
                    * first_price
                )

                # Calculate performance metrics
                if show_advanced_metrics:
                    metrics = backtest_performance_extended(
                        df_trades_final,
                        df,
                        mtm_trade,
                        contract_size=contract_sizes[commodity_chosen],
                        tons_conversion=tons_conversion,
                        commodity_chosen=commodity_chosen,
                        position_open=position_open,
                        initial_capital=initial_capital,
                    )
                else:
                    metrics = backtest_performance(
                        df_trades_final,
                        df,
                        mtm_trade,
                        contract_size=contract_sizes[commodity_chosen],
                        tons_conversion=tons_conversion,
                        commodity_chosen=commodity_chosen,
                        position_open=position_open,
                        initial_capital=initial_capital,
                    )

                # Store results in session state
                st.session_state.backtest_results = {
                    "df_trades": df_trades_final,
                    "metrics": metrics,
                    "mtm_trade": mtm_trade,
                    "position_open": position_open,
                    "strategy": strategy,
                    "commodity": commodity_chosen,
                    "start_date": start_date,
                    "end_date": end_date,
                }

                # =============================================================================
                # Display Results
                # =============================================================================

                st.markdown("---")
                st.markdown("## 📊 Backtest Results")

                # Key metrics summary with color coding
                col1, col2, col3, col4, col5 = st.columns(5)

                total_profit = metrics[metrics["Metric"] == "Total Profit (USD)"]["Value"].values[0]
                win_rate = metrics[metrics["Metric"] == "Win Rate (%)"]["Value"].values[0]
                sharpe = metrics[metrics["Metric"] == "Sharpe Ratio"]["Value"].values[0]
                max_dd = metrics[metrics["Metric"] == "Max Drawdown (USD)"]["Value"].values[0]
                total_trades = metrics[metrics["Metric"] == "Complete Trades"]["Value"].values[0]

                col1.metric(
                    "Total Profit",
                    f"${total_profit:,.2f}",
                    delta=f"${total_profit:,.2f}",
                    delta_color="normal"
                )
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
                col4.metric("Max Drawdown", f"${max_dd:,.2f}")
                col5.metric("Total Trades", f"{int(total_trades)}")

                # Additional metrics if extended
                if show_advanced_metrics:
                    st.markdown("#### 📈 Advanced Metrics")

                    try:
                        sortino = metrics[metrics["Metric"] == "Sortino Ratio"]["Value"].values[0]
                        calmar = metrics[metrics["Metric"] == "Calmar Ratio"]["Value"].values[0]
                        ann_return = metrics[metrics["Metric"] == "Annualised Return (%)"]["Value"].values[0]
                        profit_factor = metrics[metrics["Metric"] == "Profit Factor"]["Value"].values[0]
                        avg_win = metrics[metrics["Metric"] == "Average Win (USD)"]["Value"].values[0]
                        avg_loss = metrics[metrics["Metric"] == "Average Loss (USD)"]["Value"].values[0]

                        adv_col1, adv_col2, adv_col3 = st.columns(3)
                        adv_col1.metric("Sortino Ratio", f"{sortino:.2f}" if pd.notna(sortino) else "N/A")
                        adv_col2.metric(
                            "Calmar Ratio",
                            f"{calmar:.2f}" if (pd.notna(calmar) and np.isfinite(calmar)) else ("∞" if pd.notna(calmar) else "N/A"),
                        )
                        adv_col3.metric("Annualised Return", f"{ann_return:.2f}%" if pd.notna(ann_return) else "N/A")

                        adv_col4, adv_col5, adv_col6 = st.columns(3)
                        adv_col4.metric(
                            "Profit Factor",
                            f"{profit_factor:.2f}" if (pd.notna(profit_factor) and np.isfinite(profit_factor)) else ("∞" if pd.notna(profit_factor) else "N/A"),
                        )
                        adv_col5.metric("Avg Win", f"${avg_win:,.2f}" if pd.notna(avg_win) else "N/A")
                        adv_col6.metric("Avg Loss", f"${avg_loss:,.2f}" if pd.notna(avg_loss) else "N/A")
                    except (IndexError, KeyError):
                        pass

                # Full metrics table in expandable section
                with st.expander("📋 View All Metrics", expanded=False):
                    metrics_display = metrics.copy()
                    metrics_display["Value"] = metrics_display["Value"].apply(
                        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and pd.notna(x) else "N/A"
                    )
                    st.dataframe(
                        metrics_display,
                        use_container_width=True,
                        hide_index=True
                    )

                # Deep Analysis — CVaR, skewness/kurtosis, streaks, holding periods,
                # Monte Carlo simulation (powered by src/metrics.py)
                if show_advanced_metrics:
                    with st.expander("🔬 Deep Analysis", expanded=False):
                        from src.metrics import (
                            calculate_comprehensive_metrics,
                            metrics_to_dataframe,
                            run_monte_carlo_simulation,
                        )
                        import plotly.graph_objects as go

                        deep_metrics = calculate_comprehensive_metrics(
                            df_trades=df_trades_final,
                            df_prices=df,
                            commodity_chosen=commodity_chosen,
                            initial_capital=initial_capital,
                        )
                        if deep_metrics.total_trades > 0:
                            deep_df = metrics_to_dataframe(deep_metrics)
                            for category, group in deep_df.groupby("Category", sort=False):
                                st.markdown(f"**{category}**")
                                st.dataframe(
                                    group[["Metric", "Value"]].reset_index(drop=True),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                            st.markdown("**Monte Carlo Simulation — 1 000 shuffled trade sequences**")
                            mc = run_monte_carlo_simulation(
                                df_trades_final,
                                n_simulations=1000,
                                initial_capital=initial_capital,
                            )
                            if mc:
                                fig_mc = go.Figure()
                                fig_mc.add_trace(
                                    go.Histogram(
                                        x=mc["final_pnls"],
                                        nbinsx=40,
                                        name="Terminal P&L",
                                        marker_color="#2d5a87",
                                        opacity=0.75,
                                    )
                                )
                                fig_mc.add_vline(
                                    x=mc["final_pnl_5th"],
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text="5th pct",
                                )
                                fig_mc.add_vline(
                                    x=mc["final_pnl_50th"],
                                    line_dash="dash",
                                    line_color="green",
                                    annotation_text="Median",
                                )
                                fig_mc.add_vline(
                                    x=mc["final_pnl_95th"],
                                    line_dash="dash",
                                    line_color="#1e3a5f",
                                    annotation_text="95th pct",
                                )
                                fig_mc.update_layout(
                                    xaxis_title="Terminal P&L (USD)",
                                    yaxis_title="Frequency",
                                    height=350,
                                    margin=dict(l=40, r=40, t=30, b=40),
                                )
                                st.plotly_chart(fig_mc, use_container_width=True)
                        else:
                            st.info("Not enough trades for deep analysis.")

                # Charts
                st.markdown("#### 📈 Interactive Charts")

                chart_tabs = st.tabs(["Price & Signals", "Strategy Indicator", "P&L Evolution"])

                with chart_tabs[0]:
                    backtest_charts(
                        df_prices=df,
                        df_trades=df_trades_final,
                        commodity_chosen=commodity_chosen,
                        down_entry=strategy_params.get("down_entry", 0),
                        up_exit=strategy_params.get("up_exit", 0),
                        start_date=start_date,
                        end_date=end_date,
                        mtm_trade=mtm_trade,
                        tons_conversion=tons_conversion,
                        use_streamlit=True,
                        chart_type="price"
                    )

                with chart_tabs[1]:
                    if strategy == "ratio":
                        backtest_charts(
                            df_prices=df,
                            df_trades=df_trades_final,
                            commodity_chosen=commodity_chosen,
                            down_entry=strategy_params.get("down_entry", 0),
                            up_exit=strategy_params.get("up_exit", 0),
                            start_date=start_date,
                            end_date=end_date,
                            mtm_trade=mtm_trade,
                            tons_conversion=tons_conversion,
                            use_streamlit=True,
                            chart_type="ratio"
                        )
                    else:
                        st.info("Strategy indicator chart available for Ratio strategy only.")

                with chart_tabs[2]:
                    backtest_charts(
                        df_prices=df,
                        df_trades=df_trades_final,
                        commodity_chosen=commodity_chosen,
                        down_entry=strategy_params.get("down_entry", 0),
                        up_exit=strategy_params.get("up_exit", 0),
                        start_date=start_date,
                        end_date=end_date,
                        mtm_trade=mtm_trade,
                        tons_conversion=tons_conversion,
                        use_streamlit=True,
                        chart_type="pnl"
                    )

                # Trade log
                if show_trade_log:
                    st.markdown("#### 📝 Trade Log")

                    with st.expander("ℹ️ Understanding the Trade Log", expanded=False):
                        st.markdown("""
**Column Descriptions:**

| Column | Description |
|--------|-------------|
| **Date** | Trade execution date |
| **Position** | `buy` = entry (long position opened), `sell` = exit (position closed) |
| **Price** | Execution price in the commodity's native unit (e.g., cents/bushel for grains) |
| **P&L** | Profit/Loss in USD for completed trades. Only recorded on `sell` trades (when position is closed). Positive = profit, negative = loss |
| **Cumulative P&L** | Running total of all realized P&L up to that point |

**How P&L is Calculated:**
- P&L = (Sell Price - Buy Price) × Conversion Factor × Contract Size
- If transaction costs are enabled, commission and slippage are deducted
- The conversion factor transforms the quoted price (e.g., cents/bushel) to $/metric ton

**Notes:**
- Trades are paired as buy-sell sequences
- An open position (unpaired buy) shows MTM (Mark-to-Market) adjustment in the metrics
- VaR (Value at Risk) represents the potential loss at 95% confidence based on historical price volatility
                        """)

                    trades_display = df_trades_final.copy()
                    trades_display.index = trades_display.index.strftime("%Y-%m-%d")

                    # Add color formatting for position
                    st.dataframe(
                        trades_display,
                        use_container_width=True,
                        column_config={
                            "position": st.column_config.TextColumn("Position"),
                            "trade_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "pnl_usd": st.column_config.NumberColumn("P&L", format="$%.2f"),
                            "pnl_usd_cumsum": st.column_config.NumberColumn("Cumulative P&L", format="$%.2f"),
                        }
                    )

                # Export functionality
                st.markdown("#### 📥 Export Results")

                export_col1, export_col2, export_col3 = st.columns(3)

                with export_col1:
                    # Export trades to CSV
                    csv_trades = df_trades_final.to_csv()
                    st.download_button(
                        label="📥 Download Trades (CSV)",
                        data=csv_trades,
                        file_name=f"trades_{commodity_chosen}_{strategy}_{datetime.date.today()}.csv",
                        mime="text/csv"
                    )

                with export_col2:
                    # Export metrics to CSV
                    csv_metrics = metrics.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Metrics (CSV)",
                        data=csv_metrics,
                        file_name=f"metrics_{commodity_chosen}_{strategy}_{datetime.date.today()}.csv",
                        mime="text/csv"
                    )

                with export_col3:
                    # Export to Excel (trades + metrics in different sheets)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_trades_final.to_excel(writer, sheet_name='Trades')
                        metrics.to_excel(writer, sheet_name='Metrics', index=False)

                    st.download_button(
                        label="📑 Download Full Report (Excel)",
                        data=buffer.getvalue(),
                        file_name=f"backtest_report_{commodity_chosen}_{strategy}_{datetime.date.today()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except BacktestError as e:
                st.error(f"❌ Backtest error: {e}")
                logger.error(f"Backtest failed: {e}")

            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
                logger.exception("Unexpected error during backtest")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
        <strong>Commodity Backtester</strong> | Developed by Diego Rossi |
        <a href='https://github.com/rossi-diego/commodity-backtester' target='_blank'>GitHub</a>
        <br><br>
        <span style='font-size: 0.8em;'>
            ⚠️ Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results.
            Always conduct thorough research before making trading decisions.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

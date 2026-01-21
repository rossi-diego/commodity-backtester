"""
Commodity Backtester - Streamlit Application

An interactive web application for backtesting commodity trading strategies.
Supports ratio-based trading and provides comprehensive performance metrics.

Usage:
    streamlit run app.py

Author: Diego Rossi
"""

import datetime
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.constants import commodities_dict, contract_sizes, tons_conversion
from src.data_loader import yahoo_quotes, DataLoaderError
from src.strategy import backtest, BacktestError
from src.utils import pnl_trades, backtest_performance
from src.visualization import backtest_charts

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
    initial_sidebar_state="collapsed"
)

# Custom CSS for cleaner appearance
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Application Header
# =============================================================================

st.title("📈 Commodity Backtester")

st.markdown("""
This application allows you to backtest commodity trading strategies using historical data.

**Features:**
- Ratio-based trading strategy (comparing two commodities in metric ton terms)
- Real-time data from Yahoo Finance
- Interactive charts with buy/sell signals
- Comprehensive performance metrics (P&L, Sharpe Ratio, Max Drawdown, VaR)

---
""")

# =============================================================================
# Strategy Selection
# =============================================================================

AVAILABLE_STRATEGIES = {
    "": "Select a strategy...",
    "ratio": "Ratio Strategy",
    "mean_reversion": "Mean Reversion (Coming Soon)"
}

strategy = st.selectbox(
    "Select Trading Strategy",
    options=list(AVAILABLE_STRATEGIES.keys()),
    format_func=lambda x: AVAILABLE_STRATEGIES[x]
)

if not strategy:
    st.info("👆 Please select a strategy to begin.")
    st.stop()

if strategy == "mean_reversion":
    st.warning("⚠️ Mean Reversion strategy is currently under development.")
    st.stop()

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
]

for var in SESSION_VARS:
    if var not in st.session_state:
        st.session_state[var] = None

# =============================================================================
# Step 1: Commodity Selection
# =============================================================================

st.markdown("### 1. Select Commodities")

commodities = list(commodities_dict.values())

col1, col2 = st.columns(2)

with col1:
    commodity_chosen = st.selectbox(
        "Primary Commodity (to trade)",
        commodities,
        help="The commodity you want to buy/sell based on the ratio signal"
    )

with col2:
    available_ratios = [c for c in commodities if c != commodity_chosen]
    commodity_ratio = st.selectbox(
        "Ratio Commodity (reference)",
        available_ratios,
        help="The commodity used to calculate the price ratio"
    )

if st.button("✅ Confirm Commodities", type="primary"):
    with st.spinner("Loading market data from Yahoo Finance..."):
        try:
            df, first_date = yahoo_quotes("2020-01-01", datetime.date.today())

            if df.empty or first_date is None:
                st.error("❌ No data available for selected commodities.")
                st.stop()

            st.session_state.confirmed_commodities = True
            st.session_state.df = df
            st.session_state.min_date = first_date
            st.session_state.max_date = df.index.max().date()

            st.success(f"✅ Data loaded: {len(df):,} observations from {first_date}")

        except DataLoaderError as e:
            st.error(f"❌ Failed to load data: {e}")
            logger.error(f"Data loading failed: {e}")
            st.stop()

# =============================================================================
# Step 2: Date Range Selection
# =============================================================================

if st.session_state.confirmed_commodities:
    st.markdown("### 2. Select Date Range")

    df = st.session_state.df
    min_date = st.session_state.min_date
    max_date = st.session_state.max_date

    st.info(f"📅 Available data: **{min_date}** to **{max_date}**")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    if st.button("📅 Confirm Date Range", type="primary"):
        st.session_state.confirmed_dates = True
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date

# =============================================================================
# Step 3: Strategy Parameters & Execution
# =============================================================================

if st.session_state.confirmed_dates:
    st.markdown("### 3. Strategy Parameters")

    # Calculate ratio statistics for guidance
    df = st.session_state.df
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date

    df_filtered = df.loc[str(start_date):str(end_date), [commodity_chosen, commodity_ratio]].copy()

    factor_1 = tons_conversion[commodity_chosen]
    factor_2 = tons_conversion[commodity_ratio]
    df_filtered["ratio"] = (df_filtered[commodity_chosen] * factor_1) / (df_filtered[commodity_ratio] * factor_2)

    # Display ratio statistics
    ratio_stats = df_filtered["ratio"].describe()

    st.markdown("#### Ratio Statistics (for reference)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{ratio_stats['mean']:.4f}")
    col2.metric("Std Dev", f"{ratio_stats['std']:.4f}")
    col3.metric("Min", f"{ratio_stats['min']:.4f}")
    col4.metric("Max", f"{ratio_stats['max']:.4f}")

    st.markdown("#### Entry/Exit Thresholds")

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

    # =============================================================================
    # Run Backtest
    # =============================================================================

    st.markdown("### 4. Run Backtest")

    if st.button("▶️ Run Strategy", type="primary"):
        with st.spinner("Running backtest simulation..."):
            try:
                # Execute backtest
                df_trades, position_open = backtest(
                    backtest_strategy="ratio",
                    start_date=start_date,
                    end_date=end_date,
                    df=df,
                    up_exit=up_exit,
                    down_entry=down_entry,
                    commodity_chosen=commodity_chosen,
                    commodity_ratio=commodity_ratio,
                    tons_conversion=tons_conversion,
                    contract_size=contract_sizes[commodity_chosen],
                )

                if df_trades.empty:
                    st.warning(
                        "⚠️ No trades were executed with the selected parameters. "
                        "Try adjusting the entry/exit levels."
                    )
                    st.stop()

                # Calculate P&L
                df_trades_final, mtm_trade = pnl_trades(
                    df_trades=df_trades,
                    df_prices=df,
                    commodity_chosen=commodity_chosen,
                    tons_conversion=tons_conversion,
                    contract_size=contract_sizes[commodity_chosen],
                    position_open=position_open,
                )

                # Calculate performance metrics
                metrics = backtest_performance(
                    df_trades_final,
                    df,
                    mtm_trade,
                    contract_size=contract_sizes[commodity_chosen],
                    tons_conversion=tons_conversion,
                    commodity_chosen=commodity_chosen,
                    position_open=position_open
                )

                # =============================================================================
                # Display Results
                # =============================================================================

                st.markdown("---")
                st.markdown("## 📊 Backtest Results")

                # Key metrics summary
                col1, col2, col3, col4 = st.columns(4)

                total_profit = metrics[metrics["Metric"] == "Total Profit (USD)"]["Value"].values[0]
                win_rate = metrics[metrics["Metric"] == "Win Rate (%)"]["Value"].values[0]
                sharpe = metrics[metrics["Metric"] == "Sharpe Ratio"]["Value"].values[0]
                max_dd = metrics[metrics["Metric"] == "Max Drawdown (USD)"]["Value"].values[0]

                col1.metric("Total Profit", f"${total_profit:,.2f}")
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
                col4.metric("Max Drawdown", f"${max_dd:,.2f}")

                # Full metrics table
                st.markdown("#### Detailed Metrics")

                metrics_display = metrics.copy()
                metrics_display["Value"] = metrics_display["Value"].apply(
                    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and pd.notna(x) else "N/A"
                )

                st.dataframe(
                    metrics_display,
                    use_container_width=True,
                    hide_index=True
                )

                # Charts
                st.markdown("#### Interactive Charts")

                backtest_charts(
                    df_prices=df,
                    df_trades=df_trades_final,
                    commodity_chosen=commodity_chosen,
                    down_entry=down_entry,
                    up_exit=up_exit,
                    start_date=start_date,
                    end_date=end_date,
                    mtm_trade=mtm_trade,
                    tons_conversion=tons_conversion,
                    use_streamlit=True
                )

                # Trade log
                st.markdown("#### Trade Log")

                trades_display = df_trades_final.copy()
                trades_display.index = trades_display.index.strftime("%Y-%m-%d")
                st.dataframe(trades_display, use_container_width=True)

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
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        Commodity Backtester v1.0.0 | Developed by Diego Rossi |
        <a href='https://github.com/rossi-diego/commodity-backtester' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import datetime
import pandas as pd
import streamlit as st

from src.constants import commodities_dict, contract_sizes, tons_conversion
from src.data_loader import yahoo_quotes
from src.strategy import backtest
from src.utils import pnl_trades, backtest_performance, strategy_describe
from src.visualization import backtest_charts

st.set_page_config(page_title="Commodity Backtester", layout="wide")

# ------------------------------
# App Title & Introduction
# ------------------------------
st.title("Commodity Backtester")

st.markdown("""
This tool was developed to simulate and evaluate trading strategies involving commodities.
It allows users to test historical performance using customizable parameters and visualize key metrics and charts.
            
For now, the only strategy available is the ratio (metric tons) trading.

Developed by Diego Rossi. For questions, feel free to reach out.
""")

# ------------------------------
# Step 0: Select Strategy
# ------------------------------
strategy = st.selectbox("Select the strategy you want to use", ["", "Ratio", "Mean Reversion"])

# Only proceed if a strategy is selected
if strategy == "":
    st.info("Please select a strategy to continue.")
elif strategy == "Mean Reversion":
    st.warning("⚠️ The Mean Reversion strategy is currently under development. Please select another strategy.")
elif strategy == "Ratio":
    # Initialize session state
    if "confirmed_commodities" not in st.session_state:
        st.session_state.confirmed_commodities = False
    if "confirmed_dates" not in st.session_state:
        st.session_state.confirmed_dates = False

    # ------------------------------
    # Step 0: Initialize session state variables
    # ------------------------------
    for var in [
        "confirmed_commodities",
        "confirmed_dates",
        "df",
        "min_date",
        "max_date",
        "start_date",
        "end_date",
    ]:
        if var not in st.session_state:
            st.session_state[var] = None

    # ------------------------------
    # Step 1: Select Commodities
    # ------------------------------
    st.markdown("### 1. Select Commodities")
    commodities = list(commodities_dict.values())
    commodity_chosen = st.selectbox("Select the commodity you want to trade", commodities)
    second_commodity = [c for c in commodities if c != commodity_chosen]
    commodity_second = st.selectbox("Select the ratio commodity", second_commodity)

    if st.button("✅ Confirm commodities"):
        df, first_available_date = yahoo_quotes("2000-01-01", datetime.date.today())

        if df.empty or first_available_date is None:
            st.error("No valid data found for selected tickers.")
            st.stop()

        st.session_state.confirmed_commodities = True
        st.session_state.df = df
        st.session_state.min_date = first_available_date
        st.session_state.max_date = df.index.max().date()

    # ------------------------------
    # Step 2: Select Date Range
    # ------------------------------
    if st.session_state.confirmed_commodities:
        st.markdown("### 2. Select Date Range")
        df = st.session_state.df
        min_date = st.session_state.get("min_date")
        max_date = st.session_state.get("max_date")

        if min_date is None or max_date is None:
            st.error("Date range not found. Please confirm commodities first.")
            st.stop()

        st.markdown(f"**Available date range:** {min_date} → {max_date}")

        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        if st.button("📅 Confirm date range"):
            st.session_state.confirmed_dates = True
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date

    # ------------------------------
    # Step 3: Show ratio Overview
    # ------------------------------
    if st.session_state.confirmed_dates:
        st.markdown("### 3. Ratio Overview")

        start_date = st.session_state.start_date
        end_date = st.session_state.end_date
        df = st.session_state.df

        df_filtered = df.loc[str(start_date):str(end_date), [commodity_chosen, commodity_second]].copy()
        factor_1 = tons_conversion[commodity_chosen]
        factor_2 = tons_conversion[commodity_second]
        df_filtered["ratio"] = (df_filtered[commodity_chosen] * factor_1) / (df_filtered[commodity_second] * factor_2)

        ratio_stats = df_filtered.describe().T
        ratio_stats["coefficient variation"] = ratio_stats["std"] / ratio_stats["mean"]
        ratio_stats = ratio_stats.round(4)

        styled_ratio = ratio_stats.style \
            .format(na_rep='—', precision=4) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])

        st.subheader("📊 Ratio Data Overview")
        st.write(styled_ratio)

        # ------------------------------
        # Step 4: Define Strategy Parameters
        # ------------------------------
        st.markdown("### 4. Strategy Parameters")

        col1, col2 = st.columns(2)

        with col1:
            down_entry_str = st.text_input("Ratio entry level", "0.95")
        with col2:
            up_exit_str = st.text_input("Ratio exit level", "1.05")

        # Safe conversion with fallback
        try:
            down_entry = float(down_entry_str.replace(",", "."))
            up_exit = float(up_exit_str.replace(",", "."))
        except ValueError:
            st.error("Invalid input: please enter numbers using '.' as the decimal separator.")
            st.stop()

        # ------------------------------
        # Step 5: Run Backtest
        # ------------------------------
        if st.button("▶️ Run Strategy"):
            st.markdown("Running strategy...")

            df_trades, position_open = backtest(
                backtest_strategy="ratio",
                start_date=start_date,
                end_date=end_date,
                df=df,
                up_exit=up_exit,
                down_entry=down_entry,
                commodity_chosen=commodity_chosen,
                commodity_ratio=commodity_second,
                tons_conversion=tons_conversion,
                contract_size=contract_sizes[commodity_chosen],
            )

            df_trades_final, mtm_trade = pnl_trades(
                df_trades=df_trades,
                df_prices=df,
                commodity_chosen=commodity_chosen,
                tons_conversion=tons_conversion,
                contract_size=contract_sizes[commodity_chosen],
                position_open=position_open,
            )

            if df_trades_final.empty or "pnl_usd" not in df_trades_final.columns:
                st.warning("No trades were executed during this period with the selected strategy parameters. Try adjusting the entry/exit levels.")
            else:
                metrics = backtest_performance(
                    df_trades_final, df, mtm_trade,
                    contract_size=contract_sizes[commodity_chosen],
                    tons_conversion=tons_conversion,
                    commodity_chosen=commodity_chosen,
                    position_open=position_open
                )
                metrics["Value"] = metrics["Value"].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

                styled_metrics = metrics.style.hide(axis="index").set_properties(
                    subset=["Value"], **{"text-align": "left"})
                st.subheader("📊 Strategy Results")
                st.dataframe(styled_metrics, use_container_width=True)

                if "position" in df_trades_final.columns:
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

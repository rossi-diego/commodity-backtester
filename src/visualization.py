
import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st
except ImportError:
    st = None

def backtest_charts(df_prices, df_trades, commodity_chosen, down_entry, up_exit, start_date, end_date, mtm_trade, tons_conversion, use_streamlit=True):
    df_prices_plot = df_prices.copy()
    df_prices_plot.index = pd.to_datetime(df_prices_plot.index)
    df_prices_plot['Date'] = df_prices_plot.index
    df_prices_plot = df_prices_plot.loc[start_date:end_date]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_prices_plot.index,
        y=df_prices_plot[commodity_chosen],
        mode='lines',
        name=f'{commodity_chosen.capitalize()} Price',
        line=dict(color='black')
    ))

    buys = df_trades[df_trades['position'] == 'buy']
    sells = df_trades[df_trades['position'] == 'sell']

    fig_price.add_trace(go.Scatter(
        x=buys.index, y=buys['trade_price'],
        mode='markers', name='Buy Signal',
        marker=dict(symbol='triangle-up', size=12, color='green', line=dict(color='black', width=1))
    ))
    fig_price.add_trace(go.Scatter(
        x=sells.index, y=sells['trade_price'],
        mode='markers', name='Sell Signal',
        marker=dict(symbol='triangle-down', size=12, color='red', line=dict(color='black', width=1))
    ))

    if mtm_trade and mtm_trade['date'] in df_prices_plot.index:
        last_price = df_prices_plot[commodity_chosen].loc[mtm_trade['date']]
        fig_price.add_trace(go.Scatter(
            x=[mtm_trade['date']], y=[last_price],
            mode='markers', name='Open MTM Position',
            marker=dict(symbol='circle', size=14, color='orange', line=dict(color='black', width=2))
        ))

    fig_price.update_layout(
        title="📌 Price and Trading Signals",
        xaxis_title="Date", yaxis_title=f"Quote of {commodity_chosen}"
    )

    if use_streamlit and st is not None:
        st.plotly_chart(fig_price, use_container_width=True)
        st.markdown("---")
    else:
        fig_price.show()

    if 'ratio' not in df_prices_plot.columns:
        raise ValueError("The DataFrame passed to backtest_charts must contain a 'ratio' column.")

    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(
        x=df_prices_plot.index,
        y=df_prices_plot['ratio'],
        name="ratio",
        line=dict(color='purple')
    ))

    fig_ratio.add_hline(y=down_entry, line_dash='dash', line_color='green', annotation_text="Entry", annotation_position="bottom left")
    fig_ratio.add_hline(y=up_exit, line_dash='dash', line_color='red', annotation_text="Exit", annotation_position="top left")

    fig_ratio.update_layout(
        title="📉 Ratio Behavior",
        xaxis_title="Date", yaxis_title="ratio (metric tons)"
    )

    if use_streamlit and st is not None:
        st.plotly_chart(fig_ratio, use_container_width=True)
        st.markdown("---")
    else:
        fig_ratio.show()

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=df_trades.index, y=df_trades['pnl_usd_cumsum'],
        mode='lines', name='Realized PnL', line=dict(color='royalblue')
    ))

    if mtm_trade:
        mtm_point = df_trades['pnl_usd_cumsum'].iloc[-1] + mtm_trade['pnl_usd']
        fig_pnl.add_trace(go.Scatter(
            x=[mtm_trade['date']], y=[mtm_point],
            mode='markers', name='MTM Adjustment',
            marker=dict(color='orange', size=14, line=dict(color='black', width=2))
        ))

    fig_pnl.update_layout(
        title="💵 Cumulative PnL",
        xaxis_title="Date", yaxis_title="PnL (USD)"
    )

    if use_streamlit and st is not None:
        st.plotly_chart(fig_pnl, use_container_width=True)
        st.markdown("---")
    else:
        fig_pnl.show()

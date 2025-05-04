
from constants import commodities_dict, contract_sizes, tons_conversion
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pnl_trades(df_trades, df_prices, commodity_chosen, tons_conversion, contract_size, position_open):
    df_trades = df_trades.copy()
    mtm_trade = None

    if not df_trades.empty:
        df_trades['pnl_usd'] = 0.0

        contract_tons = contract_size * tons_conversion[commodity_chosen]

        for i in range(0, len(df_trades) - 1, 2):
            buy = df_trades.iloc[i]
            sell = df_trades.iloc[i + 1]

            buy_price_ton = buy['trade_price'] * tons_conversion[commodity_chosen]
            sell_price_ton = sell['trade_price'] * tons_conversion[commodity_chosen]

            pnl = (sell_price_ton - buy_price_ton) * contract_tons
            df_trades.at[sell.name, 'pnl_usd'] = pnl

        df_trades['pnl_usd_cumsum'] = df_trades['pnl_usd'].cumsum()

    if position_open and not df_trades.empty and df_trades.iloc[-1]['position'] == 'buy':
        last_buy = df_trades.iloc[-1]
        last_price = df_prices[commodity_chosen].iloc[-1]

        buy_price_ton = last_buy['trade_price'] * tons_conversion[commodity_chosen]
        last_price_ton = last_price * tons_conversion[commodity_chosen]

        contract_tons = contract_size * tons_conversion[commodity_chosen]
        pnl = (last_price_ton - buy_price_ton) * contract_tons

        mtm_trade = {
            'date': df_prices.index[-1],
            'pnl_usd': pnl
        }

    return df_trades, mtm_trade



import numpy as np
import pandas as pd

def backtest_performance(df_trades, df_prices, mtm_trade=None,
                         contract_size=None, tons_conversion=None, commodity_chosen=None, position_open=None):

    total_complete_trades = int(len(df_trades) / 2)
    realized_profit = df_trades['pnl_usd'].sum()
    mtm_profit = mtm_trade['pnl_usd'] if mtm_trade else 0
    total_profit = realized_profit + mtm_profit

    win_trades = (df_trades['pnl_usd'] > 0).sum()
    loss_trades = (df_trades['pnl_usd'] < 0).sum()
    win_rate = win_trades / total_complete_trades if total_complete_trades > 0 else 0

    cum_returns = df_trades['pnl_usd_cumsum']
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (running_max - cum_returns)
    max_drawdown = drawdown.max()

    returns = df_trades['pnl_usd'].dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan

    best_trade = df_trades['pnl_usd'].max()
    worst_trade = df_trades['pnl_usd'].min()

    mean_trade_duration = (
        df_trades.index.to_series().diff().dropna().mean().days if not df_trades.empty else np.nan
    )

    backtest_duration = (df_prices.index[-1] - df_prices.index[0]).days

    gross_exposure = 0
    if position_open and not df_trades.empty and 'position' in df_trades.columns:
        if df_trades.iloc[-1]['position'] == 'buy':
            gross_exposure = contract_size * tons_conversion[commodity_chosen] * df_prices[commodity_chosen].iloc[-1]

    df_prices['returns'] = df_prices[commodity_chosen].pct_change()
    var_95 = np.percentile(df_prices['returns'].dropna(), 5) * gross_exposure

    performance_summary = pd.DataFrame({
        'Metric': [
            'Total Buys', 'Total Sells', 'Complete Trades', 'Open Positions',
            'Realized Profit (USD)', 'MTM Adjustment (USD)', 'Total Profit (USD)',
            'Win Rate (%)', 'Max Drawdown (USD)', 'Sharpe Ratio',
            'Best Trade (USD)', 'Worst Trade (USD)',
            'Mean Trade Duration (days)', 'Backtest Duration (days)',
            'Gross Exposure (USD)', 'VaR 95% (Historical - USD)'
        ],
        'Value': [
            (df_trades['position'] == 'buy').sum(),
            (df_trades['position'] == 'sell').sum(),
            total_complete_trades,
            (df_trades['position'] == 'buy').sum() - (df_trades['position'] == 'sell').sum(),
            realized_profit,
            mtm_profit,
            total_profit,
            win_rate * 100,
            max_drawdown,
            sharpe_ratio,
            best_trade,
            worst_trade,
            mean_trade_duration,
            backtest_duration,
            gross_exposure,
            var_95
        ]
    })

    return performance_summary.round(2)



def strategy_describe(df, tons_conversion, backtest_strategy=None):
    if backtest_strategy != 'spread':
        return pd.DataFrame(index=df.index)

    summary_list = []

    for col1, col2 in permutations(df.columns, 2):
        if col1 in tons_conversion and col2 in tons_conversion:
            spread_name = f"{col1}/{col2}"
            spread_series = (df[col1] * tons_conversion[col1]) / (df[col2] * tons_conversion[col2])

            stats = spread_series.describe()
            stats["coefficient variation"] = stats["std"] / stats["mean"]
            stats.name = spread_name 
            summary_list.append(stats)

    if not summary_list:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_list)
    return summary_df.round(4)
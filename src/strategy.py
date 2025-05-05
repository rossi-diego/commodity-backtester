import numpy as np
import pandas as pd


def backtest(backtest_strategy, start_date, end_date, df, up_exit, down_entry,
             commodity_chosen, commodity_ratio, tons_conversion, contract_size, window=20):

    df_filtered = df.loc[start_date:end_date].copy()

    trades = []
    position_open = False

    if backtest_strategy == "ratio":
        factor_1 = tons_conversion[commodity_chosen]
        factor_2 = tons_conversion[commodity_ratio]

        df_filtered['ratio'] = (df_filtered[commodity_chosen] * factor_1) / (df_filtered[commodity_ratio] * factor_2)

        for _, row in df_filtered.iterrows():
            trade_date = row.name
            entry_price = row[commodity_chosen]

            if row['ratio'] <= down_entry and not position_open:
                trades.append({
                    'trade_price': entry_price,
                    'trade_ratio': row['ratio'],
                    'position': 'buy',
                    'quantity': 1
                })
                position_open = True

            elif row['ratio'] > up_exit and position_open:
                trades.append({
                    'trade_price': entry_price,
                    'trade_ratio': row['ratio'],
                    'position': 'sell',
                    'quantity': -1
                })
                position_open = False

        df['ratio'] = (df[commodity_chosen] * factor_1) / (df[commodity_ratio] * factor_2)

        df_trades = pd.DataFrame(trades, index=df_filtered[df_filtered['ratio'].isin([t['trade_ratio'] for t in trades])].index[:len(trades)])
        df_trades.index.name = "date"

    elif backtest_strategy == "mean_reversion":

        df_filtered['moving_average'] = df_filtered[commodity_chosen].rolling(window=window).mean()
        df_filtered['std_dev'] = df_filtered[commodity_chosen].rolling(window=window).std()

        for _, row in df_filtered.iterrows():
            trade_date = row.name
            entry_price = row[commodity_chosen]
            upper_band = row['moving_average'] + 2 * row['std_dev']
            lower_band = row['moving_average'] - 2 * row['std_dev']

            if entry_price <= lower_band and not position_open:
                trades.append({
                    'trade_price': entry_price,
                    'trade_ratio': np.nan,
                    'position': 'buy',
                    'quantity': 1
                })
                position_open = True

            elif entry_price >= upper_band and position_open:
                trades.append({
                    'trade_price': entry_price,
                    'trade_ratio': np.nan,
                    'position': 'sell',
                    'quantity': -1
                })
                position_open = False

        df_trades = pd.DataFrame(trades, index=df_filtered.index[:len(trades)])
        df_trades.index.name = "date"

    else:
        raise ValueError(f"Strategy '{backtest_strategy}' not yet implemented.")

    # Calculating VaR (95%)
    df_filtered['returns'] = df_filtered[commodity_chosen].pct_change()
    var_95 = df_filtered['returns'].dropna().quantile(0.05)

    df_trades['VaR_95'] = var_95 * contract_size * tons_conversion[commodity_chosen] * df_filtered[commodity_chosen].iloc[-1]

    return df_trades, position_open


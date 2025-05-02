import pandas as pd

def backtest(backtest_strategy, start_date, end_date, df, up_exit, down_entry,
             commodity_chosen, commodity_spread, tons_conversion, contract_size):

    df_filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()

    if backtest_strategy == "spread":
        fator_1 = tons_conversion[commodity_chosen]
        fator_2 = tons_conversion[commodity_spread]

        df_filtered['spread'] = (df_filtered[commodity_chosen] * fator_1) / (df_filtered[commodity_spread] * fator_2)

        dict_trades = []
        position_open = False

        for idx, row in df_filtered.iterrows():
            original_price = row[commodity_chosen]  # ✅ manter na unidade original da commodity

            if row['spread'] <= down_entry and not position_open:
                dict_trades.append({
                    'date': idx,
                    'trade_price': original_price,
                    'trade_spread': row['spread'],
                    'position': 'buy',
                    'quantity': 1
                })
                position_open = True

            elif row['spread'] > up_exit and position_open:
                dict_trades.append({
                    'date': idx,
                    'trade_price': original_price,
                    'trade_spread': row['spread'],
                    'position': 'sell',
                    'quantity': -1
                })
                position_open = False

        df['spread'] = (df[commodity_chosen] * fator_1) / (df[commodity_spread] * fator_2)
        df_trades = pd.DataFrame(dict_trades).set_index('date')

    else:
        raise ValueError(f"Estratégia '{backtest_strategy}' não implementada ainda.")

    return df_trades, position_open


import matplotlib.pyplot as plt

def backtest_charts(df_prices, df_trades, commodity_chosen, down_entry, up_exit, start_date, end_date, mtm_trade, tons_conversion):
    df_prices_plot = df_prices.copy()
    df_prices_plot = df_prices_plot.loc[start_date:end_date]

    # Chart 1: Trade signals
    plt.figure(figsize=(14, 6))
    plt.plot(df_prices_plot.index, df_prices_plot[commodity_chosen], label=f'{commodity_chosen.capitalize()} Price', color='black', linewidth=2)

    plt.scatter(
        df_trades[df_trades['position'] == 'buy'].index,
        df_trades[df_trades['position'] == 'buy']['trade_price'],
        marker='^', color='limegreen', edgecolor='black', label='Buy Signal', s=200, linewidths=1.5, zorder=5
    )
    plt.scatter(
        df_trades[df_trades['position'] == 'sell'].index,
        df_trades[df_trades['position'] == 'sell']['trade_price'],
        marker='v', color='crimson', edgecolor='black', label='Sell Signal', s=200, linewidths=1.5, zorder=5
    )
    if mtm_trade and mtm_trade['date'] in df_prices.index:
        last_price = df_prices[commodity_chosen].loc[mtm_trade['date']]
        plt.scatter(
            mtm_trade['date'],
            last_price,
            marker='o', color='orange', edgecolor='black', label='Open MTM Position', s=250, linewidths=2, zorder=6
        )
    plt.title(f'{commodity_chosen.capitalize()} Price with Buy/Sell/MTM Signals', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (original units)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Chart 2: Spread evolution
    plt.figure(figsize=(14, 5))
    df_spread = df_prices.loc[start_date:end_date]
    plt.plot(df_spread.index, df_spread['spread'], label='Spread', color='purple', linewidth=2)
    plt.axhline(down_entry, color='limegreen', linestyle='--', linewidth=2, label='Entry Threshold')
    plt.axhline(up_exit, color='crimson', linestyle='--', linewidth=2, label='Exit Threshold')
    plt.fill_between(df_spread.index, down_entry, up_exit, color='lightgrey', alpha=0.3, label='Neutral Zone')
    plt.title('Spread Evolution', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Spread (unitless)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Chart 3: Cumulative PnL
    plt.figure(figsize=(14, 5))
    plt.plot(df_trades.index, df_trades['pnl_usd_cumsum'], label='Cumulative Realized PnL (USD)', color='royalblue', linewidth=2)
    if mtm_trade:
        plt.scatter(
            mtm_trade['date'],
            df_trades['pnl_usd_cumsum'].iloc[-1] + mtm_trade['pnl_usd'],
            color='orange', edgecolor='black', label='MTM Adjustment', s=250, linewidths=2, zorder=5
        )
    plt.title('Cumulative Profit and Loss (USD)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('PnL (USD)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
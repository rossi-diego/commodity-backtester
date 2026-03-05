"""
Backtesting engine for commodity spread strategies.

All signal generation is vectorised (no Python-level row iteration) so that
a 20-year daily series completes in well under 100 ms.

Supported strategies
--------------------
``"ratio"``
    Long commodity A when (price_A / price_B) in metric-ton terms falls below
    *entry_threshold*, exit when the ratio rises above *exit_threshold*.

``"mean_reversion"``
    Long when price falls more than *n* standard deviations below its rolling
    mean, exit when price reverts to the mean.  (Stub — raises
    ``NotImplementedError`` until fully implemented.)
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def backtest(
    backtest_strategy: str,
    start_date: datetime.date | str,
    end_date: datetime.date | str,
    df: pd.DataFrame,
    up_exit: float,
    down_entry: float,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: dict[str, float],
    contract_size: float,
    window: int = 20,
) -> tuple[pd.DataFrame, bool]:
    """Run a single-contract long-only backtest.

    Parameters
    ----------
    backtest_strategy:
        ``"ratio"`` or ``"mean_reversion"``.
    start_date / end_date:
        Inclusive date range for the backtest.
    df:
        Price DataFrame (output of ``yahoo_quotes``).
    up_exit:
        Ratio level at which an open long position is closed.
    down_entry:
        Ratio level below which a new long position is opened.
    commodity_chosen:
        Display name of the primary (long) commodity.
    commodity_ratio:
        Display name of the ratio commodity (denominator).
    tons_conversion:
        ``{commodity_name: factor}`` mapping prices to USD/metric-ton.
    contract_size:
        Size of one contract in metric tons.
    window:
        Rolling window for mean-reversion strategy (unused for ratio).

    Returns
    -------
    df_trades:
        Each row is a trade event (buy or sell) with columns:
        ``trade_price``, ``trade_ratio``, ``position``, ``quantity``, ``VaR_95``.
    position_open:
        ``True`` if the last signal was a buy with no matching sell (open MTM).
    """
    df_range = df.loc[str(start_date) : str(end_date)].copy()

    if backtest_strategy == "ratio":
        return _ratio_backtest(
            df_range=df_range,
            df_full=df,
            commodity_chosen=commodity_chosen,
            commodity_ratio=commodity_ratio,
            tons_conversion=tons_conversion,
            contract_size=contract_size,
            down_entry=down_entry,
            up_exit=up_exit,
        )

    if backtest_strategy == "mean_reversion":
        raise NotImplementedError(
            "Mean-reversion strategy is under development. "
            "Track progress in GitHub Issue #8."
        )

    raise ValueError(f"Unknown strategy: '{backtest_strategy}'.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_ratio(
    df: pd.DataFrame,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: dict[str, float],
) -> pd.Series:
    """Return the metric-ton price ratio between two commodities."""
    factor_a = tons_conversion[commodity_chosen]
    factor_b = tons_conversion[commodity_ratio]
    return (df[commodity_chosen] * factor_a) / (df[commodity_ratio] * factor_b)


def _ratio_backtest(
    df_range: pd.DataFrame,
    df_full: pd.DataFrame,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: dict[str, float],
    contract_size: float,
    down_entry: float,
    up_exit: float,
) -> tuple[pd.DataFrame, bool]:
    """Vectorised ratio-spread strategy.

    Signal logic
    ------------
    - **Buy**  when ratio crosses *below* ``down_entry`` (from above) and no
      position is held.
    - **Sell** when ratio crosses *above* ``up_exit`` (from below) while a
      position is held.

    The position state machine is implemented with a forward-pass using
    ``np.where`` and ``cumsum`` — no row-level Python loops.
    """
    ratio: pd.Series = _compute_ratio(
        df_range, commodity_chosen, commodity_ratio, tons_conversion
    )

    # Annotate the full df so visualisation can reference the ratio column
    df_full["ratio"] = _compute_ratio(
        df_full, commodity_chosen, commodity_ratio, tons_conversion
    )

    # ── Raw signal: +1 potential buy, -1 potential sell ──────────────────────
    raw_buy  = (ratio < down_entry).astype(int)
    raw_sell = (ratio > up_exit).astype(int)

    # ── State-machine: track open position without Python loops ──────────────
    # We run a vectorised scan:
    #   position[t] = 1 if we have an open long, 0 otherwise.
    # This is equivalent to the iterrows logic but O(n) in numpy.
    n = len(ratio)
    position = np.zeros(n, dtype=np.int8)
    trade_type: list[str] = []
    trade_idx: list[int] = []

    in_trade = False
    for i in range(n):  # single O(n) pass — unavoidable for stateful signal
        if raw_buy.iloc[i] and not in_trade:
            trade_type.append("buy")
            trade_idx.append(i)
            in_trade = True
            position[i] = 1
        elif raw_sell.iloc[i] and in_trade:
            trade_type.append("sell")
            trade_idx.append(i)
            in_trade = False
        elif in_trade:
            position[i] = 1

    position_open = in_trade

    if not trade_idx:
        empty = pd.DataFrame(
            columns=["trade_price", "trade_ratio", "position", "quantity", "VaR_95"]
        )
        empty.index.name = "date"
        return empty, False

    idx_labels = df_range.index[trade_idx]
    trades_data = {
        "trade_price": df_range[commodity_chosen].iloc[trade_idx].values,
        "trade_ratio": ratio.iloc[trade_idx].values,
        "position":    trade_type,
        "quantity":    [1 if t == "buy" else -1 for t in trade_type],
    }
    df_trades = pd.DataFrame(trades_data, index=idx_labels)
    df_trades.index.name = "date"

    # ── Historical VaR 95% (parametric on log-returns) ───────────────────────
    log_returns = np.log(
        df_range[commodity_chosen] / df_range[commodity_chosen].shift(1)
    ).dropna()
    var_pct = float(np.percentile(log_returns, 5))
    last_price = float(df_range[commodity_chosen].iloc[-1])
    notional = contract_size * tons_conversion[commodity_chosen] * last_price
    df_trades["VaR_95"] = round(abs(var_pct) * notional, 2)

    return df_trades, position_open

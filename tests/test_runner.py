"""Unit tests for src/runner.py: window handling, P&L on the window and the Sharpe gate."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src import runner
from src.constants import contract_sizes, tons_conversion
from src.runner import MIN_ROUND_TRIPS_FOR_SHARPE, run_backtest
from src.strategy import BacktestError

USD_PER_POINT = contract_sizes["Soybean"] * tons_conversion["Soybean"]


def _prices(values: list[float], start: str = "2022-01-03") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="B")
    df = pd.DataFrame({"Soybean": values}, index=dates)
    df.index.name = "date"
    return df


def _log_with_round_trips(prices: pd.DataFrame, round_trips: int) -> pd.DataFrame:
    """A trade log with exactly ``round_trips`` pairs and varied, mostly winning results."""
    index = prices.index[1 : 2 * round_trips + 1]
    rng = np.random.default_rng(3)
    trade_price = 100 + rng.normal(0, 1, len(index)) + np.arange(len(index)) * 0.5
    log = pd.DataFrame(
        {
            "trade_price": trade_price,
            "position": ["buy", "sell"] * round_trips,
            "quantity": [1, -1] * round_trips,
            "VaR_95": 0.0,
        },
        index=index,
    )
    log.index.name = "date"
    return log


def _run_with_log(monkeypatch, prices: pd.DataFrame, round_trips: int, rule: str = "donchian"):
    log = _log_with_round_trips(prices, round_trips)
    monkeypatch.setattr(runner, "backtest", lambda **_: (log, False))
    return run_backtest(rule, "Soybean", prices.index[0], prices.index[-1], prices, 0.0, 0.0)


class TestWindow:
    def test_open_position_marked_at_window_end(self) -> None:
        """Regression for #4: mark at the window's last close, not the latest price.

        Buy & hold fills at 110 on the second bar. The window ends on the third bar,
        at 120, so the position is worth (120 - 110) points. The later 500 must not count.
        """
        prices = _prices([100.0, 110.0, 120.0, 500.0, 500.0])
        end = prices.index[2]

        result = run_backtest("buy_hold", "Soybean", prices.index[0], end, prices, 0.0, 0.0)

        expected = round((120.0 - 110.0) * USD_PER_POINT, 2)
        wrongly_marked_at_latest = round((500.0 - 110.0) * USD_PER_POINT, 2)
        assert result.mtm is not None
        assert result.mtm["pnl_usd"] == pytest.approx(expected)
        assert result.mtm["date"] == end
        assert result.summary.unadjusted_pnl == pytest.approx(expected)
        assert result.summary.unadjusted_pnl != pytest.approx(wrongly_marked_at_latest)
        assert result.window.index[-1] == end

    def test_window_drops_missing_rows(self) -> None:
        prices = _prices([100.0, np.nan, 120.0, 130.0])
        result = run_backtest(
            "buy_hold", "Soybean", prices.index[0], prices.index[-1], prices, 0.0, 0.0
        )
        assert result.window.notna().all()
        assert len(result.window) == 3

    def test_costs_reduce_pnl(self) -> None:
        prices = _prices([100.0, 110.0, 120.0])
        args = ("buy_hold", "Soybean", prices.index[0], prices.index[-1], prices)
        free = run_backtest(*args, 0.0, 0.0).summary.unadjusted_pnl
        assert run_backtest(*args, 5.0, 0.0).summary.unadjusted_pnl < free
        assert run_backtest(*args, 0.0, 0.1).summary.unadjusted_pnl < free

    def test_accepts_dates(self) -> None:
        prices = _prices([100.0, 110.0, 120.0])
        result = run_backtest(
            "buy_hold",
            "Soybean",
            datetime.date(2022, 1, 3),
            datetime.date(2022, 1, 5),
            prices,
            0.0,
            0.0,
        )
        assert result.summary.fills == 1

    def test_empty_window_raises_backtest_error(self) -> None:
        prices = _prices([100.0, 110.0, 120.0])
        with pytest.raises(BacktestError):
            run_backtest("donchian", "Soybean", "1999-01-01", "1999-12-31", prices, 0.0, 0.0)

    def test_rule_without_fills_has_zero_summary(self) -> None:
        prices = _prices([100.0] * 60)
        result = run_backtest(
            "donchian", "Soybean", prices.index[0], prices.index[-1], prices, 2.5, 0.05
        )
        assert result.trades.empty
        assert result.mtm is None
        assert result.sharpe is None
        assert result.summary.position == "Flat"
        assert result.summary.unadjusted_pnl == 0.0


class TestSharpeGate:
    def test_threshold_is_ten(self) -> None:
        assert MIN_ROUND_TRIPS_FOR_SHARPE == 10

    def test_sharpe_none_below_min_round_trips(self, monkeypatch) -> None:
        prices = _prices(list(np.linspace(100.0, 150.0, 60)))
        assert _run_with_log(monkeypatch, prices, 9).sharpe is None

    def test_sharpe_present_at_min_round_trips(self, monkeypatch) -> None:
        prices = _prices(list(np.linspace(100.0, 150.0, 60)))
        result = _run_with_log(monkeypatch, prices, 10)
        assert result.sharpe is not None
        assert np.isfinite(result.sharpe)

    def test_sharpe_none_for_buy_hold(self, monkeypatch) -> None:
        """Buy & hold's Sharpe is mechanical. Excluded even with a long enough log."""
        prices = _prices(list(np.linspace(100.0, 150.0, 60)))
        assert _run_with_log(monkeypatch, prices, 12, rule="buy_hold").sharpe is None

    def test_negative_first_price_keeps_sharpe_sign(self, monkeypatch) -> None:
        """Capital sized on a negative first price would flip the sign of the Sharpe."""
        values = list(np.linspace(100.0, 150.0, 60))
        positive = _run_with_log(monkeypatch, _prices(values), 10)
        assert positive.sharpe is not None
        assert positive.sharpe > 0

        negative_first = _prices([-37.63, *values[1:]])
        result = _run_with_log(monkeypatch, negative_first, 10)
        assert result.sharpe is not None
        assert result.sharpe > 0
        assert result.sharpe == pytest.approx(positive.sharpe, abs=1e-3)

    def test_no_positive_price_has_no_sharpe(self, monkeypatch) -> None:
        prices = _prices([-1.0] * 60)
        assert _run_with_log(monkeypatch, prices, 10).sharpe is None

"""Unit tests for src/strategy.py: signal generation and backtest engine."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    BacktestError,
    _buy_hold_signals,
    _donchian_signals,
    _tsmom_signals,
    backtest,
)

ALL_RULES = ["tsmom_12m", "donchian", "buy_hold"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    price_df: pd.DataFrame,
    tons_conv: dict[str, float],
    contract_sz: float,
    strategy: str = "tsmom_12m",
    **kwargs,
) -> tuple[pd.DataFrame, bool]:
    return backtest(
        backtest_strategy=strategy,
        start_date=price_df.index[0],
        end_date=price_df.index[-1],
        df=price_df.copy(),
        commodity_chosen="Soybean",
        tons_conversion=tons_conv,
        contract_size=contract_sz,
        **kwargs,
    )


def _with_nan_rows(price_df: pd.DataFrame) -> pd.DataFrame:
    """Blank Soybean on every 8th row, as another market's trading day would."""
    holed = price_df.copy()
    holed.iloc[3::8, holed.columns.get_loc("Soybean")] = np.nan
    return holed


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------


class TestBacktest:
    def test_returns_dataframe_and_bool(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, pos_open = _run(price_df, tons_conv, contract_sz)
        assert isinstance(df_trades, pd.DataFrame)
        assert isinstance(pos_open, bool)

    def test_required_columns_present(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _ = _run(price_df, tons_conv, contract_sz)
        assert not df_trades.empty
        assert {"trade_price", "position", "quantity", "VaR_95"} == set(df_trades.columns)

    def test_no_trades_on_flat_prices(
        self,
        flat_price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """A flat price never breaks out of a Donchian channel, so there are no trades."""
        df_trades, pos_open = _run(flat_price_df, tons_conv, contract_sz, strategy="donchian")
        assert df_trades.empty
        assert pos_open is False

    def test_t1_execution_not_on_first_bar(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """A signal on bar 0 executes at bar 1 at the earliest. Bar 0 is never a trade date."""
        df_trades, _ = _run(price_df, tons_conv, contract_sz, strategy="buy_hold")
        assert df_trades.index[0] == price_df.index[1], (
            "First trade date is not the second bar: same-bar (T+0) execution or a lost signal."
        )

    def test_var_is_non_negative(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _ = _run(price_df, tons_conv, contract_sz)
        assert (df_trades["VaR_95"] >= 0).all()

    def test_var_uses_only_past_returns(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """VaR at early trade dates should differ from late trade dates, because fewer
        returns are available. A single global percentile would mean look-ahead.
        """
        df_trades, _ = _run(price_df, tons_conv, contract_sz, strategy="donchian")
        assert len(df_trades) >= 2
        first_var = float(df_trades["VaR_95"].iloc[0])
        last_var = float(df_trades["VaR_95"].iloc[-1])
        assert first_var != last_var, (
            "VaR is identical across all trade dates, which suggests look-ahead bias."
        )

    def test_unknown_strategy_raises(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        with pytest.raises(BacktestError, match="Unknown strategy"):
            _run(price_df, tons_conv, contract_sz, strategy="invalid_strategy")

    def test_unknown_commodity_raises(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        with pytest.raises(BacktestError, match="not found"):
            backtest("buy_hold", "2020-01-02", "2021-01-02", price_df, "Gold", tons_conv, 1.0)

    def test_empty_window_raises(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        with pytest.raises(BacktestError, match="No Soybean price data"):
            backtest(
                "buy_hold", "1990-01-01", "1990-12-31", price_df, "Soybean", tons_conv, contract_sz
            )


# ---------------------------------------------------------------------------
# All rules smoke-tested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ALL_RULES)
class TestAllStrategiesSmoke:
    def test_returns_valid_trade_log(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, pos_open = _run(price_df, tons_conv, contract_sz, strategy=strategy)
        assert isinstance(pos_open, bool)
        assert not df_trades.empty
        assert {"trade_price", "position", "VaR_95"}.issubset(df_trades.columns)
        assert set(df_trades["position"].unique()).issubset({"buy", "sell"})

    def test_signal_alternation(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Buy and sell must strictly alternate. Buy & hold has a single fill."""
        df_trades, _ = _run(price_df, tons_conv, contract_sz, strategy=strategy)
        signals = df_trades["position"].tolist()
        assert signals[0] == "buy"
        if strategy != "buy_hold":
            assert len(signals) >= 2
        for i in range(len(signals) - 1):
            assert signals[i] != signals[i + 1], (
                f"[{strategy}] consecutive identical signals at positions {i}, {i + 1}: "
                f"'{signals[i]}'"
            )

    def test_trade_prices_match_closes(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Fill price must equal the closing price on the recorded trade date."""
        df_trades, _ = _run(price_df, tons_conv, contract_sz, strategy=strategy)
        for date, row in df_trades.iterrows():
            expected = float(price_df.loc[date, "Soybean"])
            actual = float(row["trade_price"])
            assert abs(actual - expected) < 1e-6, (
                f"[{strategy}] trade_price {actual} != close {expected} on {date}"
            )

    def test_signals_are_integers_without_nan(
        self,
        strategy: str,
        price_df: pd.DataFrame,
    ) -> None:
        prices = price_df["Soybean"]
        signals = {
            "tsmom_12m": _tsmom_signals(prices, 252),
            "donchian": _donchian_signals(prices, 20, 10),
            "buy_hold": _buy_hold_signals(prices),
        }[strategy]
        assert pd.api.types.is_integer_dtype(signals)
        assert signals.notna().all()
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_nan_rows_do_not_blind_rolling(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """A frame with NaN rows must trade exactly like the instrument's clean series."""
        holed = _with_nan_rows(price_df)
        clean = holed.dropna(subset=["Soybean"])
        with_nan, _ = _run(holed, tons_conv, contract_sz, strategy=strategy)
        without, _ = _run(clean, tons_conv, contract_sz, strategy=strategy)
        pd.testing.assert_frame_equal(with_nan, without)

    def test_no_fill_at_nan_price(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _ = _run(_with_nan_rows(price_df), tons_conv, contract_sz, strategy=strategy)
        assert not df_trades.empty
        assert np.isfinite(df_trades["trade_price"].astype(float)).all()

    def test_truncation_invariance(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Removing data after T must not change any trade up to T-1: no look-ahead."""
        full, _ = _run(price_df, tons_conv, contract_sz, strategy=strategy)
        for cut in (400, 613, 777, 950):
            truncated_df = price_df.iloc[:cut]
            truncated, _ = _run(truncated_df, tons_conv, contract_sz, strategy=strategy)
            before_last = truncated_df.index[-2]
            pd.testing.assert_frame_equal(
                truncated.loc[:before_last, ["trade_price", "position"]],
                full.loc[:before_last, ["trade_price", "position"]],
            )


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------


class TestTsmom:
    def test_tsmom_signals_only_on_month_end(self, price_df: pd.DataFrame) -> None:
        prices = price_df["Soybean"]
        signals = _tsmom_signals(prices, 252)
        fired = signals[signals != 0].index
        assert len(fired) > 0
        months = prices.index.to_period("M")
        last_session_of_month = prices.groupby(months).tail(1).index
        assert set(fired).issubset(set(last_session_of_month))

    def test_tsmom_negative_price_does_not_flip_sign(self) -> None:
        """A move from -37 to 20 is up. A ratio of the two would read it as down."""
        dates = pd.to_datetime(["2020-04-20", "2020-05-29"])
        prices = pd.Series([-37.0, 20.0], index=dates)
        assert _tsmom_signals(prices, 1).iloc[-1] == 1

    def test_tsmom_flat_price_is_not_long(self) -> None:
        dates = pd.to_datetime(["2020-04-30", "2020-05-29"])
        prices = pd.Series([20.0, 20.0], index=dates)
        assert _tsmom_signals(prices, 1).iloc[-1] == -1

    def test_warm_start_uses_pre_window_history(
        self, tons_conv: dict[str, float], contract_sz: float
    ) -> None:
        """A one-year window needs a year of history before it. Regression: it had no fills."""
        dates = pd.bdate_range("2015-01-01", "2020-12-31")
        df = pd.DataFrame({"Soybean": np.linspace(100.0, 300.0, len(dates))}, index=dates)
        trades, _ = backtest(
            "tsmom_12m", "2020-01-01", "2020-12-31", df, "Soybean", tons_conv, contract_sz
        )
        assert len(trades) >= 1
        assert trades.index[0] >= pd.Timestamp("2020-01-01")


class TestDonchian:
    def test_donchian_uses_prior_window_only(self) -> None:
        """A channel that included T could never be exceeded by T's own close."""
        dates = pd.date_range("2020-01-01", periods=6, freq="B")
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], index=dates)
        signals = _donchian_signals(prices, entry=3, exit_=2)
        assert signals.tolist() == [0, 0, 0, 1, 1, 1]

    def test_donchian_exit_on_prior_low(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        prices = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0], index=dates)
        signals = _donchian_signals(prices, entry=3, exit_=2)
        assert signals.tolist() == [0, 0, -1, -1, -1]

    def test_donchian_defaults_are_20_10(self) -> None:
        """Protects the owner's decision: 20/10 is the approved rule."""
        params = inspect.signature(backtest).parameters
        assert params["donchian_entry"].default == 20
        assert params["donchian_exit"].default == 10


class TestBuyHold:
    def test_buy_hold_has_one_fill_and_stays_open(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        trades, position_open = _run(price_df, tons_conv, contract_sz, strategy="buy_hold")
        assert len(trades) == 1
        assert trades["position"].iloc[0] == "buy"
        assert position_open is True


# ---------------------------------------------------------------------------
# Execution on the instrument's own calendar
# ---------------------------------------------------------------------------


class TestInstrumentCalendar:
    def test_trade_dates_come_from_clean_index(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Dates must label the session traded. A shared-frame index would mislabel them."""
        holed = _with_nan_rows(price_df)
        trades, _ = _run(holed, tons_conv, contract_sz, strategy="donchian")
        assert not trades.empty
        for date, row in trades.iterrows():
            assert holed.loc[date, "Soybean"] == pytest.approx(float(row["trade_price"]))

    def test_t1_fill_is_next_session(self, tons_conv: dict[str, float], contract_sz: float) -> None:
        """The fill is the next session of this instrument, skipping its missing rows."""
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        df = pd.DataFrame({"Soybean": [100.0, np.nan, np.nan, 130.0]}, index=dates)
        trades, _ = backtest("buy_hold", dates[0], dates[-1], df, "Soybean", tons_conv, contract_sz)
        assert trades.index.tolist() == [dates[3]]
        assert trades["trade_price"].iloc[0] == 130.0

    def test_input_frame_is_not_modified(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        before = price_df.copy()
        backtest(
            "donchian",
            price_df.index[0],
            price_df.index[-1],
            price_df,
            "Soybean",
            tons_conv,
            contract_sz,
        )
        pd.testing.assert_frame_equal(price_df, before)

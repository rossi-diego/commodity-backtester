"""Daily changes between consecutive clean sessions, and the risk they imply."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.margins import crush_table
from src.risk import (
    EWMA_LAMBDA,
    EWMA_SEED,
    MIN_CHANGES,
    NOTIONAL_TONNES,
    _ewma_variance,
    _forecasts,
    christoffersen_independence,
    clean_changes,
    excluded_changes,
    kupiec_pof,
    risk_summary,
    sample_bias,
    var_backtest,
)
from src.rolls import (
    BEAN_MONTHS,
    PRODUCT_MONTHS,
    crush_flags,
    crush_switches,
    front_contracts,
    grain_last_trade,
)


def _table(flagged: list[bool]) -> pd.DataFrame:
    """A crush table with steady legs, so every clean change is small and known."""
    dates = pd.bdate_range("2025-01-06", periods=len(flagged))
    steps = np.arange(len(flagged), dtype=float)
    table = pd.DataFrame(
        {
            "beans": 400.0 + steps,
            "meal_value": 250.0 + 2 * steps,
            "oil_value": 200.0 + 0.5 * steps,
        },
        index=dates,
    )
    table["crush"] = table["meal_value"] + table["oil_value"] - table["beans"]
    table["oil_share"] = table["oil_value"] / (table["oil_value"] + table["meal_value"])
    table["flagged"] = flagged
    return table


def _jump(table: pd.DataFrame, position: int) -> pd.DataFrame:
    """Add 50 USD/t to the meal value, and so the crush, from ``position`` on."""
    jumped = table.copy()
    jumped.iloc[position:, jumped.columns.get_loc("meal_value")] += 50.0
    jumped.iloc[position:, jumped.columns.get_loc("crush")] += 50.0
    return jumped


def test_jump_on_a_flagged_row_is_absent() -> None:
    flagged = [False] * 5 + [True] + [False] * 5
    changes = clean_changes(_jump(_table(flagged), 5))
    assert changes["crush"].max() < 50
    assert pd.Timestamp(_table(flagged).index[5]) not in changes.index


def test_jump_on_the_first_clean_row_after_a_flagged_block_is_absent() -> None:
    flagged = [False] * 4 + [True] * 3 + [False] * 5
    table = _jump(_table(flagged), 7)
    changes = clean_changes(table)
    assert table.index[7] not in changes.index
    assert changes["crush"].max() < 50
    assert table.index[8] in changes.index


def test_the_first_row_is_never_a_change() -> None:
    table = _table([False] * 4)
    assert table.index[0] not in clean_changes(table).index
    assert len(clean_changes(table)) == 3


def test_legs_add_up_to_the_crush(synthetic_prices) -> None:
    changes = clean_changes(crush_table(synthetic_prices))
    assert len(changes) > 1000
    total = changes["meal"] + changes["oil"] + changes["beans"]
    assert (total - changes["crush"]).abs().max() == pytest.approx(0.0, abs=1e-9)


def test_beans_is_the_short_leg() -> None:
    changes = clean_changes(_table([False] * 3))
    assert (changes["beans"] == -1.0).all()


def test_empty_table() -> None:
    changes = clean_changes(crush_table(pd.DataFrame()))
    assert changes.empty
    assert list(changes.columns) == ["crush", "meal", "oil", "beans"]


def test_a_change_across_a_flagged_session_is_dropped() -> None:
    """Clean, flagged, clean: no change joins the two clean ends."""
    table = _table([False, False, True, False, False])
    changes = clean_changes(table)
    assert list(changes.index) == [table.index[1], table.index[4]]


# ---------------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------------


def _from_changes(
    meal: np.ndarray,
    oil: np.ndarray | None = None,
    beans: np.ndarray | None = None,
    flagged: list[bool] | None = None,
) -> pd.DataFrame:
    """A crush table whose legs move by the given daily changes, in USD/t.

    ``beans`` is the change in soybean cost, so the crush moves by meal + oil - beans.
    """
    oil = np.zeros_like(meal) if oil is None else oil
    beans = np.zeros_like(meal) if beans is None else beans
    rows = len(meal) + 1
    dates = pd.bdate_range("2023-01-02", periods=rows)
    table = pd.DataFrame(
        {
            "beans": 400.0 + np.concatenate([[0.0], np.cumsum(beans)]),
            "meal_value": 250.0 + np.concatenate([[0.0], np.cumsum(meal)]),
            "oil_value": 200.0 + np.concatenate([[0.0], np.cumsum(oil)]),
        },
        index=dates,
    )
    table["crush"] = table["meal_value"] + table["oil_value"] - table["beans"]
    table["flagged"] = [False] * rows if flagged is None else flagged
    return table


def _risk_numbers(table: pd.DataFrame) -> tuple[float, float, float, float]:
    summary = risk_summary(table)
    assert summary is not None
    return summary.vol_20, summary.vol_1y, summary.var, summary.es


def test_jumps_on_flagged_and_first_clean_rows_leave_the_risk_unchanged() -> None:
    rng = np.random.default_rng(3)
    flagged = [False] * 401
    for start in (100, 250):
        flagged[start : start + 5] = [True] * 5
    table = _from_changes(rng.normal(0, 2, 400), flagged=flagged)
    jumped = table.copy()
    # A flagged row inside a block, and the first clean row after the other block.
    for position in (102, 255):
        jumped.iloc[position:, jumped.columns.get_loc("meal_value")] += 10_000.0
        jumped.iloc[position:, jumped.columns.get_loc("crush")] += 10_000.0
    assert _risk_numbers(jumped) == pytest.approx(_risk_numbers(table), abs=1e-6)


def test_var_and_es_match_a_direct_computation() -> None:
    """The filtered VaR and ES, rebuilt step by step from the definitions."""
    meal = np.random.default_rng(1).standard_t(4, 600)
    summary = risk_summary(_from_changes(meal))
    assert summary is not None
    x = list(meal)
    s2 = [sum(v * v for v in x[:EWMA_SEED]) / EWMA_SEED]
    for value in x[EWMA_SEED:]:
        s2.append(EWMA_LAMBDA * s2[-1] + (1 - EWMA_LAMBDA) * value * value)
    z = np.array([x[EWMA_SEED + i] / s2[i] ** 0.5 for i in range(len(x) - EWMA_SEED)])
    sigma_next = s2[-1] ** 0.5
    tail = z[z <= np.quantile(z, 0.025)]
    assert summary.vol_ewma == pytest.approx(sigma_next)
    assert summary.var == pytest.approx(-NOTIONAL_TONNES * sigma_next * np.quantile(z, 0.05))
    assert summary.es == pytest.approx(-NOTIONAL_TONNES * sigma_next * tail.mean())
    assert summary.changes == 600


def test_normal_changes_give_the_normal_var() -> None:
    sigma = 2.0
    meal = np.random.default_rng(11).normal(0, sigma, 1250)
    summary = risk_summary(_from_changes(meal))
    assert summary is not None
    assert summary.var == pytest.approx(1.645 * sigma * NOTIONAL_TONNES, rel=0.10)
    assert summary.es >= summary.var
    assert summary.vol_20 > 0
    assert summary.vol_1y == pytest.approx(sigma, rel=0.15)


def test_contributions_sum_to_one(synthetic_prices) -> None:
    summary = risk_summary(crush_table(synthetic_prices))
    assert summary is not None
    assert list(summary.contributions) == ["oil", "meal", "beans"]
    assert sum(summary.contributions.values()) == pytest.approx(1.0, abs=1e-9)


def test_a_constant_beans_leg_carries_no_risk() -> None:
    rng = np.random.default_rng(5)
    table = _from_changes(rng.normal(0, 2, 300), oil=rng.normal(0, 1, 300))
    summary = risk_summary(table)
    assert summary is not None
    assert summary.contributions["beans"] == 0
    assert summary.contributions["meal"] + summary.contributions["oil"] == pytest.approx(1.0)


def test_short_beans_offset_when_beans_move_with_the_products() -> None:
    """Beans that rise with meal cut the crush variance: their share is negative."""
    rng = np.random.default_rng(8)
    meal = rng.normal(0, 3, 300)
    summary = risk_summary(_from_changes(meal, beans=0.5 * meal + rng.normal(0, 0.5, 300)))
    assert summary is not None
    assert summary.contributions["beans"] < 0


def test_too_few_changes_give_none() -> None:
    rng = np.random.default_rng(2)
    # The first EWMA_SEED changes only seed the variance, so they are not scenarios.
    assert risk_summary(_from_changes(rng.normal(0, 1, MIN_CHANGES + EWMA_SEED - 1))) is None
    assert risk_summary(_from_changes(rng.normal(0, 1, MIN_CHANGES + EWMA_SEED))) is not None


def test_a_crush_that_never_moves_gives_none() -> None:
    assert risk_summary(_from_changes(np.zeros(400))) is None


def test_no_clean_change_gives_none() -> None:
    assert risk_summary(crush_table(pd.DataFrame())) is None
    assert risk_summary(_from_changes(np.ones(300), flagged=[True] * 301)) is None


def test_window_is_the_five_years_to_the_last_clean_change(synthetic_prices) -> None:
    table = crush_table(synthetic_prices)
    changes = clean_changes(table)
    summary = risk_summary(table)
    assert summary is not None
    assert summary.date == changes.index[-1]
    start = summary.date - pd.DateOffset(years=5)
    assert summary.changes == len(changes.loc[start:])
    assert summary.changes < len(changes)


@pytest.mark.parametrize("months", [BEAN_MONTHS, PRODUCT_MONTHS])
def test_no_retained_change_spans_a_contract_switch(synthetic_prices, months) -> None:
    """On the real calendars, both ends of every kept change hold the same contract."""
    table = crush_table(synthetic_prices)
    front = front_contracts(pd.DatetimeIndex(table.index), months, grain_last_trade)
    key = front["year"] * 100 + front["month"]
    kept = clean_changes(table).index
    assert len(kept) > 1000
    assert (key.loc[kept] == key.shift(1).loc[kept]).all()


# ---------------------------------------------------------------------------
# EWMA volatility, coverage tests and the backtest
# ---------------------------------------------------------------------------


def _series(values: np.ndarray, start: str = "2010-01-04") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), name="crush")


def test_ewma_of_a_constant_size_is_that_size() -> None:
    c = 2.5
    x = c * np.where(np.arange(200) % 2 == 0, 1.0, -1.0)
    s2 = _ewma_variance(x)
    assert len(s2) == len(x) + 1
    assert np.isnan(s2[:EWMA_SEED]).all()
    assert np.sqrt(s2[EWMA_SEED:]) == pytest.approx(c)


def test_ewma_uses_only_earlier_changes() -> None:
    x = np.random.default_rng(4).normal(0, 1, 100)
    shocked = x.copy()
    shocked[60] = 50.0
    assert np.array_equal(_ewma_variance(x)[:61], _ewma_variance(shocked)[:61], equal_nan=True)
    assert _ewma_variance(shocked)[61] > _ewma_variance(x)[61]


def test_ewma_under_the_seed_is_all_nan() -> None:
    assert np.isnan(_ewma_variance(np.ones(EWMA_SEED - 1))).all()


def test_kupiec_pof() -> None:
    assert kupiec_pof(50, 1000) == pytest.approx(1.0, abs=1e-9)
    assert kupiec_pof(65, 685) < 0.001
    assert kupiec_pof(34, 685) > 0.9
    for exceptions in (0, 100):
        p = kupiec_pof(exceptions, 100)
        assert np.isfinite(p)
        assert p < 1


def test_christoffersen_flags_clustered_exceptions() -> None:
    hits = np.zeros(210, dtype=bool)
    hits[100:110] = True
    assert christoffersen_independence(hits) < 0.01


def test_christoffersen_passes_evenly_spaced_exceptions() -> None:
    hits = np.zeros(400, dtype=bool)
    hits[::20] = True
    assert christoffersen_independence(hits) > 0.05


@pytest.mark.parametrize(
    "hits",
    [np.zeros(300, dtype=bool), np.ones(5, dtype=bool), np.r_[np.zeros(9, dtype=bool), True]],
)
def test_christoffersen_with_nothing_to_test_is_one(hits) -> None:
    assert christoffersen_independence(hits) == 1.0


def test_scaling_the_changes_scales_var_and_es_only() -> None:
    meal = np.random.default_rng(6).normal(0, 1.5, 900)
    base, double = risk_summary(_from_changes(meal)), risk_summary(_from_changes(2 * meal))
    assert base is not None and double is not None
    assert double.var == pytest.approx(2 * base.var, rel=1e-9)
    assert double.es == pytest.approx(2 * base.es, rel=1e-9)
    one, two = var_backtest(_series(meal)), var_backtest(_series(2 * meal))
    assert one is not None and two is not None
    counts = (one.days, one.exceptions, one.hs_exceptions)
    assert (two.days, two.exceptions, two.hs_exceptions) == counts


def test_no_forecast_sees_its_own_change_or_a_later_one() -> None:
    x = np.random.default_rng(9).normal(0, 1, 700)
    base = _forecasts(_series(x))
    changed = x.copy()
    changed[-1] = -40.0
    after = _forecasts(_series(changed))
    assert len(base) > 100
    pd.testing.assert_frame_equal(after.iloc[:-1], base.iloc[:-1])
    assert after.iloc[-1]["fhs"] == base.iloc[-1]["fhs"]
    assert after.iloc[-1]["hs"] == base.iloc[-1]["hs"]
    assert after.iloc[-1]["change"] == -40.0


def test_a_regime_shift_breaks_plain_hs_but_not_the_filtered_var() -> None:
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(0, 1, 1500), rng.normal(0, 3, 700)])
    backtest = var_backtest(_series(x, "2015-01-01"))
    assert backtest is not None
    assert backtest.hs_exceptions / backtest.days > 0.10
    assert 0.03 <= backtest.rate <= 0.08
    assert backtest.hs_exceptions > backtest.exceptions


def test_iid_normal_changes_give_about_five_percent() -> None:
    x = np.random.default_rng(0).normal(0, 1, 3000)
    backtest = var_backtest(_series(x))
    assert backtest is not None
    assert 0.03 <= backtest.rate <= 0.07
    assert backtest.end == _series(x).index[-1]
    assert backtest.start >= backtest.end - pd.DateOffset(years=5)


def test_without_enough_earlier_scenarios_there_is_no_backtest() -> None:
    meal = np.random.default_rng(2).normal(0, 1, MIN_CHANGES + EWMA_SEED)
    summary = risk_summary(_from_changes(meal))
    assert summary is not None
    assert summary.backtest is None
    assert var_backtest(_series(meal)) is None
    assert var_backtest(_series(np.array([1.0]))) is None


def test_the_summary_carries_the_backtest_of_its_window(synthetic_prices) -> None:
    table = crush_table(synthetic_prices)
    summary = risk_summary(table)
    assert summary is not None
    assert summary.backtest == var_backtest(clean_changes(table)["crush"])
    assert summary.backtest is not None
    assert summary.backtest.end == summary.date
    assert summary.backtest.days == summary.changes


# ---------------------------------------------------------------------------
# What the clean sample leaves out
# ---------------------------------------------------------------------------


def _delivery_table() -> pd.DataFrame:
    """Real flags over 6 years, with flagged sessions three times as volatile."""
    dates = pd.bdate_range("2020-01-02", "2025-12-31")
    flagged = crush_flags(dates).to_numpy()
    rng = np.random.default_rng(12)
    steps = rng.normal(0, 1, len(dates) - 1) * np.where(flagged[1:], 3.0, 1.0)
    table = _from_changes(steps, flagged=list(flagged))
    table.index = dates
    return table


def test_left_out_changes_are_more_volatile() -> None:
    table = _delivery_table()
    bias = sample_bias(table)
    summary = risk_summary(table)
    assert bias is not None and summary is not None
    assert bias.excluded_sd > bias.kept_sd
    assert bias.kept == summary.changes
    assert bias.excluded > 100


def test_no_left_out_change_is_next_to_a_switch() -> None:
    table = _delivery_table()
    excluded = excluded_changes(table)
    switch = crush_switches(pd.DatetimeIndex(table.index)).to_numpy()
    near = switch | np.r_[False, switch[:-1]] | np.r_[switch[1:], False]
    positions = pd.DatetimeIndex(table.index).get_indexer(excluded.index)
    assert len(positions) > 100
    assert not near[positions].any()
    assert not set(excluded.index) & set(clean_changes(table).index)
    assert table.index[0] not in excluded.index


def test_sample_bias_needs_both_samples() -> None:
    assert sample_bias(crush_table(pd.DataFrame())) is None
    assert excluded_changes(crush_table(pd.DataFrame())).empty
    assert sample_bias(_from_changes(np.random.default_rng(1).normal(0, 1, 400))) is None

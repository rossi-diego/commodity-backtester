"""Daily changes between consecutive clean sessions, and the risk they imply."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.margins import crush_table
from src.risk import MIN_CHANGES, NOTIONAL_TONNES, clean_changes, risk_summary
from src.rolls import BEAN_MONTHS, PRODUCT_MONTHS, front_contracts, grain_last_trade


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


def test_var_and_es_by_hand() -> None:
    """Changes of -1.50 to +1.49 USD/t in 0.01 steps, shuffled.

    The 5% quantile sits at position 0.05 x 299 = 14.95, so VaR = 1,000 x 1.3505. The
    2.5% quantile sits at 7.475, -1.42525, so ES averages the 8 lowest: 1,000 x 1.465.
    """
    values = (np.arange(300) - 150) / 100
    meal = np.random.default_rng(1).permutation(values)
    summary = risk_summary(_from_changes(meal))
    assert summary is not None
    assert summary.var == pytest.approx(1350.5)
    assert summary.es == pytest.approx(1465.0)
    assert summary.changes == 300


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
    assert risk_summary(_from_changes(rng.normal(0, 1, MIN_CHANGES - 1))) is None
    assert risk_summary(_from_changes(rng.normal(0, 1, MIN_CHANGES))) is not None


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

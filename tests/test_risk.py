"""Daily changes between consecutive clean sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.margins import crush_table
from src.risk import clean_changes


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

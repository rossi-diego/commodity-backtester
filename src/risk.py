"""Daily changes of the crush and its legs between consecutive clean sessions.

A change is kept only when both of its ends are clean sessions, so a roll jump or
a delivery-period move never counts as a change. The differences are taken on the
full table in row order, never after dropping flagged rows, which would join two
clean sessions across a roll.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["clean_changes"]

_LEGS = ["crush", "meal_value", "oil_value", "beans"]


def clean_changes(table: pd.DataFrame) -> pd.DataFrame:
    """Session-to-session changes in USD/t where the session and the one before are clean.

    ``table`` is ``margins.crush_table`` output. Returns ``crush``, ``meal`` and ``oil``
    changes, and ``beans`` as minus the change in soybean cost (the short leg), so
    ``crush == meal + oil + beans`` on every row.
    """
    columns = ["crush", "meal", "oil", "beans"]
    if table.empty:
        return pd.DataFrame(columns=columns, dtype=float)
    flagged = table["flagged"].astype(bool)
    keep = ~flagged & ~flagged.shift(1, fill_value=True)
    diff = table[_LEGS].astype(float).diff()
    changes = pd.DataFrame(
        {
            "crush": diff["crush"],
            "meal": diff["meal_value"],
            "oil": diff["oil_value"],
            "beans": -diff["beans"],
        },
        index=table.index,
    )
    return changes[keep].dropna()

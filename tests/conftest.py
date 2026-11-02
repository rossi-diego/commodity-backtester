"""Shared pytest fixtures for the Soybean Crush Margin test suite."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.constants import BEANS, DIESEL, MEAL, OIL

# Typical quote levels in exchange units: cents/bu, USD/short ton, cents/lb, USD/gal.
LEVELS = {BEANS: 1000.0, MEAL: 320.0, OIL: 45.0, DIESEL: 2.6}


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """Four-leg frame shaped like the real cache: 2000 to today, with isolated gaps.

    It starts in early 2000 so that it counts as a cache hit for the default request,
    and ends on the last business day so that it is not stale.
    """
    dates = pd.bdate_range("2000-01-03", datetime.date.today())
    rng = np.random.default_rng(seed=7)
    columns = {}
    for name, level in LEVELS.items():
        steps = rng.normal(0.0, 0.01, len(dates))
        walk = level * np.exp(np.cumsum(steps) - np.cumsum(steps).mean())
        walk[rng.integers(0, len(dates), size=25)] = np.nan
        columns[name] = walk
    df = pd.DataFrame(columns, index=dates)
    df.index.name = "date"
    return df

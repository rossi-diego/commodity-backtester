"""Shared pytest fixtures for the commodity-backtester test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic price data
# ---------------------------------------------------------------------------

@pytest.fixture()
def price_df() -> pd.DataFrame:
    """Daily price DataFrame for two synthetic commodities over ~4 years.

    Soybean price oscillates around 500, Corn around 400, so the metric-ton
    ratio (Soybean / Corn) oscillates around 500*0.36744 / (400*0.39368) ≈ 1.167.
    This deterministic series lets us calculate expected PnL analytically.
    """
    dates = pd.date_range("2020-01-02", periods=1000, freq="B")
    rng = np.random.default_rng(seed=42)

    soybean_prices = 500 + 60 * np.sin(np.linspace(0, 8 * np.pi, len(dates)))
    corn_prices    = 400 + 40 * np.sin(np.linspace(np.pi / 4, 8 * np.pi + np.pi / 4, len(dates)))

    # Add small noise so prices aren't perfectly sinusoidal
    soybean_prices += rng.normal(0, 2, len(dates))
    corn_prices    += rng.normal(0, 1, len(dates))

    df = pd.DataFrame(
        {"Soybean": soybean_prices, "Corn": corn_prices},
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture()
def flat_price_df() -> pd.DataFrame:
    """Price DataFrame with perfectly flat prices — zero PnL expected."""
    dates = pd.date_range("2020-01-02", periods=500, freq="B")
    df = pd.DataFrame({"Soybean": 500.0, "Corn": 400.0}, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture()
def tons_conv() -> dict[str, float]:
    """Subset of tons_conversion used in tests."""
    return {"Soybean": 0.36744, "Corn": 0.39368}


@pytest.fixture()
def contract_sz() -> float:
    return 136.08

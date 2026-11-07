"""Daily changes between consecutive clean sessions, and the 1-day risk they imply.

A change is kept only when both of its ends are clean sessions, so a roll jump or
a delivery-period move never counts as a change. The differences are taken on the
full table in row order, never after dropping flagged rows, which would join two
clean sessions across a roll.

The risk is a filtered historical simulation on a long board crush of
``NOTIONAL_TONNES`` (long meal and oil, short soybeans). Each clean change is divided
by its EWMA volatility forecast, made from the changes before it only; the
standardised changes of the past ``LOOKBACK_YEARS`` are the scenarios, scaled to the
forecast for the next change. A plain 5-year window weighs calm and volatile years
the same, so its exceptions cluster when the regime shifts. ``var_backtest`` checks
the VaR on every change of the window, each forecast without look-ahead. Front-month
closes, not a hedge book, and no square-root-of-time scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from .rolls import crush_switches
from .seasonal import LOOKBACK_YEARS

__all__ = [
    "ES_LEVEL",
    "EWMA_LAMBDA",
    "EWMA_SEED",
    "MIN_CHANGES",
    "NOTIONAL_TONNES",
    "SHORT_VOL_CHANGES",
    "VAR_LEVEL",
    "Backtest",
    "RiskSummary",
    "SampleBias",
    "christoffersen_independence",
    "clean_changes",
    "excluded_changes",
    "kupiec_pof",
    "risk_summary",
    "sample_bias",
    "var_backtest",
]

NOTIONAL_TONNES: Final = 1_000
VAR_LEVEL: Final = 0.95
ES_LEVEL: Final = 0.975
# About two years of clean changes (about 137 a year): fewer leave too few tail scenarios for a 97.5% ES.
MIN_CHANGES: Final = 250
SHORT_VOL_CHANGES: Final = 20
# RiskMetrics daily decay.
EWMA_LAMBDA: Final = 0.94
# The first 30 clean changes only seed the variance and never become scenarios.
EWMA_SEED: Final = 30

_LEGS = ["crush", "meal_value", "oil_value", "beans"]


@dataclass(frozen=True)
class Backtest:
    """The filtered VaR against the changes it forecast, over the five-year window.

    ``kupiec_p`` (unconditional coverage against 5%) and ``christoffersen_p``
    (independence of exceptions) are for the filtered VaR. ``hs_exceptions`` counts
    plain historical simulation on the same days, for comparison only.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    exceptions: int
    hs_exceptions: int
    kupiec_p: float
    christoffersen_p: float

    @property
    def rate(self) -> float:
        return self.exceptions / self.days


@dataclass(frozen=True)
class RiskSummary:
    """1-day risk of a long board crush, by filtered historical simulation.

    ``vol_20`` and ``vol_1y`` are plain daily standard deviations in USD/t, not
    annualised; ``vol_ewma`` is the EWMA forecast for the next change, which scales
    the VaR and ES. ``var`` and ``es`` are in USD per ``NOTIONAL_TONNES``, positive
    for a loss. ``contributions`` splits the crush variance by leg and sums to 1; a
    leg that offsets the others is negative.
    """

    date: pd.Timestamp
    vol_20: float
    vol_1y: float
    var: float
    es: float
    contributions: dict[str, float]
    changes: int
    vol_ewma: float = math.nan
    backtest: Backtest | None = None


@dataclass(frozen=True)
class SampleBias:
    """Clean changes against the in-contract changes the clean sample leaves out."""

    kept: int
    excluded: int
    kept_sd: float
    excluded_sd: float


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


def excluded_changes(table: pd.DataFrame) -> pd.Series:
    """Crush changes that clean_changes drops but that stay in one contract.

    The same row-order differences as ``clean_changes``, on rows it drops, leaving out
    any change within one session of a contract switch: the calendar's switch date
    can be a session off Yahoo's.
    """
    if table.empty:
        return pd.Series(dtype=float, name="crush")
    flagged = table["flagged"].astype(bool)
    keep = ~flagged & ~flagged.shift(1, fill_value=True)
    switch = pd.Series(crush_switches(pd.DatetimeIndex(table.index)).to_numpy(), table.index)
    near = switch | switch.shift(1, fill_value=False) | switch.shift(-1, fill_value=False)
    diff = table["crush"].astype(float).diff()
    return diff[~keep & ~near & diff.notna()].rename("crush")


def _ewma_variance(x: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """EWMA variance forecasts: ``s2[i]`` is for change ``x[i]``, from ``x[:i]`` only.

    ``s2[i + 1] = lam * s2[i] + (1 - lam) * x[i] ** 2``, seeded with the mean square of
    the first ``EWMA_SEED`` changes, so ``s2[:EWMA_SEED]`` is NaN. The array has one
    more entry than ``x``: ``s2[-1]`` is the forecast for the next change. It runs once
    over the whole clean history and updates per clean change, so it does not decay
    over the flagged sessions between two clean changes.
    """
    s2 = np.full(len(x) + 1, np.nan)
    if len(x) < EWMA_SEED:
        return s2
    s2[EWMA_SEED] = np.mean(x[:EWMA_SEED] ** 2)
    for i in range(EWMA_SEED, len(x)):
        s2[i + 1] = lam * s2[i] + (1 - lam) * x[i] ** 2
    return s2


def _standardised(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EWMA volatility forecasts (one more than ``x``) and the standardised changes."""
    sigma = np.sqrt(_ewma_variance(x))
    with np.errstate(divide="ignore", invalid="ignore"):
        return sigma, x / sigma[:-1]


def _window_start(dates: pd.DatetimeIndex, end: pd.Timestamp) -> int:
    """The first change of the ``LOOKBACK_YEARS`` to ``end`` that can be a scenario."""
    first = int(dates.searchsorted(end - pd.DateOffset(years=LOOKBACK_YEARS), side="left"))
    return max(first, EWMA_SEED)


def _forecasts(crush: pd.Series) -> pd.DataFrame:
    """1-day VaR forecasts, filtered (``fhs``) and plain (``hs``), for each change tested.

    The test set is every change in the five years to the last one. Change ``i`` is
    forecast from the scenarios of the five years to change ``i - 1``, so no forecast
    sees the change it is tested on. Days with under ``MIN_CHANGES`` scenarios are
    skipped. Values are in USD/t, positive for a loss.
    """
    dates = pd.DatetimeIndex(crush.index)
    x = crush.to_numpy(dtype=float)
    rows: list[tuple[pd.Timestamp, float, float, float]] = []
    if len(x) > 1:
        sigma, z = _standardised(x)
        first = int(dates.searchsorted(dates[-1] - pd.DateOffset(years=LOOKBACK_YEARS)))
        for i in range(max(first, 1), len(x)):
            start = _window_start(dates, dates[i - 1])
            if i - start < MIN_CHANGES:
                continue
            fhs = -sigma[i] * float(np.quantile(z[start:i], 1 - VAR_LEVEL))
            hs = -float(np.quantile(x[start:i], 1 - VAR_LEVEL))
            rows.append((dates[i], fhs, hs, float(x[i])))
    frame = pd.DataFrame(rows, columns=["date", "fhs", "hs", "change"])
    return frame.set_index("date")


def _xlogy(count: float, p: float) -> float:
    # 0 x ln 0 = 0, the limit, so a count of zero never takes the log of zero.
    return 0.0 if count == 0 else count * math.log(p)


def _chi2_1df_p(lr: float) -> float:
    # Upper tail of chi-squared with 1 degree of freedom: P(Z² > lr) = erfc(sqrt(lr / 2)).
    return math.erfc(math.sqrt(max(lr, 0.0) / 2))


def kupiec_pof(exceptions: int, days: int, level: float = VAR_LEVEL) -> float:
    """Kupiec proportion-of-failures p-value: is the exception rate ``1 - level``?"""
    # Rounded so that 1 - 0.95 is exactly 0.05, and 50 in 1,000 gives a ratio of 0.
    p = round(1.0 - level, 12)
    n, x = days, exceptions
    observed = x / n
    lr = -2 * (
        _xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - observed) - _xlogy(x, observed)
    )
    return _chi2_1df_p(lr)


def christoffersen_independence(hits: np.ndarray) -> float:
    """Christoffersen independence p-value: does an exception make the next one likelier?

    A first-order Markov test on the exception sequence. Returns 1.0 when there is no
    exception, or no session after one, so there is nothing to test.
    """
    h = np.asarray(hits, dtype=bool)
    before, after = h[:-1], h[1:]
    n00 = int((~before & ~after).sum())
    n01 = int((~before & after).sum())
    n10 = int((before & ~after).sum())
    n11 = int((before & after).sum())
    if not h.any() or n10 + n11 == 0 or n00 + n01 == 0:
        return 1.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi0, pi1 = n01 / (n00 + n01), n11 / (n10 + n11)

    def loglik(q: float, misses: int, exceptions: int) -> float:
        return _xlogy(misses, 1 - q) + _xlogy(exceptions, q)

    lr = -2 * (loglik(pi, n00 + n10, n01 + n11) - loglik(pi0, n00, n01) - loglik(pi1, n10, n11))
    return _chi2_1df_p(lr)


def var_backtest(crush: pd.Series) -> Backtest | None:
    """Backtest the filtered VaR on the five years of ``clean_changes(table)['crush']``.

    An exception is a change below minus its VaR. Returns None when no day has
    ``MIN_CHANGES`` earlier scenarios.
    """
    forecasts = _forecasts(crush)
    if forecasts.empty:
        return None
    change = forecasts["change"]
    hits = (change < -forecasts["fhs"]).to_numpy()
    days, exceptions = len(forecasts), int(hits.sum())
    return Backtest(
        start=pd.Timestamp(forecasts.index[0]),
        end=pd.Timestamp(forecasts.index[-1]),
        days=days,
        exceptions=exceptions,
        hs_exceptions=int((change < -forecasts["hs"]).sum()),
        kupiec_p=kupiec_pof(exceptions, days),
        christoffersen_p=christoffersen_independence(hits),
    )


def risk_summary(table: pd.DataFrame) -> RiskSummary | None:
    """Volatility, VaR, ES and risk by leg over the five years to the last clean change.

    Filtered historical simulation: the standardised changes of the window, times the
    EWMA volatility for the next change. VaR is minus the 5% quantile of the scenarios
    (numpy's linear interpolation) and ES is minus their mean at or below the 2.5%
    quantile. A leg's contribution is its covariance with the crush over the crush
    variance, equal-weighted over the window. Returns None with fewer than
    ``MIN_CHANGES`` scenarios in the window, or when the crush never moves.
    """
    changes = clean_changes(table)
    if changes.empty:
        return None
    date = pd.Timestamp(changes.index[-1])
    window = changes.loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    crush = window["crush"]
    variance = float(crush.var())
    sigma, z = _standardised(changes["crush"].to_numpy(dtype=float))
    scenarios = z[_window_start(pd.DatetimeIndex(changes.index), date) :]
    sigma_next = float(sigma[-1])
    if len(scenarios) < MIN_CHANGES or not variance > 0 or not sigma_next > 0:
        return None
    tail = np.quantile(scenarios, 1 - ES_LEVEL)
    year = crush.loc[date - pd.DateOffset(years=1) : date]
    return RiskSummary(
        date=date,
        vol_20=float(crush.iloc[-SHORT_VOL_CHANGES:].std()),
        vol_1y=float(year.std()),
        var=-NOTIONAL_TONNES * sigma_next * float(np.quantile(scenarios, 1 - VAR_LEVEL)),
        es=-NOTIONAL_TONNES * sigma_next * float(scenarios[scenarios <= tail].mean()),
        contributions={
            leg: float(window[leg].cov(crush)) / variance for leg in ("oil", "meal", "beans")
        },
        changes=len(window),
        vol_ewma=sigma_next,
        backtest=var_backtest(changes["crush"]),
    )


def sample_bias(table: pd.DataFrame) -> SampleBias | None:
    """Clean against excluded in-contract changes, over the risk window.

    Both samples cover the ``LOOKBACK_YEARS`` to the last clean change. Returns None when
    either has fewer than 2 changes.
    """
    changes = clean_changes(table)
    if changes.empty:
        return None
    date = pd.Timestamp(changes.index[-1])
    start = date - pd.DateOffset(years=LOOKBACK_YEARS)
    kept = changes["crush"].loc[start:date]
    excluded = excluded_changes(table).loc[start:date]
    if len(kept) < 2 or len(excluded) < 2:
        return None
    return SampleBias(
        kept=len(kept),
        excluded=len(excluded),
        kept_sd=float(kept.std()),
        excluded_sd=float(excluded.std()),
    )

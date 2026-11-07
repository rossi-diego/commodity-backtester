"""Daily changes between consecutive clean sessions, and the 1-day risk they imply.

A change is kept only when both of its ends are clean sessions, so a roll jump or
a delivery-period move never counts as a change. The differences are taken on the
full table in row order, never after dropping flagged rows, which would join two
clean sessions across a roll.

The risk is a filtered historical simulation on the crusher's margin on
``NOTIONAL_TONNES``, unhedged (long meal and oil, short soybeans). Each clean change
is divided by its EWMA volatility forecast, made from the changes before it only; the
standardised changes of the past ``LOOKBACK_YEARS`` are the scenarios, scaled to the
forecast for the next change. A plain 5-year window weighs calm and volatile years
the same, so its exceptions cluster when the regime shifts. ``var_backtest`` checks
the VaR, and the ES behind it, on every change of the window, each forecast without
look-ahead. Each leg's share of the ES: its mean in the scenarios at or below the
2.5% quantile, over the crush's (Euler allocation). Front-month closes, not a hedge
book, and no square-root-of-time scaling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from .rolls import crush_switches
from .seasonal import LOOKBACK_YEARS

__all__ = [
    "ES_BACKTEST_DRAWS",
    "ES_BACKTEST_SEED",
    "ES_LEVEL",
    "EWMA_LAMBDA",
    "EWMA_SEED",
    "MIN_CHANGES",
    "NOTIONAL_TONNES",
    "SHORT_VOL_CHANGES",
    "VAR_LEVEL",
    "WORST_DAYS",
    "Backtest",
    "RiskSummary",
    "SampleBias",
    "christoffersen_independence",
    "clean_changes",
    "excluded_changes",
    "kupiec_pof",
    "percent_shares",
    "risk_summary",
    "sample_bias",
    "var_backtest",
    "var_forecasts",
    "worst_changes",
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
# Simulated samples for the ES backtest p-value, drawn from a fixed seed so it repeats.
ES_BACKTEST_DRAWS: Final = 2_000
ES_BACKTEST_SEED: Final = 0
WORST_DAYS: Final = 3

_LEGS = ["crush", "meal_value", "oil_value", "beans"]
_SHARES = ("oil", "meal", "beans")


@dataclass(frozen=True)
class Backtest:
    """The filtered VaR and ES against the changes they forecast, over the five-year window.

    ``kupiec_p`` (unconditional coverage against 5%) and ``christoffersen_p``
    (independence of exceptions) are for the filtered VaR. ``hs_exceptions`` counts
    plain historical simulation on the same days, for comparison only.

    The ``es_`` fields test the 97.5% ES: ``es_exceptions`` are changes below minus the
    97.5% VaR, ``es_ratio`` is their mean loss over the ES forecast (NaN with none),
    and ``z2`` is Acerbi and Szekely's (2014) statistic, 0 in expectation, negative
    when the ES is understated and exactly 1 with no exception. ``z2_p`` is its
    one-sided p-value, simulated from each day's own scenarios.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    exceptions: int
    hs_exceptions: int
    kupiec_p: float
    christoffersen_p: float
    es_exceptions: int
    es_kupiec_p: float
    es_christoffersen_p: float
    es_ratio: float
    z2: float
    z2_p: float

    @property
    def rate(self) -> float:
        return self.exceptions / self.days

    @property
    def es_rate(self) -> float:
        return self.es_exceptions / self.days


@dataclass(frozen=True)
class SampleBias:
    """Clean changes against the in-contract changes the clean sample leaves out.

    ``combined_var`` is the same filtered VaR, in USD per ``NOTIONAL_TONNES``, on the
    clean and left-out changes together, over the ``combined_changes`` of its window;
    NaN when they are too few. ``excluded_exceptions`` counts the left-out changes
    below minus the clean VaR forecast for their day, of ``excluded_tested``.
    """

    kept: int
    excluded: int
    kept_sd: float
    excluded_sd: float
    combined_changes: int
    combined_var: float
    excluded_tested: int
    excluded_exceptions: int
    excluded_kupiec_p: float


@dataclass(frozen=True)
class RiskSummary:
    """1-day risk of the crusher's margin on 1,000 t, unhedged (long meal and oil, short
    soybeans), by filtered historical simulation.

    ``vol_20`` and ``vol_1y`` are plain daily standard deviations in USD/t, not
    annualised; ``vol_ewma`` is the EWMA forecast for the next change, which scales
    the VaR and ES. ``var`` and ``es`` are in USD per ``NOTIONAL_TONNES``, positive
    for a loss. ``contributions`` is each leg's share of the ES: its mean in the
    scenarios at or below the 2.5% quantile, over the crush's (Euler allocation), from
    ``tail_scenarios`` scenarios. The shares sum to 1; a leg that offsets the others
    is negative. ``worst`` holds the largest clean losses of the window, in USD.
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
    tail_scenarios: int = 0
    bias: SampleBias | None = None
    worst: tuple[tuple[pd.Timestamp, float], ...] = ()


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


def _fhs_var(sigma_i: float, scenarios: np.ndarray) -> float:
    """The filtered VaR in USD/t: minus the 5% quantile of the scenarios, times ``sigma_i``."""
    return -sigma_i * float(np.quantile(scenarios, 1 - VAR_LEVEL))


def _scenarios(changes: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """The scenario legs of the window in USD/t, and the volatility they are scaled to.

    Each leg of a clean change is divided by that change's EWMA volatility and times
    the forecast for the next change, so the legs of a row sum to its crush scenario.
    """
    sigma, _ = _standardised(changes["crush"].to_numpy(dtype=float))
    start = _window_start(pd.DatetimeIndex(changes.index), pd.Timestamp(changes.index[-1]))
    sigma_next = float(sigma[-1])
    legs = changes[list(_SHARES)].iloc[start:].mul(sigma_next / sigma[start:-1], axis=0)
    return legs, sigma_next


def _forecast_rows(crush: pd.Series) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """The forecasts of ``var_forecasts``, and each tested day's scenarios in USD/t."""
    dates = pd.DatetimeIndex(crush.index)
    x = crush.to_numpy(dtype=float)
    rows: list[tuple[pd.Timestamp, float, float, float, float, float]] = []
    scenarios: list[np.ndarray] = []
    if len(x) > 1:
        sigma, z = _standardised(x)
        first = int(dates.searchsorted(dates[-1] - pd.DateOffset(years=LOOKBACK_YEARS)))
        for i in range(max(first, 1), len(x)):
            start = _window_start(dates, dates[i - 1])
            if i - start < MIN_CHANGES:
                continue
            window = z[start:i]
            tail = float(np.quantile(window, 1 - ES_LEVEL))
            rows.append(
                (
                    dates[i],
                    _fhs_var(sigma[i], window),
                    -float(np.quantile(x[start:i], 1 - VAR_LEVEL)),
                    float(x[i]),
                    -sigma[i] * tail,
                    -sigma[i] * float(window[window <= tail].mean()),
                )
            )
            scenarios.append(sigma[i] * window)
    columns = ["date", "fhs", "hs", "change", "var975", "es975"]
    return pd.DataFrame(rows, columns=columns).set_index("date"), scenarios


def var_forecasts(crush: pd.Series) -> pd.DataFrame:
    """1-day forecasts for each change tested: VaR filtered (``fhs``) and plain (``hs``).

    ``var975`` and ``es975`` are the filtered 97.5% VaR and ES. The test set is every
    change in the five years to the last one. Change ``i`` is forecast from the
    scenarios of the five years to change ``i - 1``, so no forecast sees the change it
    is tested on. Days with under ``MIN_CHANGES`` scenarios are skipped. Values are in
    USD/t, positive for a loss.
    """
    return _forecast_rows(crush)[0]


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


def _z2(change: np.ndarray, var: np.ndarray, es: np.ndarray) -> np.ndarray:
    """Acerbi and Szekely's Z2 over the last axis: sum(X I / (T alpha ES)) + 1."""
    hits = change < -var
    days = change.shape[-1]
    return np.asarray(np.sum(change * hits / (days * (1 - ES_LEVEL) * es), axis=-1) + 1)


def _z2_p(observed: float, var: np.ndarray, es: np.ndarray, scenarios: list[np.ndarray]) -> float:
    """Share of Z2 values simulated under the model at or below ``observed``.

    Each draw takes every day's outcome from that day's own scenarios, with replacement,
    and keeps its 97.5% VaR and ES. The published critical value of -0.70 is for
    T = 250 and Z2 is a mean, so on another number of days the p-value is simulated.
    """
    lengths = np.array([len(s) for s in scenarios])
    padded = np.zeros((len(scenarios), int(lengths.max())))
    for row, values in enumerate(scenarios):
        padded[row, : len(values)] = values
    rng = np.random.default_rng(ES_BACKTEST_SEED)
    picks = rng.integers(0, lengths, size=(ES_BACKTEST_DRAWS, len(scenarios)))
    outcomes = padded[np.arange(len(scenarios)), picks]
    return float(np.mean(_z2(outcomes, var, es) <= observed))


def var_backtest(crush: pd.Series) -> Backtest | None:
    """Backtest the filtered VaR and ES on the five years of ``clean_changes(table)['crush']``.

    An exception is a change below minus its VaR. Returns None when no day has
    ``MIN_CHANGES`` earlier scenarios.
    """
    forecasts, scenarios = _forecast_rows(crush)
    if forecasts.empty:
        return None
    change = forecasts["change"].to_numpy()
    hits = change < -forecasts["fhs"].to_numpy()
    var975, es975 = forecasts["var975"].to_numpy(), forecasts["es975"].to_numpy()
    es_hits = change < -var975
    days, exceptions, es_exceptions = len(forecasts), int(hits.sum()), int(es_hits.sum())
    z2 = float(_z2(change, var975, es975))
    return Backtest(
        start=pd.Timestamp(forecasts.index[0]),
        end=pd.Timestamp(forecasts.index[-1]),
        days=days,
        exceptions=exceptions,
        hs_exceptions=int((change < -forecasts["hs"].to_numpy()).sum()),
        kupiec_p=kupiec_pof(exceptions, days),
        christoffersen_p=christoffersen_independence(hits),
        es_exceptions=es_exceptions,
        es_kupiec_p=kupiec_pof(es_exceptions, days, ES_LEVEL),
        es_christoffersen_p=christoffersen_independence(es_hits),
        es_ratio=float(np.mean(-change[es_hits] / es975[es_hits])) if es_exceptions else math.nan,
        z2=z2,
        z2_p=_z2_p(z2, var975, es975, scenarios),
    )


def worst_changes(
    changes: pd.Series, n: int = WORST_DAYS
) -> tuple[tuple[pd.Timestamp, float], ...]:
    """The ``n`` largest clean losses of the ``LOOKBACK_YEARS`` window, in USD, worst first.

    ``changes`` is ``clean_changes(table)['crush']``; each loss is times
    ``NOTIONAL_TONNES``. Only falls count, so there can be fewer than ``n``.
    """
    if changes.empty:
        return ()
    date = pd.Timestamp(changes.index[-1])
    window = changes.loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    worst = window[window < 0].nsmallest(n)
    days = pd.DatetimeIndex(worst.index)
    return tuple(
        (day, float(v) * NOTIONAL_TONNES) for day, v in zip(days, worst.to_numpy(), strict=True)
    )


def risk_summary(table: pd.DataFrame) -> RiskSummary | None:
    """Volatility, VaR, ES and each leg's share of the ES, over the five years to the last
    clean change.

    Filtered historical simulation: the standardised changes of the window, times the
    EWMA volatility for the next change. VaR is minus the 5% quantile of the scenarios
    (numpy's linear interpolation) and ES is minus their mean at or below the 2.5%
    quantile. Each leg's share of the ES: its mean in the scenarios at or below the
    2.5% quantile, over the crush's (Euler allocation). Returns None with fewer than
    ``MIN_CHANGES`` scenarios in the window, or when the crush never moves.
    """
    changes = clean_changes(table)
    if changes.empty:
        return None
    date = pd.Timestamp(changes.index[-1])
    window = changes.loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    crush = window["crush"]
    sigma, z = _standardised(changes["crush"].to_numpy(dtype=float))
    scenarios = z[_window_start(pd.DatetimeIndex(changes.index), date) :]
    sigma_next = float(sigma[-1])
    if len(scenarios) < MIN_CHANGES or not crush.std() > 0 or not sigma_next > 0:
        return None
    legs, _ = _scenarios(changes)
    crush_scenarios = sigma_next * scenarios
    tail = crush_scenarios <= np.quantile(crush_scenarios, 1 - ES_LEVEL)
    tail_mean = float(crush_scenarios[tail].mean())
    year = crush.loc[date - pd.DateOffset(years=1) : date]
    return RiskSummary(
        date=date,
        vol_20=float(crush.iloc[-SHORT_VOL_CHANGES:].std()),
        vol_1y=float(year.std()),
        var=NOTIONAL_TONNES * _fhs_var(sigma_next, scenarios),
        es=-NOTIONAL_TONNES * tail_mean,
        contributions={
            leg: float(legs[leg].to_numpy()[tail].mean()) / tail_mean for leg in _SHARES
        },
        changes=len(window),
        vol_ewma=sigma_next,
        backtest=var_backtest(changes["crush"]),
        tail_scenarios=int(tail.sum()),
        bias=_sample_bias(table, changes),
        worst=worst_changes(changes["crush"]),
    )


def _combined_var(series: pd.Series) -> tuple[int, float]:
    """The filtered VaR, in USD per ``NOTIONAL_TONNES``, and its scenario count."""
    sigma, z = _standardised(series.to_numpy(dtype=float))
    dates = pd.DatetimeIndex(series.index)
    scenarios = z[_window_start(dates, pd.Timestamp(dates[-1])) :]
    if len(scenarios) < MIN_CHANGES or not sigma[-1] > 0:
        return len(scenarios), math.nan
    return len(scenarios), NOTIONAL_TONNES * _fhs_var(float(sigma[-1]), scenarios)


def _excluded_hits(clean: pd.Series, excluded: pd.Series) -> np.ndarray:
    """For each left-out change, whether it falls below minus the clean VaR for its day.

    A change dated ``t`` takes the forecast the clean model makes for the next clean
    change on or after ``t``, from the scenarios before it only. Changes with under
    ``MIN_CHANGES`` scenarios are skipped.
    """
    dates = pd.DatetimeIndex(clean.index)
    sigma, z = _standardised(clean.to_numpy(dtype=float))
    hits: list[bool] = []
    for t, change in zip(pd.DatetimeIndex(excluded.index), excluded.to_numpy(), strict=True):
        j = int(dates.searchsorted(t, side="left"))
        if j == 0:
            continue
        start = _window_start(dates, dates[j - 1])
        if j - start < MIN_CHANGES:
            continue
        # sigma has one entry more than the changes: sigma[len] is the next forecast.
        hits.append(bool(change < -_fhs_var(float(sigma[j]), z[start:j])))
    return np.array(hits, dtype=bool)


def _sample_bias(table: pd.DataFrame, changes: pd.DataFrame) -> SampleBias | None:
    if changes.empty:
        return None
    date = pd.Timestamp(changes.index[-1])
    start = date - pd.DateOffset(years=LOOKBACK_YEARS)
    left_out = excluded_changes(table)
    kept = changes["crush"].loc[start:date]
    excluded = left_out.loc[start:date]
    if len(kept) < 2 or len(excluded) < 2:
        return None
    # Cut at the clean date, so the combined VaR is measured on the same day as the clean one.
    both = pd.concat([changes["crush"], left_out]).sort_index().loc[:date]
    combined, var = _combined_var(both)
    hits = _excluded_hits(changes["crush"], excluded)
    return SampleBias(
        kept=len(kept),
        excluded=len(excluded),
        kept_sd=float(kept.std()),
        excluded_sd=float(excluded.std()),
        combined_changes=combined,
        combined_var=var,
        excluded_tested=len(hits),
        excluded_exceptions=int(hits.sum()),
        excluded_kupiec_p=kupiec_pof(int(hits.sum()), len(hits)) if len(hits) else math.nan,
    )


def sample_bias(table: pd.DataFrame) -> SampleBias | None:
    """Clean against excluded in-contract changes, over the risk window.

    Both samples cover the ``LOOKBACK_YEARS`` to the last clean change. The same filtered
    VaR is also run on the two together, with the EWMA over the combined series, and
    the clean VaR is tested on the left-out changes. Returns None when either sample
    has fewer than 2 changes.
    """
    return _sample_bias(table, clean_changes(table))


def percent_shares(shares: Mapping[str, float]) -> dict[str, int]:
    """Shares as whole percents that add up to their rounded total.

    Largest-remainder rounding: each percent is floored, and the points still missing go
    to the largest remainders, so ES shares that sum to 1 always show as 100%.
    """
    percents = {leg: share * 100 for leg, share in shares.items()}
    shown = {leg: math.floor(p) for leg, p in percents.items()}
    missing = round(sum(percents.values())) - sum(shown.values())
    by_remainder = sorted(percents, key=lambda leg: percents[leg] - shown[leg], reverse=True)
    for leg in by_remainder[:missing]:
        shown[leg] += 1
    return shown

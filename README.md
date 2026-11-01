# Commodity Backtester

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://commodity-backtester.streamlit.app/)
[![CI](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive backtester and ratio monitor for CBOT grain and NYMEX energy futures, built with Python and Streamlit. It runs on daily closes from Yahoo Finance continuous front-month series (from 2000). The app answers one question without a click: what is this rule saying now, in this market, and how far can that number be trusted?

![Backtest view: position, round trips, time in market, unadjusted P&L and the price chart](docs/screenshot.png)

> **Data source:** continuous front-month futures via Yahoo Finance (`ZS=F`, `ZC=F`, `KE=F`, `ZM=F`, `ZL=F`, `HO=F`, `CL=F`).
> See [Known Limitations](#known-limitations) for important notes on roll adjustment.

> [!WARNING]
> **Known issues under revalidation ([v0.3 correctness milestone](https://github.com/rossi-diego/commodity-backtester/milestone/1)).** P&L in the app is indicative and is labelled that way. Do not rely on these numbers until the milestone closes:
> - Sharpe, Sortino, max drawdown and Calmar are computed on realised P&L only, without daily mark-to-market ([#2](https://github.com/rossi-diego/commodity-backtester/issues/2)); Sortino uses the wrong dispersion measure ([#9](https://github.com/rossi-diego/commodity-backtester/issues/9)).
> - Partly fixed: each rule now trades its instrument's own series, so it no longer fills at missing prices. The data-quality report is still pending ([#3](https://github.com/rossi-diego/commodity-backtester/issues/3)).
> - Contract rolls and expiry-day moves are booked as P&L ([#6](https://github.com/rossi-diego/commodity-backtester/issues/6)), so the sign of a result can flip against a ledger with roll days removed.
> - Energy contract P&L is about 1% off for heating oil and crude ([#7](https://github.com/rossi-diego/commodity-backtester/issues/7)).
> - Hidden from the app, not fixed: the VaR calculation ([#8](https://github.com/rossi-diego/commodity-backtester/issues/8)) and stop-loss and take-profit, which can fire on roll gaps and re-enter immediately ([#11](https://github.com/rossi-diego/commodity-backtester/issues/11)).

---

## Features

- **First screen with results** - opens on Soybean, 12M momentum, 10 years, with no button to press
- **Three rules** - 12M momentum, Donchian 20/10 and buy & hold as a fixed reference
- **Ratio monitor** - level, trailing 1-year z-score and trailing 5-year percentile for four curated pairs. No trades and no P&L
- **Custom windows** - fixed periods by default, or any start and end date to study 2008 or 2020
- **Round-trip P&L** - in USD using CME/CBOT contract specifications, with commission and slippage on by default
- **T+1 execution model** - signals generated at close T are filled at close T+1; no same-bar look-ahead
- **Trade log** - per-fill log with CSV download
- **Parquet data cache** - validated before every write, replaced atomically, with a fallback to the existing cache when Yahoo fails
- **Works on a phone** - no sidebar, and the results appear without opening a menu

---

## Installation

### Quickstart (app only)

```bash
git clone https://github.com/rossi-diego/commodity-backtester.git
cd commodity-backtester
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Developer setup (includes tests, linting, type-checking)

```bash
pip install -r requirements-dev.txt
pytest                                  # run test suite
ruff format --check src/ tests/ app.py  # formatting
ruff check src/ tests/ app.py           # lint
mypy src/                               # type-check
```

---

## Project Structure

```
commodity-backtester/
├── app.py                      # Streamlit UI only
├── .streamlit/config.toml      # Theme
├── docs/screenshot.png         # README screenshot
├── src/
│   ├── __init__.py
│   ├── catalog.py              # The rules offered in the app, with their copy
│   ├── config.py               # Paths, cache slack, annualisation, risk-free rate
│   ├── constants.py            # Continuous futures tickers, contract specs, ratio pairs
│   ├── data_loader.py          # Yahoo Finance fetch, response validation, Parquet cache
│   ├── monitor.py              # Ratio series, rolling statistics, snapshot
│   ├── runner.py               # One backtest for the app: window, engine, P&L, summary
│   ├── strategy.py             # Signal generators + trade state machine (T+1 execution)
│   ├── utils.py                # P&L calculation, summary and performance metrics
│   └── visualization.py        # Plotly figure builders, no UI imports
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic price data, no network)
│   ├── test_app.py             # End-to-end with Streamlit AppTest, no network
│   ├── test_catalog.py         # Rule catalogue matches the engine
│   ├── test_constants.py       # Contract reference data integrity
│   ├── test_data_loader.py     # Cache rules and response validation (mocked)
│   ├── test_monitor.py         # Ratio and rolling statistics, by hand
│   ├── test_runner.py          # Window handling, mark at window end, Sharpe gate
│   ├── test_strategy.py        # Signals, T+1 execution, truncation invariance
│   ├── test_utils.py           # P&L correctness (CME-verified), summary, metrics
│   └── test_visualization.py   # Figure content, marker limit, band speed
├── .github/workflows/ci.yml    # CI: lint → typecheck → test
├── pyproject.toml              # Ruff, mypy, pytest configuration
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Dev/test dependencies
```

---

## Rules

Each rule is long or flat, one contract, with fixed parameters. They are baselines to exercise the engine, not claims of edge. Neither active rule has been shown to beat holding the front month over 2000 to 2026: the difference is within noise. Every result in the app is in-sample.

| Rule | Description |
|---|---|
| **12M momentum** | Long when the price is above its level 252 sessions ago, checked at each month end. Otherwise flat. The signal uses the price difference, so a negative price cannot flip it. |
| **Donchian 20/10** | Long on a close above the prior 20-session high. Flat on a close below the prior 10-session low. |
| **Buy & hold** | Reference: long from the first fill to the window end, at the same costs. No comparison is computed yet. |

### Ratio monitor

`Soybean / Corn`, `KC Wheat / Corn`, `Soy Oil / Soy Meal` and `Heating Oil / WTI`, in USD per metric ton. The z-score uses 252 sessions and the percentile 1260, always on the history up to the window end; the period only changes what is charted. Sessions where either price is not positive are dropped. Ratios use unadjusted closes, so contract rolls cause jumps.

---

## What the App Shows

| Shown | Note |
|---|---|
| **Position now** (or **Position at end**, if the window ends before the last close) | Long or flat, since the last fill |
| **Round trips** and fills | Completed buy and sell pairs |
| **Time in market** | Share of window sessions in a long position |
| **Unadjusted P&L** | Per contract, on unadjusted prices, rolls included, open position marked at the window end |
| **Sharpe**, in the `Indicative statistics` expander | Only with at least 10 round trips and not for buy & hold. Below that it is mechanical: for one open position it reduces to `±√(252/n)` |

`Sharpe = (mean_daily_return − rf/252) / σ_daily × √252`, with daily P&L divided by one contract's notional at the first positive price of the window. The daily series still contains realised P&L only ([#2](https://github.com/rossi-diego/commodity-backtester/issues/2)), so the Sharpe differs from one on a daily mark-to-market. In a static study of random custom windows with 10 or more round trips, that gap had a median of 0.03, a 95th percentile of 0.23 and a maximum of 0.48, with no sign flips, so treat it as indicative and not as bounded.

**Computed but not shown until the daily ledger ships:** Sortino, Calmar, VaR, profit factor, recovery factor, win rate and max drawdown. On a realised-only series, buy & hold has a drawdown of $0, so showing it would mislead. The code and its tests stay in `src/utils.py`.

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **T+1 next-bar execution** | Signals are generated from the close of day T and filled at the close of day T+1. This eliminates same-bar look-ahead bias, where the closing price that fires the signal would simultaneously be used as the fill price. |
| **Signals on history, then slice** | Signals run on the full history up to the window end and are then cut to the window. A one-year window therefore has a year of momentum history behind it, instead of starting blind. |
| **Per-instrument series** | Each rule runs on `df[market].dropna()`, not on the shared seven-market frame. A holiday in another market cannot create a NaN that blinds a rolling window for weeks or produces a fill at a missing price. |
| **P&L is not the headline** | The first metric is the position. On unadjusted prices, the sign of the P&L can disagree with a ledger that removes roll days, so the number is labelled `Unadjusted P&L`, shown last, and comes with a warning. |
| **Capital-normalised returns** | Daily PnL is divided by the initial contract notional before computing Sharpe. This avoids the ill-defined `pct_change()` of a cumulative PnL series that starts at zero. |
| **USD/metric-ton normalisation** | Commodities trade in incompatible units (cents/bushel, USD/gallon, USD/barrel). Converting everything to $/MT using CME/CBOT contract specs makes cross-commodity ratios economically meaningful. |
| **Validated, atomic cache** | A download replaces `data/yfinance_raw.parquet` only if each column has at least 90% of its cached observations and does not end earlier. A partial Yahoo response cannot overwrite good data, and a failed refresh serves the existing cache, labelled as possibly out of date. |

---

## Known Limitations

These are deliberate simplifications for a portfolio project. A production system would address each one.

| Limitation | Detail |
|---|---|
| **No roll adjustment on continuous tickers** | Yahoo Finance `=F` tickers roll to the nearest front-month contract but apply **no Panama, proportional, or ratio back-adjustment**. Each quarterly roll introduces a price discontinuity (typically 5–20+ cents/bushel for grains), which is booked as P&L and also moves ratios. A production system would use a properly back-adjusted continuous series (e.g., from Refinitiv, Barchart, or a self-built Panama-adjusted series). |
| **Close-to-close T+1 execution** | Fills are modelled at the next day's closing price. In practice, a systematic strategy would execute at the open or use a VWAP algo, often improving on the close-to-close assumption. |
| **Flat position sizing** | Each signal trades exactly one contract. A real system would size positions by volatility target (e.g., 1 % of AUM per unit of daily volatility). |
| **No margin or funding cost** | The PnL engine deducts commission and slippage but not overnight financing or initial margin requirements. |
| **Fixed parameters** | The rules have no editable parameters until the daily ledger ships, because parameter searches on this data would be curve fitting. |

---

## Architecture

```
Yahoo =F (7 tickers)
   |  validated per column, written atomically, falls back to the cache on failure
   v
data/yfinance_raw.parquet  (local, gitignored)
   |  load_prices() -> LoadResult, cached for one hour by the app
   v
series per instrument: df[market].loc[:end].dropna()
   |                                   |
   v                                   v
src/runner.py                      src/monitor.py
 signal on history -> slice         ratio on positive prices only
 trades (T+1), P&L on the window    z-score 252, percentile 1260
 summary, gated Sharpe              snapshot
   |                                   |
   +----------------+------------------+
                    v
src/visualization.py (pure Plotly figures) -> app.py (Streamlit UI)
```

Logs use `key=value` fields, for example `event=data_load source=cache rows=6711 as_of=2026-10-30`, `event=cache_rejected problems=...` and `event=data_load_failed reason=...`.

---

## Contact

Developed by **Diego Rossi** — [GitHub](https://github.com/rossi-diego) · [LinkedIn](https://www.linkedin.com/in/diego-rossi-santanna/)

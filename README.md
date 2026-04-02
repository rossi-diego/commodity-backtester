# Commodity Backtester

[![CI](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![View App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://commodity-backtester.streamlit.app/)

An interactive backtesting framework for commodity trading strategies, built with Python and Streamlit. Test 25 years of historical performance using real futures data, visualise trade signals, and evaluate risk-adjusted returns with institutional-standard metrics.

> **Data source:** continuous front-month futures via Yahoo Finance (`ZS=F`, `ZC=F`, `KE=F`, `ZM=F`, `ZL=F`, `HO=F`, `CL=F`).
> See [Known Limitations](#known-limitations) for important notes on roll adjustment.

---

## Features

- **Five strategies** — Ratio (spread trigger), Bollinger Mean Reversion, Dual-EMA Momentum, Channel Breakout, MACD Crossover
- **Correct PnL engine** — round-trip PnL in USD using CME/CBOT contract specifications; optional commission and slippage
- **Risk analytics** — Sharpe, Sortino, Calmar, Profit Factor, Recovery Factor, VaR 95%, Max Drawdown — all computed on a capital-normalised equity curve
- **T+1 execution model** — signals generated at close T are filled at close T+1; no same-bar look-ahead
- **Interactive charts** — price + signal markers, ratio behaviour, cumulative PnL (Plotly)
- **Vectorised engine** — entire signal generation in numpy/pandas; 25-year daily backtest < 100 ms
- **Parquet data cache** — first run fetches from Yahoo Finance and writes a local cache; subsequent runs are instant
- **Modular architecture** — each concern isolated in its own module; easy to add new strategies

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
pytest          # run test suite
ruff check src/ # lint
mypy src/       # type-check
```

---

## Project Structure

```
commodity-backtester/
├── app.py                      # Streamlit entry point (UI only, no calculations)
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, annualisation factor, risk-free rate
│   ├── constants.py            # Continuous futures tickers, contract specs
│   ├── data_loader.py          # Yahoo Finance fetch + Parquet cache
│   ├── strategy.py             # Vectorised backtest engine (T+1 execution)
│   ├── utils.py                # PnL calculation + performance metrics
│   └── visualization.py        # Plotly chart builders
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic price data, no network)
│   ├── test_strategy.py        # Signal generation, T+1 execution, RSI, all strategies
│   ├── test_utils.py           # PnL correctness (CME-verified), Sharpe/Calmar finite
│   └── test_constants.py       # Contract reference data integrity
├── notebooks/                  # Exploratory analysis
├── .github/workflows/ci.yml    # CI: lint → typecheck → test
├── pyproject.toml              # Ruff, mypy, pytest configuration
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Dev/test dependencies
```

---

## Supported Strategies

| Strategy | Description |
|---|---|
| **Ratio** | Long commodity A when `(price_A / price_B)` in $/MT terms falls below an entry threshold; exit above an exit threshold. Ratio is normalised using CME/CBOT contract specs so cross-commodity comparisons are unit-consistent. |
| **Mean Reversion** | Bollinger Bands — long when price closes below the lower band (rolling mean − N σ); exit above the upper band. |
| **Momentum** | Dual EMA crossover with Wilder's RSI filter. Buys when fast EMA > slow EMA and RSI is between the oversold and overbought levels; exits on MA reversal or RSI overbought. |
| **Breakout** | Channel breakout — buy when price closes above the rolling N-day high by a confirmation threshold; exit below resistance. |
| **MACD** | MACD line crossover — long on bullish histogram sign change (MACD crosses above signal line); exit on bearish cross. |

---

## Performance Metrics

All metrics are computed on a **capital-normalised daily return series** (`daily_pnl / initial_contract_notional`), ensuring dimensionally consistent Sharpe, Sortino, and Calmar ratios.

| Metric | Formula |
|---|---|
| **Sharpe Ratio** | `(mean_daily_return − rf/252) / σ_daily × √252` |
| **Sortino Ratio** | `(mean_daily_return − rf/252) / σ_downside × √252` |
| **Calmar Ratio** | `annualised_return / max_drawdown_pct` (both dimensionless) |
| **Profit Factor** | `gross_profit / gross_loss` (per completed round trip) |
| **Recovery Factor** | `total_pnl / max_drawdown_USD` |
| **VaR 95%** | Historical log-return percentile × contract notional at trade date (no look-ahead) |
| **Max Drawdown** | Peak-to-trough in USD on cumulative realised PnL |

`rf` = `config.RISK_FREE_RATE` (default `0.0`; set to e.g. `0.04` for excess-return Sharpe).

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **T+1 next-bar execution** | Signals are generated from the close of day T and filled at the close of day T+1. This eliminates same-bar look-ahead bias, where the closing price that fires the signal would simultaneously be used as the fill price. |
| **Capital-normalised returns** | Daily PnL is divided by the initial contract notional (one lot at first-day price) before computing Sharpe, Sortino, and Calmar. This avoids the ill-defined `pct_change()` of a cumulative PnL series that starts at zero. |
| **USD/metric-ton normalisation** | Commodities trade in incompatible units (cents/bushel, USD/gallon, USD/barrel). Converting everything to $/MT using CME/CBOT contract specs makes cross-commodity ratios economically meaningful. Conversion factors are derived from official contract specifications. |
| **Wilder's RSI** | The Momentum strategy uses `ewm(com=period−1, adjust=False)` for gain/loss smoothing — matching TA-Lib, Bloomberg, and TradingView. The naive `rolling().mean()` approximation miscalibrates overbought/oversold thresholds. |
| **Per-trade rolling VaR** | VaR at each trade date uses only log-returns available up to that date, preventing future return data from leaking into early-backtest risk estimates. |
| **Parquet cache with range validation** | The first run writes `data/yfinance_raw.parquet`. Subsequent calls check whether the cache covers the requested date range before re-fetching, making the app usable offline and reducing Yahoo Finance dependency. |
| **Streamlit as a presentation layer only** | All business logic lives in `src/`. The app imports and calls; it contains no calculations. This keeps the core library testable, importable, and framework-agnostic. |

---

## Known Limitations

These are deliberate simplifications for a portfolio project. A production system would address each one.

| Limitation | Detail |
|---|---|
| **No roll adjustment on continuous tickers** | Yahoo Finance `=F` tickers roll to the nearest front-month contract but apply **no Panama, proportional, or ratio back-adjustment**. Each quarterly roll introduces a price discontinuity (typically 5–20+ cents/bushel for grains). For the Ratio strategy this can produce spurious signals around roll dates. A production system would use a properly back-adjusted continuous series (e.g., from Refinitiv, Barchart, or a self-built Panama-adjusted series). |
| **Single-leg "ratio" strategy** | The Ratio strategy goes long commodity A when the ratio is cheap. It does **not** simultaneously short commodity B. A true spread trade is delta-neutral across both legs; this implementation is a directional bet on A, triggered by the ratio level. |
| **Close-to-close T+1 execution** | Fills are modelled at the next day's closing price. In practice, a systematic strategy would execute at the open or use a VWAP algo, often improving on the close-to-close assumption. |
| **Flat position sizing** | Each signal trades exactly one contract. A real system would size positions by volatility target (e.g., 1 % of AUM per unit of daily volatility). |
| **No margin or funding cost** | The PnL engine deducts commission and slippage but not overnight financing or initial margin requirements. |
| **Historical VaR only** | VaR uses the empirical return distribution with no fat-tail adjustment (Cornish-Fisher, Monte Carlo, or EVT). |

---

## Architecture

```
app.py (Streamlit UI — presentation only)
    │
    ├── src/config.py       ─── paths, annualisation, risk-free rate
    ├── src/data_loader.py  ─── yfinance → Parquet cache → pd.DataFrame
    ├── src/constants.py    ─── tickers, contract sizes (MT), unit conversions
    ├── src/strategy.py     ─── vectorised T+1 signal generation
    ├── src/utils.py        ─── PnL engine + capital-normalised metrics
    └── src/visualization.py─── Plotly figure builders
```

---

## Contact

Developed by **Diego Rossi** — [GitHub](https://github.com/rossi-diego) · [LinkedIn](https://www.linkedin.com/in/diego-rossi-santanna/)

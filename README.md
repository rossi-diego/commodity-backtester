# Commodity Backtester

[![CI](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![View App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://commodity-backtester.streamlit.app/)

An interactive backtesting framework for commodity spread strategies, built with Python and Streamlit. Test historical performance using real futures data, visualise trade signals, and evaluate risk-adjusted returns with institutional-grade metrics.

> **Data source:** continuous futures via Yahoo Finance (`ZS=F`, `ZC=F`, `KE=F`, `ZM=F`, `ZL=F`, `HO=F`, `CL=F`) — tickers that auto-roll and never expire.

---

## Features

- **Ratio strategy** — long when the metric-ton price ratio crosses below an entry threshold; exit when it recovers above an exit threshold
- **Performance analytics** — Sharpe, Sortino, Calmar, Profit Factor, Recovery Factor, VaR 95%, Max Drawdown
- **Interactive charts** — price + signal markers, ratio behaviour, cumulative PnL (Plotly)
- **Vectorised engine** — entire signal generation in numpy/pandas; 20-year daily backtest < 100 ms
- **Modular architecture** — each concern isolated in its own module; easy to add new strategies

---

## Screenshots

> _Run `streamlit run app.py` locally and add a screenshot here._

| Strategy Results | Ratio Chart |
|---|---|
| _(screenshot)_ | _(screenshot)_ |

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
├── app.py                      # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── constants.py            # Continuous futures tickers, contract specs
│   ├── data_loader.py          # Yahoo Finance data acquisition
│   ├── strategy.py             # Vectorised backtest engine
│   ├── utils.py                # PnL calculation + performance metrics
│   ├── visualization.py        # Plotly chart builders
│   └── config.py               # File path constants
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic price data)
│   ├── test_strategy.py        # Signal generation tests
│   ├── test_utils.py           # PnL and metrics tests
│   └── test_constants.py       # Contract reference data integrity
├── notebooks/                  # Exploratory analysis
├── .github/workflows/ci.yml    # CI: lint → typecheck → test
├── pyproject.toml              # Ruff, mypy, pytest configuration
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Dev/test dependencies
```

---

## Supported Strategies

| Strategy | Status | Description |
|---|---|---|
| Ratio | ✅ Live | Long commodity A when `(price_A / price_B)` in $/MT terms falls below entry threshold |
| Mean Reversion | ✅ Live | Bollinger Bands — long when price drops > N std devs below rolling mean |
| Momentum | ✅ Live | Dual MA crossover with RSI confirmation filter |
| Breakout | ✅ Live | Channel breakout — buy on resistance breach with threshold confirmation |
| MACD | ✅ Live | MACD line crossover — buy on bullish signal cross |

---

## Performance Metrics

| Metric | Description |
|---|---|
| Sharpe Ratio | Annualised excess return / volatility on daily equity curve |
| Sortino Ratio | Annualised excess return / downside deviation |
| Calmar Ratio | Annualised return / max drawdown |
| Profit Factor | Gross profit / gross loss |
| Recovery Factor | Total profit / max drawdown |
| VaR 95% | Historical value-at-risk on daily log-returns |

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **Continuous futures tickers** (`ZS=F`, not `ZSN25.CBT`) | Expiring front-month contracts break silently after each expiry cycle. Continuous tickers auto-roll to the nearest active contract and work for any date range. |
| **USD/metric-ton normalisation** | Commodities trade in incompatible units (cents/bushel, USD/gallon, USD/barrel). Converting everything to $/MT using CME/CBOT contract specs makes cross-commodity ratios economically meaningful. |
| **Vectorised signal generation** | `numpy`/`pandas` operations over the full time-series — no `iterrows()` loops. A 25-year daily backtest completes in under 100 ms. |
| **Streamlit as a presentation layer only** | All business logic lives in `src/`. The app imports and calls; it contains no calculations. This keeps the core library testable, importable, and framework-agnostic. |
| **Parquet for caching** | Columnar format enables fast selective reads and reduces file size vs CSV, while `openpyxl` Excel export keeps results accessible to non-technical stakeholders. |

---

## Architecture

```
app.py (Streamlit UI)
    │
    ├── src/data_loader.py  ─── yfinance → pd.DataFrame
    ├── src/constants.py    ─── tickers, contract sizes, unit conversions
    ├── src/strategy.py     ─── vectorised signal generation
    ├── src/utils.py        ─── PnL engine + performance metrics
    └── src/visualization.py─── Plotly figure builders
```

---

## Contact

Developed by **Diego Rossi** — [GitHub](https://github.com/rossi-diego) · [LinkedIn](https://www.linkedin.com/in/diego-rossi-santanna/)

For questions or collaboration, feel free to reach out.

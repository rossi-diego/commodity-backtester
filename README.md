# Commodity Backtester

[![CI](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/commodity-backtester/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive backtester for rules on CBOT grain and NYMEX energy futures, built with Python and Streamlit. It runs on daily closes from Yahoo Finance continuous front-month series (from 2000), visualises trade signals and reports trade and risk statistics.

> **Data source:** continuous front-month futures via Yahoo Finance (`ZS=F`, `ZC=F`, `KE=F`, `ZM=F`, `ZL=F`, `HO=F`, `CL=F`).
> See [Known Limitations](#known-limitations) for important notes on roll adjustment.

> [!WARNING]
> **Known issues under revalidation ([v0.3 correctness milestone](https://github.com/rossi-diego/commodity-backtester/milestone/1)).** Several reported numbers are currently computed incorrectly. Do not rely on them until the milestone closes:
> - Sharpe, Sortino, max drawdown and Calmar are computed on realised P&L only, without daily mark-to-market ([#2](https://github.com/rossi-diego/commodity-backtester/issues/2)); Sortino uses the wrong dispersion measure ([#9](https://github.com/rossi-diego/commodity-backtester/issues/9)).
> - An open position is marked at the latest available price, not at the backtest end date ([#4](https://github.com/rossi-diego/commodity-backtester/issues/4)).
> - Trades can fill at missing prices on instrument-specific holidays ([#3](https://github.com/rossi-diego/commodity-backtester/issues/3)).
> - Contract rolls and expiry-day moves are booked as P&L ([#6](https://github.com/rossi-diego/commodity-backtester/issues/6)).
> - The breakout exit fires on almost every day ([#10](https://github.com/rossi-diego/commodity-backtester/issues/10)); stops can fire on roll gaps and re-enter immediately ([#11](https://github.com/rossi-diego/commodity-backtester/issues/11)).
> - Ratio-strategy default thresholds use the whole backtest window ([#5](https://github.com/rossi-diego/commodity-backtester/issues/5)).

---

## Features

- **Signal library (baselines)** — Ratio (ratio-triggered outright, single leg), Bollinger Mean Reversion, Dual-EMA Momentum, Channel Breakout, MACD Crossover
- **Round-trip P&L** — in USD using CME/CBOT contract specifications; commission and slippage on by default
- **Risk analytics** — Sharpe, Sortino, Calmar, Profit Factor, Recovery Factor, VaR 95%, Max Drawdown (see the known issues above)
- **T+1 execution model** — signals generated at close T are filled at close T+1; no same-bar look-ahead
- **Interactive charts** — price + signal markers, ratio behaviour, cumulative PnL (Plotly)
- **Vectorised signals** — signal generation in numpy/pandas; the trade state machine is a plain Python loop
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
├── app.py                      # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, annualisation factor, risk-free rate
│   ├── constants.py            # Continuous futures tickers, contract specs
│   ├── data_loader.py          # Yahoo Finance fetch + Parquet cache
│   ├── strategy.py             # Signal generators + trade state machine (T+1 execution)
│   ├── utils.py                # PnL calculation + performance metrics
│   └── visualization.py        # Plotly chart builders
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic price data, no network)
│   ├── test_strategy.py        # Signal generation, T+1 execution, RSI, all strategies
│   ├── test_utils.py           # PnL correctness (CME-verified), Sharpe/Calmar finite
│   └── test_constants.py       # Contract reference data integrity
├── .github/workflows/ci.yml    # CI: lint → typecheck → test
├── pyproject.toml              # Ruff, mypy, pytest configuration
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Dev/test dependencies
```

---

## Signal Library (baselines)

These are classic technical rules, included as baselines to exercise the engine, not as claims of edge. Every result shown in the app is in-sample unless you hold out a later period yourself, so treat parameter searches as curve fitting.

| Strategy | Description |
|---|---|
| **Ratio** | Long commodity A when `(price_A / price_B)` in $/MT terms falls below an entry threshold; exit above an exit threshold. Ratio is normalised using CME/CBOT contract specs so cross-commodity comparisons are unit-consistent. |
| **Mean Reversion** | Bollinger Bands — long when price closes below the lower band (rolling mean − N σ); exit above the upper band. |
| **Momentum** | Dual EMA crossover with Wilder's RSI filter. Buys when fast EMA > slow EMA and RSI is between the oversold and overbought levels; exits on MA reversal or RSI overbought. |
| **Breakout** | Channel breakout — buy when price closes above the rolling N-day high by a confirmation threshold; exit below resistance. |
| **MACD** | MACD line crossover — long on bullish histogram sign change (MACD crosses above signal line); exit on bearish cross. |

---

## Performance Metrics

Risk metrics are normalised by the initial contract notional (`daily_pnl / initial_contract_notional`). The daily P&L series currently contains realised P&L only; see [#2](https://github.com/rossi-diego/commodity-backtester/issues/2).

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
app.py (Streamlit UI)
    │
    ├── src/config.py       ─── paths, annualisation, risk-free rate
    ├── src/data_loader.py  ─── yfinance → Parquet cache → pd.DataFrame
    ├── src/constants.py    ─── tickers, contract sizes (MT), unit conversions
    ├── src/strategy.py     ─── T+1 signal generation + trade state machine
    ├── src/utils.py        ─── PnL engine + capital-normalised metrics
    └── src/visualization.py─── Plotly figure builders
```

---

## Contact

Developed by **Diego Rossi** — [GitHub](https://github.com/rossi-diego) · [LinkedIn](https://www.linkedin.com/in/diego-rossi-santanna/)

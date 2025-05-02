# 🧠 Commodity Trading Strategy Backtester

A modular and extensible Python project designed for backtesting commodity trading strategies, starting with spread-based logic and prepared for future expansion into other models and data sources.

## 🚀 Features

- 📥 Market data collection via Yahoo Finance
- ⚖️ Spread calculation between multiple commodities
- 🧠 Strategy simulation with customizable entry/exit thresholds
- 💰 Profit & Loss computation including open positions (MTM)
- 📈 Interactive visualizations with Matplotlib
- 🧩 Modular architecture ready for integration with Streamlit

## 🗂 Project Structure

```
project-root/
├── notebooks/           # Interactive notebooks for exploration and backtesting
├── src/                 # Main source code
│   ├── config.py            # Directory paths and file references
│   ├── constants.py         # Commodity metadata (tickers, contract sizes, unit converters)
│   ├── data_loader.py       # Data ingestion (e.g., Yahoo Finance)
│   ├── strategy.py          # Strategy execution (initial: spread-based)
│   ├── utils.py             # PnL calculations, performance metrics
│   ├── visualization.py     # Backtest charts (price, spread, PnL)
├── requirements.txt     # Project dependencies
```

## ⚙️ Requirements

- Python 3.11+

## ▶️ How to Run

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 📘 Notebooks

Explore the strategy in action using the interactive notebooks inside the `notebooks/` directory. The main one is:

- `02-backtest.ipynb` – Full backtest pipeline and chart generation

---

Built by Diego Rossi – a personal initiative to model and evaluate commodity trading strategies. 💼📉📊
# 📈 Commodity Backtester

[![View App](https://img.shields.io/badge/🚀%20Open%20App%20on-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://commodity-backtester.streamlit.app/)


This project provides an interactive environment for backtesting trading strategies focused on commodities. Users can test historical performance using customizable parameters, review detailed metrics, and visualize trade signals on charts.

> 💡 While the initial implementation supports spread-based strategies, the architecture is designed to accommodate various other trading models in the future.

---

## 🔍 Features

- Interactive backtest tool for commodity trading strategies  
- Modular and extensible design to support multiple strategies (e.g. Spread, Mean Reversion, Momentum)  
- Clean, responsive interface built with **Streamlit**  
- Historical trade metrics: Realized PnL, Drawdown, Sharpe Ratio, Win Rate  
- Interactive trade signal charts with buy/sell markers  
- Descriptive statistics of selected commodity prices and spreads

---

## 📊 Data Sources

The app currently retrieves historical price data using the **Yahoo Finance API** via `yfinance`.  

> 📌 Future versions will include support for additional data providers (e.g. **Quandl**, **USDA**, **Bloomberg**, **Eikon**) to enable broader market coverage and integrate fundamental data (e.g. crush margins, export volumes, stocks/use).

---

## 🛠️ Installation

Clone the repository and install the requirements:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
├── app.py                # Streamlit app interface
├── src/
│   ├── strategy.py       # Trading strategy logic
│   ├── utils.py          # PnL and metrics calculations
│   ├── constants.py      # Contract sizes, naming, conversion factors
│   ├── data_loader.py    # Data loading logic (Yahoo Finance)
│   ├── visualization.py  # Plotting and chart rendering
├── requirements.txt
├── README.md
```

---

## 🧠 Purpose

This project was built to simulate and evaluate commodity-based trading strategies in a flexible and visual environment. It serves both as a practical tool for backtesting and as a growing foundation for more advanced algorithmic research in commodities.

---

## 📬 Contact

Developed by **Diego Rossi**  
For questions, suggestions, or collaboration opportunities, feel free to reach out.
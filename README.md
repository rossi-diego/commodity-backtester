# Commodity Backtester

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://commodity-backtester.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional-grade backtesting framework for commodity trading strategies. Built with Python, featuring an interactive Streamlit interface and comprehensive performance analytics.

![Backtester Screenshot](https://via.placeholder.com/800x400?text=Commodity+Backtester+Screenshot)

## Features

- **Ratio-Based Trading Strategy**: Trade commodities based on price ratio analysis in metric ton terms
- **Real-Time Data**: Automatic data retrieval from Yahoo Finance
- **Interactive Visualization**: Dynamic charts with buy/sell signals using Plotly
- **Comprehensive Metrics**: P&L analysis, Sharpe Ratio, Max Drawdown, VaR, Win Rate
- **Mark-to-Market**: Track open positions with MTM adjustments
- **Modular Architecture**: Clean, extensible codebase ready for new strategies

## Supported Commodities

| Commodity | Exchange | Ticker |
|-----------|----------|--------|
| Soybean | CBOT | ZSN25.CBT |
| Corn | CBOT | ZCN25.CBT |
| Wheat | CBOT | KEN25.CBT |
| Soybean Meal | CBOT | ZMN25.CBT |
| Soybean Oil | CBOT | ZLN25.CBT |
| Heating Oil | NYMEX | HON25.NYM |
| Crude Oil | NYMEX | CLN25.NYM |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rossi-diego/commodity-backtester.git
cd commodity-backtester

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Project Structure

```
commodity-backtester/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md
│
├── src/                    # Core modules
│   ├── __init__.py         # Package exports
│   ├── config.py           # Configuration & paths
│   ├── constants.py        # Commodity specifications
│   ├── data_loader.py      # Yahoo Finance data retrieval
│   ├── strategy.py         # Trading strategy implementations
│   ├── utils.py            # P&L & performance calculations
│   └── visualization.py    # Plotly chart generation
│
├── data/                   # Data cache (auto-generated)
├── tests/                  # Unit tests
└── notebooks/              # Jupyter notebooks for exploration
```

## How It Works

### Ratio Trading Strategy

The ratio strategy compares two commodities by converting their prices to a common unit (metric tons). Trading signals are generated based on the ratio crossing predefined thresholds:

1. **Entry Signal (Buy)**: When the ratio drops below the entry threshold
2. **Exit Signal (Sell)**: When the ratio rises above the exit threshold

```
Ratio = (Commodity_A × Conversion_A) / (Commodity_B × Conversion_B)
```

### Performance Metrics

| Metric | Description |
|--------|-------------|
| Total Profit | Sum of realized P&L + MTM adjustment |
| Win Rate | Percentage of profitable trades |
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Max Drawdown | Largest peak-to-trough decline |
| VaR (95%) | Value at Risk at 95% confidence |
| Mean Trade Duration | Average holding period |

## API Reference

### Data Loading

```python
from src.data_loader import yahoo_quotes

df, first_date = yahoo_quotes(
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Running a Backtest

```python
from src.strategy import backtest
from src.constants import tons_conversion, contract_sizes

df_trades, position_open = backtest(
    backtest_strategy="ratio",
    start_date=start_date,
    end_date=end_date,
    df=price_data,
    up_exit=1.05,
    down_entry=0.95,
    commodity_chosen="Soybean",
    commodity_ratio="Corn",
    tons_conversion=tons_conversion,
    contract_size=contract_sizes["Soybean"]
)
```

### Calculating Performance

```python
from src.utils import pnl_trades, backtest_performance

df_trades_final, mtm_trade = pnl_trades(
    df_trades=df_trades,
    df_prices=df,
    commodity_chosen="Soybean",
    tons_conversion=tons_conversion,
    contract_size=136.0,
    position_open=position_open
)

metrics = backtest_performance(
    df_trades_final,
    df,
    mtm_trade,
    contract_size=136.0,
    tons_conversion=tons_conversion,
    commodity_chosen="Soybean",
    position_open=position_open
)
```

## Development

### Adding a New Commodity

Edit `src/constants.py`:

```python
COMMODITIES["NEW_TICKER"] = CommoditySpec(
    ticker="NEW_TICKER",
    name="New Commodity",
    exchange="EXCHANGE",
    contract_size=100.0,
    tons_conversion=1.0,
    unit="$/unit"
)
```

### Adding a New Strategy

1. Add strategy type to `StrategyType` enum in `src/strategy.py`
2. Implement `_execute_<strategy>_strategy()` function
3. Add case handling in `backtest()` function

## Roadmap

- [ ] Mean Reversion strategy implementation
- [ ] Multi-leg spread trading
- [ ] Database integration for historical storage
- [ ] API endpoint for programmatic access
- [ ] Machine learning signal generation
- [ ] Portfolio optimization module

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Diego Rossi**

- GitHub: [@rossi-diego](https://github.com/rossi-diego)
- LinkedIn: [Diego Rossi](https://linkedin.com/in/diego-rossi)

---

*Built with Python, Streamlit, and Plotly*

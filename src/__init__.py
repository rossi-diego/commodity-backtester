"""
Commodity Backtester - Source Package

A modular backtesting framework for commodity trading strategies.

Modules:
    - config: Application configuration and paths
    - constants: Commodity specifications and metadata
    - data_loader: Yahoo Finance data retrieval
    - strategy: Trading strategy implementations
    - utils: P&L calculations and performance metrics
    - visualization: Interactive charting with Plotly
"""

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    TRADING_DAYS_PER_YEAR,
    VAR_CONFIDENCE_LEVEL,
)

from src.constants import (
    COMMODITIES,
    CommoditySpec,
    commodities_dict,
    contract_sizes,
    tons_conversion,
    tickers,
)

from src.data_loader import yahoo_quotes, DataLoaderError
from src.strategy import backtest, BacktestError, StrategyType
from src.utils import pnl_trades, backtest_performance, strategy_describe
from src.visualization import backtest_charts

__version__ = "1.0.0"
__author__ = "Diego Rossi"

__all__ = [
    # Config
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "TRADING_DAYS_PER_YEAR",
    "VAR_CONFIDENCE_LEVEL",
    # Constants
    "COMMODITIES",
    "CommoditySpec",
    "commodities_dict",
    "contract_sizes",
    "tons_conversion",
    "tickers",
    # Data
    "yahoo_quotes",
    "DataLoaderError",
    # Strategy
    "backtest",
    "BacktestError",
    "StrategyType",
    # Utils
    "pnl_trades",
    "backtest_performance",
    "strategy_describe",
    # Visualization
    "backtest_charts",
]

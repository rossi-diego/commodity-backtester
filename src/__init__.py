"""
Commodity Backtester Pro - Source Package

A professional-grade modular backtesting framework for commodity trading strategies.

Modules:
    - config: Application configuration and paths
    - constants: Commodity specifications and metadata
    - data_loader: Yahoo Finance data retrieval with caching
    - strategy: Trading strategy implementations (5 strategies)
    - utils: P&L calculations, performance metrics, and risk analysis
    - visualization: Interactive charting with Plotly

Version: 2.0.0
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

from src.data_loader import (
    yahoo_quotes,
    DataLoaderError,
    clear_cache,
    get_cache_info,
    get_available_commodities,
)

from src.strategy import (
    backtest,
    BacktestError,
    StrategyType,
    get_available_strategies,
    get_strategy_parameters,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_support_resistance,
    calculate_atr,
)

from src.utils import (
    pnl_trades,
    backtest_performance,
    backtest_performance_extended,
    strategy_describe,
    calculate_drawdown_series,
    calculate_monthly_returns,
    calculate_risk_metrics,
)

from src.visualization import (
    backtest_charts,
    create_strategy_comparison_chart,
    create_equity_curve,
    create_trade_distribution,
    create_monthly_returns_heatmap,
)

__version__ = "2.0.0"
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
    "clear_cache",
    "get_cache_info",
    "get_available_commodities",
    # Strategy
    "backtest",
    "BacktestError",
    "StrategyType",
    "get_available_strategies",
    "get_strategy_parameters",
    "calculate_rsi",
    "calculate_bollinger_bands",
    "calculate_macd",
    "calculate_moving_averages",
    "calculate_support_resistance",
    "calculate_atr",
    # Utils
    "pnl_trades",
    "backtest_performance",
    "backtest_performance_extended",
    "strategy_describe",
    "calculate_drawdown_series",
    "calculate_monthly_returns",
    "calculate_risk_metrics",
    # Visualization
    "backtest_charts",
    "create_strategy_comparison_chart",
    "create_equity_curve",
    "create_trade_distribution",
    "create_monthly_returns_heatmap",
]

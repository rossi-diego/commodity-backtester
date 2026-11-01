"""
Configuration module for the Commodity Backtester application.

This module centralizes all configuration paths and constants used throughout
the application, following the single source of truth principle.
"""

from pathlib import Path
from typing import Final

# =============================================================================
# Directory Configuration
# =============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

# Create the data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# Data File Paths
# =============================================================================

YFINANCE_RAW: Final[Path] = DATA_DIR / "yfinance_raw.parquet"

# =============================================================================
# Application Settings
# =============================================================================

# Default date range for data fetching (25-year window gives statistically
# meaningful drawdown and regime coverage across full commodity cycles)
DEFAULT_START_DATE: Final[str] = "2000-01-01"

# VaR confidence level
VAR_CONFIDENCE_LEVEL: Final[float] = 0.95

# Trading days per year (for annualized metrics)
TRADING_DAYS_PER_YEAR: Final[int] = 252

# Annualised risk-free rate used in Sharpe / Sortino (0.0 = no adjustment).
# For commodity futures the risk-free component is embedded in the futures basis;
# set to e.g. 0.04 (4 %) to compute true excess-return Sharpe.
RISK_FREE_RATE: Final[float] = 0.0

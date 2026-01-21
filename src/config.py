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
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Data File Paths
# =============================================================================

YFINANCE_RAW: Final[Path] = DATA_DIR / "yfinance_raw.parquet"
YFINANCE_CLEAN: Final[Path] = DATA_DIR / "yfinance_clean.parquet"
YFINANCE_DESCRIBE: Final[Path] = DATA_DIR / "yfinance_describe.csv"

# =============================================================================
# Application Settings
# =============================================================================

# Default date range for data fetching
DEFAULT_START_DATE: Final[str] = "2020-01-01"

# VaR confidence level
VAR_CONFIDENCE_LEVEL: Final[float] = 0.95

# Trading days per year (for annualized metrics)
TRADING_DAYS_PER_YEAR: Final[int] = 252

# Logging configuration
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE: Final[Path] = LOGS_DIR / "backtester.log"

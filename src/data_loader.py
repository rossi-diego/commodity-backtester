"""
Data loading module for fetching commodity price data.

This module handles all data retrieval operations from Yahoo Finance,
including caching, validation, and error handling.
"""

import logging
from datetime import date
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from src.constants import commodities_dict, tickers

# Configure module logger
logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


def yahoo_quotes(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    progress: bool = False
) -> Tuple[pd.DataFrame, Optional[date]]:
    """
    Fetch historical commodity prices from Yahoo Finance.

    Retrieves closing prices for all configured commodity tickers and returns
    a cleaned DataFrame with human-readable column names.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format. Defaults to '2020-01-01'.
        end_date: End date in 'YYYY-MM-DD' format. Defaults to today.
        progress: Whether to show download progress bar.

    Returns:
        Tuple containing:
            - DataFrame with commodity prices indexed by date
            - First available date in the dataset (or None if empty)

    Raises:
        DataLoaderError: If data retrieval fails after retries.

    Example:
        >>> df, first_date = yahoo_quotes("2023-01-01", "2023-12-31")
        >>> print(df.columns.tolist())
        ['Soybean', 'Corn', 'Wheat', ...]
    """
    # Set default dates
    if start_date is None:
        start_date = "2020-01-01"
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info(f"Fetching data from {start_date} to {end_date}")

    try:
        # Download data from Yahoo Finance
        raw_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=progress,
            auto_adjust=True
        )

        # Handle multi-level columns if present
        if isinstance(raw_data.columns, pd.MultiIndex):
            df = raw_data["Close"].copy()
        else:
            df = raw_data.copy()

        # Remove rows with all NaN values
        df = df.dropna(how="all")

        if df.empty:
            logger.warning("No data retrieved from Yahoo Finance")
            return pd.DataFrame(), None

        # Rename columns to human-readable names
        df = df.rename(columns=lambda col: commodities_dict.get(col, col))

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # Sort by date
        df = df.sort_index()

        first_available_date = df.index.min().date()
        logger.info(f"Successfully loaded {len(df)} rows from {first_available_date}")

        return df, first_available_date

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise DataLoaderError(f"Failed to fetch commodity data: {e}") from e


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present

    Returns:
        True if all required columns are present, False otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.warning(f"Missing required columns: {missing}")
        return False
    return True

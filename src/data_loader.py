"""
Data loading module for fetching commodity price data.

This module handles all data retrieval operations from Yahoo Finance,
including caching, validation, and error handling.
"""

import logging
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from src.constants import commodities_dict, tickers
from src.config import DATA_DIR

# Configure module logger
logger = logging.getLogger(__name__)

# Cache settings
CACHE_DIR = DATA_DIR / "cache"
CACHE_EXPIRY_HOURS = 6  # Cache expires after 6 hours


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


def _get_cache_path(start_date: str, end_date: str) -> Path:
    """
    Generate a unique cache file path based on date range.

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        Path to the cache file
    """
    # Create a hash of the date range for unique filename
    cache_key = f"{start_date}_{end_date}_{','.join(sorted(tickers))}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"commodity_data_{cache_hash}.parquet"


def _is_cache_valid(cache_path: Path) -> bool:
    """
    Check if cache file exists and is not expired.

    Args:
        cache_path: Path to the cache file

    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False

    # Check if cache is expired
    cache_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    expiry_time = cache_mtime + timedelta(hours=CACHE_EXPIRY_HOURS)

    if datetime.now() > expiry_time:
        logger.info(f"Cache expired: {cache_path}")
        return False

    return True


def _load_from_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """
    Load data from cache file.

    Args:
        cache_path: Path to the cache file

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Loaded {len(df)} rows from cache")
        return df
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def _save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """
    Save data to cache file.

    Args:
        df: DataFrame to cache
        cache_path: Path to the cache file
    """
    try:
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, engine="pyarrow")
        logger.info(f"Saved {len(df)} rows to cache")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def yahoo_quotes(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    progress: bool = False,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, Optional[date]]:
    """
    Fetch historical commodity prices from Yahoo Finance.

    Retrieves closing prices for all configured commodity tickers and returns
    a cleaned DataFrame with human-readable column names. Supports caching
    for faster subsequent loads.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format. Defaults to '2020-01-01'.
        end_date: End date in 'YYYY-MM-DD' format. Defaults to today.
        progress: Whether to show download progress bar.
        use_cache: Whether to use cached data if available.

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

    logger.info(f"Requesting data from {start_date} to {end_date}")

    # Check cache first
    cache_path = _get_cache_path(start_date, end_date)

    if use_cache and _is_cache_valid(cache_path):
        df = _load_from_cache(cache_path)
        if df is not None:
            first_available_date = df.index.min().date()
            return df, first_available_date

    # Fetch from Yahoo Finance
    logger.info(f"Fetching fresh data from Yahoo Finance...")

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

        # Save to cache
        if use_cache:
            _save_to_cache(df, cache_path)

        return df, first_available_date

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise DataLoaderError(f"Failed to fetch commodity data: {e}") from e


def clear_cache() -> int:
    """
    Clear all cached data files.

    Returns:
        Number of files deleted
    """
    if not CACHE_DIR.exists():
        return 0

    count = 0
    for cache_file in CACHE_DIR.glob("*.parquet"):
        try:
            cache_file.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file}: {e}")

    logger.info(f"Cleared {count} cache files")
    return count


def get_cache_info() -> dict:
    """
    Get information about cached data.

    Returns:
        Dictionary with cache statistics
    """
    if not CACHE_DIR.exists():
        return {"files": 0, "total_size_mb": 0}

    cache_files = list(CACHE_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in cache_files)

    return {
        "files": len(cache_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": str(CACHE_DIR),
        "expiry_hours": CACHE_EXPIRY_HOURS
    }


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


def get_available_commodities(df: pd.DataFrame) -> list:
    """
    Get list of available commodities in the dataset.

    Args:
        df: DataFrame with commodity data

    Returns:
        List of commodity names with data
    """
    available = []
    for col in df.columns:
        if df[col].notna().sum() > 0:
            available.append(col)
    return available

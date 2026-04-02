"""Unit tests for src/data_loader.py — Parquet cache logic.

All tests mock out yfinance and the filesystem so that no network calls are
made and the production cache file is never touched.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src import config as dl_config
from src.data_loader import DataLoaderError, yahoo_quotes  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_close_df(start: str = "2000-01-03", periods: int = 20) -> pd.DataFrame:
    """Return a minimal Close-price DataFrame (ticker columns) as yfinance would."""
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {"ZS=F": [500.0] * periods, "ZC=F": [400.0] * periods},
        index=dates,
    )


def _fake_cached_df(start: str = "1999-01-04", end: str = "2021-12-31") -> pd.DataFrame:
    """Return an already-renamed cache DataFrame (display-name columns)."""
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    df = pd.DataFrame({"Soybean": [500.0] * n, "Corn": [400.0] * n}, index=pd.to_datetime(dates))
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestYahooQuotes:
    def test_cache_miss_downloads_and_writes(self, tmp_path, monkeypatch):
        """No cache file → yf.download called once; result written to parquet."""
        cache_file = tmp_path / "cache.parquet"
        monkeypatch.setattr(dl_config, "YFINANCE_RAW", cache_file)

        close_df = _fake_close_df()
        with patch("src.data_loader.yf.download") as mock_dl:
            mock_dl.return_value = {"Close": close_df}
            df, first_date = yahoo_quotes("2000-01-03", "2000-01-28")

        mock_dl.assert_called_once()
        assert cache_file.exists(), "Parquet cache should have been written after a cache miss."
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(first_date, datetime.date)

    def test_cache_hit_skips_download(self, monkeypatch):
        """Cache that fully covers the requested range → yf.download never called."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        monkeypatch.setattr(dl_config, "YFINANCE_RAW", mock_path)

        cached = _fake_cached_df("1999-01-04", "2021-12-31")

        with (
            patch("pandas.read_parquet", return_value=cached),
            patch("src.data_loader.yf.download") as mock_dl,
        ):
            df, first_date = yahoo_quotes("2020-01-02", "2020-12-31")

        mock_dl.assert_not_called()
        assert not df.empty

    def test_stale_cache_re_downloads(self, tmp_path, monkeypatch):
        """Cache that ends well before the requested end date triggers a re-download."""
        cache_file = tmp_path / "cache.parquet"

        # Write a stale cache ending in 2015
        stale = _fake_cached_df("2000-01-03", "2015-12-31")
        stale.to_parquet(cache_file)

        monkeypatch.setattr(dl_config, "YFINANCE_RAW", cache_file)

        close_df = _fake_close_df()
        with patch("src.data_loader.yf.download") as mock_dl:
            mock_dl.return_value = {"Close": close_df}
            # Request end date is today — far past the 2015 cache end → stale
            yahoo_quotes("2000-01-03", str(datetime.date.today()))

        mock_dl.assert_called_once()

    def test_empty_yfinance_response_raises(self, monkeypatch):
        """yfinance returning an empty DataFrame must raise DataLoaderError."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        monkeypatch.setattr(dl_config, "YFINANCE_RAW", mock_path)

        with patch("src.data_loader.yf.download") as mock_dl:
            mock_dl.return_value = {"Close": pd.DataFrame()}
            with pytest.raises(DataLoaderError):
                yahoo_quotes("2020-01-02", "2020-12-31")

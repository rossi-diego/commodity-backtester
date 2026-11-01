"""Unit tests for src/data_loader.py: Parquet cache logic and response validation.

All tests mock out yfinance and redirect the cache to a temporary path, so that
no network calls are made and the production cache file is never touched.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src import config as dl_config
from src.constants import commodities_dict
from src.data_loader import DataLoaderError, _response_problems, load_prices, yahoo_quotes

TICKER_BY_NAME = {name: ticker for ticker, name in commodities_dict.items()}
NAMES = list(commodities_dict.values())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_close_df(start: str = "2000-01-03", periods: int = 20) -> pd.DataFrame:
    """Return a minimal Close-price DataFrame (ticker columns) as yfinance would."""
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame({ticker: [500.0] * periods for ticker in commodities_dict}, index=dates)


def _fake_cached_df(start: str = "1999-01-04", end: str = "2021-12-31") -> pd.DataFrame:
    """Return an already-renamed cache DataFrame (display-name columns)."""
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    df = pd.DataFrame({"Soybean": [500.0] * n, "Corn": [400.0] * n}, index=pd.to_datetime(dates))
    df.index.name = "date"
    return df


def _full_cached_df(start: str = "2000-03-15", end: str = "2024-06-28") -> pd.DataFrame:
    """Return a cache frame with every tracked column."""
    dates = pd.date_range(start, end, freq="B")
    df = pd.DataFrame(dict.fromkeys(NAMES, 100.0), index=dates)
    df.index.name = "date"
    return df


def _as_response(names_frame: pd.DataFrame) -> pd.DataFrame:
    """Rename display-name columns back to tickers, as yfinance returns them."""
    return names_frame.rename(columns=TICKER_BY_NAME)


def _extended(cached: pd.DataFrame, extra_days: int = 30) -> pd.DataFrame:
    """Return a good refresh of ``cached``: same history plus newer rows."""
    new_index = pd.date_range(cached.index[0], periods=len(cached) + extra_days, freq="B")
    fresh = pd.DataFrame(dict.fromkeys(NAMES, 101.0), index=new_index)
    fresh.index.name = "date"
    return fresh


@pytest.fixture()
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "cache.parquet"
    monkeypatch.setattr(dl_config, "YFINANCE_RAW", path)
    return path


@pytest.fixture()
def stale_cache(cache_file):
    """A valid cache that ends long before today, so every call tries to download."""
    cached = _full_cached_df()
    cached.to_parquet(cache_file)
    return cached


def _refresh_with(response: pd.DataFrame) -> tuple[pd.DataFrame, MagicMock]:
    with patch("src.data_loader.yf.download") as mock_dl:
        mock_dl.return_value = {"Close": _as_response(response)}
        df, _ = yahoo_quotes()
    return df, mock_dl


def _assert_same_frame(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(left, right, check_freq=False, check_index_type=False)


# ---------------------------------------------------------------------------
# Cache hit or miss
# ---------------------------------------------------------------------------


class TestYahooQuotes:
    def test_cache_miss_downloads_and_writes(self, cache_file):
        """No cache file: yf.download called once; result written to parquet."""
        with patch("src.data_loader.yf.download") as mock_dl:
            mock_dl.return_value = {"Close": _fake_close_df()}
            df, first_date = yahoo_quotes("2000-01-03", "2000-01-28")

        mock_dl.assert_called_once()
        assert cache_file.exists(), "Parquet cache should have been written after a cache miss."
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(first_date, datetime.date)

    def test_cache_hit_skips_download(self, monkeypatch):
        """Cache that fully covers the requested range: yf.download never called."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        monkeypatch.setattr(dl_config, "YFINANCE_RAW", mock_path)

        cached = _fake_cached_df("1999-01-04", "2021-12-31")

        with (
            patch("pandas.read_parquet", return_value=cached),
            patch("src.data_loader.yf.download") as mock_dl,
        ):
            df, _first_date = yahoo_quotes("2020-01-02", "2020-12-31")

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
            # Request end date is today, far past the 2015 cache end: stale
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

    def test_cache_hit_with_default_start(self, cache_file):
        """Regression: Yahoo's first date is 2000-03-15, so a 2000-01-01 request must still hit."""
        today = datetime.date.today()
        _full_cached_df("2000-03-15", str(today)).to_parquet(cache_file)
        with patch("src.data_loader.yf.download") as mock_dl:
            yahoo_quotes("2000-01-01", today)
        mock_dl.assert_not_called()

    def test_cache_starting_much_later_misses(self, cache_file):
        today = datetime.date.today()
        _full_cached_df("2015-01-05", str(today)).to_parquet(cache_file)
        with patch("src.data_loader.yf.download") as mock_dl:
            mock_dl.return_value = {"Close": pd.DataFrame()}
            yahoo_quotes("2000-01-01", today)
        mock_dl.assert_called_once()

    def test_unreadable_cache_is_treated_as_missing(self, cache_file):
        cache_file.write_bytes(b"not a parquet file")
        with patch("src.data_loader.yf.download") as mock_dl:
            mock_dl.return_value = {"Close": _fake_close_df()}
            df, _ = yahoo_quotes("2000-01-03", "2000-01-28")
        mock_dl.assert_called_once()
        assert len(df) == 20


# ---------------------------------------------------------------------------
# Validation without a cache
# ---------------------------------------------------------------------------


class TestColdResponse:
    def test_cold_response_with_real_column_starts_is_accepted(self, cache_file):
        """Regression: real columns start up to 264 days after 2000-01-01, and that must pass."""
        response = _full_cached_df("2000-01-03", "2024-06-28")
        for offset, name in zip([74, 135, 198, 235, 244, 258, 264], NAMES, strict=True):
            cutoff = pd.Timestamp("2000-01-01") + pd.Timedelta(days=offset)
            response.loc[response.index < cutoff, name] = np.nan

        _refresh_with(response)
        assert cache_file.exists()

    def test_cold_response_starting_too_late_is_rejected(self, cache_file):
        response = _full_cached_df("2000-03-15", "2024-06-28")
        response.loc[response.index < "2002-01-01", "Soybean"] = np.nan

        with pytest.raises(DataLoaderError, match="Soybean starts after"):
            _refresh_with(response)
        assert not cache_file.exists()

    def test_cold_response_missing_column_is_rejected(self, cache_file):
        response = _full_cached_df("2000-03-15", "2024-06-28").drop(columns="Corn")
        with pytest.raises(DataLoaderError, match="Corn has no values"):
            _refresh_with(response)
        assert not cache_file.exists()


# ---------------------------------------------------------------------------
# Validation against an existing cache
# ---------------------------------------------------------------------------


class TestWarmResponse:
    def _assert_rejected(self, cache_file, stale_cache, response) -> None:
        before = cache_file.read_bytes()
        df, mock_dl = _refresh_with(response)
        mock_dl.assert_called_once()
        assert cache_file.read_bytes() == before, "A rejected response must not touch the cache."
        _assert_same_frame(df, stale_cache)

    def test_partial_response_keeps_cache(self, cache_file, stale_cache):
        response = _extended(stale_cache)
        response["Soybean"] = np.nan
        self._assert_rejected(cache_file, stale_cache, response)

    def test_short_response_keeps_cache(self, cache_file, stale_cache):
        response = _extended(stale_cache).iloc[-1000:]
        self._assert_rejected(cache_file, stale_cache, response)

    def test_column_with_recent_history_only_keeps_cache(self, cache_file, stale_cache):
        response = _extended(stale_cache)
        cutoff = response.index[-1] - pd.DateOffset(months=22)
        response.loc[response.index < cutoff, "Soybean"] = np.nan
        self._assert_rejected(cache_file, stale_cache, response)

    def test_column_ending_early_keeps_cache(self, cache_file, stale_cache):
        response = _extended(stale_cache)
        cutoff = stale_cache.index[-1] - pd.DateOffset(months=10)
        response.loc[response.index > cutoff, "Corn"] = np.nan
        self._assert_rejected(cache_file, stale_cache, response)

    def test_frame_ending_early_keeps_cache(self, cache_file, stale_cache):
        cutoff = stale_cache.index[-1] - pd.DateOffset(months=4)
        response = stale_cache.loc[:cutoff]
        self._assert_rejected(cache_file, stale_cache, response)

    def test_valid_refresh_overwrites_cache(self, cache_file, stale_cache):
        response = _extended(stale_cache)
        df, _ = _refresh_with(response)
        stored = pd.read_parquet(cache_file)
        assert stored.index.max() == response.index.max()
        assert df.index.max() == response.index.max()

    def test_empty_response_returns_stale_cache(self, cache_file, stale_cache):
        with patch("src.data_loader.yf.download") as mock_dl:
            mock_dl.return_value = {"Close": pd.DataFrame()}
            df, _ = yahoo_quotes()
        _assert_same_frame(df, stale_cache)

    def test_download_exception_returns_stale_cache(self, cache_file, stale_cache):
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("rate limited")):
            df, _ = yahoo_quotes()
        _assert_same_frame(df, stale_cache)

    def test_no_cache_and_failure_raises(self, cache_file):
        with (
            patch("src.data_loader.yf.download", side_effect=RuntimeError("rate limited")),
            pytest.raises(DataLoaderError, match="rate limited"),
        ):
            yahoo_quotes()

    def test_write_is_atomic(self, cache_file, stale_cache, monkeypatch):
        """A failed write leaves the original cache intact and no temporary file behind."""
        before = cache_file.read_bytes()

        def failing_write(self, path, *args, **kwargs):
            path.write_bytes(b"partial")
            raise OSError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_write)
        df, _ = _refresh_with(_extended(stale_cache))

        assert cache_file.read_bytes() == before
        assert not cache_file.with_suffix(".tmp").exists()
        assert not df.empty


class TestImperfectCache:
    """A missing or empty column in the cache imposes no constraint on the response."""

    def _problems(self, cached: pd.DataFrame, response: pd.DataFrame) -> list[str]:
        return _response_problems(response, cached, datetime.date(2000, 1, 1))

    def test_column_missing_from_cache_does_not_raise(self):
        cached = _full_cached_df().drop(columns="Corn")
        assert self._problems(cached, _extended(_full_cached_df())) == []

    def test_all_nan_column_in_cache_does_not_raise(self):
        cached = _full_cached_df()
        cached["Corn"] = np.nan
        assert self._problems(cached, _extended(_full_cached_df())) == []

    def test_constraints_still_apply_to_other_columns(self):
        cached = _full_cached_df()
        cached["Corn"] = np.nan
        response = _extended(_full_cached_df())
        response["Soybean"] = np.nan
        assert self._problems(cached, response) == ["Soybean has no values"]

    def test_good_response_repairs_poisoned_cache(self, cache_file):
        """A cache with an empty column is replaced end to end, without an exception."""
        poisoned = _full_cached_df()
        poisoned["Corn"] = np.nan
        poisoned.to_parquet(cache_file)

        df, mock_dl = _refresh_with(_extended(_full_cached_df()))

        mock_dl.assert_called_once()
        assert df["Corn"].notna().all()
        assert pd.read_parquet(cache_file)["Corn"].notna().all()


# ---------------------------------------------------------------------------
# load_prices
# ---------------------------------------------------------------------------


class TestLoadPrices:
    def test_failure_is_returned_as_a_value(self, cache_file):
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("down")):
            result = load_prices()
        assert result.prices is None
        assert result.as_of is None
        assert result.is_stale is False
        assert "down" in str(result.error)

    def test_recent_cache_is_not_stale(self, cache_file):
        today = datetime.date.today()
        _full_cached_df("2000-03-15", str(today)).to_parquet(cache_file)
        with patch("src.data_loader.yf.download") as mock_dl:
            result = load_prices()
        mock_dl.assert_not_called()
        assert result.error is None
        assert result.is_stale is False

    def test_old_cache_after_failed_refresh_is_stale(self, cache_file, stale_cache):
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("down")):
            result = load_prices()
        assert result.prices is not None
        assert result.as_of == stale_cache.index.max().date()
        assert result.is_stale is True
        assert result.error is None

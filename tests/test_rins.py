"""D4 RIN loading: parsing, validation and the order of fallbacks. No network."""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import patch

import pandas as pd
import pytest

from src import config, rins

WEEKS = pd.date_range("2018-01-01", periods=150, freq="W-MON")


def _rows(weeks=WEEKS, price: float = 1.5) -> list[list[object]]:
    rows: list[list[object]] = []
    for week in weeks:
        text = f"{week.month}/{week.day}/{week.year}"
        rows += [[text, "6", 0.9], [text, "4", price], [text, "5", 1.1]]
    return rows


def _write(path, weeks=WEEKS, price: float = 1.5) -> None:
    frame = pd.DataFrame({"week": weeks.strftime("%Y-%m-%d"), "usd_per_rin": price})
    frame.to_csv(path, index=False)


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    cache, seed = tmp_path / "rin.csv", tmp_path / "seed.csv"
    monkeypatch.setattr(config, "RIN_CACHE", cache)
    monkeypatch.setattr(config, "RIN_SEED", seed)
    return cache, seed


class TestParse:
    def test_keeps_d4_sorted_by_week(self) -> None:
        rows = list(reversed(_rows()))
        series = rins.parse_rows(rows)
        assert len(series) == len(WEEKS)
        assert series.index.is_monotonic_increasing
        assert (series == 1.5).all()

    def test_skips_rows_without_a_number(self) -> None:
        series = rins.parse_rows([["1/1/2024", "4", None], ["1/8/2024", "4", 1.2]])
        assert list(series) == [1.2]


class TestLoad:
    def test_fresh_download_is_cached(self, paths) -> None:
        cache, _ = paths
        with patch("src.rins._fetch_rows", return_value=_rows()):
            result = rins.load_d4()
        assert result.source == "epa"
        assert result.as_of == WEEKS[-1].date()
        assert cache.exists()

    def test_recent_cache_skips_the_network(self, paths) -> None:
        cache, _ = paths
        _write(cache)
        with patch("src.rins._fetch_rows") as fetch:
            result = rins.load_d4()
        fetch.assert_not_called()
        assert result.source == "cache"

    def test_old_cache_is_refreshed(self, paths) -> None:
        cache, _ = paths
        _write(cache, WEEKS[:-1])
        old = time.time() - 4 * 24 * 3600
        os.utime(cache, (old, old))
        with patch("src.rins._fetch_rows", return_value=_rows()):
            assert rins.load_d4().source == "epa"

    def test_failure_falls_back_to_the_cache_then_the_seed(self, paths) -> None:
        cache, seed = paths
        _write(seed)
        with patch("src.rins._fetch_rows", side_effect=OSError("offline")):
            result = rins.load_d4()
        assert result.source == "seed"
        assert "offline" in result.error

        _write(cache)
        old = time.time() - 4 * 24 * 3600
        os.utime(cache, (old, old))
        with patch("src.rins._fetch_rows", side_effect=OSError("offline")):
            assert rins.load_d4().source == "cache"

    def test_nothing_at_all(self, paths) -> None:
        with patch("src.rins._fetch_rows", side_effect=OSError("offline")):
            result = rins.load_d4()
        assert result.prices is None
        assert result.as_of is None
        assert result.source == "none"

    @pytest.mark.parametrize(
        ("rows", "problem"),
        [
            (_rows(WEEKS[:50]), "only 50 D4 weeks"),
            (_rows(price=4.0), "outside EPA's D4 filter"),
            (_rows(WEEKS[:-5]), "ends before the saved series"),
        ],
    )
    def test_bad_responses_are_rejected(self, paths, rows, problem) -> None:
        cache, seed = paths
        _write(seed)
        with patch("src.rins._fetch_rows", return_value=rows):
            result = rins.load_d4()
        assert result.source == "seed"
        assert problem in result.error
        assert not cache.exists()

    def test_shorter_history_is_rejected(self, paths) -> None:
        _, seed = paths
        _write(seed)
        with patch("src.rins._fetch_rows", return_value=_rows(WEEKS[30:])):
            assert "fewer weeks" in rins.load_d4().error

    def test_duplicate_weeks_are_rejected(self, paths) -> None:
        rows = _rows()
        rows += [rows[1]]
        with patch("src.rins._fetch_rows", return_value=rows):
            assert "duplicate weeks" in rins.load_d4().error

    def test_unreadable_seed_is_ignored(self, paths) -> None:
        _, seed = paths
        seed.write_text("not,a\nvalid,file\n", encoding="utf-8")
        with patch("src.rins._fetch_rows", side_effect=OSError("offline")):
            assert rins.load_d4().prices is None

    @pytest.mark.parametrize(
        "text",
        [
            "week,price\n2024-01-01,1.2\n",
            "week,usd_per_rin\n2024-01-01,abc\n",
            "week,usd_per_rin\nx,1.2\n",
            "week,usd_per_rin\n",
            "week,usd_per_rin\n2024-01-01,\n",
        ],
        ids=[
            "no-price-column",
            "price-not-a-number",
            "week-not-a-date",
            "header-only",
            "price-missing",
        ],
    )
    def test_bad_cache_is_ignored(self, paths, caplog, text) -> None:
        cache, seed = paths
        cache.write_text(text, encoding="utf-8")
        with caplog.at_level("WARNING", logger="src.rins"):
            assert rins._read(cache) is None
        assert "rin_unreadable" in caplog.text

        _write(seed)
        with patch("src.rins._fetch_rows", side_effect=OSError("offline")):
            result = rins.load_d4()
        assert result.source == "seed"
        assert result.as_of == WEEKS[-1].date()

    def test_a_failed_cache_write_keeps_the_result(self, paths) -> None:
        with (
            patch("src.rins._fetch_rows", return_value=_rows()),
            patch("pandas.DataFrame.to_csv", side_effect=OSError("disk full")),
        ):
            assert rins.load_d4().source == "epa"


class _FakeSocket:
    """Answers the Qlik engine calls the module makes, with two pages of rows."""

    def __init__(self, rows: list[list[object]], error: bool = False) -> None:
        self.rows, self.error, self.replies = rows, error, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def send(self, text: str) -> None:
        request = json.loads(text)
        method, params = request["method"], request["params"]
        if self.error:
            self.replies.append({"id": request["id"], "error": {"message": "denied"}})
            return
        if method in ("OpenDoc", "GetObject"):
            result = {"qReturn": {"qHandle": 1}}
        elif method == "GetLayout":
            result = {"qLayout": {"qHyperCube": {"qSize": {"qcy": len(self.rows)}}}}
        else:
            top, height = params[1][0]["qTop"], params[1][0]["qHeight"]
            chunk = self.rows[top : top + min(height, 200)]
            matrix = [[{"qText": w}, {"qText": c}, {"qNum": p}] for w, c, p in chunk]
            result = {"qDataPages": [{"qMatrix": matrix}]}
        self.replies += [{"method": "OnConnected"}, {"id": request["id"], "result": result}]

    async def recv(self) -> str:
        return json.dumps(self.replies.pop(0))


class TestEngine:
    def test_reads_every_page(self) -> None:
        rows = _rows()
        with patch("src.rins.websockets.connect", return_value=_FakeSocket(rows)):
            fetched = rins._fetch_rows()
        assert fetched == rows
        assert len(rins.parse_rows(fetched)) == len(WEEKS)

    def test_engine_errors_raise(self) -> None:
        with (
            patch("src.rins.websockets.connect", return_value=_FakeSocket([], error=True)),
            pytest.raises(RuntimeError, match="OpenDoc"),
        ):
            asyncio.run(rins._fetch())

    def test_an_empty_page_ends_the_read(self) -> None:
        socket = _FakeSocket(_rows()[:3])
        original = socket.send

        async def short_page(text: str) -> None:
            await original(text)
            if json.loads(text)["method"] == "GetLayout":
                socket.replies[-1]["result"]["qLayout"]["qHyperCube"]["qSize"]["qcy"] = 10

        socket.send = short_page
        with patch("src.rins.websockets.connect", return_value=socket):
            assert len(rins._fetch_rows()) == 3


class TestSeedFile:
    def test_saved_copy_is_valid_and_dated(self) -> None:
        seed = pd.read_csv(config.PROJECT_ROOT / "data" / "seed" / "epa_d4_rin_weekly.csv")
        assert list(seed.columns) == ["week", "usd_per_rin"]
        assert len(seed) >= 800
        assert seed["usd_per_rin"].between(rins.D4_MIN_PRICE, rins.D4_MAX_PRICE).all()
        assert pd.to_datetime(seed["week"]).is_monotonic_increasing

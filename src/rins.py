"""Weekly D4 RIN prices from EPA's public Renewable Fuel Standard data.

EPA publishes weekly volume-weighted average prices of separated RINs, from
EMTS transactions, in a public Qlik Sense app on its "RIN Trades and Price
Information" page. There is no file to download, so this module reads the
chart through the same engine API the page uses.

Order of sources: a recent local cache, then EPA, then the dated copy saved in
the repository. A download replaces the cache only if it passes ``_problems``,
so a changed or partial EPA response cannot replace good data.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import websockets

from . import config

__all__ = ["RinResult", "load_d4", "parse_rows"]

logger = logging.getLogger(__name__)

EPA_PAGE = (
    "https://www.epa.gov/fuels-registration-reporting-and-compliance-help/"
    "rin-trades-and-price-information"
)
_APP_ID = "73b2b6a5-70c6-4820-b3fa-186ac094f10d"
_CHART_ID = "7c8498d9-0ecb-4580-9d5c-9ede039ea654"
_ENGINE_URL = f"wss://edap.epa.gov/public/app/{_APP_ID}"
_PAGE_ROWS = 3000

# EPA drops D4 transfers outside this range as outliers, so a value outside it means
# the response is not what this module expects.
D4_MIN_PRICE = 0.05
D4_MAX_PRICE = 3.00
MIN_WEEKS = 100
MIN_ROW_SHARE = 0.9
# EPA reloads the data monthly, so a cache younger than this is served as is.
CACHE_MAX_AGE_SECONDS = 3 * 24 * 3600
FETCH_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class RinResult:
    """Weekly D4 prices in USD per RIN, indexed by week start, and where they came from."""

    prices: pd.Series | None
    source: str
    error: str | None = None

    @property
    def as_of(self) -> datetime.date | None:
        return None if self.prices is None else self.prices.index[-1].date()


def load_d4() -> RinResult:
    """Return the weekly D4 series without raising."""
    cached = _read(config.RIN_CACHE)
    if cached is not None and _age_seconds(config.RIN_CACHE) < CACHE_MAX_AGE_SECONDS:
        return RinResult(cached, "cache")

    reference = cached if cached is not None else _read(config.RIN_SEED)
    try:
        fresh = parse_rows(_fetch_rows())
        problems = _problems(fresh, reference)
        if problems:
            raise ValueError("; ".join(problems))
    except Exception as exc:  # noqa: BLE001
        logger.warning("event=rin_load_failed reason=%s", exc)
        if cached is not None:
            return RinResult(cached, "cache", str(exc))
        seed = _read(config.RIN_SEED)
        return RinResult(seed, "seed" if seed is not None else "none", str(exc))

    _write(fresh, config.RIN_CACHE)
    logger.info("event=rin_load source=epa weeks=%d as_of=%s", len(fresh), fresh.index[-1].date())
    return RinResult(fresh, "epa")


def parse_rows(rows: list[list[Any]]) -> pd.Series:
    """Keep the D4 rows of ``[week as M/D/YYYY, D code, price]`` as a weekly series."""
    weeks, prices = [], []
    for week, code, price in rows:
        if str(code).strip() == "4" and isinstance(price, int | float):
            weeks.append(pd.Timestamp(datetime.datetime.strptime(str(week), "%m/%d/%Y")))
            prices.append(float(price))
    series = pd.Series(prices, index=pd.DatetimeIndex(weeks, name="week"), name="usd_per_rin")
    return series.sort_index()


def _problems(fresh: pd.Series, reference: pd.Series | None) -> list[str]:
    problems: list[str] = []
    if len(fresh) < MIN_WEEKS:
        return [f"only {len(fresh)} D4 weeks"]
    if not fresh.index.is_unique:
        problems.append("duplicate weeks")
    if ((fresh < D4_MIN_PRICE) | (fresh > D4_MAX_PRICE)).any():
        problems.append("prices outside EPA's D4 filter")
    if reference is not None:
        if len(fresh) < MIN_ROW_SHARE * len(reference):
            problems.append("fewer weeks than the saved series")
        if fresh.index[-1] < reference.index[-1]:
            problems.append("ends before the saved series")
    return problems


def _fetch_rows() -> list[list[Any]]:
    return asyncio.run(asyncio.wait_for(_fetch(), FETCH_TIMEOUT_SECONDS))


async def _fetch() -> list[list[Any]]:
    headers = {"Origin": "https://edap.epa.gov"}
    async with websockets.connect(_ENGINE_URL, additional_headers=headers, max_size=2**26) as ws:
        engine = _Engine(ws)
        doc = await engine.call(-1, "OpenDoc", _APP_ID)
        chart = await engine.call(doc["qReturn"]["qHandle"], "GetObject", _CHART_ID)
        handle = chart["qReturn"]["qHandle"]
        layout = await engine.call(handle, "GetLayout")
        total = layout["qLayout"]["qHyperCube"]["qSize"]["qcy"]
        rows: list[list[Any]] = []
        while len(rows) < total:
            page = [{"qTop": len(rows), "qLeft": 0, "qWidth": 3, "qHeight": _PAGE_ROWS}]
            data = await engine.call(handle, "GetHyperCubeData", "/qHyperCubeDef", page)
            matrix = data["qDataPages"][0]["qMatrix"]
            if not matrix:
                break
            rows += [[row[0]["qText"], row[1]["qText"], row[2].get("qNum")] for row in matrix]
        return rows


class _Engine:
    """Minimal JSON-RPC client for the Qlik engine."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.next_id = 0

    async def call(self, handle: int, method: str, *params: Any) -> dict[str, Any]:
        self.next_id += 1
        request = {"jsonrpc": "2.0", "id": self.next_id, "handle": handle, "method": method}
        await self.ws.send(json.dumps({**request, "params": list(params)}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                result: dict[str, Any] = message["result"]
                return result


def _read(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["week"]).set_index("week")
    except Exception as exc:  # noqa: BLE001
        logger.warning("event=rin_unreadable path=%s reason=%s", path.name, exc)
        return None
    series = frame["usd_per_rin"].astype(float).sort_index()
    return None if series.empty else series


def _write(series: pd.Series, path: Path) -> None:
    temp = path.with_suffix(".tmp")
    try:
        series.rename("usd_per_rin").to_frame().to_csv(temp, date_format="%Y-%m-%d")
        os.replace(temp, path)
    except Exception as exc:  # noqa: BLE001
        temp.unlink(missing_ok=True)
        logger.warning("event=rin_cache_write_failed reason=%s", exc)


def _age_seconds(path: Path) -> float:
    return time.time() - path.stat().st_mtime

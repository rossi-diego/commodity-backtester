"""Paths and data settings for the Soybean Crush Margin app, in one place."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

# Parquet cache of daily closes. Git-ignored, rebuilt from Yahoo Finance when stale.
YFINANCE_RAW: Final[Path] = DATA_DIR / "prices.parquet"

# Weekly D4 RIN prices from EPA: a local cache, and a dated copy kept in the repository
# as the fallback when EPA cannot be reached.
RIN_CACHE: Final[Path] = DATA_DIR / "rin_d4.csv"
RIN_SEED: Final[Path] = DATA_DIR / "seed" / "epa_d4_rin_weekly.csv"

# First requested date. The crush legs all start trading on Yahoo in early 2000.
DEFAULT_START_DATE: Final[str] = "2000-01-01"

# Slack between the first cached date and the requested start. The first date in
# the Yahoo frame is 2000-03-15 when 2000-01-01 is requested.
CACHE_START_SLACK_DAYS: Final[int] = 120

# Slack between each column's first date and the requested start when there is no
# cache to compare with. Real columns start up to 264 days after 2000-01-01.
COLUMN_START_SLACK_DAYS: Final[int] = 365

# Days without a new close before the data is labelled as possibly out of date, and
# the most a column may end before the rest of a download. Covers weekends and exchange
# holidays. It no longer gates the refetch: the cache is refreshed once it misses the
# last business day.
CACHE_STALE_AFTER_DAYS: Final[int] = 5

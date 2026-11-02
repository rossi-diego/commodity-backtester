# Soybean Crush Margin

**What is a soybean crusher earning today, and how much of it is biodiesel?**

Built by **Diego Rossi** · [LinkedIn](https://www.linkedin.com/in/rossi-diego/) · [Live app](https://commodity-backtester.streamlit.app/)

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://commodity-backtester.streamlit.app/)
[![CI](https://github.com/rossi-diego/soybean-crush-margin/actions/workflows/ci.yml/badge.svg)](https://github.com/rossi-diego/soybean-crush-margin/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A crusher buys soybeans and sells the meal and oil they yield. Since 2021, demand from biodiesel and renewable diesel has pushed soybean oil from about a third to about half of that product value, so the crush margin is now partly a fuel story. This app reads that story from public futures prices, in metric units, and says what it leaves out.

![Soybean crush margin page: today's margin and its 5-year rank, oil share, soybean oil against diesel, and this year's margin over the range of the past five years](docs/screenshot.png)

---

## What the page answers

| Question | Measure | Unit |
|---|---|---|
| Is crushing paying well or badly now? | Crush margin, with its rank against the past 5 years and this year drawn over the monthly range of the five previous years | USD per tonne of soybeans |
| How much of the value is oil? | Oil share of the product value, and meal and oil value against the soybean cost | % and USD per tonne |
| What does the oil in biodiesel cost next to diesel? | The soybean oil needed for 1 m³ of biodiesel against 1 m³ of heating oil, and the gap between them. This is the feedstock alone, not the biodiesel price: methanol, energy and plant costs come on top | USD per m³ |

Every number on the page states its date and unit. The page opens on the latest data with no button to press.

## How it is calculated

| Measure | Formula | Source |
|---|---|---|
| Crush margin | 0.733 × meal + 0.183 × oil − soybeans, all in USD per tonne | CME board crush: 44 lb of meal and 11 lb of oil per 60 lb bushel, which is 0.022 × meal + 0.11 × oil − soybeans in USD per bushel ([CME guide](https://www.cmegroup.com/education/files/soybean-crush-reference-guide.pdf)) |
| Oil share | oil value / (oil value + meal value) | Derived |
| Feedstock cost | 0.90 t of soybean oil per m³ of biodiesel × oil in USD per tonne | EIA plant data, 2017 to 2020, average 7.50 lb per gallon, which is 0.899 t per m³ ([farmdoc daily](https://farmdocdaily.illinois.edu/wp-content/uploads/2022/02/fdd021622.pdf)) |
| Oil cost vs diesel | feedstock cost − heating oil, in USD per m³ | NYMEX heating oil stands in for diesel. The industry quotes the same comparison per gallon as the BOHO spread |

Unit conversions use the exact definitions (1 lb = 0.45359237 kg, 1 US gallon = 3.785411784 L) and are tested in `tests/test_constants.py`.

On 2026-10-27 the crush margin was 88.43 USD/t (2.41 USD/bu), oil was 50% of the product value, and soybean oil cost 1,324 USD per m³ of biodiesel against 1,262 USD/m³ for heating oil. `tests/test_margins.py` asserts these figures from that day's closes.

## Rolls and delivery

Yahoo Finance's continuous `=F` series stay in the nearest contract until its last trading day, so they trade through the delivery period and then jump to the next contract. A trader who does not take delivery would have rolled before first notice.

`src/rolls.py` rebuilds approximate CBOT and NYMEX contract calendars and flags every session where a leg is in its delivery period or has just switched contract. Flagged sessions are drawn as a thin grey line and left out of every range and percentile: about 4 in 10 sessions for the crush. Measured on 2000 to 2026, they hold 19 of the 20 largest daily moves in the crush, including the record low of −1.82 USD/bu on 2008-09-12, which was an expiring September contract and not a real margin.

From mid-September to mid-December the front months are not the board crush pair: November beans trade against October, then December, products, and January beans against December products. Those sessions are kept, because their daily changes match clean sessions (median 0.86 against 0.89 USD/t), but their level includes a product calendar spread.

## What it leaves out

These are deliberate limits for a project built on free data. A production desk would add each one.

| Left out | Why it matters |
|---|---|
| Plant costs, energy, freight, local basis | The page shows a gross board margin, not plant profit |
| Biofuel credits and mandates (RINs, LCFS, 45Z) | They often decide biodiesel profitability, and need paid or policy data |
| Energy content | Biodiesel holds about 7% less energy than diesel per unit of volume ([AFDC](https://afdc.energy.gov/fuels/properties)); the gap compares volumes |
| The forward curve | Only the front month is free with history, so there is no forward crush yet |
| Exchange holidays | Calendars use business days, so a flag can be one session off around a holiday |

## Architecture

```
Yahoo Finance: ZS=F, ZM=F, ZL=F, HO=F (daily closes from 2000)
   |  validated per column, written atomically, falls back to the cache on failure
   v
data/prices.parquet  (local, git-ignored)
   |  load_prices() -> LoadResult, cached for one hour by the app; a failure is never cached
   v
src/margins.py   metric units, crush, oil share, feedstock gap; a missing leg drops the session
src/rolls.py     contract calendars: delivery and switch flags
src/seasonal.py  latest clean reading, 5-year percentile, monthly range of the five previous years
   |
   v
src/visualization.py (pure Plotly figures) -> app.py (Streamlit page)
```

Logs use `key=value` fields, for example `event=data_load source=cache rows=6711 as_of=2026-10-30`, `event=cache_rejected problems=...` and `event=data_load_failed reason=...`.

## Quick start

```bash
git clone https://github.com/rossi-diego/soybean-crush-margin.git
cd soybean-crush-margin
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads about 26 years of daily closes from Yahoo Finance, which takes a few seconds.

### Developer setup

```bash
pip install -r requirements-dev.txt
pytest                                  # tests and coverage
ruff format --check src/ tests/ app.py  # formatting
ruff check src/ tests/ app.py           # lint
mypy src/                               # type-check
```

## Project structure

```
soybean-crush-margin/
├── app.py                      # Streamlit page only
├── .streamlit/config.toml      # Dark theme
├── docs/screenshot.png         # README screenshot
├── src/
│   ├── config.py               # Paths and cache settings
│   ├── constants.py            # Tickers, units, crush and biodiesel factors with sources
│   ├── data_loader.py          # Yahoo Finance fetch, response validation, Parquet cache
│   ├── rolls.py                # Contract calendars, delivery and switch flags
│   ├── margins.py              # Crush margin, oil share, feedstock gap
│   ├── seasonal.py             # Percentile rank and monthly range on clean sessions
│   └── visualization.py        # Plotly figure builders, no UI imports
├── tests/                      # Unit tests per module, and the page end to end with AppTest
├── .github/workflows/ci.yml    # CI: lint, type-check, test
├── pyproject.toml              # Ruff, mypy, pytest configuration
├── requirements.txt            # Runtime dependencies, pinned
└── requirements-dev.txt        # Development dependencies
```

## Key design decisions

| Decision | Why |
|---|---|
| **Flag, do not fix, the rolls** | A back-adjusted series needs every contract's history, which is not free. Flagging delivery and switch sessions keeps every number traceable to a real close and stops expiry artefacts from setting records. |
| **Metric units** | European and Brazilian desks think in tonnes and m³. The CME per-bushel value is shown beside the headline for anyone who quotes the board crush. |
| **A missing leg drops the session** | A forward-filled price would invent a margin. |
| **Amber means the crush margin, nothing else** | Controls, badges and the warning stay neutral, so the colour keeps one meaning. No green or red: a margin level is not a verdict. |
| **The warning sits beside the number** | What the margin leaves out is read with the margin, not after it. |
| **Validated, atomic cache** | A download replaces the cache only if each column keeps at least 90% of its cached observations and does not end earlier, so a partial Yahoo response cannot overwrite good data. |

## History

This repository started as a rule-based simulator on seven commodity futures. An audit showed that roll handling could flip the sign of its results and that its rules were not how processors and merchants work, so the project was refocused on the margin a crusher actually manages. The earlier code is in the git history.

---

Personal project, not affiliated with any employer. Not investment advice.

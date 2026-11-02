"""Contract reference data and unit conversions for the soybean crush.

Tickers use Yahoo's continuous front-month notation (``=F``). Conversion factors
follow the CME/CBOT contract specifications; the biodiesel factor is documented
where it is defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "BEANS",
    "BUSHELS_PER_TONNE",
    "COMMODITIES",
    "DIESEL",
    "GALLONS_PER_M3",
    "LB_PER_TONNE",
    "MEAL",
    "MEAL_YIELD",
    "OIL",
    "OIL_PER_M3_BIODIESEL",
    "OIL_YIELD",
    "RINS_PER_GALLON_BIODIESEL",
    "SHORT_TONS_PER_TONNE",
    "CommoditySpec",
    "commodities_dict",
    "tickers",
]


@dataclass(frozen=True)
class CommoditySpec:
    """Metadata for a single futures contract."""

    name: str
    ticker: str
    exchange: str
    unit: str


BEANS: Final[str] = "Soybean"
MEAL: Final[str] = "Soybean Meal"
OIL: Final[str] = "Soybean Oil"
DIESEL: Final[str] = "Heating Oil"

COMMODITIES: dict[str, CommoditySpec] = {
    "ZS=F": CommoditySpec(name=BEANS, ticker="ZS=F", exchange="CBOT", unit="US cents/bushel"),
    "ZM=F": CommoditySpec(name=MEAL, ticker="ZM=F", exchange="CBOT", unit="USD/short ton"),
    "ZL=F": CommoditySpec(name=OIL, ticker="ZL=F", exchange="CBOT", unit="US cents/lb"),
    "HO=F": CommoditySpec(name=DIESEL, ticker="HO=F", exchange="NYMEX", unit="USD/gallon"),
}

tickers: list[str] = list(COMMODITIES)
commodities_dict: dict[str, str] = {ticker: spec.name for ticker, spec in COMMODITIES.items()}

# Exact definitions: 1 lb = 0.45359237 kg, a bushel of soybeans is 60 lb, a short ton
# 2,000 lb and a US gallon 3.785411784 litres.
LB_PER_TONNE: Final[float] = 1000 / 0.45359237
BUSHELS_PER_TONNE: Final[float] = LB_PER_TONNE / 60
SHORT_TONS_PER_TONNE: Final[float] = LB_PER_TONNE / 2000
GALLONS_PER_M3: Final[float] = 1000 / 3.785411784

# CME board crush yields: 44 lb of meal and 11 lb of oil per 60 lb bushel.
MEAL_YIELD: Final[float] = 44 / 60
OIL_YIELD: Final[float] = 11 / 60

# Tonnes of soybean oil per m3 of biodiesel. EIA plant data for 2017 to 2020 average
# 7.50 lb per gallon (farmdoc daily, 16 Feb 2022), which is 0.899 t/m3.
OIL_PER_M3_BIODIESEL: Final[float] = 0.90

# EPA equivalence value for biodiesel under the Renewable Fuel Standard (40 CFR 80.1415).
RINS_PER_GALLON_BIODIESEL: Final[float] = 1.5

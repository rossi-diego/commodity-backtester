"""
Commodity constants and contract specifications.

This module contains all commodity-related constants including:
- Ticker symbols for Yahoo Finance
- Contract specifications (sizes, conversion factors)
- Commodity metadata

The architecture supports easy extension for new commodities.
"""

from dataclasses import dataclass
from typing import Dict, List, Final


@dataclass(frozen=True)
class CommoditySpec:
    """
    Specification for a single commodity contract.

    Attributes:
        ticker: Yahoo Finance ticker symbol
        name: Human-readable commodity name
        exchange: Exchange where the commodity trades
        contract_size: Size of one contract in exchange units
        tons_conversion: Factor to convert price to $/metric ton
        currency: Price currency
        unit: Price quotation unit
    """
    ticker: str
    name: str
    exchange: str
    contract_size: float
    tons_conversion: float
    currency: str = "USD"
    unit: str = "cents/bushel"


# =============================================================================
# Commodity Specifications Database
# =============================================================================

COMMODITIES: Dict[str, CommoditySpec] = {
    # CBOT Grains
    "ZSN25.CBT": CommoditySpec(
        ticker="ZSN25.CBT",
        name="Soybean",
        exchange="CBOT",
        contract_size=136.0,
        tons_conversion=36.74 / 100,
        unit="cents/bushel"
    ),
    "ZCN25.CBT": CommoditySpec(
        ticker="ZCN25.CBT",
        name="Corn",
        exchange="CBOT",
        contract_size=127.0,
        tons_conversion=39.37 / 100,
        unit="cents/bushel"
    ),
    "KEN25.CBT": CommoditySpec(
        ticker="KEN25.CBT",
        name="Wheat",
        exchange="CBOT",
        contract_size=136.0,
        tons_conversion=36.74 / 100,
        unit="cents/bushel"
    ),

    # CBOT Soy Complex
    "ZMN25.CBT": CommoditySpec(
        ticker="ZMN25.CBT",
        name="Soybean Meal",
        exchange="CBOT",
        contract_size=0.9072,
        tons_conversion=1.1023,
        unit="$/short ton"
    ),
    "ZLN25.CBT": CommoditySpec(
        ticker="ZLN25.CBT",
        name="Soybean Oil",
        exchange="CBOT",
        contract_size=27.22,
        tons_conversion=22.0462,
        unit="cents/lb"
    ),

    # NYMEX Energy
    "HON25.NYM": CommoditySpec(
        ticker="HON25.NYM",
        name="Heating Oil",
        exchange="NYMEX",
        contract_size=133.0,
        tons_conversion=312.5,
        unit="$/gallon"
    ),
    "CLN25.NYM": CommoditySpec(
        ticker="CLN25.NYM",
        name="Crude Oil",
        exchange="NYMEX",
        contract_size=136.4,
        tons_conversion=7.4,
        unit="$/barrel"
    ),
}


# =============================================================================
# Derived Constants (for backward compatibility)
# =============================================================================

def get_tickers() -> List[str]:
    """Return list of all available ticker symbols."""
    return list(COMMODITIES.keys())


def get_commodities_dict() -> Dict[str, str]:
    """Return mapping of ticker to display name."""
    return {spec.ticker: spec.name for spec in COMMODITIES.values()}


def get_contract_sizes() -> Dict[str, float]:
    """Return mapping of commodity name to contract size."""
    return {spec.name: spec.contract_size for spec in COMMODITIES.values()}


def get_tons_conversion() -> Dict[str, float]:
    """Return mapping of commodity name to tons conversion factor."""
    return {spec.name: spec.tons_conversion for spec in COMMODITIES.values()}


# Legacy exports for backward compatibility
tickers: Final[List[str]] = get_tickers()
commodities_dict: Final[Dict[str, str]] = get_commodities_dict()
contract_sizes: Final[Dict[str, float]] = get_contract_sizes()
tons_conversion: Final[Dict[str, float]] = get_tons_conversion()

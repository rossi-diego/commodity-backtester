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
    "ZS=F": CommoditySpec(
        ticker="ZS=F",
        name="Soybean",
        exchange="CBOT",
        contract_size=136.0,
        tons_conversion=36.74 / 100,
        unit="cents/bushel"
    ),
    "ZC=F": CommoditySpec(
        ticker="ZC=F",
        name="Corn",
        exchange="CBOT",
        contract_size=127.0,
        tons_conversion=39.37 / 100,
        unit="cents/bushel"
    ),
    "ZW=F": CommoditySpec(
        ticker="ZW=F",
        name="Wheat",
        exchange="CBOT",
        contract_size=136.0,
        tons_conversion=36.74 / 100,
        unit="cents/bushel"
    ),

    # CBOT Soy Complex
    "ZM=F": CommoditySpec(
        ticker="ZM=F",
        name="Soybean Meal",
        exchange="CBOT",
        contract_size=0.9072,
        tons_conversion=1.1023,
        unit="$/short ton"
    ),
    "ZL=F": CommoditySpec(
        ticker="ZL=F",
        name="Soybean Oil",
        exchange="CBOT",
        contract_size=27.22,
        tons_conversion=22.0462,
        unit="cents/lb"
    ),

    # NYMEX Energy
    "HO=F": CommoditySpec(
        ticker="HO=F",
        name="Heating Oil",
        exchange="NYMEX",
        contract_size=133.0,
        tons_conversion=312.5,
        unit="$/gallon"
    ),
    "CL=F": CommoditySpec(
        ticker="CL=F",
        name="Crude Oil",
        exchange="NYMEX",
        contract_size=136.4,
        tons_conversion=7.4,
        unit="$/barrel"
    ),
    "NG=F": CommoditySpec(
        ticker="NG=F",
        name="Natural Gas",
        exchange="NYMEX",
        contract_size=10000.0,
        tons_conversion=1.0,
        unit="$/MMBtu"
    ),
    "GC=F": CommoditySpec(
        ticker="GC=F",
        name="Gold",
        exchange="COMEX",
        contract_size=100.0,
        tons_conversion=32.1507,
        unit="$/troy oz"
    ),
    "SI=F": CommoditySpec(
        ticker="SI=F",
        name="Silver",
        exchange="COMEX",
        contract_size=5000.0,
        tons_conversion=32.1507,
        unit="$/troy oz"
    ),
    "HG=F": CommoditySpec(
        ticker="HG=F",
        name="Copper",
        exchange="COMEX",
        contract_size=25000.0,
        tons_conversion=2204.62,
        unit="$/lb"
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

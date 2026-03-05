"""
Commodity contract reference data.

Tickers use continuous futures notation (``=F``) so they automatically roll
to the nearest active contract and never expire.  All unit conversions are
based on official CME/CBOT contract specifications.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommoditySpec:
    """Metadata for a single commodity futures contract."""

    name: str
    ticker: str
    exchange: str
    unit: str


tickers: list[str] = [
    "ZS=F", "ZC=F", "KE=F", "ZM=F", "ZL=F", "HO=F", "CL=F",
]

COMMODITIES: dict[str, CommoditySpec] = {
    "ZS=F": CommoditySpec(name="Soybean",        ticker="ZS=F", exchange="CBOT",  unit="cents/bushel"),
    "ZC=F": CommoditySpec(name="Corn",            ticker="ZC=F", exchange="CBOT",  unit="cents/bushel"),
    "KE=F": CommoditySpec(name="KC HRW Wheat",    ticker="KE=F", exchange="CBOT",  unit="cents/bushel"),
    "ZM=F": CommoditySpec(name="Soybean Meal",    ticker="ZM=F", exchange="CBOT",  unit="USD/short-ton"),
    "ZL=F": CommoditySpec(name="Soybean Oil",     ticker="ZL=F", exchange="CBOT",  unit="cents/lb"),
    "HO=F": CommoditySpec(name="Heating Oil",     ticker="HO=F", exchange="NYMEX", unit="USD/gallon"),
    "CL=F": CommoditySpec(name="Crude Oil (WTI)", ticker="CL=F", exchange="NYMEX", unit="USD/barrel"),
}

commodities_dict: dict[str, str] = {t: s.name for t, s in COMMODITIES.items()}

contract_sizes: dict[str, float] = {
    "Soybean":         136.08,
    "Corn":            127.01,
    "KC HRW Wheat":    136.08,
    "Soybean Meal":     90.72,
    "Soybean Oil":      27.22,
    "Heating Oil":     133.60,
    "Crude Oil (WTI)": 135.10,
}

tons_conversion: dict[str, float] = {
    "Soybean":          0.36744,
    "Corn":             0.39368,
    "KC HRW Wheat":     0.36744,
    "Soybean Meal":     1.10231,
    "Soybean Oil":     22.04620,
    "Heating Oil":    317.77000,
    "Crude Oil (WTI)":  7.33000,
}

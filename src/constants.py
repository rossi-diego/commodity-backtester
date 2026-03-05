"""
Commodity contract reference data.

Tickers use continuous futures notation (``=F``) so they automatically roll
to the nearest active contract and never expire.  All unit conversions are
based on official CME/CBOT contract specifications.

Unit convention
---------------
- Grain prices (ZS, ZC, KE, ZM, ZL): quoted in USD/bushel (cents) or USD/short-ton
- Energy prices (HO, CL): quoted in USD/gallon or USD/barrel
- ``tons_conversion[name]`` converts the native price unit to USD/metric-ton
- ``contract_sizes[name]`` is the number of metric tons per contract
"""

# ---------------------------------------------------------------------------
# Continuous futures tickers (yfinance)
# ---------------------------------------------------------------------------
tickers: list[str] = [
    "ZS=F",   # Soybean
    "ZC=F",   # Corn
    "KE=F",   # KC HRW Wheat (hard red winter)
    "ZM=F",   # Soybean Meal
    "ZL=F",   # Soybean Oil
    "HO=F",   # Heating Oil (NY Harbor ULSD)
    "CL=F",   # Crude Oil (WTI)
]

# ---------------------------------------------------------------------------
# Human-readable commodity names (keyed by ticker)
# ---------------------------------------------------------------------------
commodities_dict: dict[str, str] = {
    "ZS=F": "Soybean",
    "ZC=F": "Corn",
    "KE=F": "KC HRW Wheat",
    "ZM=F": "Soybean Meal",
    "ZL=F": "Soybean Oil",
    "HO=F": "Heating Oil",
    "CL=F": "Crude Oil (WTI)",
}

# ---------------------------------------------------------------------------
# Contract size in metric tons per contract
#
# CBOT grains: 5 000 bushels/contract
#   Soybean:      5 000 bu × 0.027 216 MT/bu  = 136.08 MT
#   Corn:         5 000 bu × 0.025 401 MT/bu  = 127.01 MT
#   HRW Wheat:    5 000 bu × 0.027 216 MT/bu  = 136.08 MT
#   Soybean Meal: 100 short tons × 0.907 185 MT/st = 90.72 MT
#   Soybean Oil:  60 000 lbs × 0.000 453 592 MT/lb = 27.22 MT
# Energy (NYM):
#   Heating Oil:  42 000 US gallons × 0.003 785 m³/gal × 0.840 MT/m³ ≈ 133.6 MT
#   Crude Oil:    1 000 barrels × 0.158 987 m³/bbl × 0.850 MT/m³ ≈ 135.1 MT
# ---------------------------------------------------------------------------
contract_sizes: dict[str, float] = {
    "Soybean":          136.08,
    "Corn":             127.01,
    "KC HRW Wheat":     136.08,
    "Soybean Meal":      90.72,
    "Soybean Oil":       27.22,
    "Heating Oil":      133.60,
    "Crude Oil (WTI)":  135.10,
}

# ---------------------------------------------------------------------------
# Price-to-USD-per-metric-ton conversion factors
#
# Grain prices are quoted in cents/bushel by yfinance.
#   factor = 100 (cents→dollars) / bushel-weight-in-MT
#   e.g. Soybean: 100 / 27.216 kg/bu = 3.6744 → divide cents/bu by 100
#        then multiply by (1 / 0.027216) to get $/MT
#   Simplified: $/MT = (cents/bu) × (1 bu / 0.027216 MT) / 100
#             = (cents/bu) × 0.36744
#
# Soybean Meal is quoted in $/short-ton → $/MT = price / 0.907185 = price × 1.10231
# Soybean Oil is quoted in cents/lb → $/MT = (cents/lb) × 22.0462
# Heating Oil is quoted in USD/gallon → $/MT ≈ price × 317.77 (density 0.840 MT/m³)
# Crude Oil is quoted in USD/barrel → $/MT ≈ price × 7.33 (API 35°, ~0.851 MT/m³)
# ---------------------------------------------------------------------------
tons_conversion: dict[str, float] = {
    "Soybean":          0.36744,    # cents/bu → $/MT
    "Corn":             0.39368,    # cents/bu → $/MT
    "KC HRW Wheat":     0.36744,    # cents/bu → $/MT
    "Soybean Meal":     1.10231,    # $/short-ton → $/MT
    "Soybean Oil":     22.04620,    # cents/lb → $/MT
    "Heating Oil":    317.77000,    # $/gallon → $/MT
    "Crude Oil (WTI)":  7.33000,    # $/barrel → $/MT
}

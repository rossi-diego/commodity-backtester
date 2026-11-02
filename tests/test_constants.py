"""Contract reference data and conversion factors."""

from __future__ import annotations

import re

import pytest

from src.constants import (
    BUSHELS_PER_TONNE,
    COMMODITIES,
    GALLONS_PER_M3,
    LB_PER_TONNE,
    MEAL_YIELD,
    OIL_PER_M3_BIODIESEL,
    OIL_YIELD,
    SHORT_TONS_PER_TONNE,
    commodities_dict,
    tickers,
)

KG_PER_LB = 0.45359237
LITRES_PER_GALLON = 3.785411784


class TestTickers:
    def test_four_crush_legs(self) -> None:
        assert tickers == ["ZS=F", "ZM=F", "ZL=F", "HO=F"]
        assert list(commodities_dict.values()) == [
            "Soybean",
            "Soybean Meal",
            "Soybean Oil",
            "Heating Oil",
        ]

    def test_continuous_without_month_codes(self) -> None:
        for ticker in tickers:
            assert ticker.endswith("=F")
            assert not re.search(r"[FGHJKMNQUVXZ]\d{2}", ticker)

    def test_units_match_the_exchange_quotes(self) -> None:
        units = {spec.name: spec.unit for spec in COMMODITIES.values()}
        assert units["Soybean"] == "US cents/bushel"
        assert units["Soybean Meal"] == "USD/short ton"
        assert units["Soybean Oil"] == "US cents/lb"
        assert units["Heating Oil"] == "USD/gallon"


class TestConversions:
    def test_bushels_per_tonne(self) -> None:
        assert pytest.approx(1000, rel=1e-4) == 60 * KG_PER_LB * BUSHELS_PER_TONNE

    def test_mass_and_volume(self) -> None:
        assert pytest.approx(1000, rel=1e-5) == LB_PER_TONNE * KG_PER_LB
        assert pytest.approx(1000, rel=1e-4) == SHORT_TONS_PER_TONNE * 2000 * KG_PER_LB
        assert pytest.approx(1000, rel=1e-5) == GALLONS_PER_M3 * LITRES_PER_GALLON

    def test_board_crush_yields_match_cme_factors(self) -> None:
        """CME: 0.022 x meal (USD/short ton) and 0.11 x oil (cents/lb) per bushel."""
        meal_short_tons_per_bushel = MEAL_YIELD * 60 / 2000
        oil_lb_per_bushel = OIL_YIELD * 60
        assert meal_short_tons_per_bushel == pytest.approx(0.022)
        assert oil_lb_per_bushel == pytest.approx(11)

    def test_biodiesel_factor_is_7_5_lb_per_gallon(self) -> None:
        tonnes_per_m3 = 7.5 * KG_PER_LB / LITRES_PER_GALLON
        assert pytest.approx(tonnes_per_m3, abs=0.005) == OIL_PER_M3_BIODIESEL

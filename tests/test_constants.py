"""Unit tests for src/constants.py — contract reference data integrity."""

from __future__ import annotations

from src.constants import commodities_dict, contract_sizes, tickers, tons_conversion


class TestTickers:
    def test_all_tickers_are_continuous(self) -> None:
        """All tickers must use the continuous futures ``=F`` suffix."""
        for ticker in tickers:
            assert ticker.endswith("=F"), (
                f"Ticker '{ticker}' is not a continuous contract. "
                "Use the '=F' suffix (e.g. 'ZS=F') to avoid expiry issues."
            )

    def test_no_month_year_in_ticker(self) -> None:
        """Hardcoded month codes (N25, Z24, etc.) must not appear in tickers."""
        import re

        month_pattern = re.compile(r"[FGHJKMNQUVXZ]\d{2}")
        for ticker in tickers:
            assert not month_pattern.search(ticker), (
                f"Ticker '{ticker}' contains a month/year code. "
                "Switch to a continuous contract ticker."
            )


class TestCommoditiesDict:
    def test_every_ticker_has_a_display_name(self) -> None:
        for ticker in tickers:
            assert ticker in commodities_dict, f"No display name for ticker '{ticker}'."

    def test_display_names_are_non_empty(self) -> None:
        for ticker, name in commodities_dict.items():
            assert name.strip(), f"Empty display name for ticker '{ticker}'."

    def test_no_month_year_in_display_names(self) -> None:
        """Display names must not contain month-year strings like 'july 25'."""
        import re

        date_pattern = re.compile(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{2,4}\b", re.IGNORECASE
        )
        for ticker, name in commodities_dict.items():
            assert not date_pattern.search(name), (
                f"Display name '{name}' for ticker '{ticker}' contains a month/year. "
                "Use generic names (e.g. 'Soybean' not 'Soybean - july 25')."
            )


class TestContractSizes:
    def test_every_commodity_has_contract_size(self) -> None:
        display_names = set(commodities_dict.values())
        for name in display_names:
            assert name in contract_sizes, f"No contract size for commodity '{name}'."

    def test_contract_sizes_are_positive(self) -> None:
        for name, size in contract_sizes.items():
            assert size > 0, f"Contract size for '{name}' must be positive, got {size}."

    def test_grain_contract_sizes_within_range(self) -> None:
        """CBOT grain contracts are 5000 bushels; MT equivalent should be 90–140 MT."""
        grain_commodities = {"Soybean", "Corn", "KC HRW Wheat"}
        for name in grain_commodities:
            if name in contract_sizes:
                assert 90 < contract_sizes[name] < 145, (
                    f"Contract size for '{name}' ({contract_sizes[name]} MT) is outside "
                    "the expected range for a 5000-bushel CBOT grain contract (90–145 MT)."
                )


class TestTonsConversion:
    def test_every_commodity_has_conversion_factor(self) -> None:
        display_names = set(commodities_dict.values())
        for name in display_names:
            assert name in tons_conversion, f"No conversion factor for commodity '{name}'."

    def test_conversion_factors_are_positive(self) -> None:
        for name, factor in tons_conversion.items():
            assert factor > 0, f"Conversion factor for '{name}' must be positive, got {factor}."

    def test_grain_factors_in_expected_range(self) -> None:
        """Grain factors convert cents/bushel → $/MT (expected ~0.35–0.45)."""
        grain_commodities = {"Soybean", "Corn", "KC HRW Wheat"}
        for name in grain_commodities:
            if name in tons_conversion:
                assert 0.30 < tons_conversion[name] < 0.50, (
                    f"Conversion factor for '{name}' ({tons_conversion[name]}) is outside "
                    "the expected range for a grain commodity (cents/bu → $/MT: ~0.35–0.45)."
                )

    def test_soybean_oil_factor_in_expected_range(self) -> None:
        """Soybean oil factor converts cents/lb → $/MT (~22.0)."""
        if "Soybean Oil" in tons_conversion:
            assert 20 < tons_conversion["Soybean Oil"] < 25

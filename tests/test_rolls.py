"""Contract calendars and the sessions they flag."""

from __future__ import annotations

import pandas as pd

from src.rolls import (
    BEAN_MONTHS,
    BOARD_PAIRS,
    DIESEL_MONTHS,
    crush_flags,
    front_contracts,
    gap_flags,
    grain_first_notice,
    grain_last_trade,
    monthly_last_trade,
    off_pair,
)


class TestCalendars:
    def test_grain_last_trade_is_the_business_day_before_the_15th(self) -> None:
        assert grain_last_trade(2025, 11) == pd.Timestamp("2025-11-14")
        # 15 March 2026 is a Sunday, so the day before is Friday the 13th.
        assert grain_last_trade(2026, 3) == pd.Timestamp("2026-03-13")

    def test_grain_first_notice_is_the_last_business_day_before_the_month(self) -> None:
        assert grain_first_notice(2025, 11) == pd.Timestamp("2025-10-31")
        assert grain_first_notice(2026, 8) == pd.Timestamp("2026-07-31")

    def test_heating_oil_last_trade(self) -> None:
        assert monthly_last_trade(2026, 11) == pd.Timestamp("2026-10-30")
        assert monthly_last_trade(2026, 3) == pd.Timestamp("2026-02-27")


class TestFrontContracts:
    def test_soybeans_switch_after_the_november_2025_expiry(self) -> None:
        """Yahoo's ZS=F jumped from 1112.75 to 1157.25 between these two sessions."""
        dates = pd.DatetimeIndex(["2025-11-14", "2025-11-17"])
        front = front_contracts(dates, BEAN_MONTHS, grain_last_trade)
        assert list(zip(front["year"], front["month"], strict=True)) == [(2025, 11), (2026, 1)]

    def test_heating_oil_switch_after_month_end(self) -> None:
        dates = pd.DatetimeIndex(["2026-09-30", "2026-10-01"])
        front = front_contracts(dates, DIESEL_MONTHS, monthly_last_trade)
        assert list(front["month"]) == [10, 11]


class TestFlags:
    def test_delivery_and_switch_sessions_are_flagged(self) -> None:
        dates = pd.bdate_range("2025-10-20", "2025-11-28")
        flags = crush_flags(dates)
        # November beans are in delivery from first notice to the last trading day.
        assert flags.loc["2025-10-31":"2025-11-14"].all()
        # The first session on January beans is a switch.
        assert flags.loc["2025-11-17"]
        assert not flags.loc["2025-10-27"]
        assert not flags.loc["2025-11-20"]

    def test_known_expiry_artefact_is_flagged(self) -> None:
        """The front-month crush hit -1.82 USD/bu on 2008-09-12, inside September delivery."""
        flags = crush_flags(pd.bdate_range("2008-09-01", "2008-09-30"))
        assert flags.loc["2008-09-12"]

    def test_board_pair_from_mid_october_is_kept(self) -> None:
        """From 15 October November beans meet December products, the CME board pair; not flagged."""
        flags = crush_flags(pd.bdate_range("2026-10-15", "2026-10-29"))
        assert not flags.any()

    def test_crush_flags_cover_about_four_in_ten_sessions(self) -> None:
        share = crush_flags(pd.bdate_range("2010-01-01", "2019-12-31")).mean()
        assert 0.35 < share < 0.5

    def test_gap_flags_include_heating_oil_switches(self) -> None:
        flags = gap_flags(pd.bdate_range("2026-09-21", "2026-10-09"))
        assert flags.loc["2026-10-01"]
        # Soybean oil is in October delivery from 30 September, so the day before is clean.
        assert not flags.loc["2026-09-29"]


class TestBoardPairs:
    def test_board_pairs_cover_every_bean_month(self) -> None:
        assert set(BOARD_PAIRS) == set(BEAN_MONTHS)

    def test_off_pair_windows(self) -> None:
        """Late September (Nov beans, Oct products) and late November (Jan beans, Dec products)."""
        dates = pd.bdate_range("2025-09-01", "2025-12-31")
        off = off_pair(dates)
        flags = crush_flags(dates)
        assert off.name == "off_pair"
        for day in ("2025-09-22", "2025-11-20"):
            assert off.loc[day]
            assert not flags.loc[day]
        assert not off.loc["2025-10-20"]

    def test_off_pair_on_no_dates(self) -> None:
        assert off_pair(pd.DatetimeIndex([])).empty

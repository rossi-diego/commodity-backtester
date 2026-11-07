"""HTML builders for cards, rails and method notes."""

from __future__ import annotations

import re
from dataclasses import replace

import pandas as pd
import pytest

from src.cards import (
    Kpi,
    MethodCard,
    backtest_fails,
    backtest_line,
    kpi_row_html,
    method_row_html,
    ordinal,
    percentile_html,
    rail_html,
    risk_strip_html,
    section_head_html,
    split_bar_html,
)
from src.risk import Backtest, RiskSummary
from src.text import BACKTEST_LINE


@pytest.mark.parametrize(
    ("value", "text"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12.4, "12th"), (22, "22nd")],
)
def test_ordinal(value, text) -> None:
    assert ordinal(value) == text


class TestPercentile:
    def test_marker_sits_at_the_rank(self) -> None:
        markup = percentile_html(75.04)
        assert 'style="left:75.0%"' in markup
        assert 'aria-label="75th percentile of 5 years"' in markup

    def test_rank_names_its_series(self) -> None:
        markup = percentile_html(40.0, "before the RIN")
        assert 'aria-label="40th percentile of 5 years, before the RIN"' in markup
        assert "<span>40th percentile of 5 years, before the RIN</span>" in markup
        card = Kpi("gap", 3, "Gap", "$1", "", "", 40.0, "", rank_of="before the RIN")
        assert kpi_row_html([card]).count(", before the RIN") == 2

    def test_no_rank_keeps_the_row(self) -> None:
        markup = percentile_html(None)
        assert "cm-mark" not in markup
        assert "cm-track" in markup
        assert "n/a" in markup


def test_split_bar_matches_the_share() -> None:
    markup = split_bar_html(0.5, "$265/t", "$266/t")
    assert 'style="width:50.0%"' in markup
    assert "Oil $265/t" in markup
    assert "Meal $266/t" in markup


class TestKpiRow:
    def _card(self, **kwargs) -> Kpi:
        values = {
            "key": "crush",
            "number": 1,
            "label": "Crush margin",
            "value": "$83/t",
            "note": "per tonne",
            "detail": "a < b",
            "percentile": 50.0,
            "caption": "caption",
        }
        return Kpi(**(values | kwargs))

    def test_three_cards_link_to_their_sections(self) -> None:
        cards = [self._card(key=key) for key in ("crush", "share", "gap")]
        markup = kpi_row_html(cards)
        assert markup.count('class="cm-kpi ') == 3
        for key in ("crush", "share", "gap"):
            assert f'href="#{key}"' in markup

    def test_text_is_escaped_and_visual_replaces_detail(self) -> None:
        assert "a &lt; b" in kpi_row_html([self._card()])
        markup = kpi_row_html([self._card(visual="<b>bar</b>")])
        assert "<b>bar</b>" in markup
        assert "a &lt; b" not in markup

    def test_tag_only_when_given(self) -> None:
        assert "cm-tag" not in kpi_row_html([self._card()])
        assert "roll day" in kpi_row_html([self._card(tag="roll day")])


def test_section_head_is_the_anchor() -> None:
    markup = section_head_html("gap", 3, "Oil cost vs diesel", "Oil & diesel")
    assert 'id="gap"' in markup
    assert "Oil &amp; diesel" in markup


def test_rail_has_three_blocks() -> None:
    markup = rail_html("What", "Read", [("Board crush", "$2.27/bu")])
    assert markup.count("<h3>") == 3
    assert "<dt>Board crush</dt><dd>$2.27/bu</dd>" in markup


BACKTEST = Backtest(
    start=pd.Timestamp("2021-11-17"),
    end=pd.Timestamp("2026-10-29"),
    days=685,
    exceptions=33,
    hs_exceptions=65,
    kupiec_p=0.8312,
    christoffersen_p=0.2871,
)


class TestRiskStrip:
    def _summary(self, **kwargs) -> RiskSummary:
        values = {
            "date": pd.Timestamp("2026-11-05"),
            "vol_20": 4.567,
            "vol_1y": 5.0,
            "var": 8123.4,
            "es": 11234.6,
            "contributions": {"oil": 0.62, "meal": 0.71, "beans": -0.33},
            "changes": 700,
        }
        return RiskSummary(**(values | kwargs))

    def test_four_cells_with_signed_leg_shares(self) -> None:
        markup = risk_strip_html(self._summary(), "note", BACKTEST_LINE)
        assert markup.count('class="cm-risk-cell"') == 4
        assert "<dd>$4.57/t / $5.00/t</dd>" in markup
        assert "Daily σ, last 20 changes / 1 year" in markup
        assert "sessions" not in markup
        assert "1-day VaR 95%, 1,000 t</dt><dd>$8,123</dd>" in markup
        assert "ES 97.5%</dt><dd>$11,235</dd>" in markup
        assert re.sub(r"<[^>]+>", "", markup).count("oil +62% · meal +71% · beans −33%") == 1
        assert "n/a" not in markup

    def test_each_leg_is_kept_on_one_line(self) -> None:
        markup = risk_strip_html(self._summary(), "note", BACKTEST_LINE)
        expected = (
            '<dd><span class="cm-leg">oil +62%</span> · <span class="cm-leg">meal +71%</span>'
            ' · <span class="cm-leg">beans −33%</span></dd>'
        )
        assert expected in markup

    def test_note_starts_with_the_as_of_date(self) -> None:
        summary = self._summary(date=pd.Timestamp("2026-10-09"))
        markup = risk_strip_html(summary, "Note.", BACKTEST_LINE)
        expected = '<p class="cm-risk-note">As of 9 Oct 2026, the last clean change. Note.</p>'
        assert expected in markup

    def test_text_is_escaped(self) -> None:
        markup = risk_strip_html(self._summary(), "a < b & c", BACKTEST_LINE)
        assert "the last clean change. a &lt; b &amp; c</p>" in markup
        assert "a < b" not in markup

    def test_no_summary_is_one_line(self) -> None:
        markup = risk_strip_html(None, "note", BACKTEST_LINE)
        assert '<p class="cm-risk-note">note</p>' in markup
        assert "As of" not in markup
        assert "cm-risk-cell" not in markup
        assert "cm-risk-test" not in markup
        assert "n/a: under 250 clean daily changes" in markup

    def test_a_rounded_zero_has_no_sign(self) -> None:
        summary = self._summary(contributions={"oil": 0.002, "meal": 1.0, "beans": -0.002})
        text = re.sub(r"<[^>]+>", "", risk_strip_html(summary, "", BACKTEST_LINE))
        assert "oil 0% · meal +100% · beans 0%" in text


class TestBacktestLine:
    def _strip(self, test: Backtest | None, template: str = BACKTEST_LINE) -> str:
        summary = TestRiskStrip()._summary(backtest=test)
        return risk_strip_html(summary, "Note.", template)

    def test_a_passing_backtest_holds(self) -> None:
        expected = (
            '</dl><p class="cm-risk-test"><span class="cm-chip cm-chip-pass">Holds</span> '
            "Backtest, 5 years: 33 of 685 changes beyond the VaR (4.8%, 5% expected); "
            'Kupiec p 0.83, independence p 0.29.</p><p class="cm-risk-note">'
        )
        markup = self._strip(BACKTEST)
        assert expected in markup
        assert markup.count("cm-risk-test") == 1
        assert markup.count('class="cm-risk-cell"') == 4

    @pytest.mark.parametrize(
        ("kupiec", "independence"), [(0.049, 0.5), (0.5, 0.049), (0.001, 0.001)]
    )
    def test_either_test_failing_under_covers(self, kupiec, independence) -> None:
        test = replace(BACKTEST, kupiec_p=kupiec, christoffersen_p=independence)
        assert backtest_fails(test)
        markup = self._strip(test)
        assert '<span class="cm-chip cm-chip-fail">Under-covers</span>' in markup
        assert "Holds" not in markup

    def test_the_threshold_itself_holds(self) -> None:
        assert not backtest_fails(replace(BACKTEST, kupiec_p=0.05, christoffersen_p=0.05))

    def test_small_p_values_read_under_a_hundredth(self) -> None:
        test = replace(BACKTEST, exceptions=65, kupiec_p=0.0004, christoffersen_p=0.0099)
        line = backtest_line(test, BACKTEST_LINE)
        assert line.endswith("(9.5%, 5% expected); Kupiec p < 0.01, independence p < 0.01.")
        assert "Kupiec p &lt; 0.01, independence p &lt; 0.01." in self._strip(test)
        assert backtest_line(replace(BACKTEST, kupiec_p=0.01), BACKTEST_LINE).count("p 0.01") == 1

    def test_no_backtest_no_line(self) -> None:
        markup = self._strip(None)
        assert "cm-risk-test" not in markup
        assert "cm-chip" not in markup
        assert "Backtest" not in markup

    def test_line_is_escaped(self) -> None:
        markup = self._strip(BACKTEST, "<b>{exceptions}</b> & {days:,}")
        assert "&lt;b&gt;33&lt;/b&gt; &amp; 685</p>" in markup
        assert "<b>" not in markup

    def test_line_is_built_only_from_the_backtest(self) -> None:
        test = replace(BACKTEST, days=12_345, exceptions=617)
        line = backtest_line(test, BACKTEST_LINE)
        assert "617 of 12,345 changes" in line
        assert f"({test.rate:.1%}," in line


def test_method_row() -> None:
    card = MethodCard("Title", (("Lead", "text"),), "closing")
    markup = method_row_html([card, card, card])
    assert markup.count("<section>") == 3
    assert "<b>Lead:</b> text" in markup

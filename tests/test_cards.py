"""HTML builders for cards, rails and method notes."""

from __future__ import annotations

import pandas as pd
import pytest

from src.cards import (
    Kpi,
    MethodCard,
    kpi_row_html,
    method_row_html,
    ordinal,
    percentile_html,
    rail_html,
    risk_strip_html,
    section_head_html,
    split_bar_html,
)
from src.risk import RiskSummary


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
        markup = risk_strip_html(self._summary(), "note")
        assert markup.count('class="cm-risk-cell"') == 4
        assert "<dd>$4.57/t / $5.00/t</dd>" in markup
        assert "Daily σ, last 20 changes / 1 year" in markup
        assert "sessions" not in markup
        assert "1-day VaR 95%, 1,000 t</dt><dd>$8,123</dd>" in markup
        assert "ES 97.5%</dt><dd>$11,235</dd>" in markup
        assert "oil +62% · meal +71% · beans −33%" in markup
        assert "n/a" not in markup

    def test_note_starts_with_the_as_of_date(self) -> None:
        markup = risk_strip_html(self._summary(date=pd.Timestamp("2026-10-09")), "Note.")
        expected = '<p class="cm-risk-note">As of 9 Oct 2026, the last clean change. Note.</p>'
        assert expected in markup

    def test_text_is_escaped(self) -> None:
        markup = risk_strip_html(self._summary(), "a < b & c")
        assert "the last clean change. a &lt; b &amp; c</p>" in markup
        assert "a < b" not in markup

    def test_no_summary_is_one_line(self) -> None:
        markup = risk_strip_html(None, "note")
        assert '<p class="cm-risk-note">note</p>' in markup
        assert "As of" not in markup
        assert "cm-risk-cell" not in markup
        assert "n/a: under 250 clean daily changes" in markup

    def test_a_rounded_zero_has_no_sign(self) -> None:
        summary = self._summary(contributions={"oil": 0.002, "meal": 1.0, "beans": -0.002})
        assert "oil 0% · meal +100% · beans 0%" in risk_strip_html(summary, "")


def test_method_row() -> None:
    card = MethodCard("Title", (("Lead", "text"),), "closing")
    markup = method_row_html([card, card, card])
    assert markup.count("<section>") == 3
    assert "<b>Lead:</b> text" in markup

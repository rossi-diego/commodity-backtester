"""HTML builders for cards, rails and method notes."""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pandas as pd
import pytest

from src.cards import (
    VERDICT_CLASS,
    Kpi,
    MethodCard,
    backtest_line,
    backtest_verdict,
    es_backtest_line,
    kpi_row_html,
    method_row_html,
    ordinal,
    percentile_html,
    rail_html,
    risk_effect,
    risk_strip_html,
    section_head_html,
    split_bar_html,
    worst_line,
)
from src.risk import Backtest, RiskSummary, SampleBias, kupiec_pof
from src.text import BACKTEST_LINE, ES_BACKTEST_LINE, RISK_EFFECT, RISK_STRIP, WORST_LINE


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
    es_exceptions=14,
    es_kupiec_p=0.4299,
    es_christoffersen_p=0.0285,
    es_ratio=0.9745,
    z2=0.2034,
    z2_p=0.842,
)

BIAS = SampleBias(
    kept=685,
    excluded=436,
    kept_sd=3.53,
    excluded_sd=4.49,
    combined_changes=1121,
    combined_var=5450.6,
    excluded_tested=436,
    excluded_exceptions=27,
    excluded_kupiec_p=0.27,
)

WORST = (
    (pd.Timestamp("2023-07-26"), -11743.3),
    (pd.Timestamp("2023-03-23"), -10078.8),
    (pd.Timestamp("2026-06-05"), -10034.7),
)


def _copy(note: str = "note", test: str = BACKTEST_LINE):
    return replace(RISK_STRIP, note=note, var_test=test)


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
        markup = risk_strip_html(self._summary(), _copy())
        assert "Share of ES by leg" in markup
        assert markup.count('class="cm-risk-cell"') == 4
        assert "<dd>$4.57/t / $5.00/t</dd>" in markup
        assert "Daily σ, last 20 changes / 1 year" in markup
        assert "sessions" not in markup
        assert "1-day VaR 95%, 1,000 t</dt><dd>$8,123</dd>" in markup
        assert "ES 97.5%</dt><dd>$11,235</dd>" in markup
        assert re.sub(r"<[^>]+>", "", markup).count("oil +62% · meal +71% · beans −33%") == 1
        assert "n/a" not in markup

    def test_each_leg_is_kept_on_one_line(self) -> None:
        markup = risk_strip_html(self._summary(), _copy())
        expected = (
            '<dd><span class="cm-leg">oil +62%</span> · <span class="cm-leg">meal +71%</span>'
            ' · <span class="cm-leg">beans −33%</span></dd>'
        )
        assert expected in markup

    def test_note_starts_with_the_as_of_date(self) -> None:
        summary = self._summary(date=pd.Timestamp("2026-10-09"))
        markup = risk_strip_html(summary, _copy("Note."))
        expected = '<p class="cm-risk-note">As of 9 Oct 2026, the last clean change. Note.</p>'
        assert expected in markup

    def test_text_is_escaped(self) -> None:
        markup = risk_strip_html(self._summary(), _copy("a < b & c"))
        assert "the last clean change. a &lt; b &amp; c</p>" in markup
        assert "a < b" not in markup

    def test_no_summary_is_one_line(self) -> None:
        markup = risk_strip_html(None, _copy())
        assert '<p class="cm-risk-note">note</p>' in markup
        assert "As of" not in markup
        assert "cm-risk-cell" not in markup
        assert "cm-risk-test" not in markup
        assert "n/a: under 250 clean daily changes" in markup

    def test_a_rounded_zero_has_no_sign(self) -> None:
        summary = self._summary(contributions={"oil": 0.002, "meal": 1.0, "beans": -0.002})
        text = re.sub(r"<[^>]+>", "", risk_strip_html(summary, _copy("")))
        assert "oil 0% · meal +100% · beans 0%" in text


class TestRiskNote:
    @pytest.mark.parametrize(
        ("combined", "effect"),
        [(5450.6, "lower"), (9000.0, "raise"), (math.nan, "none")],
    )
    def test_the_effect_follows_the_combined_var(self, combined, effect) -> None:
        summary = TestRiskStrip()._summary(bias=replace(BIAS, combined_var=combined))
        assert risk_effect(summary) == effect
        markup = risk_strip_html(summary, RISK_STRIP)
        assert f"are left out; {RISK_EFFECT[effect]}. Crusher&#x27;s margin" in markup

    def test_no_bias_or_no_summary_is_none(self) -> None:
        assert risk_effect(None) == "none"
        assert risk_effect(TestRiskStrip()._summary()) == "none"
        assert "too few to test" in risk_strip_html(None, RISK_STRIP)
        assert "{effect}" not in risk_strip_html(None, RISK_STRIP)


class TestWorstLine:
    def test_the_worst_days_come_after_the_tests_and_before_the_note(self) -> None:
        summary = TestRiskStrip()._summary(backtest=BACKTEST, worst=WORST)
        markup = risk_strip_html(summary, RISK_STRIP)
        expected = (
            '<p class="cm-risk-test">Worst clean days, 5 years, 1,000 t: −$11,743 (26 Jul 2023), '
            '−$10,079 (23 Mar 2023), −$10,035 (5 Jun 2026).</p><p class="cm-risk-note">'
        )
        assert expected in markup
        assert markup.count('class="cm-risk-test"') == 3
        assert markup.index("ES backtest") < markup.index("Worst clean days")
        assert markup.count('class="cm-risk-cell"') == 4

    def test_no_worst_day_no_line(self) -> None:
        markup = risk_strip_html(TestRiskStrip()._summary(backtest=BACKTEST), RISK_STRIP)
        assert "Worst" not in markup
        assert markup.count('class="cm-risk-test"') == 2

    def test_the_line_is_built_from_the_days(self) -> None:
        line = worst_line(WORST[:1], WORST_LINE)
        assert line == "Worst clean days, 5 years, 1,000 t: −$11,743 (26 Jul 2023)."


def _verdict(exceptions: int, days: int = 685, christoffersen: float = 0.5) -> Backtest:
    kupiec = kupiec_pof(exceptions, days)
    return replace(
        BACKTEST,
        days=days,
        exceptions=exceptions,
        kupiec_p=kupiec,
        christoffersen_p=christoffersen,
    )


class TestBacktestLine:
    def _strip(self, test: Backtest | None, template: str = BACKTEST_LINE) -> str:
        summary = TestRiskStrip()._summary(backtest=test)
        return risk_strip_html(summary, _copy("Note.", template))

    def test_a_passing_backtest_holds(self) -> None:
        expected = (
            '</dl><p class="cm-risk-test"><span class="cm-chip cm-chip-pass">Holds</span> '
            "Backtest, 5 years: 33 of 685 changes beyond the VaR (4.8%, 5% expected); "
            'Kupiec p 0.83, independence p 0.29.</p><p class="cm-risk-test">'
        )
        markup = self._strip(BACKTEST)
        assert expected in markup
        assert markup.count("cm-risk-test") == 2
        assert markup.count('class="cm-risk-cell"') == 4

    @pytest.mark.parametrize(
        ("test", "verdict"),
        [
            (_verdict(70), "Under-covers"),
            (_verdict(15), "Over-conservative"),
            (_verdict(34, christoffersen=0.01), "Clusters"),
            (_verdict(34), "Holds"),
            # Too many exceptions and clustered: coverage comes first.
            (_verdict(70, christoffersen=0.01), "Under-covers"),
        ],
    )
    def test_each_verdict(self, test, verdict) -> None:
        assert backtest_verdict(test) == verdict
        chip = f'<span class="cm-chip {VERDICT_CLASS[verdict]}">{verdict}</span> Backtest,'
        assert chip in self._strip(test)

    def test_verdict_classes(self) -> None:
        assert VERDICT_CLASS == {
            "Holds": "cm-chip-pass",
            "Under-covers": "cm-chip-fail",
            "Clusters": "cm-chip-fail",
            "Over-conservative": "cm-chip-over",
        }

    def test_the_threshold_itself_holds(self) -> None:
        test = replace(BACKTEST, kupiec_p=0.05, christoffersen_p=0.05)
        assert backtest_verdict(test) == "Holds"

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
        assert "ES backtest" not in markup

    def test_line_is_escaped(self) -> None:
        markup = self._strip(BACKTEST, "<b>{exceptions}</b> & {days:,}")
        assert "&lt;b&gt;33&lt;/b&gt; &amp; 685</p>" in markup
        assert "<b>" not in markup

    def test_line_is_built_only_from_the_backtest(self) -> None:
        test = replace(BACKTEST, days=12_345, exceptions=617)
        line = backtest_line(test, BACKTEST_LINE)
        assert "617 of 12,345 changes" in line
        assert f"({test.rate:.1%}," in line


class TestEsBacktestLine:
    def _strip(self, test: Backtest) -> str:
        return risk_strip_html(TestRiskStrip()._summary(backtest=test), RISK_STRIP)

    def test_the_es_line_sits_under_the_var_line(self) -> None:
        expected = (
            'Kupiec p 0.83, independence p 0.29.</p><p class="cm-risk-test">'
            '<span class="cm-chip cm-chip-pass">Holds</span> ES backtest, 5 years: 14 of 685 '
            "changes beyond the 97.5% VaR (2.0%, 2.5% expected); mean loss 0.97× ES; "
            "Acerbi–Szekely p\N{NO-BREAK SPACE}0.84.</p>"
        )
        assert expected in self._strip(BACKTEST)

    @pytest.mark.parametrize(("p", "verdict"), [(0.05, "Holds"), (0.0499, "Under-covers")])
    def test_the_es_chip_reads_the_simulated_p(self, p, verdict) -> None:
        test = replace(BACKTEST, z2_p=p)
        chip = f'<span class="cm-chip {VERDICT_CLASS[verdict]}">{verdict}</span> ES backtest'
        assert chip in self._strip(test)

    def test_no_es_exception_has_no_severity(self) -> None:
        test = replace(BACKTEST, es_exceptions=0, es_ratio=math.nan, z2=1.0, z2_p=1.0)
        line = es_backtest_line(test, ES_BACKTEST_LINE)
        assert "0 of 685 changes" in line
        assert "(0.0%, 2.5% expected); mean loss n/a;" in line
        assert line.endswith("p\N{NO-BREAK SPACE}1.00.")


def test_method_row() -> None:
    card = MethodCard("Title", (("Lead", "text"),), "closing")
    markup = method_row_html([card, card, card])
    assert markup.count("<section>") == 3
    assert "<b>Lead:</b> text" in markup

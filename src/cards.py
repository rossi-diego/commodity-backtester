"""HTML for the cards, rails and method notes. Pure functions, no UI imports.

The page renders these with ``st.html``, which strips scripts and handlers, so
everything here is plain markup styled by the page CSS. Runtime text goes
through ``html.escape``; markup arguments come only from this module.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd

from .risk import (
    ES_LEVEL,
    MIN_CHANGES,
    NOTIONAL_TONNES,
    SHORT_VOL_CHANGES,
    VAR_LEVEL,
    Backtest,
    RiskSummary,
    percent_shares,
)
from .seasonal import LOOKBACK_YEARS

__all__ = [
    "BACKTEST_SIGNIFICANCE",
    "VERDICT_CLASS",
    "Kpi",
    "MethodCard",
    "RiskCopy",
    "Verdict",
    "backtest_line",
    "backtest_verdict",
    "es_backtest_line",
    "kpi_row_html",
    "method_row_html",
    "ordinal",
    "percentile_html",
    "rail_html",
    "risk_effect",
    "risk_strip_html",
    "section_head_html",
    "split_bar_html",
    "worst_line",
]

# A p-value under this rejects the VaR: too many or too few exceptions, or clusters.
BACKTEST_SIGNIFICANCE: Final = 0.05

Verdict = Literal["Holds", "Under-covers", "Over-conservative", "Clusters"]
VERDICT_CLASS: Final[dict[Verdict, str]] = {
    "Holds": "cm-chip-pass",
    "Under-covers": "cm-chip-fail",
    "Clusters": "cm-chip-fail",
    "Over-conservative": "cm-chip-over",
}


@dataclass(frozen=True)
class Kpi:
    """One headline card. ``visual`` is markup from this module; the rest is text.

    ``rank_of`` names the series the percentile ranks, when it is not the headline value.
    """

    key: str
    number: int
    label: str
    value: str
    note: str
    detail: str
    percentile: float | None
    caption: str
    visual: str = ""
    tag: str = ""
    rank_of: str = ""


@dataclass(frozen=True)
class MethodCard:
    """A method note: a title, three lead/sentence entries (markup) and a closing line."""

    title: str
    entries: tuple[tuple[str, str], ...]
    closing: str


@dataclass(frozen=True)
class RiskCopy:
    """The risk strip's copy, from ``src.text``: templates filled by this module.

    ``note`` takes an ``{effect}``, one of ``effects`` keyed lower, raise or none.
    """

    note: str
    var_test: str
    es_test: str
    worst: str
    effects: Mapping[str, str]


def ordinal(value: float) -> str:
    number = round(value)
    suffix = (
        "th" if 11 <= number % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )
    return f"{number}{suffix}"


def percentile_html(percentile: float | None, rank_of: str = "") -> str:
    """A track with a marker where today ranks among clean sessions in 5 years.

    ``rank_of`` is appended to the label, to name the series that is ranked.
    """
    if percentile is None:
        return (
            '<div class="cm-rank" role="img" aria-label="Not enough history to rank">'
            '<div class="cm-track"></div><span>Rank n/a: under a year of history</span></div>'
        )
    text = f"{ordinal(percentile)} percentile of 5 years" + (f", {rank_of}" if rank_of else "")
    text = html.escape(text)
    return (
        f'<div class="cm-rank" role="img" aria-label="{text}">'
        f'<div class="cm-track"><span class="cm-mark" style="left:{percentile:.1f}%"></span></div>'
        f"<span>{text}</span></div>"
    )


def split_bar_html(share: float, oil: str, meal: str) -> str:
    """Oil against meal in the product value, as one bar."""
    label = f"Oil {share:.0%} of product value, meal {1 - share:.0%}"
    return (
        f'<div class="cm-split" role="img" aria-label="{label}">'
        f'<span class="cm-split-oil" style="width:{share * 100:.1f}%"></span>'
        '<span class="cm-split-meal"></span></div>'
        f'<p class="cm-legend"><i class="cm-key-oil"></i>Oil {html.escape(oil)}'
        f' <i class="cm-key-meal"></i>Meal {html.escape(meal)}</p>'
    )


def kpi_row_html(cards: list[Kpi]) -> str:
    return '<div class="cm-kpis">' + "".join(_kpi(card) for card in cards) + "</div>"


def _kpi(card: Kpi) -> str:
    tag = f' <em class="cm-tag">{html.escape(card.tag)}</em>' if card.tag else ""
    return (
        f'<a class="cm-kpi cm-kpi-{card.key}" href="#{card.key}">'
        f'<p class="cm-label"><i>{card.number}</i>{html.escape(card.label)}{tag}</p>'
        f'<p class="cm-value">{html.escape(card.value)}'
        f"<small>{html.escape(card.note)}</small></p>"
        f'<div class="cm-detail">{card.visual or html.escape(card.detail)}</div>'
        f"{percentile_html(card.percentile, card.rank_of)}"
        f'<p class="cm-caption">{html.escape(card.caption)}</p></a>'
    )


def section_head_html(key: str, number: int, title: str, sentence: str) -> str:
    return (
        f'<div class="cm-head" id="{key}"><p class="cm-label"><i>{number}</i>'
        f"{html.escape(title)}</p><p>{html.escape(sentence)}</p></div>"
    )


def rail_html(what: str, reading: str, today: list[tuple[str, str]]) -> str:
    """The text beside a chart: what it is, how to read it, and today's levels."""
    rows = "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>" for label, value in today
    )
    return (
        '<div class="cm-rail">'
        f"<div><h3>What it is</h3><p>{html.escape(what)}</p></div>"
        f"<div><h3>Reading the chart</h3><p>{html.escape(reading)}</p></div>"
        f'<div class="cm-today"><h3>Today</h3><dl>{rows}</dl></div></div>'
    )


def _signed_percent(shown: int) -> str:
    # A typographic minus, as in the page's USD values.
    sign = "−" if shown < 0 else ("+" if shown > 0 else "")
    return f"{sign}{abs(shown)}%"


def _dollars(value: float) -> str:
    return f"{'−' if round(value) < 0 else ''}${abs(value):,.0f}"


def _p_value(p: float) -> str:
    # A no-break space, so "< 0.01" never splits across lines.
    return "<\N{NO-BREAK SPACE}0.01" if p < 0.01 else f"{p:.2f}"


def backtest_verdict(test: Backtest) -> Verdict:
    """What the VaR backtest says, Kupiec's coverage first, then independence."""
    if test.kupiec_p < BACKTEST_SIGNIFICANCE and test.rate > 1 - VAR_LEVEL:
        return "Under-covers"
    if test.kupiec_p < BACKTEST_SIGNIFICANCE and test.rate < 1 - VAR_LEVEL:
        return "Over-conservative"
    if test.christoffersen_p < BACKTEST_SIGNIFICANCE:
        return "Clusters"
    return "Holds"


def _es_verdict(test: Backtest) -> Verdict:
    return "Holds" if test.z2_p >= BACKTEST_SIGNIFICANCE else "Under-covers"


def _chip(verdict: Verdict) -> str:
    return f'<span class="cm-chip {VERDICT_CLASS[verdict]}">{verdict}</span>'


def backtest_line(test: Backtest, template: str) -> str:
    """The backtest sentence as plain text, filled only from ``test`` and the risk constants."""
    return template.format(
        years=LOOKBACK_YEARS,
        exceptions=test.exceptions,
        days=test.days,
        rate=test.rate,
        expected=1 - VAR_LEVEL,
        kupiec=_p_value(test.kupiec_p),
        independence=_p_value(test.christoffersen_p),
    )


def es_backtest_line(test: Backtest, template: str) -> str:
    """The ES backtest sentence as plain text, filled only from ``test`` and the constants."""
    return template.format(
        years=LOOKBACK_YEARS,
        exceptions=test.es_exceptions,
        days=test.days,
        level=ES_LEVEL,
        rate=test.es_rate,
        expected=1 - ES_LEVEL,
        severity="n/a" if math.isnan(test.es_ratio) else f"{test.es_ratio:.2f}× ES",
        z2p=_p_value(test.z2_p),
    )


def _day(d: pd.Timestamp) -> str:
    # d.day, not %-d, which is not portable to Windows.
    return f"{d.day} {d:%b %Y}"


def worst_line(worst: tuple[tuple[pd.Timestamp, float], ...], template: str) -> str:
    """The largest clean losses of the window as plain text, worst first.

    Each loss and its date stay on one line: no-break spaces inside the item, and a word
    joiner after the minus, which a browser would otherwise break after.
    """
    days = ", ".join(
        f"{_dollars(loss)} ({_day(date)})".replace("−", "−\N{WORD JOINER}").replace(
            " ", "\N{NO-BREAK SPACE}"
        )
        for date, loss in worst
    )
    return template.format(years=LOOKBACK_YEARS, tonnes=NOTIONAL_TONNES, days=days)


def risk_effect(summary: RiskSummary | None) -> str:
    """Whether the left-out roll and delivery changes would lower or raise the VaR."""
    if summary is None or summary.bias is None:
        return "none"
    if summary.bias.combined_var < summary.var:
        return "lower"
    # A NaN combined VaR, from too few changes, compares false both ways.
    return "raise" if summary.bias.combined_var > summary.var else "none"


def risk_strip_html(summary: RiskSummary | None, copy: RiskCopy) -> str:
    """The 1-day risk of the crush in one row of four cells, under its chart.

    When the summary has a backtest, two lines under the cells state the VaR and the ES
    test, each after a chip with its verdict; a third lists the worst clean days.
    """
    note = copy.note.format(effect=copy.effects[risk_effect(summary)])
    if summary is None:
        footer = f'<p class="cm-risk-note">{html.escape(note)}</p>'
        line = f"n/a: under {MIN_CHANGES} clean daily changes"
        return f'<div class="cm-risk"><p class="cm-risk-na">{html.escape(line)}</p>{footer}</div>'
    as_of = f"As of {_day(summary.date)}, the last clean change. "
    footer = f'<p class="cm-risk-note">{html.escape(as_of + note)}</p>'
    if summary.worst:
        text = html.escape(worst_line(summary.worst, copy.worst))
        footer = f'<p class="cm-risk-test">{text}</p>{footer}'
    test = summary.backtest
    if test is not None:
        lines = (
            (backtest_verdict(test), backtest_line(test, copy.var_test)),
            (_es_verdict(test), es_backtest_line(test, copy.es_test)),
        )
        tests = "".join(
            f'<p class="cm-risk-test">{_chip(verdict)} {html.escape(text)}</p>'
            for verdict, text in lines
        )
        footer = tests + footer
    # Each leg stays on one line and the legs wrap as a flex row, spaced by the CSS gap, so a
    # narrow cell breaks only between legs and no line ends on a separator.
    legs = (
        '<span class="cm-legs">'
        + "".join(
            f'<span class="cm-leg">{html.escape(f"{leg} {_signed_percent(share)}")}</span>'
            for leg, share in percent_shares(summary.contributions).items()
        )
        + "</span>"
    )
    cells = [
        (
            f"Daily σ, last {SHORT_VOL_CHANGES} changes / 1 year",
            html.escape(f"${summary.vol_20:,.2f}/t / ${summary.vol_1y:,.2f}/t"),
        ),
        (f"1-day VaR {VAR_LEVEL:.0%}, {NOTIONAL_TONNES:,} t", html.escape(_dollars(summary.var))),
        (f"ES {ES_LEVEL:.1%}", html.escape(_dollars(summary.es))),
        ("Share of ES by leg", legs),
    ]
    rows = "".join(
        f'<div class="cm-risk-cell"><dt>{html.escape(label)}</dt><dd>{value}</dd></div>'
        for label, value in cells
    )
    return f'<div class="cm-risk"><dl>{rows}</dl>{footer}</div>'


def method_row_html(cards: list[MethodCard]) -> str:
    return '<div class="cm-method">' + "".join(_method(card) for card in cards) + "</div>"


def _method(card: MethodCard) -> str:
    entries = "".join(f"<p><b>{lead}:</b> {text}</p>" for lead, text in card.entries)
    return (
        f"<section><h2>{html.escape(card.title)}</h2>{entries}"
        f'<p class="cm-closing">{card.closing}</p></section>'
    )

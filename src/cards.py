"""HTML for the cards, rails and method notes. Pure functions, no UI imports.

The page renders these with ``st.html``, which strips scripts and handlers, so
everything here is plain markup styled by the page CSS. Runtime text goes
through ``html.escape``; markup arguments come only from this module.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from .risk import (
    ES_LEVEL,
    MIN_CHANGES,
    NOTIONAL_TONNES,
    SHORT_VOL_SESSIONS,
    VAR_LEVEL,
    RiskSummary,
)

__all__ = [
    "Kpi",
    "MethodCard",
    "kpi_row_html",
    "method_row_html",
    "ordinal",
    "percentile_html",
    "rail_html",
    "risk_strip_html",
    "section_head_html",
    "split_bar_html",
]


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


def _signed_percent(share: float) -> str:
    # A typographic minus, as in the page's USD values.
    shown = round(share * 100)
    sign = "−" if shown < 0 else ("+" if shown > 0 else "")
    return f"{sign}{abs(shown)}%"


def _dollars(value: float) -> str:
    return f"{'−' if round(value) < 0 else ''}${abs(value):,.0f}"


def risk_strip_html(summary: RiskSummary | None, note: str) -> str:
    """The 1-day risk of the crush in one row of four cells, under its chart."""
    footer = f'<p class="cm-risk-note">{html.escape(note)}</p>'
    if summary is None:
        line = f"n/a: under {MIN_CHANGES} clean daily changes"
        return f'<div class="cm-risk"><p class="cm-risk-na">{html.escape(line)}</p>{footer}</div>'
    legs = " · ".join(
        f"{leg} {_signed_percent(share)}" for leg, share in summary.contributions.items()
    )
    cells = [
        (
            f"Daily σ, {SHORT_VOL_SESSIONS} sessions / 1 year",
            f"${summary.vol_20:,.2f}/t / ${summary.vol_1y:,.2f}/t",
        ),
        (f"1-day VaR {VAR_LEVEL:.0%}, {NOTIONAL_TONNES:,} t", _dollars(summary.var)),
        (f"ES {ES_LEVEL:.1%}", _dollars(summary.es)),
        ("Risk by leg", legs),
    ]
    rows = "".join(
        f'<div class="cm-risk-cell"><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>'
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

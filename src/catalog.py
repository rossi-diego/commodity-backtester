"""Catalogue of the rules offered in the app, with their user-facing copy."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RULES", "RuleSpec"]


@dataclass(frozen=True)
class RuleSpec:
    """A rule the app can run: engine key, select label and plain-language description."""

    key: str
    label: str
    description: str


RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        key="tsmom_12m",
        label="12M momentum",
        description=(
            "Long when the price is above its level 252 sessions ago, "
            "checked at each month end. Otherwise flat."
        ),
    ),
    RuleSpec(
        key="donchian",
        label="Donchian 20/10",
        description=(
            "Long on a close above the prior 20-session high. "
            "Flat on a close below the prior 10-session low."
        ),
    ),
    RuleSpec(
        key="buy_hold",
        label="Buy & hold",
        description=(
            "Reference: long from the first fill to the window end, at the same costs. "
            "No comparison is computed yet."
        ),
    ),
)

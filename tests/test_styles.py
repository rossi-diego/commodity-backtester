"""The rendered page CSS: the rules built in ``src.styles``, as the page receives them."""

from __future__ import annotations

import re

from src.styles import APP_CSS, year_pill_css
from src.visualization import YEAR_COLORS


def test_css_is_one_style_block() -> None:
    assert APP_CSS.startswith("<style>")
    assert APP_CSS.endswith("</style>")
    assert APP_CSS.count("<style>") == 1


def test_selected_toggle_knob_is_dark() -> None:
    """On, the track is near-white, so the knob inside it takes the page colour."""
    rule = (
        "[class*='st-key-rin-'] label[data-selected]:not([data-disabled])>div:first-of-type>div"
        "{background:var(--page)}"
    )
    assert rule in APP_CSS
    assert "--page:#0b1016" in APP_CSS


class TestYearPills:
    def test_newest_pill_takes_last_years_colour(self) -> None:
        rules = year_pill_css(YEAR_COLORS).splitlines()
        assert len(rules) == 10
        assert ":nth-child(1){" in rules[0]
        assert "#2bb39b" in rules[0]
        assert "solid" in rules[0]

    def test_repeats_are_dashed_in_the_same_colours(self) -> None:
        rules = year_pill_css(YEAR_COLORS).splitlines()
        assert ":nth-child(6){" in rules[5]
        assert "dashed" in rules[5]
        assert "#2bb39b" in rules[5]
        for i, rule in enumerate(rules, start=1):
            assert YEAR_COLORS[(i - 1) % len(YEAR_COLORS)] in rule
            assert ("dashed" in rule) == (i > len(YEAR_COLORS))

    def test_only_selected_pills_are_outlined(self) -> None:
        rules = year_pill_css(YEAR_COLORS, count=3).splitlines()
        assert len(rules) == 3
        assert all("button[data-selected]" in rule for rule in rules)

    def test_rules_are_in_the_page_css(self) -> None:
        assert year_pill_css(YEAR_COLORS) in APP_CSS
        assert len(re.findall(r"st-key-years-", APP_CSS)) == 10

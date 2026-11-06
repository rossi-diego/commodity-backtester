"""End-to-end tests of app.py with AppTest: no browser, no network, no real cache."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config, visualization
from src.margins import crush_table
from src.seasonal import latest_reading

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
SECTIONS = ["crush", "share", "gap"]
LABELS = ["Crush margin", "Oil share", "Oil cost vs diesel"]
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


def _new_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)


def _cached_app(monkeypatch, tmp_path, frame: pd.DataFrame):
    path = tmp_path / "cache.parquet"
    frame.to_parquet(path)
    monkeypatch.setattr(config, "YFINANCE_RAW", path)
    st.cache_data.clear()
    return _new_app()


@pytest.fixture(autouse=True)
def no_rin_network(tmp_path, monkeypatch):
    """EPA is never called in tests: the page reads the saved D4 copy in the repository."""
    monkeypatch.setattr(config, "RIN_CACHE", tmp_path / "rin.csv")
    with patch("src.rins._fetch_rows", side_effect=OSError("offline")):
        yield


@pytest.fixture()
def app(tmp_path, monkeypatch, synthetic_prices):
    """The app on a synthetic cache that is a hit, so nothing may touch the network.

    The check comes after the yield. A side_effect would be swallowed by the
    loader's fallback, and the test would pass without noticing the download.
    """
    at = _cached_app(monkeypatch, tmp_path, synthetic_prices)
    with patch("src.data_loader.yf.download") as download:
        yield at
    download.assert_not_called()
    st.cache_data.clear()


def _html(at: AppTest) -> str:
    return " ".join(element.proto.body for element in at.get("html"))


def _text(markup: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup))


def _block(at: AppTest, marker: str) -> str:
    return next(e.proto.body for e in at.get("html") if marker in e.proto.body)


def _kpis(at: AppTest) -> str:
    return _block(at, 'class="cm-kpis"')


def _head(at: AppTest, key: str) -> str:
    return _text(_block(at, f'id="{key}"'))


def _chart(at: AppTest, key: str) -> str:
    return next(c for c in at.get("plotly_chart") if f"chart-{key}" in c.proto.id).proto.spec


def _no_exceptions(at: AppTest) -> None:
    assert not at.exception, [exc.value for exc in at.exception]


# ---------------------------------------------------------------------------
# First load
# ---------------------------------------------------------------------------


class TestFirstLoad:
    def test_three_cards_and_three_charts_without_a_click(self, app) -> None:
        app.run()
        _no_exceptions(app)
        cards = _kpis(app)
        labels = re.findall(r'class="cm-label"><i>\d</i>([^<]+)', cards)
        assert [label.strip() for label in labels] == LABELS
        assert [c.proto.id.rsplit("chart-", 1)[-1] for c in app.get("plotly_chart")] == SECTIONS
        assert not app.button
        assert not app.metric

    def test_headline_matches_the_engine(self, app, synthetic_prices) -> None:
        app.run()
        table = crush_table(synthetic_prices)
        reading = latest_reading(table["crush"], table["flagged"])
        sign = "−" if round(reading.value) < 0 else ""
        value = f"{sign}${abs(reading.value):,.0f}/t"
        assert f"{value} on {reading.date.day} {reading.date:%b %Y}" in _head(app, "crush")
        assert f"{reading.percentile:.0f}%" in _head(app, "crush")
        assert f">{value}<" in _kpis(app)

    def test_no_unescaped_dollar_reaches_markdown(self, app) -> None:
        """Two unescaped dollar signs in one block render the text between them as LaTeX."""
        app.run()
        for block in [*app.markdown, *app.warning, *app.caption]:
            assert not re.search(r"(?<!\\)\$", block.value), block.value

    def test_rin_card_is_dated_and_static(self, app) -> None:
        app.run()
        seed = pd.read_csv(config.RIN_SEED)
        week = pd.Timestamp(seed["week"].iloc[-1])
        price = seed["usd_per_rin"].iloc[-1]
        cards = _text(_kpis(app))
        assert f"D4 ${price:.2f}, week of {week.day} {week:%b} (EPA, saved copy" in cards
        assert "With the D4 RIN:" in cards
        app.toggle(key="rin-gap").set_value(True).run()
        _no_exceptions(app)
        assert _text(_kpis(app)) == cards
        assert f"D4 RIN prices from EPA to the week of {week.date()}, saved copy." in (
            app.caption[-1].value
        )

    def test_rin_toggle_changes_the_chart_and_the_sentence(self, app) -> None:
        app.run()
        assert "Oil minus heating oil, USD per m" in _chart(app, "gap")
        app.toggle(key="rin-gap").set_value(True).run()
        _no_exceptions(app)
        assert "Oil minus heating oil and D4 RIN" in _chart(app, "gap")
        assert "Before the RIN" in _chart(app, "gap")
        assert "Oil cost vs diesel With the D4 RIN," in _head(app, "gap")

    def test_page_works_without_any_rin_data(self, app, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(config, "RIN_SEED", tmp_path / "missing.csv")
        app.run()
        _no_exceptions(app)
        assert "D4 RIN prices could not be loaded" in _text(_kpis(app))
        assert app.toggle(key="rin-gap").proto.disabled

    def test_negative_values_use_a_typographic_minus(self, app) -> None:
        app.run()
        cards = _kpis(app)
        assert "-$" not in cards
        assert re.search(r"[−+]\$[\d,.]+/gal", cards)

    def test_gap_is_never_presented_as_a_biodiesel_price(self, app) -> None:
        app.run()
        assert "not the biodiesel price" in _html(app)
        assert "Feedstock only" in _text(_kpis(app))

    def test_scope_names_what_is_left_out(self, app) -> None:
        app.run()
        hero = _block(app, "cm-hero")
        for text in ("Gross margins", "D4 RIN", "basis", "rolls or deliveries"):
            assert text in hero
        assert not app.warning

    def test_footer_dates_the_data_and_disclaims(self, app, synthetic_prices) -> None:
        app.run()
        footer = app.caption[-1].value
        assert f"Data to {synthetic_prices.index.max().date()}." in footer
        assert (
            "Personal project, not affiliated with any employer. Not investment advice." in footer
        )


class TestControls:
    @pytest.mark.parametrize("key", SECTIONS)
    def test_each_chart_has_its_own_window(self, app, key) -> None:
        app.run()
        before = {other: len(_chart(app, other)) for other in SECTIONS}
        assert app.button_group(key=f"win-{key}").proto.required is True
        app.button_group(key=f"win-{key}").set_value("5Y").run()
        _no_exceptions(app)
        assert len(_chart(app, key)) < before[key]
        for other in set(SECTIONS) - {key}:
            assert len(_chart(app, other)) == before[other]

    @pytest.mark.parametrize("key", SECTIONS)
    def test_by_month_compares_years(self, app, key) -> None:
        app.run()
        app.button_group(key=f"view-{key}").set_value("By month").run()
        _no_exceptions(app)
        year = pd.Timestamp.today().year
        spec = _chart(app, key)
        for name in (str(year), str(year - 1), str(year - 2), "averages"):
            assert f'"name":"{name}' in spec or name in spec
        assert "against" in _head(app, key) or "too few" in _head(app, key)
        app.checkbox(key=f"band-{key}").uncheck().run()
        assert "averages" not in _chart(app, key)

    def test_years_are_capped(self, app) -> None:
        app.run()
        app.button_group(key="view-crush").set_value("By month").run()
        year = pd.Timestamp.today().year
        app.button_group(key="years-crush").set_value([year - n for n in range(1, 8)]).run()
        _no_exceptions(app)
        assert any("Up to 5 years" in caption.value for caption in app.caption)
        spec = _chart(app, "crush")
        assert f'"name":"{year - 5}"' in spec
        assert f'"name":"{year - 6}"' not in spec

    def test_rin_by_month_uses_epa_weeks(self, app) -> None:
        app.run()
        app.toggle(key="rin-gap").set_value(True).run()
        app.button_group(key="view-gap").set_value("By month").run()
        _no_exceptions(app)
        assert "RIN months use EPA weeks only" in _block(
            app, 'cm-rail"><div><h3>What it is</h3><p>About'
        )


class TestIntroduction:
    def test_hero_has_question_steps_and_author(self, app) -> None:
        app.run()
        hero = _block(app, "cm-hero")
        for text in (
            "Soybean Crush Margin",
            "how much of it is biodiesel",
            'href="#crush"',
            'href="#share"',
            'href="#gap"',
            "Diego Rossi",
            "linkedin.com/in/rossi-diego",
            "Yahoo Finance",
        ):
            assert text in hero

    def test_no_employer_or_title_is_named(self) -> None:
        for text in (SOURCE, README):
            for word in ("Oleoplan", "Inpasa", "Coordinator", "Manager at"):
                assert word not in text

    def test_every_chart_is_explained_beside_it_and_method_closes(self, app) -> None:
        app.run()
        page = _html(app)
        assert page.count('class="cm-rail"') == 3
        for heading in ("How it is calculated", "Rolls and delivery", "What it leaves out"):
            assert heading in page
        assert [e.label for e in app.expander] == ["Full methodology and sources"]


# ---------------------------------------------------------------------------
# Data states
# ---------------------------------------------------------------------------


class TestDataStates:
    def test_load_failure_is_plain_and_not_cached(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, "YFINANCE_RAW", tmp_path / "absent.parquet")
        st.cache_data.clear()
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")) as dl:
            at = _new_app().run()
            _no_exceptions(at)
            assert "could not be loaded" in at.error[0].value
            assert "yahoo down" not in at.error[0].value
            assert [button.label for button in at.button] == ["Retry"]
            assert 'class="cm-kpis"' not in _html(at)
            first = dl.call_count
            at.button[0].click().run()
            assert dl.call_count > first, "A failed load must not be served from cache."
        assert "None" not in _block(at, "cm-hero")
        st.cache_data.clear()

    def test_retry_recovers(self, tmp_path, monkeypatch, synthetic_prices) -> None:
        path = tmp_path / "late.parquet"
        monkeypatch.setattr(config, "YFINANCE_RAW", path)
        st.cache_data.clear()
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at = _new_app().run()
            assert at.error
            synthetic_prices.to_parquet(path)
            at.button[0].click().run()
        _no_exceptions(at)
        assert not at.error
        assert 'class="cm-kpis"' in _html(at)
        st.cache_data.clear()

    def test_old_data_labelled(self, tmp_path, monkeypatch, synthetic_prices) -> None:
        at = _cached_app(monkeypatch, tmp_path, synthetic_prices.loc[:"2024-06-28"])
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at.run()
        _no_exceptions(at)
        assert "2024-06-28 (may be out of date)" in _block(at, "cm-hero")
        assert "2024-06-28 (may be out of date)" in at.caption[-1].value
        st.cache_data.clear()

    def test_missing_crush_leg_warns(self, tmp_path, monkeypatch, synthetic_prices) -> None:
        frame = synthetic_prices.copy()
        frame["Soybean Meal"] = np.nan
        at = _cached_app(monkeypatch, tmp_path, frame)
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at.run()
        _no_exceptions(at)
        assert any("crush margin cannot be computed" in w.value for w in at.warning)
        assert 'class="cm-kpis"' not in _html(at)
        st.cache_data.clear()

    def test_missing_diesel_keeps_the_crush(self, tmp_path, monkeypatch, synthetic_prices) -> None:
        frame = synthetic_prices.copy()
        frame["Heating Oil"] = np.nan
        at = _cached_app(monkeypatch, tmp_path, frame)
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at.run()
        _no_exceptions(at)
        assert any("feedstock gap is not shown" in w.value for w in at.warning)
        assert "Prices missing." in _kpis(at)
        assert [c.proto.id.rsplit("chart-", 1)[-1] for c in at.get("plotly_chart")] == [
            "crush",
            "share",
        ]
        st.cache_data.clear()

    def test_card_says_when_the_latest_close_is_flagged(self, tmp_path, monkeypatch) -> None:
        """A frame ending inside November delivery: the reading uses an earlier clean date."""
        dates = pd.bdate_range("2019-01-02", "2025-11-12")
        frame = pd.DataFrame(
            {"Soybean": 1000.0, "Soybean Meal": 320.0, "Soybean Oil": 45.0, "Heating Oil": 2.6},
            index=dates,
        )
        at = _cached_app(monkeypatch, tmp_path, frame)
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at.run()
            for key in SECTIONS:
                at.button_group(key=f"view-{key}").set_value("By month").run()
        _no_exceptions(at)
        assert "last close is a roll day" in _kpis(at)
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Source rules
# ---------------------------------------------------------------------------


def _luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.fixture()
def tokens() -> dict[str, str]:
    return dict(re.findall(r"--(\w+):(#[0-9a-f]{6})", SOURCE))


class TestSource:
    def test_no_emoji_no_unsafe_html_one_css_block(self) -> None:
        assert not EMOJI.search(SOURCE)
        assert "unsafe_allow_html" not in SOURCE
        assert SOURCE.count("<style>") == 1
        assert "<script" not in SOURCE

    def test_charts_use_no_streamlit_theme(self) -> None:
        """Without theme=None Streamlit silently replaces the axis colours of the figures."""
        assert SOURCE.count("st.plotly_chart(") == 1
        assert SOURCE.count("theme=None") == 1

    def test_no_backtest_left(self) -> None:
        for text in (SOURCE, README):
            for word in ("Donchian", "momentum", "Buy & hold", "Backtest"):
                assert word not in text

    def test_page_clears_the_streamlit_cloud_bar(self) -> None:
        """The bar sits in the parent page, covers 56 px and hid the title in production."""
        tops = re.findall(r"stMainBlockContainer'\]\{[^}]*?padding:([\d.]+)rem", SOURCE)
        assert len(tops) == 2
        assert all(float(top) * 15 >= 56 for top in tops)


class TestDarkTheme:
    def test_theme_is_dark_and_matches_the_tokens(self, tokens) -> None:
        with (ROOT / ".streamlit" / "config.toml").open("rb") as file:
            theme = tomllib.load(file)["theme"]
        assert theme["base"] == "dark"
        assert theme["backgroundColor"] == tokens["page"]
        assert theme["secondaryBackgroundColor"] == tokens["card"]
        assert tokens["crush"] == visualization.CRUSH
        assert tokens["oil"] == visualization.OIL
        assert tokens["card"] == visualization.SURFACE

    def test_amber_is_reserved_for_the_crush(self, tokens) -> None:
        with (ROOT / ".streamlit" / "config.toml").open("rb") as file:
            theme = tomllib.load(file)["theme"]
        assert theme["primaryColor"] != tokens["crush"]
        rules = [line for line in SOURCE.splitlines() if "var(--crush)" in line]
        assert rules and all(line.startswith(".cm-kpi-crush") for line in rules)

    @pytest.mark.parametrize(
        ("text", "surface"),
        [("ink", "card"), ("mut", "card"), ("mut", "page"), ("crush", "card")],
    )
    def test_text_contrast_is_at_least_aa(self, tokens, text, surface) -> None:
        assert _contrast(tokens[text], tokens[surface]) >= 4.5

    def test_chart_marks_are_visible_on_the_surface(self) -> None:
        for color in (
            visualization.CRUSH,
            visualization.OIL,
            visualization.MEAL,
            visualization.FLAT,
            *visualization.YEAR_COLORS,
        ):
            assert _contrast(color, visualization.SURFACE) >= 3

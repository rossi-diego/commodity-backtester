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
LABELS = ["Crush margin", "Against 5 years", "Oil share", "Oil cost vs diesel"]
CHARTS = {
    "seasonal_chart",
    "history_chart",
    "value_chart",
    "share_chart",
    "feedstock_chart",
    "gap_chart",
}
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


def _new_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


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


def _chart_keys(at: AppTest) -> set[str]:
    return {key for chart in at.get("plotly_chart") for key in CHARTS if key in chart.proto.id}


def _markdown(at: AppTest) -> str:
    return " ".join(block.value for block in at.markdown)


def _hero(at: AppTest) -> str:
    return " ".join(element.proto.body for element in at.get("html"))


def _metric(at: AppTest, label: str):
    return next(metric for metric in at.metric if metric.label == label)


def _no_exceptions(at: AppTest) -> None:
    assert not at.exception, [exc.value for exc in at.exception]


# ---------------------------------------------------------------------------
# First load
# ---------------------------------------------------------------------------


class TestFirstLoad:
    def test_everything_shows_without_a_click(self, app) -> None:
        app.run()
        _no_exceptions(app)
        assert [metric.label for metric in app.metric] == LABELS
        assert _chart_keys(app) == CHARTS
        assert not app.button
        assert not app.selectbox

    def test_headline_matches_the_engine(self, app, synthetic_prices) -> None:
        app.run()
        table = crush_table(synthetic_prices)
        reading = latest_reading(table["crush"], table["flagged"])
        headline = next(
            block.value for block in app.markdown if "Soybean crush margin" in block.value
        )
        sign = "−" if round(reading.value) < 0 else ""
        assert f"{sign}\\${abs(reading.value):,.0f}/t" in headline
        assert str(reading.date.date()) in headline
        assert f"{reading.percentile:.0f}%" in headline
        assert _metric(app, "Crush margin").value == f"{sign}${abs(reading.value):,.0f}/t"

    def test_no_unescaped_dollar_reaches_markdown(self, app) -> None:
        """Two unescaped dollar signs in one block render the text between them as LaTeX."""
        app.run()
        for block in [*app.markdown, *app.warning, *app.caption]:
            assert not re.search(r"(?<!\\)\$", block.value), block.value

    def test_rin_credit_is_shown_with_its_date_and_source(self, app) -> None:
        app.run()
        seed = pd.read_csv(config.RIN_SEED)
        week, price = seed["week"].iloc[-1], seed["usd_per_rin"].iloc[-1]
        head = next(block.value for block in app.markdown if "soybean oil in 1 m³" in block.value)
        assert f"latest D4 RIN, \\${price:,.2f} in the week of {week} (EPA, saved copy)" in head
        assert "After the D4 RIN credit" in app.caption[3].value
        assert f"D4 RIN prices from EPA to the week of {week}, saved copy." in app.caption[-1].value

    def test_page_works_without_any_rin_data(self, app, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(config, "RIN_SEED", tmp_path / "missing.csv")
        app.run()
        _no_exceptions(app)
        assert any("credit is not shown" in block.value for block in app.markdown)
        assert "Not the biodiesel price" in app.caption[3].value

    def test_negative_values_use_a_typographic_minus(self, app) -> None:
        app.run()
        caption = app.caption[3].value
        assert "−\\$" in caption
        assert "-\\$" not in caption

    def test_gap_is_never_presented_as_a_biodiesel_price(self, app) -> None:
        app.run()
        head = next(block.value for block in app.markdown if "soybean oil in 1 m³" in block.value)
        assert "not the biodiesel price" in head
        assert "It is not the biodiesel price" in _markdown(app)
        assert "oil only" in _metric(app, "Oil cost vs diesel").delta

    def test_warning_names_what_is_left_out(self, app) -> None:
        app.run()
        warning = app.warning[0].value
        for text in ("before costs", "D4 RIN", "45Z", "basis", "roll or delivery"):
            assert text in warning

    def test_window_changes_the_history(self, app) -> None:
        app.run()
        before = len(
            next(c for c in app.get("plotly_chart") if "history_chart" in c.proto.id).proto.spec
        )
        app.button_group(key="window").set_value("5Y").run()
        _no_exceptions(app)
        after = len(
            next(c for c in app.get("plotly_chart") if "history_chart" in c.proto.id).proto.spec
        )
        assert after < before

    def test_window_control_cannot_be_cleared(self, app) -> None:
        app.run()
        assert app.button_group(key="window").proto.required is True

    def test_footer_dates_the_data_and_disclaims(self, app, synthetic_prices) -> None:
        app.run()
        footer = app.caption[-1].value
        assert f"Data to {synthetic_prices.index.max().date()}." in footer
        assert (
            "Personal project, not affiliated with any employer. Not investment advice." in footer
        )


class TestIntroduction:
    def test_hero_has_question_steps_and_author(self, app) -> None:
        app.run()
        hero = _hero(app)
        for text in (
            "Soybean Crush Margin",
            "how much of it is biodiesel",
            "Today's margin",
            "Where it comes from",
            "Oil against diesel",
            "Diego Rossi",
            "linkedin.com/in/rossi-diego",
            "Yahoo Finance",
        ):
            assert text in hero

    def test_no_employer_or_title_is_named(self) -> None:
        for text in (SOURCE, README):
            for word in ("Oleoplan", "Inpasa", "Coordinator", "Manager at"):
                assert word not in text

    def test_every_section_is_explained_beside_its_chart(self, app) -> None:
        app.run()
        text = _markdown(app)
        for heading in (
            "How to read it",
            "Why oil matters more now",
            "Oil as a fuel feedstock",
            "How it is calculated",
            "Rolls and delivery",
            "What it leaves out",
        ):
            assert heading in text
        assert not app.expander


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
            assert not at.metric
            first = dl.call_count
            at.button[0].click().run()
            assert dl.call_count > first, "A failed load must not be served from cache."
        assert "None" not in _hero(at)
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
        assert [metric.label for metric in at.metric] == LABELS
        st.cache_data.clear()

    def test_old_data_labelled(self, tmp_path, monkeypatch, synthetic_prices) -> None:
        at = _cached_app(monkeypatch, tmp_path, synthetic_prices.loc[:"2024-06-28"])
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at.run()
        _no_exceptions(at)
        assert "2024-06-28 (may be out of date)" in _hero(at)
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
        assert not at.metric
        st.cache_data.clear()

    def test_missing_diesel_keeps_the_crush(self, tmp_path, monkeypatch, synthetic_prices) -> None:
        frame = synthetic_prices.copy()
        frame["Heating Oil"] = np.nan
        at = _cached_app(monkeypatch, tmp_path, frame)
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at.run()
        _no_exceptions(at)
        assert any("feedstock gap is not shown" in w.value for w in at.warning)
        assert _metric(at, "Oil cost vs diesel").value == "n/a"
        assert "seasonal_chart" in _chart_keys(at)
        st.cache_data.clear()

    def test_reading_says_when_the_latest_close_is_flagged(self, tmp_path, monkeypatch) -> None:
        """A frame ending inside November delivery: the headline uses an earlier clean date."""
        dates = pd.bdate_range("2019-01-02", "2025-11-12")
        frame = pd.DataFrame(
            {"Soybean": 1000.0, "Soybean Meal": 320.0, "Soybean Oil": 45.0, "Heating Oil": 2.6},
            index=dates,
        )
        at = _cached_app(monkeypatch, tmp_path, frame)
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at.run()
        _no_exceptions(at)
        headline = next(
            block.value for block in at.markdown if "Soybean crush margin" in block.value
        )
        assert "The latest close, 2025-11-12, falls in a roll or delivery window" in headline
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
        calls = SOURCE.count("st.plotly_chart(")
        assert calls == len(CHARTS)
        assert SOURCE.count("theme=None") == calls

    def test_no_backtest_left(self) -> None:
        for text in (SOURCE, README):
            for word in ("Donchian", "momentum", "Buy & hold", "Backtest"):
                assert word not in text

    def test_page_clears_the_streamlit_cloud_bar(self) -> None:
        """The bar sits in the parent page, covers 56 px and hid the title in production."""
        tops = re.findall(
            r"stMainBlockContainer'\]\{(?:max-width:none;)?padding:([\d.]+)rem", SOURCE
        )
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
        assert tokens["card"] == visualization.SURFACE

    def test_amber_is_reserved_for_the_crush(self, tokens) -> None:
        with (ROOT / ".streamlit" / "config.toml").open("rb") as file:
            theme = tomllib.load(file)["theme"]
        assert theme["primaryColor"] != tokens["crush"]
        assert "orange-badge" not in SOURCE
        rules = [line for line in SOURCE.splitlines() if "var(--crush)" in line]
        assert rules and all(line.startswith(".st-key-card-kpi-crush{") for line in rules)

    @pytest.mark.parametrize(
        ("text", "surface"),
        [
            ("ink", "card"),
            ("mut", "card"),
            ("mut", "page"),
            ("ink", "raised"),
            ("oncrush", "crush"),
        ],
    )
    def test_text_contrast_is_at_least_aa(self, tokens, text, surface) -> None:
        assert _contrast(tokens[text], tokens[surface]) >= 4.5

    def test_chart_marks_are_visible_on_the_surface(self) -> None:
        for color in (
            visualization.CRUSH,
            visualization.OIL,
            visualization.MEAL,
            visualization.FLAT,
        ):
            assert _contrast(color, visualization.SURFACE) >= 3

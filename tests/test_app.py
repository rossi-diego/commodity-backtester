"""End-to-end tests of app.py with AppTest: no browser, no network, no real cache."""

from __future__ import annotations

import base64
import datetime
import itertools
import json
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
from src.catalog import RULES
from src.constants import RATIO_PAIRS, commodities_dict
from src.runner import run_backtest

ROOT = Path(__file__).resolve().parents[1]
MARKETS = list(commodities_dict.values())
RULE_LABELS = [rule.label for rule in RULES]
FIXED_PERIODS = ["1Y", "3Y", "5Y", "10Y", "All"]
HEADLINES = ["Round trips", "Time in market", "Unadjusted P&L"]
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


def _new_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


@pytest.fixture()
def app(tmp_path, monkeypatch, synthetic_prices):
    """The app on a synthetic cache that is a hit, so nothing may touch the network.

    The check comes after the yield. A side_effect would be swallowed by the
    loader's fallback, and the test would pass without noticing the download.
    """
    path = tmp_path / "cache.parquet"
    synthetic_prices.to_parquet(path)
    monkeypatch.setattr(config, "YFINANCE_RAW", path)
    st.cache_data.clear()
    with patch("src.data_loader.yf.download") as download:
        yield _new_app()
    download.assert_not_called()
    st.cache_data.clear()


def _labels(at: AppTest) -> list[str]:
    return [metric.label for metric in at.metric]


def _headline_labels(at: AppTest) -> list[str]:
    """Metric labels outside the expanders. The Sharpe lives in one, even while closed."""
    in_expanders = [metric.label for expander in at.expander for metric in expander.metric]
    labels = _labels(at)
    for label in in_expanders:
        labels.remove(label)
    return labels


def _chart_spec(at: AppTest, key: str) -> dict:
    chart = next(chart for chart in at.get("plotly_chart") if key in chart.proto.id)
    return json.loads(chart.proto.spec)


def _trace(spec: dict, name: str) -> dict:
    return next(trace for trace in spec["data"] if trace.get("name") == name)


def _trace_y(trace: dict) -> np.ndarray:
    """Plotly serialises float arrays as base64 with a dtype."""
    y = trace["y"]
    if isinstance(y, dict):
        return np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"]).astype(float)
    return np.asarray(y, dtype=float)


def _metric(at: AppTest, label: str):
    return next(metric for metric in at.metric if metric.label == label)


def _usd(metric) -> float:
    text = metric.value.replace("$", "").replace(",", "")
    return float(text)


def _no_exceptions(at: AppTest) -> None:
    assert not at.exception, [exc.value for exc in at.exception]


def _assert_headlines(at: AppTest) -> None:
    labels = _labels(at)
    assert labels[0] in {"Position now", "Position at end"}
    assert labels[1:4] == HEADLINES


def _switch_view(at: AppTest, view: str) -> None:
    at.button_group(key="view").set_value(view).run()


def _charts(at: AppTest) -> list:
    return list(at.get("plotly_chart"))


def _set_custom(at: AppTest, start: datetime.date, end: datetime.date) -> None:
    at.selectbox(key="period").set_value("Custom").run()
    at.date_input(key="custom_start").set_value(start)
    at.date_input(key="custom_end").set_value(end).run()


# ---------------------------------------------------------------------------
# First load and the main controls
# ---------------------------------------------------------------------------


class TestFirstLoad:
    def test_first_load_shows_results_without_clicks(self, app) -> None:
        app.run()
        _no_exceptions(app)
        _assert_headlines(app)
        assert len(_headline_labels(app)) == 4
        assert _charts(app), "A chart must be on the first screen."
        assert not any("run" in button.label.lower() for button in app.button)

    def test_defaults(self, app) -> None:
        app.run()
        assert app.selectbox(key="market").value == "Soybean"
        assert app.selectbox(key="rule").value == "12M momentum"
        assert app.selectbox(key="period").value == "10Y"
        assert app.button_group(key="view").value == "Backtest"

    def test_header_shows_data_date(self, app, synthetic_prices) -> None:
        app.run()
        as_of = synthetic_prices.index.max().date()
        assert any(f"Data to {as_of}." in caption.value for caption in app.caption)

    def test_warning_always_visible(self, app) -> None:
        app.run()
        for rule, period in itertools.product(RULE_LABELS, ["1Y", "All"]):
            app.selectbox(key="rule").set_value(rule)
            app.selectbox(key="period").set_value(period).run()
            assert any("indicative" in warning.value for warning in app.warning), (rule, period)

    def test_results_persist_across_widget_changes(self, app) -> None:
        app.run()
        steps = [
            ("market", "Corn"),
            ("rule", "Donchian 20/10"),
            ("period", "3Y"),
            ("rule", "Buy & hold"),
            ("market", "Heating Oil"),
        ]
        for key, value in steps:
            app.selectbox(key=key).set_value(value).run()
            _no_exceptions(app)
            _assert_headlines(app)
            assert _charts(app)

    def test_costs_change_pnl(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Donchian 20/10").run()
        before = _usd(_metric(app, "Unadjusted P&L"))
        app.number_input(key="commission").set_value(50.0).run()
        assert _usd(_metric(app, "Unadjusted P&L")) < before

    def test_slippage_changes_pnl(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Donchian 20/10").run()
        before = _usd(_metric(app, "Unadjusted P&L"))
        app.number_input(key="slippage").set_value(1.0).run()
        assert _usd(_metric(app, "Unadjusted P&L")) < before

    def test_view_control_cannot_be_cleared(self, app) -> None:
        """Without required=True, clicking the active view selects nothing at all."""
        app.run()
        assert app.button_group(key="view").proto.required is True

    def test_price_axis_uses_the_market_unit(self, app) -> None:
        app.run()
        assert _chart_spec(app, "price_chart")["layout"]["yaxis"]["title"]["text"] == "cents/bushel"
        app.selectbox(key="market").set_value("Crude Oil (WTI)").run()
        assert _chart_spec(app, "price_chart")["layout"]["yaxis"]["title"]["text"] == "USD/barrel"

    def test_pnl_chart_shows_the_open_position(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Buy & hold").run()
        assert _metric(app, "Position now").value == "Long"
        open_trace = _trace(_chart_spec(app, "pnl_chart"), "With open position")
        headline = _usd(_metric(app, "Unadjusted P&L"))
        assert _trace_y(open_trace)[-1] == pytest.approx(headline, abs=1.0)

    def test_all_market_rule_period_combinations(self, app) -> None:
        app.run()
        for market, rule, period in itertools.product(MARKETS, RULE_LABELS, FIXED_PERIODS):
            app.selectbox(key="market").set_value(market)
            app.selectbox(key="rule").set_value(rule)
            app.selectbox(key="period").set_value(period).run()
            _no_exceptions(app)
            _assert_headlines(app)


# ---------------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------------


class TestIntroduction:
    def test_intro_is_visible(self, app) -> None:
        app.run()
        hero = " ".join(element.proto.body for element in app.get("html"))
        for text in (
            "Pick a market",
            "Pick a rule",
            "Read the result",
            "Diego Rossi",
            "Grupo Oleoplan",
            "Yahoo Finance",
        ):
            assert text in hero

    def test_start_here_is_visible(self, app) -> None:
        app.run()
        assert any("Start here" in block.value for block in app.markdown)

    def test_every_rule_is_explained_without_a_click(self, app) -> None:
        app.run()
        visible = " ".join(block.value for block in app.markdown)
        for rule in RULES:
            assert rule.description in visible

    def test_selected_rule_is_marked_once(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Donchian 20/10").run()
        rail = next(block.value for block in app.markdown if "The three rules" in block.value)
        assert rail.count("[Selected]") == 1
        assert "**Donchian 20/10** :orange-badge[Selected]" in rail

    def test_only_expander_is_the_statistics(self, app) -> None:
        app.run()
        assert [expander.label for expander in app.expander] == ["Indicative statistics"]

    def test_study_figures_are_gone(self, app) -> None:
        """The drawdown study is removed from the app and the README until the daily ledger."""
        app.run()
        visible = " ".join(block.value for block in app.markdown)
        sources = (
            visible,
            (ROOT / "app.py").read_text(encoding="utf-8"),
            (ROOT / "README.md").read_text(encoding="utf-8"),
        )
        for text in sources:
            for figure in ("64%", "26%", "79%"):
                assert figure not in text

    def test_chart_copy_names_the_colour_on_screen(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        assert "Amber while" in source
        assert "turns amber" in source
        assert "turns blue" not in source
        assert "Blue while" not in source

    def test_load_failure_keeps_the_intro(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, "YFINANCE_RAW", tmp_path / "absent.parquet")
        st.cache_data.clear()
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at = _new_app().run()
        hero = " ".join(element.proto.body for element in at.get("html"))
        assert "Pick a market" in hero
        assert "None" not in hero
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class TestRulesOnScreen:
    def test_buy_hold_shows_one_fill_and_no_sharpe(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Buy & hold")
        app.selectbox(key="period").set_value("All").run()
        assert _metric(app, "Round trips").value == "0"
        assert _metric(app, "Round trips").delta == "1 fill"
        assert "Sharpe" not in _labels(app)
        statistics = app.expander[0]
        assert any("Sharpe needs the daily ledger" in c.value for c in statistics.caption)

    def test_buy_hold_has_no_few_trades_help(self, app) -> None:
        """It would fire with 0 round trips, on a rule that has one fill by construction."""
        app.run()
        app.selectbox(key="rule").set_value("Buy & hold").run()
        assert not _metric(app, "Round trips").proto.help

    def test_rules_show_few_trades_help(self, app) -> None:
        app.run()
        assert "Few trades" in _metric(app, "Round trips").proto.help

    def test_sharpe_shown_with_enough_round_trips(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Donchian 20/10")
        app.selectbox(key="period").set_value("All").run()
        statistics = app.expander[0]
        assert [metric.label for metric in statistics.metric] == ["Sharpe"]
        assert statistics.metric[0].delta == "unadjusted, incl. rolls"
        assert not statistics.caption

    def test_dense_fills_get_a_caption(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Donchian 20/10")
        app.selectbox(key="period").set_value("All").run()
        assert any("markers hidden" in caption.value for caption in app.caption)

    def test_no_stop_or_take_profit_controls(self, app) -> None:
        app.run()
        controls = [
            *app.selectbox,
            *app.number_input,
            *app.slider,
            *app.checkbox,
            *app.text_input,
            *app.button,
        ]
        for control in controls:
            assert "stop" not in control.label.lower()
            assert "take" not in control.label.lower()

    def test_no_removed_strategy_names(self, app) -> None:
        app.run()
        assert app.selectbox(key="rule").options == RULE_LABELS
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        for removed in ("Bollinger", "MACD", "RSI", "Breakout"):
            assert removed not in source
            assert removed not in " ".join(RULE_LABELS)

    def test_no_hidden_metrics_on_screen(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Donchian 20/10")
        app.selectbox(key="period").set_value("All").run()
        labels = _labels(app)
        for hidden in ("Sortino", "Calmar", "VaR", "drawdown", "Win", "Profit", "Recovery"):
            assert not any(hidden.lower() in label.lower() for label in labels), hidden
        outside_expander = labels[:4]
        assert "Sharpe" not in outside_expander
        assert labels.count("Sharpe") == 1

    def test_warning_escapes_dollars(self, app) -> None:
        """An unescaped pair of dollar signs would render the text between them as LaTeX."""
        app.run()
        body = app.warning[0].value
        assert "-\\$12.5k" in body
        assert "+\\$63k" in body

    def test_warning_keeps_its_three_claims_together(self, app) -> None:
        app.run()
        body = app.warning[0].value
        for claim in ("indicative", "2026-10-27", "Neither rule"):
            assert claim in body

    def test_no_sharpe_gap_limit_in_copy(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        assert "0.12" not in source
        assert "0.12" not in (ROOT / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Custom window
# ---------------------------------------------------------------------------


class TestCustomWindow:
    def test_date_inputs_appear_only_with_custom(self, app) -> None:
        app.run()
        assert len(app.date_input) == 0
        app.selectbox(key="period").set_value("Custom").run()
        _no_exceptions(app)
        assert len(app.date_input) == 2

    def test_date_inputs_default_to_2008_and_cover_full_history(
        self, app, synthetic_prices
    ) -> None:
        app.run()
        app.selectbox(key="period").set_value("Custom").run()
        start, end = app.date_input(key="custom_start"), app.date_input(key="custom_end")
        assert start.value == datetime.date(2008, 1, 2)
        assert end.value == datetime.date(2012, 12, 31)
        for widget in (start, end):
            assert widget.min == synthetic_prices.index.min().date()
            assert widget.max == synthetic_prices.index.max().date()

    def test_2008_window_is_reachable_with_dates_far_from_the_defaults(self, app) -> None:
        """Without explicit limits, date_input only allows about 10 years around its value."""
        app.run()
        _set_custom(app, datetime.date(2000, 2, 1), datetime.date(2026, 1, 2))
        _no_exceptions(app)
        _assert_headlines(app)

    def test_default_custom_window_runs(self, app) -> None:
        app.run()
        app.selectbox(key="period").set_value("Custom").run()
        _no_exceptions(app)
        assert _metric(app, "Position at end")
        assert _charts(app)

    def test_custom_past_end_labels_position_at_end(self, app, synthetic_prices) -> None:
        """Regression for #4 through the UI: P&L is computed on the window, not the latest price."""
        start, end = datetime.date(2008, 1, 2), datetime.date(2012, 12, 31)
        app.run()
        app.selectbox(key="rule").set_value("Buy & hold").run()
        _set_custom(app, start, end)

        assert "Position at end" in _labels(app)
        assert "Position now" not in _labels(app)
        expected = run_backtest(
            "buy_hold", "Soybean", start, end, synthetic_prices, 2.5, 0.05
        ).summary.unadjusted_pnl
        shown = _metric(app, "Unadjusted P&L").value
        sign = "-" if round(expected) < 0 else ""
        assert shown == f"{sign}${abs(expected):,.0f}"

    def test_custom_to_last_close_says_position_now(self, app, synthetic_prices) -> None:
        app.run()
        _set_custom(
            app, datetime.date(2020, 1, 2), synthetic_prices["Soybean"].dropna().index[-1].date()
        )
        assert "Position now" in _labels(app)

    def test_custom_start_after_end_warns(self, app) -> None:
        app.run()
        _set_custom(app, datetime.date(2012, 1, 2), datetime.date(2008, 1, 2))
        _no_exceptions(app)
        assert any("before End" in warning.value for warning in app.warning)
        assert len(app.selectbox) >= 3, "Controls must stay available."
        assert not app.metric

    def test_custom_window_without_data_warns(self, app) -> None:
        """Business-day data has no rows on a weekend."""
        app.run()
        _set_custom(app, datetime.date(2010, 1, 2), datetime.date(2010, 1, 3))
        _no_exceptions(app)
        assert any("No Soybean prices" in warning.value for warning in app.warning)
        assert len(app.selectbox) >= 3

    def test_custom_dates_survive_leaving_custom(self, app) -> None:
        """Fixed periods stop rendering the date inputs, which would otherwise reset them."""
        start, end = datetime.date(2019, 1, 2), datetime.date(2021, 1, 4)
        app.run()
        _set_custom(app, start, end)
        app.selectbox(key="period").set_value("5Y").run()
        assert len(app.date_input) == 0
        app.selectbox(key="period").set_value("Custom").run()
        assert app.date_input(key="custom_start").value == start
        assert app.date_input(key="custom_end").value == end

    def test_custom_dates_survive_view_switch(self, app) -> None:
        start, end = datetime.date(2019, 1, 2), datetime.date(2021, 1, 4)
        app.run()
        _set_custom(app, start, end)
        _switch_view(app, "Ratio monitor")
        assert app.date_input(key="custom_start").value == start
        _switch_view(app, "Backtest")
        assert app.date_input(key="custom_end").value == end


# ---------------------------------------------------------------------------
# Ratio monitor and shared state
# ---------------------------------------------------------------------------


class TestMonitor:
    def test_market_survives_view_switch(self, app) -> None:
        app.run()
        app.selectbox(key="market").set_value("Corn").run()
        _switch_view(app, "Ratio monitor")
        _switch_view(app, "Backtest")
        assert app.selectbox(key="market").value == "Corn"

    def test_rule_and_costs_survive_view_switch(self, app) -> None:
        app.run()
        app.selectbox(key="rule").set_value("Buy & hold")
        app.number_input(key="commission").set_value(9.0).run()
        _switch_view(app, "Ratio monitor")
        _switch_view(app, "Backtest")
        assert app.selectbox(key="rule").value == "Buy & hold"
        assert app.number_input(key="commission").value == 9.0

    def test_period_shared_between_views(self, app) -> None:
        app.run()
        app.selectbox(key="period").set_value("5Y").run()
        _switch_view(app, "Ratio monitor")
        assert app.selectbox(key="period").value == "5Y"
        app.selectbox(key="period").set_value("3Y").run()
        _switch_view(app, "Backtest")
        assert app.selectbox(key="period").value == "3Y"

    def test_monitor_metrics_have_no_pnl(self, app) -> None:
        app.run()
        _switch_view(app, "Ratio monitor")
        _no_exceptions(app)
        assert _labels(app) == ["Z-score", "Percentile", "Level"]
        for metric in app.metric:
            assert "P&L" not in metric.label
            assert "P&L" not in metric.delta
        assert not app.expander
        assert any("no P&L" in info.value for info in app.info)

    def test_monitor_has_no_costs_or_rule(self, app) -> None:
        app.run()
        _switch_view(app, "Ratio monitor")
        assert [box.label for box in app.selectbox] == ["Pair", "Period"]
        assert not app.number_input

    def test_monitor_all_pairs(self, app) -> None:
        app.run()
        _switch_view(app, "Ratio monitor")
        for pair in RATIO_PAIRS:
            app.selectbox(key="pair").set_value(pair).run()
            _no_exceptions(app)
            assert _labels(app) == ["Z-score", "Percentile", "Level"]
            assert len(_charts(app)) == 2
            assert _metric(app, "Z-score").value != "n/a"
            assert _metric(app, "Percentile").value != "n/a"

    def test_monitor_all_periods_and_custom(self, app) -> None:
        app.run()
        _switch_view(app, "Ratio monitor")
        for period in FIXED_PERIODS:
            app.selectbox(key="period").set_value(period).run()
            _no_exceptions(app)
        _set_custom(app, datetime.date(2008, 1, 2), datetime.date(2012, 12, 31))
        _no_exceptions(app)
        assert "2012-12-31" in _metric(app, "Level").proto.help

    def test_level_is_labelled_as_a_basis_not_a_price(self, app) -> None:
        app.run()
        _switch_view(app, "Ratio monitor")
        assert _metric(app, "Level").delta == "USD/MT basis"

    def test_zscore_uses_history_before_a_short_window(self, app) -> None:
        """Statistics run on the full history. On the window alone, 1Y would have no z-score."""
        app.run()
        _switch_view(app, "Ratio monitor")
        app.selectbox(key="period").set_value("1Y").run()
        spec = _chart_spec(app, "zscore_chart")
        assert np.isfinite(_trace_y(_trace(spec, "Z-score"))).all()
        mean = _trace(_chart_spec(app, "ratio_chart"), "1y mean")
        assert np.isfinite(_trace_y(mean)).all()

    def test_monitor_window_without_data_warns(self, app) -> None:
        app.run()
        _switch_view(app, "Ratio monitor")
        _set_custom(app, datetime.date(2010, 1, 2), datetime.date(2010, 1, 3))
        _no_exceptions(app)
        assert any("No Soybean / Corn data" in warning.value for warning in app.warning)


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------


class TestTradeLog:
    def test_log_has_labelled_columns_and_no_engine_fields(self, app) -> None:
        app.run()
        log = app.dataframe[0].value
        assert list(log.columns) == [
            "Date",
            "Position",
            "Price (cents/bushel)",
            "P&L (USD)",
            "Cumulative (USD)",
        ]

    def test_download_offered(self, app) -> None:
        app.run()
        downloads = app.get("download_button")
        assert len(downloads) == 1
        assert downloads[0].proto.label == "Download trades (CSV)"

    def test_no_fills_has_no_log_and_no_exception(self, app, synthetic_prices, monkeypatch) -> None:
        """A one-week window has no Donchian fill unless a breakout falls exactly there."""
        app.run()
        app.selectbox(key="rule").set_value("12M momentum").run()
        _set_custom(app, datetime.date(2010, 1, 4), datetime.date(2010, 1, 8))
        _no_exceptions(app)
        assert _metric(app, "Position at end").value == "Flat"
        assert _metric(app, "Position at end").delta == "no fills"
        assert _metric(app, "Unadjusted P&L").value == "$0"
        assert _metric(app, "Round trips").delta == "0 fills"
        assert _metric(app, "Time in market").value == "0%"
        assert not app.dataframe
        assert any("No fills" in caption.value for caption in app.caption)
        assert _charts(app)


# ---------------------------------------------------------------------------
# Data failure states
# ---------------------------------------------------------------------------


class TestDamagedCache:
    """A cache hit with a bad column must warn in that market and leave the rest usable."""

    @pytest.fixture(params=["all_nan", "missing"])
    def damaged_app(self, request, tmp_path, monkeypatch, synthetic_prices):
        frame = synthetic_prices.copy()
        if request.param == "all_nan":
            frame["Corn"] = np.nan
        else:
            frame = frame.drop(columns="Corn")
        path = tmp_path / "damaged.parquet"
        frame.to_parquet(path)
        monkeypatch.setattr(config, "YFINANCE_RAW", path)
        st.cache_data.clear()
        with patch("src.data_loader.yf.download") as download:
            yield _new_app()
        download.assert_not_called()
        st.cache_data.clear()

    def test_backtest_warns_and_keeps_the_controls(self, damaged_app) -> None:
        damaged_app.run()
        _no_exceptions(damaged_app)
        _assert_headlines(damaged_app)

        damaged_app.selectbox(key="market").set_value("Corn").run()

        _no_exceptions(damaged_app)
        assert any("No Corn prices are available" in w.value for w in damaged_app.warning)
        assert not damaged_app.metric
        assert not _charts(damaged_app)
        assert len(damaged_app.selectbox) >= 3, "Controls must stay available."

        damaged_app.selectbox(key="market").set_value("Soybean").run()
        _no_exceptions(damaged_app)
        _assert_headlines(damaged_app)

    def test_monitor_warns_and_keeps_the_controls(self, damaged_app) -> None:
        damaged_app.run()
        _switch_view(damaged_app, "Ratio monitor")

        _no_exceptions(damaged_app)
        assert any("No Soybean / Corn data is available" in w.value for w in damaged_app.warning)
        assert not damaged_app.metric
        assert not _charts(damaged_app)
        assert [box.label for box in damaged_app.selectbox] == ["Pair", "Period"]

        damaged_app.selectbox(key="pair").set_value("Heating Oil / WTI").run()
        _no_exceptions(damaged_app)
        assert _labels(damaged_app) == ["Z-score", "Percentile", "Level"]

    def test_default_market_damaged_on_first_load(self, tmp_path, monkeypatch, synthetic_prices):
        """Regression: an empty default market raised IndexError on the first screen."""
        frame = synthetic_prices.copy()
        frame["Soybean"] = np.nan
        path = tmp_path / "damaged.parquet"
        frame.to_parquet(path)
        monkeypatch.setattr(config, "YFINANCE_RAW", path)
        st.cache_data.clear()
        with patch("src.data_loader.yf.download") as download:
            at = _new_app().run()
        download.assert_not_called()
        _no_exceptions(at)
        assert any("No Soybean prices are available" in w.value for w in at.warning)
        st.cache_data.clear()


class TestDataStates:
    def test_load_failure_shows_retry(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(config, "YFINANCE_RAW", tmp_path / "absent.parquet")
        st.cache_data.clear()
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at = _new_app().run()
            _no_exceptions(at)
            assert any("yahoo down" in error.value for error in at.error)
            assert [button.label for button in at.button] == ["Retry"]
            assert not at.metric
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
        _assert_headlines(at)
        st.cache_data.clear()

    def test_old_data_labelled(self, tmp_path, monkeypatch, synthetic_prices) -> None:
        path = tmp_path / "old.parquet"
        synthetic_prices.loc[: pd.Timestamp("2024-06-28")].to_parquet(path)
        monkeypatch.setattr(config, "YFINANCE_RAW", path)
        st.cache_data.clear()
        with patch("src.data_loader.yf.download", side_effect=RuntimeError("yahoo down")):
            at = _new_app().run()
        _no_exceptions(at)
        assert any("2024-06-28 (may be out of date)" in caption.value for caption in at.caption)
        _assert_headlines(at)
        assert _charts(at)
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Source rules
# ---------------------------------------------------------------------------


class TestSource:
    def test_no_emoji_in_source(self) -> None:
        for relative in ("app.py", "src/catalog.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            assert not EMOJI.search(source), relative

    def test_no_unsafe_html_and_single_css_block(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        assert "unsafe_allow_html" not in source
        assert source.count("<style>") == 1
        assert "<script" not in source

    def test_charts_use_no_streamlit_theme(self) -> None:
        """Without theme=None Streamlit silently replaces the axis colours of the figures."""
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        calls = source.count("st.plotly_chart(")
        assert calls == 4
        assert source.count("theme=None") == calls


def _luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class TestDarkTheme:
    """The dark direction was chosen on the condition that it keeps these guarantees."""

    @pytest.fixture()
    def tokens(self) -> dict[str, str]:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        return dict(re.findall(r"--(\w+):(#[0-9a-f]{6})", source))

    def test_theme_is_dark_and_matches_the_tokens(self, tokens) -> None:
        with (ROOT / ".streamlit" / "config.toml").open("rb") as file:
            theme = tomllib.load(file)["theme"]
        assert theme["base"] == "dark"
        assert theme["backgroundColor"] == tokens["page"]
        assert theme["secondaryBackgroundColor"] == tokens["card"]
        assert theme["primaryColor"] == tokens["posl"] == visualization.ENTRY
        assert tokens["card"] == visualization.SURFACE

    @pytest.mark.parametrize(
        ("text", "surface"),
        [
            ("ink", "card"),
            ("mut", "card"),
            ("mut", "page"),
            ("ink", "raised"),
            ("mut", "raised"),
            ("onpos", "posl"),
        ],
    )
    def test_text_contrast_is_at_least_aa(self, tokens, text, surface) -> None:
        assert _contrast(tokens[text], tokens[surface]) >= 4.5

    def test_chart_marks_are_visible_on_the_surface(self) -> None:
        for color in (visualization.ENTRY, visualization.EXIT, visualization.FLAT):
            assert _contrast(color, visualization.SURFACE) >= 3

    def test_warning_does_not_borrow_the_position_colour(self) -> None:
        """Amber means long. A warning in amber would make a long position read as an alert."""
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        rule = re.search(r"\[data-testid='stAlertContainer'\]\{([^}]*)\}", source).group(1)
        assert "--posl" not in rule

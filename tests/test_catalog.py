"""Unit tests for src/catalog.py: the rules offered in the app."""

from __future__ import annotations

import inspect
import re
import typing

from src.catalog import RULES
from src.strategy import StrategyType, backtest

EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


class TestRules:
    def test_three_rules_in_display_order(self) -> None:
        assert [rule.label for rule in RULES] == ["12M momentum", "Donchian 20/10", "Buy & hold"]

    def test_keys_match_strategy_type(self) -> None:
        assert {rule.key for rule in RULES} == set(typing.get_args(StrategyType))

    def test_descriptions_are_not_empty(self) -> None:
        for rule in RULES:
            assert rule.description.strip()

    def test_copy_has_no_emoji(self) -> None:
        for rule in RULES:
            assert not EMOJI.search(rule.label)
            assert not EMOJI.search(rule.description)

    def test_donchian_label_matches_engine_defaults(self) -> None:
        params = inspect.signature(backtest).parameters
        entry, exit_ = params["donchian_entry"].default, params["donchian_exit"].default
        donchian = next(rule for rule in RULES if rule.key == "donchian")
        assert donchian.label == f"Donchian {entry}/{exit_}"
        assert f"prior {entry}-session high" in donchian.description
        assert f"prior {exit_}-session low" in donchian.description

    def test_momentum_description_matches_engine_default(self) -> None:
        lookback = inspect.signature(backtest).parameters["tsmom_lookback"].default
        momentum = next(rule for rule in RULES if rule.key == "tsmom_12m")
        assert f"{lookback} sessions" in momentum.description

"""Tests for large-scale generated strategies."""

from __future__ import annotations

from autonomous_trading_researcher.core.enums import SignalDirection
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.research.strategy_generator import MassiveStrategyGenerator
from autonomous_trading_researcher.strategies.registry import get_strategy


def test_massive_strategy_generator_creates_unique_population(
    app_config,
    synthetic_market_data,
) -> None:
    """The generator should create a deduplicated strategy population."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    generator = MassiveStrategyGenerator(seed=app_config.research.generated_strategy_seed)
    strategies = generator.generate(features, "BTC/USDT", candidate_count=20)

    assert len(strategies) == 20
    assert len({strategy.name for strategy in strategies}) == 20
    signals = strategies[0].generate_signals(features)
    assert set(signals.unique()).issubset({signal.value for signal in SignalDirection})


def test_generated_strategy_can_be_rehydrated_without_registry_file(
    app_config,
    synthetic_market_data,
) -> None:
    """Generated strategies should validate from in-memory parameters."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    generator = MassiveStrategyGenerator(seed=app_config.research.generated_strategy_seed)
    strategy = generator.generate(features, "BTC/USDT", candidate_count=1)[0]

    rehydrated = get_strategy(strategy.name, strategy.parameters)
    signals = rehydrated.generate_signals(features)

    assert rehydrated.name == strategy.name
    assert len(signals) == len(features)

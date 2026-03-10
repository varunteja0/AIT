"""Tests for automated strategy discovery."""

from __future__ import annotations

from autonomous_trading_researcher.backtesting.engine import EventDrivenBacktestEngine
from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.research.discovery import StrategyDiscoveryService


def test_strategy_discovery_returns_ranked_candidates(
    app_config,
    synthetic_market_data,
) -> None:
    """The discovery service should rank and validate strategy candidates."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    service = StrategyDiscoveryService(
        config=app_config.research,
        validation_config=app_config.validation,
        vectorized_backtester=VectorizedBacktestEngine(app_config.backtesting),
        event_driven_backtester=EventDrivenBacktestEngine(app_config.backtesting),
    )

    candidates = service.discover_for_symbol("BTC/USDT", features)

    assert candidates
    assert candidates[0].backtest_result.validation_engine == "event_driven"
    assert candidates[0].score >= candidates[-1].score
    assert (
        candidates[0].strategy_name in app_config.research.enabled_strategies
        or candidates[0].strategy_name.startswith("generated_")
    )

"""Tests for vectorized and event-driven backtesting engines."""

from __future__ import annotations

import pandas as pd

from autonomous_trading_researcher.backtesting.engine import EventDrivenBacktestEngine
from autonomous_trading_researcher.backtesting.validation import WalkForwardValidator
from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.strategies.base import BaseStrategy
from autonomous_trading_researcher.strategies.momentum import MomentumStrategy


class AlwaysLongStrategy(BaseStrategy):
    """Minimal always-long test strategy."""

    name = "always_long"

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series("LONG", index=features.index)


def test_backtesting_engines_return_metrics(app_config, synthetic_market_data) -> None:
    """Both backtesting engines should produce populated results."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    strategy = MomentumStrategy(threshold=0.001, leverage=1.0)

    vectorized = VectorizedBacktestEngine(app_config.backtesting).run(
        features,
        strategy,
        symbol="BTC/USDT",
    )
    event_driven = EventDrivenBacktestEngine(app_config.backtesting).run(
        features,
        strategy,
        symbol="BTC/USDT",
    )

    assert vectorized.validation_engine == "vectorized"
    assert event_driven.validation_engine == "event_driven"
    assert len(vectorized.equity_curve) == len(features)
    assert len(event_driven.equity_curve) == len(features)
    assert vectorized.metrics.max_drawdown >= 0
    assert event_driven.metrics.total_return > -1.0


def test_vectorized_backtester_runs_parallel_batch(
    app_config,
    synthetic_market_data,
) -> None:
    """The vectorized engine should batch-evaluate strategies in parallel."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    engine = VectorizedBacktestEngine(app_config.backtesting)
    strategies = [
        MomentumStrategy(threshold=0.001, leverage=1.0),
        MomentumStrategy(threshold=0.002, leverage=0.5),
        MomentumStrategy(threshold=0.003, leverage=1.5),
    ]

    results = engine.run_batch(features, strategies, symbol="BTC/USDT", max_workers=2)

    assert len(results) == 3
    assert all(result.symbol == "BTC/USDT" for result in results)


def test_event_driven_backtester_lags_signals_to_avoid_lookahead(app_config) -> None:
    """The event-driven engine should only trade on the next bar."""

    features = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [110.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [110.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
    )
    strategy = AlwaysLongStrategy(leverage=1.0)

    result = EventDrivenBacktestEngine(app_config.backtesting).run(
        features,
        strategy,
        symbol="BTC/USDT",
    )

    assert result.equity_curve[0] == app_config.backtesting.starting_cash
    assert result.trade_log[0]["timestamp"] == features.index[1]


def test_walk_forward_validator_produces_out_of_sample_metrics(
    app_config,
    synthetic_market_data,
) -> None:
    """Walk-forward validation should produce aggregate out-of-sample metrics."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    strategy = MomentumStrategy(threshold=0.001, leverage=1.0)
    backtester = VectorizedBacktestEngine(app_config.backtesting)

    report = WalkForwardValidator(app_config.backtesting, backtester).run(
        features,
        strategy,
        symbol="BTC/USDT",
    )

    assert report is not None
    assert report.fold_count >= 1
    assert report.metrics.max_drawdown >= 0.0

"""Tests for the built-in strategies."""

from __future__ import annotations

from autonomous_trading_researcher.core.enums import SignalDirection
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.strategies.breakout import BreakoutStrategy
from autonomous_trading_researcher.strategies.mean_reversion import MeanReversionStrategy
from autonomous_trading_researcher.strategies.momentum import MomentumStrategy


def test_builtin_strategies_emit_supported_signals(app_config, synthetic_market_data) -> None:
    """Each built-in strategy should only emit LONG, SHORT, or FLAT signals."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    strategies = [
        MomentumStrategy(threshold=0.001, leverage=1.0),
        MeanReversionStrategy(z_score_threshold=0.75, leverage=1.0),
        BreakoutStrategy(lookback=8, leverage=1.0),
    ]
    allowed = {signal.value for signal in SignalDirection}

    for strategy in strategies:
        signals = strategy.generate_signals(features)
        assert set(signals.unique()).issubset(allowed)
        assert len(signals) == len(features)


"""Tests for statistical validation and ensemble execution."""

from __future__ import annotations

from autonomous_trading_researcher.backtesting.statistics import StrategyStatisticsValidator
from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.execution.ensemble import StrategyEnsembleEngine
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.strategies.momentum import MomentumStrategy


def test_statistical_validator_accepts_reasonable_backtest(
    app_config,
    synthetic_market_data,
) -> None:
    """A sensible backtest should pass permissive validation thresholds."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    result = VectorizedBacktestEngine(app_config.backtesting).run(
        features,
        MomentumStrategy(threshold=0.001, leverage=1.0),
        symbol="BTC/USDT",
    )

    validation = StrategyStatisticsValidator(app_config.validation).validate(result)

    assert validation.passed is True
    assert isinstance(validation.alpha_t_stat, float)


def test_ensemble_engine_aggregates_member_signals(
    app_config,
    synthetic_market_data,
) -> None:
    """The ensemble engine should combine the top candidate signals."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    backtester = VectorizedBacktestEngine(app_config.backtesting)
    strategies = [
        MomentumStrategy(threshold=0.001, leverage=1.0),
        MomentumStrategy(threshold=0.002, leverage=0.5),
    ]
    candidates: list[StrategyCandidate] = []
    for rank, strategy in enumerate(strategies, start=1):
        result = backtester.run(features, strategy, symbol="BTC/USDT")
        candidates.append(
            StrategyCandidate(
                symbol="BTC/USDT",
                strategy_name=strategy.name,
                parameters=strategy.parameters,
                score=float(rank),
                backtest_result=result,
            )
        )

    decision = StrategyEnsembleEngine(ensemble_size=2).aggregate_signal(candidates, features)

    assert decision.symbol == "BTC/USDT"
    assert 0.0 <= decision.confidence <= 1.0
    assert len(decision.members) == 2

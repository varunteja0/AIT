"""Tests for the genetic strategy evolution engine."""

from __future__ import annotations

import math

from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.research.genetic_evolution import (
    GeneticStrategyEvolutionEngine,
)
from autonomous_trading_researcher.research.ranking import CandidateRanker
from autonomous_trading_researcher.research.strategy_generator import MassiveStrategyGenerator


def test_genetic_strategy_evolution_returns_ranked_candidates(
    app_config,
    synthetic_market_data,
) -> None:
    """Evolution should produce a scored strategy population."""

    features = FeaturePipeline(app_config.feature_engineering).build(synthetic_market_data)
    generator = MassiveStrategyGenerator(seed=app_config.research.generated_strategy_seed)
    seed_population = generator.generate(features, "BTC/USDT", candidate_count=8)
    evolution = GeneticStrategyEvolutionEngine(
        vectorized_backtester=VectorizedBacktestEngine(app_config.backtesting),
        ranker=CandidateRanker(app_config.research.ranking_weights),
        generator=generator,
        population_size=6,
        generations=2,
        max_workers=2,
    )

    candidates = evolution.evolve("BTC/USDT", features, seed_population)

    assert candidates
    assert candidates[0].score >= candidates[-1].score
    assert not math.isnan(candidates[0].backtest_result.metrics.sharpe_ratio)

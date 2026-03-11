"""Genetic evolution engine for generated strategy populations."""

from __future__ import annotations

import random

from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.infra.distributed.backends import ExecutionBackend
from autonomous_trading_researcher.research.ranking import CandidateRanker
from autonomous_trading_researcher.research.strategy_generator import (
    GeneratedStrategy,
    MassiveStrategyGenerator,
)


class GeneticStrategyEvolutionEngine:
    """Evolve generated strategies using selection, mutation, and crossover."""

    def __init__(
        self,
        vectorized_backtester: VectorizedBacktestEngine,
        ranker: CandidateRanker,
        generator: MassiveStrategyGenerator,
        population_size: int,
        generations: int,
        max_workers: int,
        execution_backend: ExecutionBackend | None = None,
        mutation_rate: float = 0.25,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.2,
        seed: int = 23,
    ) -> None:
        self.vectorized_backtester = vectorized_backtester
        self.ranker = ranker
        self.generator = generator
        self.population_size = max(2, population_size)
        self.generations = max(1, generations)
        self.max_workers = max(1, max_workers)
        self.execution_backend = execution_backend
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.rng = random.Random(seed)  # noqa: S311

    def _evaluate_population(
        self,
        symbol: str,
        features,
        population: list[GeneratedStrategy],
    ) -> list[StrategyCandidate]:
        """Evaluate a population and return ranked strategy candidates."""

        results = self.vectorized_backtester.run_batch(
            features,
            population,
            symbol=symbol,
            max_workers=self.max_workers,
            backend=self.execution_backend,
        )
        candidates = [
            StrategyCandidate(
                symbol=symbol,
                strategy_name=result.strategy_name,
                parameters=result.parameters,
                score=self.ranker.score(result.metrics),
                backtest_result=result,
            )
            for result in results
        ]
        return self.ranker.rank(candidates)

    def _tournament_select(
        self,
        ranked: list[StrategyCandidate],
        size: int = 3,
    ) -> GeneratedStrategy:
        """Select one strategy via tournament selection."""

        pool = self.rng.sample(ranked, k=min(size, len(ranked)))
        winner = max(pool, key=lambda candidate: candidate.score)
        return GeneratedStrategy(**winner.parameters)

    def evolve(
        self,
        symbol: str,
        features,
        seed_population: list[GeneratedStrategy],
    ) -> list[StrategyCandidate]:
        """Run the genetic evolution process and return ranked final candidates."""

        population = list(seed_population[: self.population_size])
        if len(population) < self.population_size:
            population.extend(
                self.generator.generate(
                    features,
                    symbol,
                    candidate_count=self.population_size - len(population),
                )
            )

        ranked = self._evaluate_population(symbol, features, population)
        for _ in range(self.generations):
            elite_count = max(1, int(self.population_size * self.elite_fraction))
            next_population = [
                GeneratedStrategy(**candidate.parameters) for candidate in ranked[:elite_count]
            ]
            while len(next_population) < self.population_size:
                left = self._tournament_select(ranked)
                if self.rng.random() < self.crossover_rate:
                    right = self._tournament_select(ranked)
                    child = self.generator.crossover(left, right)
                else:
                    child = left
                if self.rng.random() < self.mutation_rate:
                    child = self.generator.mutate(child, features)
                next_population.append(child)
            ranked = self._evaluate_population(symbol, features, next_population)
        return ranked

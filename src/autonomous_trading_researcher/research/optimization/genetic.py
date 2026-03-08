"""Simple discrete genetic algorithm optimizer."""

from __future__ import annotations

import random
from collections.abc import Callable

from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.research.generator import ParameterSpace, sample_parameters


class GeneticAlgorithmOptimizer:
    """Search parameter spaces via tournament selection and mutation."""

    def __init__(self, seed: int = 11) -> None:
        self.rng = random.Random(seed)  # noqa: S311

    def _mutate(
        self,
        chromosome: dict[str, float | int | str],
        parameter_space: ParameterSpace,
        mutation_rate: float = 0.2,
    ) -> dict[str, float | int | str]:
        """Randomly mutate a chromosome in-place and return it."""

        mutated = chromosome.copy()
        for name, values in parameter_space.items():
            if self.rng.random() < mutation_rate:
                mutated[name] = self.rng.choice(values)
        return mutated

    def _crossover(
        self,
        left: dict[str, float | int | str],
        right: dict[str, float | int | str],
    ) -> dict[str, float | int | str]:
        """Create a child chromosome from two parents."""

        child: dict[str, float | int | str] = {}
        for name in left.keys():
            child[name] = left[name] if self.rng.random() < 0.5 else right[name]
        return child

    def optimize(
        self,
        strategy_name: str,
        parameter_space: ParameterSpace,
        population_size: int,
        generations: int,
        evaluator: Callable[[str, dict[str, float | int | str]], StrategyCandidate],
    ) -> list[StrategyCandidate]:
        """Return scored candidates from a simple GA search."""

        if not parameter_space:
            return [evaluator(strategy_name, {})]

        def key_for(
            params: dict[str, float | int | str],
        ) -> tuple[tuple[str, float | int | str], ...]:
            return tuple(sorted(params.items()))

        cache: dict[tuple[tuple[str, float | int | str], ...], StrategyCandidate] = {}

        def evaluate(params: dict[str, float | int | str]) -> StrategyCandidate:
            cache_key = key_for(params)
            if cache_key not in cache:
                cache[cache_key] = evaluator(strategy_name, params)
            return cache[cache_key]

        population = [
            sample_parameters(parameter_space, self.rng) for _ in range(max(2, population_size))
        ]
        elite_count = max(1, population_size // 5)

        for _ in range(max(1, generations)):
            scored = sorted(
                (evaluate(chromosome) for chromosome in population),
                key=lambda candidate: candidate.score,
                reverse=True,
            )
            parents = [candidate.parameters for candidate in scored[: max(2, elite_count * 2)]]
            next_population = [candidate.parameters for candidate in scored[:elite_count]]
            while len(next_population) < population_size:
                left = self.rng.choice(parents)
                right = self.rng.choice(parents)
                child = self._crossover(left, right)
                next_population.append(self._mutate(child, parameter_space))
            population = next_population

        return sorted(cache.values(), key=lambda candidate: candidate.score, reverse=True)

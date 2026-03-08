"""Discrete grid-search optimizer."""

from __future__ import annotations

from collections.abc import Callable

from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.research.generator import ParameterSpace, iter_parameter_grid


class GridSearchOptimizer:
    """Evaluate every combination in a discrete parameter grid."""

    def optimize(
        self,
        strategy_name: str,
        parameter_space: ParameterSpace,
        limit: int,
        evaluator: Callable[[str, dict[str, float | int | str]], StrategyCandidate],
    ) -> list[StrategyCandidate]:
        """Return scored candidates from exhaustive search."""

        candidates = [
            evaluator(strategy_name, parameters)
            for parameters in iter_parameter_grid(parameter_space, limit=limit)
        ]
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

"""Bayesian optimization backed by Optuna's TPE sampler."""

from __future__ import annotations

from collections.abc import Callable

import optuna

from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.research.generator import ParameterSpace


class BayesianOptimizer:
    """Optimize strategy parameters with a Bayesian sampler."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def optimize(
        self,
        strategy_name: str,
        parameter_space: ParameterSpace,
        trials: int,
        evaluator: Callable[[str, dict[str, float | int | str]], StrategyCandidate],
    ) -> list[StrategyCandidate]:
        """Return scored candidates from a Bayesian search run."""

        if not parameter_space or trials <= 0:
            return [evaluator(strategy_name, {})]

        trial_results: dict[int, StrategyCandidate] = {}

        def objective(trial: optuna.Trial) -> float:
            parameters: dict[str, float | int | str] = {}
            for name, values in parameter_space.items():
                parameters[name] = trial.suggest_categorical(name, values)
            candidate = evaluator(strategy_name, parameters)
            trial_results[trial.number] = candidate
            return candidate.score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        return sorted(trial_results.values(), key=lambda candidate: candidate.score, reverse=True)

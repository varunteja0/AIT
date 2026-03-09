"""Bayesian optimization backed by Optuna's TPE sampler."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import optuna

from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.research.generator import ParameterSpace


class BayesianOptimizer:
    """Optimize strategy parameters with a Bayesian sampler."""

    def __init__(self, seed: int = 7, storage_path: str | None = None) -> None:
        self.seed = seed
        self.storage_path = storage_path

    def _storage_url(self) -> str | None:
        """Return the Optuna storage URL when persistence is configured."""

        if self.storage_path is None:
            return None
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"

    def optimize(
        self,
        strategy_name: str,
        parameter_space: ParameterSpace,
        trials: int,
        evaluator: Callable[[str, dict[str, float | int | str]], StrategyCandidate],
        study_key: str | None = None,
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
            storage=self._storage_url(),
            study_name=study_key or strategy_name,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        return sorted(trial_results.values(), key=lambda candidate: candidate.score, reverse=True)

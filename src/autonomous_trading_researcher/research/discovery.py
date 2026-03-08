"""High-level automated strategy discovery service."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from autonomous_trading_researcher.backtesting.engine import EventDrivenBacktestEngine
from autonomous_trading_researcher.backtesting.validation import WalkForwardValidator
from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.config import ResearchConfig
from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.research.experiment_db import ExperimentDatabase
from autonomous_trading_researcher.research.generator import iter_parameter_grid
from autonomous_trading_researcher.research.genetic_evolution import (
    GeneticStrategyEvolutionEngine,
)
from autonomous_trading_researcher.research.optimization.bayesian import BayesianOptimizer
from autonomous_trading_researcher.research.ranking import CandidateRanker
from autonomous_trading_researcher.research.strategy_generator import (
    MassiveStrategyGenerator,
    build_strategy_id,
)
from autonomous_trading_researcher.strategies.generated.loader import save_strategy_payload
from autonomous_trading_researcher.strategies.registry import BUILTIN_STRATEGIES, get_strategy

CandidateKey = tuple[str, tuple[tuple[str, float | int | str], ...]]


class StrategyDiscoveryService:
    """Discover and validate strategy candidates for a single symbol dataset."""

    def __init__(
        self,
        config: ResearchConfig,
        vectorized_backtester: VectorizedBacktestEngine,
        event_driven_backtester: EventDrivenBacktestEngine,
    ) -> None:
        self.config = config
        self.vectorized_backtester = vectorized_backtester
        self.event_driven_backtester = event_driven_backtester
        self.ranker = CandidateRanker(config.ranking_weights)
        self.bayesian = BayesianOptimizer()
        self.generator = MassiveStrategyGenerator(seed=config.generated_strategy_seed)
        self.walk_forward = WalkForwardValidator(
            config=vectorized_backtester.config,
            backtester=vectorized_backtester,
        )
        self.evolution = GeneticStrategyEvolutionEngine(
            vectorized_backtester=vectorized_backtester,
            ranker=self.ranker,
            generator=self.generator,
            population_size=config.genetic_population,
            generations=config.genetic_generations,
            max_workers=config.max_parallel_workers,
        )
        self.experiment_db = ExperimentDatabase(config.experiment_db_path)
        self.generated_strategy_dir = Path(config.generated_strategy_dir)

    def _evaluate(
        self,
        symbol: str,
        strategy,
        features,
    ) -> StrategyCandidate:
        """Run a fast vectorized evaluation for one candidate."""

        result = self.vectorized_backtester.run(features, strategy, symbol=symbol)
        score = self.ranker.score(result.metrics)
        return StrategyCandidate(
            symbol=symbol,
            strategy_name=strategy.name,
            parameters=strategy.parameters,
            score=score,
            backtest_result=result,
        )

    def _evaluate_batch(
        self,
        symbol: str,
        features,
        strategies,
    ) -> list[StrategyCandidate]:
        """Run a vectorized batch backtest over a strategy population."""

        results = self.vectorized_backtester.run_batch(
            features,
            strategies,
            symbol=symbol,
            max_workers=self.config.max_parallel_workers,
        )
        return [
            StrategyCandidate(
                symbol=symbol,
                strategy_name=result.strategy_name,
                parameters=result.parameters,
                score=self.ranker.score(result.metrics),
                backtest_result=result,
            )
            for result in results
        ]

    def _validate_top_candidate(self, candidate: StrategyCandidate, features) -> StrategyCandidate:
        """Re-run the best candidate through the event-driven validator."""

        strategy = get_strategy(candidate.strategy_name, candidate.parameters)
        validated_result = self.event_driven_backtester.run(
            features,
            strategy,
            symbol=candidate.symbol,
        )
        walk_forward_report = self.walk_forward.run(features, strategy, symbol=candidate.symbol)
        event_score = self.ranker.score(validated_result.metrics)
        walk_forward_score = (
            self.ranker.score(walk_forward_report.metrics)
            if walk_forward_report is not None
            else event_score
        )
        combined_score = (0.7 * event_score) + (0.3 * walk_forward_score)
        return StrategyCandidate(
            symbol=candidate.symbol,
            strategy_name=candidate.strategy_name,
            parameters={
                **candidate.parameters,
                "walk_forward_score": round(walk_forward_score, 6),
                "walk_forward_folds": (
                    walk_forward_report.fold_count if walk_forward_report is not None else 0
                ),
            },
            score=combined_score,
            backtest_result=validated_result,
        )

    def _traditional_candidates(self, symbol: str, features) -> list[StrategyCandidate]:
        """Discover traditional indicator strategies from built-in templates."""

        raw_candidates: list[StrategyCandidate] = []
        for strategy_name in self.config.enabled_strategies:
            if strategy_name not in BUILTIN_STRATEGIES:
                continue
            parameter_space = self.config.strategy_parameter_space.get(strategy_name, {})
            strategy_population = [
                get_strategy(strategy_name, parameters)
                for parameters in iter_parameter_grid(
                    parameter_space,
                    self.config.grid_search_limit,
                )
            ]
            if strategy_population:
                raw_candidates.extend(
                    self._evaluate_batch(symbol, features, strategy_population)
                )
            raw_candidates.extend(
                self.bayesian.optimize(
                    strategy_name,
                    parameter_space,
                    self.config.bayesian_trials,
                    lambda name, params: self._evaluate(
                        symbol,
                        get_strategy(name, params),
                        features,
                    ),
                )
            )
        return raw_candidates

    def _generated_candidates(self, symbol: str, features) -> list[StrategyCandidate]:
        """Discover generated and microstructure-heavy strategies."""

        generated_population = self.generator.generate(
            features,
            symbol,
            candidate_count=self.config.generated_strategy_count,
        )
        raw_candidates = self._evaluate_batch(symbol, features, generated_population)
        evolved_candidates = self.evolution.evolve(
            symbol,
            features,
            seed_population=generated_population[: self.config.genetic_population],
        )
        return raw_candidates + evolved_candidates

    def _persist_top_candidates(self, candidates: list[StrategyCandidate]) -> None:
        """Persist top strategies to disk and the experiment database."""

        for candidate in candidates:
            strategy_id = str(candidate.parameters.get("strategy_id", ""))
            if not strategy_id:
                enriched_parameters = dict(candidate.parameters)
                enriched_parameters["strategy_name"] = candidate.strategy_name
                strategy_id = build_strategy_id(enriched_parameters).replace(
                    "generated_",
                    f"{candidate.strategy_name}_",
                    1,
                )
                candidate.parameters["strategy_id"] = strategy_id
            family = (
                "generated"
                if isinstance(candidate.parameters.get("template"), str)
                else "builtin"
            )
            payload = {
                "strategy_id": strategy_id,
                "strategy_name": candidate.strategy_name,
                "family": family,
                "symbol": candidate.symbol,
                "parameters": candidate.parameters,
                "score": candidate.score,
                "metrics": asdict(candidate.backtest_result.metrics),
            }
            save_strategy_payload(payload, self.generated_strategy_dir)
        self.experiment_db.record_candidates(candidates)

    def discover_for_symbol(self, symbol: str, features) -> list[StrategyCandidate]:
        """Run large-scale discovery and return the top-ranked strategies."""

        raw_candidates = self._traditional_candidates(symbol, features)
        raw_candidates.extend(self._generated_candidates(symbol, features))
        deduplicated: dict[CandidateKey, StrategyCandidate] = {}
        for candidate in raw_candidates:
            key = (candidate.strategy_name, tuple(sorted(candidate.parameters.items())))
            existing = deduplicated.get(key)
            if existing is None or candidate.score > existing.score:
                deduplicated[key] = candidate
        ranked = self.ranker.rank(list(deduplicated.values()))
        shortlisted = ranked[: self.config.top_n_strategies]
        validated = [
            self._validate_top_candidate(candidate, features)
            for candidate in shortlisted[: self.config.validate_top_n]
        ]
        combined = self.ranker.rank(validated + shortlisted[self.config.validate_top_n :])
        top_candidates = combined[: self.config.top_n_strategies]
        self._persist_top_candidates(top_candidates)
        return top_candidates

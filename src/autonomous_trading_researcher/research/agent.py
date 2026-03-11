"""Autonomous research agent that proposes new experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from autonomous_trading_researcher.research.experiment_db import ExperimentDatabase
from autonomous_trading_researcher.research.knowledge_graph.queries import (
    get_feature_combinations_that_improve_returns,
    get_top_features_by_sharpe,
)
from autonomous_trading_researcher.research.strategy_generator import GeneratedStrategy


@dataclass(slots=True)
class Hypothesis:
    """Research hypothesis generated from prior results."""

    hypothesis_id: str
    description: str
    parameters: dict[str, Any]


class HypothesisGenerator:
    """Generate hypotheses from experiment outcomes and knowledge graph."""

    def __init__(self, experiment_db: ExperimentDatabase) -> None:
        self.experiment_db = experiment_db

    def generate(self, *, top_n: int = 5) -> list[Hypothesis]:
        top_strategies = self.experiment_db.top_strategies(limit=top_n)
        hypotheses: list[Hypothesis] = []
        for strategy in top_strategies:
            params = strategy.get("parameters", {})
            primary = params.get("primary_feature")
            if isinstance(primary, str):
                hypotheses.append(
                    Hypothesis(
                        hypothesis_id=f"focus_{primary}",
                        description=f"Increase sensitivity to {primary}.",
                        parameters={"primary_feature": primary, "template": "feature_gt_threshold"},
                    )
                )
        return hypotheses


class ExperimentPlanner:
    """Convert hypotheses into concrete strategy candidates."""

    def build_strategies(
        self,
        *,
        hypotheses: Iterable[Hypothesis],
        features: pd.DataFrame,
        symbol: str,
    ) -> list[GeneratedStrategy]:
        strategies: list[GeneratedStrategy] = []
        for hypothesis in hypotheses:
            primary = hypothesis.parameters.get("primary_feature")
            if not isinstance(primary, str) or primary not in features.columns:
                continue
            series = features[primary].replace([np.inf, -np.inf], np.nan).dropna()
            threshold = float(series.quantile(0.6)) if not series.empty else 0.0
            parameters = {
                "template": hypothesis.parameters.get("template", "feature_gt_threshold"),
                "primary_feature": primary,
                "threshold": threshold,
                "leverage": 1.0,
                "holding_period": 2,
                "stop_loss": 0.01,
                "take_profit": 0.02,
                "symbol": symbol,
            }
            strategies.append(GeneratedStrategy(**parameters))
        return strategies


class ResultAnalyzer:
    """Analyze experiment results to inform future research."""

    def summarize(self, top_strategies: list[dict[str, Any]]) -> dict[str, Any]:
        if not top_strategies:
            return {"best_score": 0.0, "count": 0}
        return {
            "best_score": float(top_strategies[0].get("score", 0.0)),
            "count": len(top_strategies),
        }


class ResearchAgent:
    """High-level agent that proposes new experiments."""

    def __init__(self, experiment_db: ExperimentDatabase) -> None:
        self.hypothesis_generator = HypothesisGenerator(experiment_db)
        self.planner = ExperimentPlanner()
        self.analyzer = ResultAnalyzer()
        self.experiment_db = experiment_db

    def propose_strategies(
        self,
        *,
        features: pd.DataFrame,
        symbol: str,
        top_n_features: int = 5,
    ) -> list[GeneratedStrategy]:
        hypotheses = self.hypothesis_generator.generate(top_n=top_n_features)
        strategies = self.planner.build_strategies(
            hypotheses=hypotheses,
            features=features,
            symbol=symbol,
        )
        return strategies

    def knowledge_graph_suggestions(self, graph_store) -> list[GeneratedStrategy]:
        del graph_store
        return []

    def top_feature_hypotheses(self, graph_store, *, top_n: int = 5) -> list[Hypothesis]:
        top_features = get_top_features_by_sharpe(graph_store, top_n=top_n)
        hypotheses: list[Hypothesis] = []
        for entry in top_features:
            feature_id = entry["feature_id"].replace("feature:", "")
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"kg_{feature_id}",
                    description=f"Focus on {feature_id} from knowledge graph.",
                    parameters={"primary_feature": feature_id, "template": "feature_gt_threshold"},
                )
            )
        return hypotheses

    def feature_combo_hypotheses(self, graph_store, *, min_delta: float = 0.1) -> list[Hypothesis]:
        combos = get_feature_combinations_that_improve_returns(
            graph_store,
            min_delta=min_delta,
        )
        hypotheses: list[Hypothesis] = []
        for combo in combos[:5]:
            feature_a = combo["feature_a"].replace("feature:", "")
            feature_b = combo["feature_b"].replace("feature:", "")
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"combo_{feature_a}_{feature_b}",
                    description="Combine complementary features.",
                    parameters={
                        "primary_feature": feature_a,
                        "secondary_feature": feature_b,
                        "template": "feature_combo_long",
                    },
                )
            )
        return hypotheses

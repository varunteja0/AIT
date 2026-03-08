"""Candidate scoring and ranking logic."""

from __future__ import annotations

from autonomous_trading_researcher.core.models import PerformanceMetrics, StrategyCandidate


class CandidateRanker:
    """Score and sort strategy candidates using weighted metrics."""

    def __init__(self, metric_weights: dict[str, float]) -> None:
        self.metric_weights = metric_weights

    def score(self, metrics: PerformanceMetrics) -> float:
        """Compute a scalar candidate score."""

        score = 0.0
        for metric_name, weight in self.metric_weights.items():
            score += getattr(metrics, metric_name, 0.0) * weight
        score -= metrics.max_drawdown * 0.25
        return float(score)

    def rank(self, candidates: list[StrategyCandidate]) -> list[StrategyCandidate]:
        """Sort candidates in descending score order."""

        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


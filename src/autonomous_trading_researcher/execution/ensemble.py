"""Strategy ensemble selection and signal aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autonomous_trading_researcher.core.enums import SignalDirection
from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.strategies.registry import get_strategy


@dataclass(frozen=True, slots=True)
class EnsembleDecision:
    """Aggregated ensemble trading decision for one symbol."""

    symbol: str
    direction: SignalDirection
    confidence: float
    members: list[str]


class StrategyEnsembleEngine:
    """Select top strategies and combine their signals via weighted voting."""

    def __init__(self, ensemble_size: int = 10) -> None:
        self.ensemble_size = max(1, ensemble_size)

    def select(self, candidates: list[StrategyCandidate]) -> list[StrategyCandidate]:
        """Select the top-N validated strategies for deployment."""

        return candidates[: self.ensemble_size]

    def _weights(self, candidates: list[StrategyCandidate]) -> list[float]:
        """Build non-negative voting weights from candidate scores."""

        positive_scores = [max(candidate.score, 0.0) for candidate in candidates]
        total = sum(positive_scores)
        if total <= 0.0:
            return [1.0 / len(candidates)] * len(candidates)
        return [score / total for score in positive_scores]

    def aggregate_signal(
        self,
        candidates: list[StrategyCandidate],
        features: pd.DataFrame,
    ) -> EnsembleDecision:
        """Aggregate member signals into one directional ensemble decision."""

        if not candidates:
            raise ValueError("ensemble_requires_candidates")
        weights = self._weights(candidates)
        vote = 0.0
        signal_map = {
            SignalDirection.LONG.value: 1.0,
            SignalDirection.SHORT.value: -1.0,
            SignalDirection.FLAT.value: 0.0,
        }
        members: list[str] = []
        for weight, candidate in zip(weights, candidates, strict=True):
            strategy = get_strategy(candidate.strategy_name, candidate.parameters)
            signal = str(strategy.generate_signals(features).iloc[-1])
            vote += weight * signal_map.get(signal, 0.0)
            members.append(candidate.strategy_name)
        if vote > 0:
            direction = SignalDirection.LONG
        elif vote < 0:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.FLAT
        return EnsembleDecision(
            symbol=candidates[0].symbol,
            direction=direction,
            confidence=min(1.0, abs(vote)),
            members=members,
        )

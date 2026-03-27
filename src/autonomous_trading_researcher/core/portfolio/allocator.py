"""Portfolio allocation logic for strategy ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from autonomous_trading_researcher.core.models import StrategyCandidate


@dataclass(slots=True)
class AllocationResult:
    """Portfolio allocation output."""

    weights: dict[str, float]
    symbol_weights: dict[str, float]


class PortfolioAllocator:
    """Compute strategy weights with correlation control and risk budgeting."""

    def __init__(
        self,
        *,
        annualization_factor: int = 252,
        min_weight: float = 0.0,
    ) -> None:
        self.annualization_factor = annualization_factor
        self.min_weight = min_weight

    def _candidate_id(self, candidate: StrategyCandidate) -> str:
        return str(candidate.parameters.get("strategy_id", candidate.strategy_name))

    def _returns_frame(self, candidates: Iterable[StrategyCandidate]) -> pd.DataFrame:
        payload: dict[str, list[float]] = {}
        for candidate in candidates:
            returns = candidate.backtest_result.returns
            if not returns:
                continue
            payload[self._candidate_id(candidate)] = list(returns)
        if not payload:
            return pd.DataFrame()
        return pd.DataFrame(payload).fillna(0.0)

    def allocate(self, candidates: list[StrategyCandidate]) -> AllocationResult:
        """Return normalized weights per strategy and per symbol."""

        if not candidates:
            return AllocationResult(weights={}, symbol_weights={})

        returns_frame = self._returns_frame(candidates)
        correlations = (
            returns_frame.corr().fillna(0.0)
            if not returns_frame.empty
            else pd.DataFrame()
        )
        weights: dict[str, float] = {}
        for candidate in candidates:
            candidate_id = self._candidate_id(candidate)
            base_score = max(candidate.score, 0.0)
            if returns_frame.empty:
                volatility = 1.0
                avg_corr = 0.0
            else:
                series = returns_frame.get(candidate_id)
                volatility = (
                    float(series.std(ddof=0)) * np.sqrt(self.annualization_factor)
                    if series is not None
                    else 1.0
                )
                correlations_for_candidate = correlations.get(candidate_id)
                avg_corr = (
                    float(correlations_for_candidate.abs().mean())
                    if correlations_for_candidate is not None
                    else 0.0
                )
            corr_adjustment = 1.0 / (1.0 + avg_corr)
            risk_adjustment = 1.0 / max(volatility, 1e-9)
            weight = base_score * corr_adjustment * risk_adjustment
            weights[candidate_id] = max(self.min_weight, weight)

        total = sum(weights.values())
        if total <= 0.0:
            equal_weight = 1.0 / len(weights)
            weights = {key: equal_weight for key in weights}
        else:
            weights = {key: value / total for key, value in weights.items()}

        symbol_weights: dict[str, float] = {}
        for candidate in candidates:
            candidate_id = self._candidate_id(candidate)
            symbol_weights[candidate.symbol] = symbol_weights.get(candidate.symbol, 0.0) + weights.get(
                candidate_id, 0.0
            )

        return AllocationResult(weights=weights, symbol_weights=symbol_weights)

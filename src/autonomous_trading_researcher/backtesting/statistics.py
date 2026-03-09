"""Statistical validation helpers for strategy acceptance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from autonomous_trading_researcher.config import ValidationConfig
from autonomous_trading_researcher.core.models import BacktestResult


@dataclass(frozen=True, slots=True)
class StatisticalValidationResult:
    """Outcome of statistical validation for one backtest."""

    passed: bool
    alpha_t_stat: float
    rejection_reasons: list[str]


def alpha_t_statistic(period_returns: pd.Series) -> float:
    """Compute a one-sample t-statistic for mean strategy alpha."""

    clean_returns = period_returns.dropna()
    observations = len(clean_returns)
    if observations < 2:
        return 0.0
    standard_error = float(clean_returns.std(ddof=1)) / math.sqrt(observations)
    if standard_error == 0.0:
        return 0.0
    return float(clean_returns.mean()) / standard_error


class StrategyStatisticsValidator:
    """Apply statistical thresholds to validated backtests."""

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def validate(self, backtest_result: BacktestResult) -> StatisticalValidationResult:
        """Validate a backtest result against configured thresholds."""

        period_returns = pd.Series(backtest_result.returns)
        metrics = backtest_result.metrics
        alpha_t_stat = alpha_t_statistic(period_returns)
        rejection_reasons: list[str] = []
        if metrics.sharpe_ratio < self.config.min_sharpe:
            rejection_reasons.append("min_sharpe")
        if metrics.sortino_ratio < self.config.min_sortino:
            rejection_reasons.append("min_sortino")
        if metrics.profit_factor < self.config.min_profit_factor:
            rejection_reasons.append("min_profit_factor")
        if metrics.max_drawdown > self.config.max_drawdown:
            rejection_reasons.append("max_drawdown")
        if alpha_t_stat < self.config.min_alpha_t_stat:
            rejection_reasons.append("min_alpha_t_stat")
        return StatisticalValidationResult(
            passed=not rejection_reasons,
            alpha_t_stat=alpha_t_stat,
            rejection_reasons=rejection_reasons,
        )

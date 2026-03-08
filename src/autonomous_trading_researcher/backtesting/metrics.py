"""Performance metric calculations for strategy evaluation."""

from __future__ import annotations

import math

import pandas as pd

from autonomous_trading_researcher.core.models import PerformanceMetrics


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio while protecting against division by zero."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(
    equity_curve: pd.Series,
    period_returns: pd.Series,
    trade_pnls: list[float],
    annualization_factor: int,
) -> PerformanceMetrics:
    """Compute summary performance statistics."""

    clean_returns = period_returns.fillna(0.0)
    total_return = _safe_ratio(equity_curve.iloc[-1], equity_curve.iloc[0]) - 1.0
    returns_std = float(clean_returns.std(ddof=0))
    sharpe_ratio = (
        (float(clean_returns.mean()) / returns_std) * math.sqrt(annualization_factor)
        if returns_std > 0
        else 0.0
    )
    downside = clean_returns[clean_returns < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    sortino_ratio = (
        (float(clean_returns.mean()) / downside_std) * math.sqrt(annualization_factor)
        if downside_std > 0
        else 0.0
    )
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else 0.0
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
    profit_factor = _safe_ratio(gross_profit, gross_loss) if gross_loss > 0 else gross_profit
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    win_rate = _safe_ratio(wins, len(trade_pnls)) if trade_pnls else 0.0
    return PerformanceMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        profit_factor=float(profit_factor),
        win_rate=float(win_rate),
    )


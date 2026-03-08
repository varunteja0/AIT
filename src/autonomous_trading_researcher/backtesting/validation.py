"""Walk-forward validation utilities for out-of-sample strategy testing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autonomous_trading_researcher.backtesting.metrics import compute_metrics
from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.config import BacktestingConfig
from autonomous_trading_researcher.core.models import PerformanceMetrics
from autonomous_trading_researcher.strategies.base import BaseStrategy


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """Expanding-window train/test split definition."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """Summary of walk-forward out-of-sample validation."""

    fold_count: int
    metrics: PerformanceMetrics


class WalkForwardValidator:
    """Evaluate strategies across expanding walk-forward splits."""

    def __init__(
        self,
        config: BacktestingConfig,
        backtester: VectorizedBacktestEngine,
    ) -> None:
        self.config = config
        self.backtester = backtester

    def splits(self, features: pd.DataFrame) -> list[WalkForwardSplit]:
        """Build expanding walk-forward splits for a dataset."""

        total_bars = len(features)
        if total_bars < (self.config.walk_forward_train_size + self.config.walk_forward_test_size):
            return []

        test_size = max(
            self.config.walk_forward_test_size,
            total_bars // (self.config.walk_forward_splits + 1),
        )
        initial_train = max(
            self.config.walk_forward_train_size,
            total_bars - (self.config.walk_forward_splits * test_size),
        )

        splits: list[WalkForwardSplit] = []
        for fold in range(self.config.walk_forward_splits):
            train_end = initial_train + (fold * test_size)
            test_start = train_end
            test_end = min(test_start + test_size, total_bars)
            if train_end < self.config.walk_forward_train_size:
                continue
            if (test_end - test_start) < self.config.walk_forward_test_size:
                continue
            splits.append(
                WalkForwardSplit(
                    train_start=0,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
        return splits

    def run(
        self,
        features: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str,
    ) -> WalkForwardReport | None:
        """Run walk-forward validation and return aggregate out-of-sample metrics."""

        splits = self.splits(features)
        if not splits:
            return None

        test_returns: list[pd.Series] = []
        test_equity: list[pd.Series] = []
        trade_pnls: list[float] = []

        for split in splits:
            combined = features.iloc[split.train_start : split.test_end]
            test_slice = features.iloc[split.test_start : split.test_end]
            result = self.backtester.run(combined, strategy, symbol=symbol)
            test_length = len(test_slice)
            fold_returns = pd.Series(result.returns[-test_length:], index=test_slice.index)
            fold_equity = pd.Series(result.equity_curve[-test_length:], index=test_slice.index)
            test_returns.append(fold_returns)
            test_equity.append(fold_equity)
            trade_pnls.extend(value for value in fold_returns.tolist() if value != 0.0)

        combined_returns = pd.concat(test_returns)
        combined_equity = pd.concat(test_equity)
        metrics = compute_metrics(
            equity_curve=combined_equity,
            period_returns=combined_returns,
            trade_pnls=trade_pnls,
            annualization_factor=self.config.annualization_factor,
        )
        return WalkForwardReport(fold_count=len(splits), metrics=metrics)

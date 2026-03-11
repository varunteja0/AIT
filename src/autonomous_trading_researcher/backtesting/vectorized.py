"""Fast vectorized backtesting engine for broad parameter sweeps."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from autonomous_trading_researcher.backtesting.costs import CostModel
from autonomous_trading_researcher.backtesting.metrics import compute_metrics
from autonomous_trading_researcher.backtesting.rules import apply_position_rules
from autonomous_trading_researcher.config import BacktestingConfig
from autonomous_trading_researcher.core.models import BacktestResult
from autonomous_trading_researcher.infra.distributed.backends import (
    ExecutionBackend,
    LocalExecutionBackend,
)
from autonomous_trading_researcher.strategies.base import BaseStrategy


def _run_vectorized_backtest(
    config: BacktestingConfig,
    features: pd.DataFrame,
    strategy: BaseStrategy,
    symbol: str,
) -> BacktestResult:
    """Execute one vectorized backtest job."""

    frame = features.copy()
    target_exposure = strategy.target_exposure(frame) * config.position_size
    target_exposure = apply_position_rules(frame, target_exposure, strategy.parameters)
    lagged_exposure = target_exposure.shift(1).fillna(0.0)
    market_returns = frame["close"].pct_change().fillna(0.0)
    turnover = target_exposure.diff().abs().fillna(abs(target_exposure))
    cost_rate = config.fee_rate + (config.slippage_bps / 10_000)
    net_returns = lagged_exposure * market_returns - turnover * cost_rate
    equity_curve = config.starting_cash * (1 + net_returns).cumprod()
    trade_pnls = [value for value in net_returns.tolist() if value != 0]
    trade_log = [
        {
            "timestamp": timestamp,
            "target_exposure": exposure,
            "net_return": strategy_return,
        }
        for timestamp, exposure, strategy_return in zip(
            frame.index,
            target_exposure,
            net_returns,
            strict=True,
        )
        if strategy_return != 0 or exposure != 0
    ]
    metrics = compute_metrics(
        equity_curve=equity_curve,
        period_returns=net_returns,
        trade_pnls=trade_pnls,
        annualization_factor=config.annualization_factor,
    )
    return BacktestResult(
        symbol=symbol,
        strategy_name=strategy.name,
        parameters=strategy.parameters,
        metrics=metrics,
        equity_curve=equity_curve.tolist(),
        returns=net_returns.tolist(),
        trade_log=trade_log,
        validation_engine="vectorized",
    )


class VectorizedBacktestEngine:
    """Vectorized backtest suitable for high-throughput research."""

    def __init__(self, config: BacktestingConfig) -> None:
        self.config = config
        self.cost_model = CostModel(config.fee_rate, config.slippage_bps)

    def run(
        self,
        features: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str,
    ) -> BacktestResult:
        """Run a vectorized strategy simulation."""

        return _run_vectorized_backtest(
            self.config,
            features,
            strategy,
            symbol,
        )

    def run_batch(
        self,
        features: pd.DataFrame,
        strategies: Sequence[BaseStrategy],
        symbol: str,
        max_workers: int = 1,
        backend: ExecutionBackend | None = None,
    ) -> list[BacktestResult]:
        """Run many vectorized backtests, optionally in parallel."""

        strategy_population = list(strategies)
        if not strategy_population:
            return []
        executor_backend = backend or LocalExecutionBackend(max_workers=max_workers)
        payload = [(self.config, features, strategy, symbol) for strategy in strategy_population]
        return executor_backend.map(_run_vectorized_backtest, payload)

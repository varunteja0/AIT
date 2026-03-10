"""Event-driven backtesting engine with position and cash accounting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autonomous_trading_researcher.backtesting.costs import CostModel
from autonomous_trading_researcher.backtesting.metrics import compute_metrics
from autonomous_trading_researcher.backtesting.rules import apply_position_rules
from autonomous_trading_researcher.config import BacktestingConfig
from autonomous_trading_researcher.core.models import BacktestResult
from autonomous_trading_researcher.strategies.base import BaseStrategy


@dataclass(slots=True)
class PositionAccounting:
    """Mutable position accounting state for the event-driven simulator."""

    quantity: float = 0.0
    entry_price: float = 0.0


class EventDrivenBacktestEngine:
    """Backtest engine that processes bars sequentially."""

    def __init__(self, config: BacktestingConfig) -> None:
        self.config = config
        self.cost_model = CostModel(config.fee_rate, config.slippage_bps)

    def _update_accounting(
        self,
        accounting: PositionAccounting,
        quantity_delta: float,
        execution_price: float,
    ) -> float:
        """Update position accounting and return realized PnL from any closed size."""

        current_qty = accounting.quantity
        if current_qty == 0 or current_qty * quantity_delta > 0:
            new_qty = current_qty + quantity_delta
            if new_qty != 0:
                accounting.entry_price = (
                    (current_qty * accounting.entry_price) + (quantity_delta * execution_price)
                ) / new_qty
            accounting.quantity = new_qty
            return 0.0

        close_qty = min(abs(quantity_delta), abs(current_qty))
        realized_pnl = close_qty * (execution_price - accounting.entry_price) * (
            1 if current_qty > 0 else -1
        )
        accounting.quantity = current_qty + quantity_delta
        if accounting.quantity == 0:
            accounting.entry_price = 0.0
        elif current_qty * accounting.quantity < 0:
            accounting.entry_price = execution_price
        return realized_pnl

    def run(
        self,
        features: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str,
    ) -> BacktestResult:
        """Run an event-driven backtest over sequential bars."""

        target_exposures = strategy.target_exposure(features) * self.config.position_size
        target_exposures = apply_position_rules(features, target_exposures, strategy.parameters)
        execution_targets = target_exposures.shift(1).fillna(0.0)
        cash = self.config.starting_cash
        accounting = PositionAccounting()
        equity_curve: list[float] = []
        returns: list[float] = []
        trade_pnls: list[float] = []
        trade_log: list[dict[str, object]] = []
        last_equity = self.config.starting_cash

        for timestamp, row in features.iterrows():
            market_price = float(row["close"])
            execution_reference = float(row.get("open", market_price))
            pre_trade_equity = cash + (accounting.quantity * execution_reference)
            target_qty = (
                pre_trade_equity * float(execution_targets.loc[timestamp])
            ) / execution_reference
            quantity_delta = target_qty - accounting.quantity

            if abs(quantity_delta) > 1e-10:
                execution_price = self.cost_model.execution_price(
                    execution_reference,
                    quantity_delta,
                )
                notional = quantity_delta * execution_price
                fees = self.cost_model.transaction_cost(notional)
                realized_pnl = self._update_accounting(accounting, quantity_delta, execution_price)
                cash -= notional
                cash -= fees
                if realized_pnl != 0:
                    trade_pnls.append(realized_pnl - fees)
                trade_log.append(
                    {
                        "timestamp": timestamp,
                        "quantity_delta": quantity_delta,
                        "execution_price": execution_price,
                        "fees": fees,
                        "realized_pnl": realized_pnl,
                        "position_quantity": accounting.quantity,
                    }
                )

            equity = cash + (accounting.quantity * market_price)
            equity_curve.append(equity)
            period_return = (equity / last_equity) - 1.0 if last_equity else 0.0
            returns.append(period_return)
            last_equity = equity

        equity_series = pd.Series(equity_curve, index=features.index)
        returns_series = pd.Series(returns, index=features.index)
        metrics = compute_metrics(
            equity_curve=equity_series,
            period_returns=returns_series,
            trade_pnls=trade_pnls,
            annualization_factor=self.config.annualization_factor,
        )
        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            parameters=strategy.parameters,
            metrics=metrics,
            equity_curve=equity_curve,
            returns=returns,
            trade_log=trade_log,
            validation_engine="event_driven",
        )

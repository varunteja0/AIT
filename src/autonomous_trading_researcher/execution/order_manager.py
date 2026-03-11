"""Order management service that applies risk checks before execution."""

from __future__ import annotations

from autonomous_trading_researcher.core.models import (
    OrderRequest,
    OrderResult,
    PortfolioState,
    Position,
)
from autonomous_trading_researcher.execution.broker import ExecutionBroker
from autonomous_trading_researcher.risk.manager import RiskManager


class ExecutionService:
    """Coordinate order routing and local portfolio updates."""

    def __init__(self, broker: ExecutionBroker, risk_manager: RiskManager) -> None:
        self.broker = broker
        self.risk_manager = risk_manager

    async def place_order(
        self,
        request: OrderRequest,
        market_price: float,
        portfolio_state: PortfolioState,
    ) -> OrderResult:
        """Apply risk checks and route an order through the broker."""

        self.risk_manager.validate_order(request, market_price, portfolio_state)
        await self.broker.connect()
        result = await self.broker.place_order(request)
        self._apply_fill(portfolio_state, result, market_price)
        return result

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """Cancel an order through the broker."""

        await self.broker.connect()
        return await self.broker.cancel_order(order_id, symbol)

    async def fetch_positions(self) -> dict[str, float]:
        """Return broker-reported positions."""

        await self.broker.connect()
        return await self.broker.fetch_positions()

    def _apply_fill(
        self,
        portfolio_state: PortfolioState,
        result: OrderResult,
        market_price: float,
    ) -> None:
        """Reflect a filled order into the local portfolio state."""

        if result.filled <= 0:
            return
        signed_quantity = result.filled if result.side.value == "BUY" else -result.filled
        execution_price = float(result.average_price or market_price)
        transaction_cost = float(result.raw.get("transaction_cost", 0.0))
        position = portfolio_state.positions.get(result.symbol, Position(symbol=result.symbol))
        new_quantity = position.quantity + signed_quantity
        if new_quantity == 0:
            position.average_price = 0.0
        elif position.quantity == 0 or position.quantity * signed_quantity > 0:
            position.average_price = (
                (position.quantity * position.average_price) + (signed_quantity * execution_price)
            ) / new_quantity
        else:
            if position.quantity * new_quantity < 0:
                position.average_price = execution_price
        position.quantity = new_quantity
        position.market_price = market_price
        portfolio_state.positions[result.symbol] = position
        portfolio_state.cash -= signed_quantity * execution_price
        portfolio_state.cash -= transaction_cost
        portfolio_state.realized_pnl -= transaction_cost
        portfolio_state.unrealized_pnl = sum(
            (pos.market_price - pos.average_price) * pos.quantity
            for pos in portfolio_state.positions.values()
        )
        portfolio_state.equity = portfolio_state.cash + sum(
            pos.market_price * pos.quantity for pos in portfolio_state.positions.values()
        )
        portfolio_state.peak_equity = max(portfolio_state.peak_equity, portfolio_state.equity)

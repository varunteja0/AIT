"""Paper trading broker with configurable latency and market impact."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from autonomous_trading_researcher.core.enums import OrderSide, OrderStatus
from autonomous_trading_researcher.core.models import OrderRequest, OrderResult
from autonomous_trading_researcher.execution.broker import ExecutionBroker


class PaperExecutionBroker(ExecutionBroker):
    """In-memory broker for paper trading with simple fill simulation."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        fee_rate: float = 0.0005,
        latency_ms: int = 50,
    ) -> None:
        self.positions: dict[str, float] = {}
        self.slippage_bps = slippage_bps
        self.fee_rate = fee_rate
        self.latency_ms = latency_ms

    async def connect(self) -> None:
        """No-op connection hook for paper execution."""

    def _execution_price(self, request: OrderRequest) -> float | None:
        """Apply deterministic slippage to the requested price."""

        if request.price is None:
            return None
        slip = self.slippage_bps / 10_000
        multiplier = (1.0 + slip) if request.side == OrderSide.BUY else (1.0 - slip)
        return float(request.price) * multiplier

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Fill a paper order after simulated latency and costs."""

        await asyncio.sleep(self.latency_ms / 1000)
        execution_price = self._execution_price(request)
        signed_amount = request.amount if request.side == OrderSide.BUY else -request.amount
        self.positions[request.symbol] = self.positions.get(request.symbol, 0.0) + signed_amount
        transaction_cost = (
            abs(request.amount * execution_price) * self.fee_rate
            if execution_price is not None
            else 0.0
        )
        return OrderResult(
            order_id=str(uuid4()),
            symbol=request.symbol,
            side=request.side,
            amount=request.amount,
            status=OrderStatus.FILLED,
            filled=request.amount,
            average_price=execution_price,
            raw={
                "paper_trading": True,
                "latency_ms": self.latency_ms,
                "slippage_bps": self.slippage_bps,
                "transaction_cost": transaction_cost,
            },
        )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """Return a successful paper cancellation."""

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=None,
            amount=0.0,
            status=OrderStatus.CANCELLED,
            raw={"paper_trading": True},
        )

    async def fetch_positions(self) -> dict[str, float]:
        """Return current in-memory positions."""

        return dict(self.positions)

    async def close(self) -> None:
        """No-op close hook for paper execution."""

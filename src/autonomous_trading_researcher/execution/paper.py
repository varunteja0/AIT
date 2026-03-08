"""Paper trading broker implementation."""

from __future__ import annotations

from uuid import uuid4

from autonomous_trading_researcher.core.enums import OrderStatus
from autonomous_trading_researcher.core.models import OrderRequest, OrderResult
from autonomous_trading_researcher.execution.broker import ExecutionBroker


class PaperExecutionBroker(ExecutionBroker):
    """In-memory paper broker used for safe dry runs."""

    def __init__(self) -> None:
        self.positions: dict[str, float] = {}

    async def connect(self) -> None:
        """No-op connection hook for the paper broker."""

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Fill a paper order immediately."""

        signed_amount = request.amount if request.side.value == "BUY" else -request.amount
        self.positions[request.symbol] = self.positions.get(request.symbol, 0.0) + signed_amount
        return OrderResult(
            order_id=str(uuid4()),
            symbol=request.symbol,
            side=request.side,
            amount=request.amount,
            status=OrderStatus.FILLED,
            filled=request.amount,
            average_price=request.price,
            raw={"paper_trading": True},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """Return a successful paper cancellation."""

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=None,  # type: ignore[arg-type]
            amount=0.0,
            status=OrderStatus.CANCELLED,
            raw={"paper_trading": True},
        )

    async def fetch_positions(self) -> dict[str, float]:
        """Return current in-memory positions."""

        return dict(self.positions)

    async def close(self) -> None:
        """No-op close hook for the paper broker."""


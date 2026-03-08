"""CCXT-backed execution broker with retry handling."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import ccxt.async_support as ccxt_async

from autonomous_trading_researcher.core.enums import OrderSide, OrderStatus
from autonomous_trading_researcher.core.models import OrderRequest, OrderResult
from autonomous_trading_researcher.execution.broker import ExecutionBroker

LOGGER = logging.getLogger(__name__)


class CCXTExecutionBroker(ExecutionBroker):
    """Place and manage orders through a `ccxt` exchange client."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str | None,
        api_secret: str | None,
        sandbox: bool = False,
        max_retries: int = 3,
    ) -> None:
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.max_retries = max_retries
        self._client: Any | None = None

    async def connect(self) -> None:
        """Open the exchange client."""

        if self._client is not None:
            return
        exchange_class = getattr(ccxt_async, self.exchange_id)
        self._client = exchange_class(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
            }
        )
        if self.sandbox and hasattr(self._client, "set_sandbox_mode"):
            self._client.set_sandbox_mode(True)
        await self._client.load_markets()

    async def _retry(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Retry network-sensitive exchange calls."""

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await operation()
            except (ccxt_async.NetworkError, ccxt_async.RequestTimeout) as exc:
                last_error = exc
                LOGGER.warning(
                    "execution_retry",
                    extra={"operation": operation_name, "attempt": attempt},
                )
                await asyncio.sleep(attempt)
        if last_error is None:
            raise RuntimeError(f"retry_failed_without_error:{operation_name}")
        raise last_error

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order through the exchange."""

        await self.connect()
        assert self._client is not None
        side = request.side.value.lower()

        async def operation() -> dict[str, Any]:
            return await self._client.create_order(
                request.symbol,
                request.order_type,
                side,
                request.amount,
                request.price,
            )

        raw = await self._retry("create_order", operation)
        return OrderResult(
            order_id=str(raw["id"]),
            symbol=request.symbol,
            side=request.side,
            amount=float(raw["amount"]),
            status=self._map_status(raw.get("status")),
            filled=float(raw.get("filled", 0.0)),
            average_price=raw.get("average"),
            raw=raw,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """Cancel an order through the exchange."""

        await self.connect()
        assert self._client is not None

        async def operation() -> dict[str, Any]:
            return await self._client.cancel_order(order_id, symbol)

        raw = await self._retry("cancel_order", operation)
        side = OrderSide.BUY if str(raw.get("side", "buy")).lower() == "buy" else OrderSide.SELL
        return OrderResult(
            order_id=str(raw["id"]),
            symbol=symbol,
            side=side,
            amount=float(raw.get("amount", 0.0)),
            status=OrderStatus.CANCELLED,
            filled=float(raw.get("filled", 0.0)),
            average_price=raw.get("average"),
            raw=raw,
        )

    async def fetch_positions(self) -> dict[str, float]:
        """Return broker positions keyed by symbol."""

        await self.connect()
        assert self._client is not None

        async def operation() -> list[dict[str, Any]]:
            if hasattr(self._client, "fetch_positions"):
                return await self._client.fetch_positions()
            return []

        raw_positions = await self._retry("fetch_positions", operation)
        return {
            position["symbol"]: float(
                position.get("contracts") or position.get("positionAmt") or 0.0
            )
            for position in raw_positions
        }

    async def close(self) -> None:
        """Close the exchange client."""

        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    def _map_status(raw_status: Any) -> OrderStatus:
        """Normalize ccxt order statuses into the local enum."""

        status = str(raw_status or "open").lower()
        mapping = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(status, OrderStatus.FAILED)

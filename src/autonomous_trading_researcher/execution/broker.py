"""Execution broker interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from autonomous_trading_researcher.core.models import OrderRequest, OrderResult


class ExecutionBroker(ABC):
    """Abstract broker interface for order routing."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize any broker resources."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a market or limit order."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """Cancel an existing order."""

    @abstractmethod
    async def fetch_positions(self) -> dict[str, float]:
        """Return current broker positions by symbol."""

    @abstractmethod
    async def close(self) -> None:
        """Close broker resources."""


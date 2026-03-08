"""Enumerations shared across the system."""

from __future__ import annotations

from enum import StrEnum


class SignalDirection(StrEnum):
    """Directional signal emitted by strategies."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class MarketEventType(StrEnum):
    """Types of market events observed by the system."""

    TRADE = "TRADE"
    ORDER_BOOK = "ORDER_BOOK"
    BAR = "BAR"


class OrderSide(StrEnum):
    """Execution order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    """Lifecycle state for an order."""

    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

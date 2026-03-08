"""Custom exceptions for the platform."""

from __future__ import annotations


class PlatformError(Exception):
    """Base exception for all platform-specific errors."""


class DataCollectionError(PlatformError):
    """Raised when market data collection fails."""


class ResearchError(PlatformError):
    """Raised when strategy research cannot be completed."""


class RiskLimitBreachError(PlatformError):
    """Raised when risk constraints are violated."""

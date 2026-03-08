"""Domain models used across data, research, risk, and execution layers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from autonomous_trading_researcher.core.enums import OrderSide, OrderStatus, SignalDirection


@dataclass(slots=True)
class Trade:
    """Normalized trade event."""

    exchange_id: str
    symbol: str
    timestamp: datetime
    price: float
    amount: float
    side: str | None = None
    trade_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Serialize the trade into a flat record for parquet persistence."""

        return {
            "exchange_id": self.exchange_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": self.price,
            "amount": self.amount,
            "side": self.side,
            "trade_id": self.trade_id,
        }


@dataclass(slots=True)
class OrderBookSnapshot:
    """Normalized order book snapshot with aggregated depth metrics."""

    exchange_id: str
    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]

    @property
    def best_bid(self) -> float:
        """Return the best bid price."""

        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Return the best ask price."""

        return self.asks[0][0] if self.asks else 0.0

    @property
    def bid_depth(self) -> float:
        """Return the total top-of-book bid depth."""

        return sum(size for _, size in self.bids)

    @property
    def ask_depth(self) -> float:
        """Return the total top-of-book ask depth."""

        return sum(size for _, size in self.asks)

    @property
    def spread(self) -> float:
        """Return the top-of-book bid/ask spread."""

        if not self.best_bid or not self.best_ask:
            return 0.0
        return self.best_ask - self.best_bid

    def to_record(self) -> dict[str, Any]:
        """Serialize the order book snapshot into a flat record for parquet persistence."""

        return {
            "exchange_id": self.exchange_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "spread": self.spread,
            "bids_json": json.dumps(self.bids),
            "asks_json": json.dumps(self.asks),
        }


@dataclass(slots=True)
class Signal:
    """Trading signal for a symbol at a point in time."""

    symbol: str
    timestamp: datetime
    direction: SignalDirection
    strength: float = 1.0


@dataclass(slots=True)
class Position:
    """Current portfolio position for a symbol."""

    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0

    @property
    def notional(self) -> float:
        """Return current notional exposure."""

        return self.quantity * self.market_price


@dataclass(slots=True)
class PortfolioState:
    """Portfolio-level tracking model."""

    cash: float
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0


@dataclass(slots=True)
class OrderRequest:
    """Request to place an order through the execution service."""

    symbol: str
    side: OrderSide
    amount: float
    order_type: str = "market"
    price: float | None = None


@dataclass(slots=True)
class OrderResult:
    """Outcome of an order placement or cancellation."""

    order_id: str
    symbol: str
    side: OrderSide | None
    amount: float
    status: OrderStatus
    filled: float = 0.0
    average_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PerformanceMetrics:
    """Computed strategy performance metrics."""

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float


@dataclass(slots=True)
class BacktestResult:
    """Result bundle returned by a backtest engine."""

    symbol: str
    strategy_name: str
    parameters: dict[str, float | int | str]
    metrics: PerformanceMetrics
    equity_curve: list[float]
    returns: list[float]
    trade_log: list[dict[str, Any]]
    validation_engine: str


@dataclass(slots=True)
class StrategyCandidate:
    """Research candidate composed of a strategy and parameter set."""

    symbol: str
    strategy_name: str
    parameters: dict[str, float | int | str]
    score: float
    backtest_result: BacktestResult


@dataclass(slots=True)
class RiskSnapshot:
    """Current risk posture of the platform."""

    current_exposure: float
    daily_loss: float
    drawdown: float
    halted: bool
    breach_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MonitoringSnapshot:
    """Monitoring view emitted by the observability layer."""

    timestamp: datetime
    pnl: float
    equity: float
    open_positions: int
    risk_exposure: float
    system_healthy: bool
    details: dict[str, Any] = field(default_factory=dict)

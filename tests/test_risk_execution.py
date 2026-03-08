"""Tests for risk controls and execution service behavior."""

from __future__ import annotations

import asyncio

import pytest

from autonomous_trading_researcher.core.enums import OrderSide
from autonomous_trading_researcher.core.exceptions import RiskLimitBreachError
from autonomous_trading_researcher.core.models import OrderRequest, PortfolioState, Position
from autonomous_trading_researcher.execution.order_manager import ExecutionService
from autonomous_trading_researcher.execution.paper import PaperExecutionBroker
from autonomous_trading_researcher.risk.limits import RiskLimits
from autonomous_trading_researcher.risk.manager import RiskManager


def test_risk_manager_halts_on_drawdown() -> None:
    """Trading should halt when drawdown exceeds the configured threshold."""

    manager = RiskManager(
        RiskLimits(
            max_position_size=0.25,
            max_portfolio_exposure=1.0,
            max_daily_loss=0.10,
            max_drawdown=0.10,
        )
    )
    portfolio = PortfolioState(cash=10_000, equity=10_000, peak_equity=10_000)
    manager.evaluate_portfolio(portfolio)
    portfolio.equity = 8_500

    snapshot = manager.evaluate_portfolio(portfolio)

    assert snapshot.halted is True
    assert "max_drawdown" in snapshot.breach_reasons


def test_execution_service_updates_portfolio() -> None:
    """Paper execution should update local positions after a fill."""

    manager = RiskManager(
        RiskLimits(
            max_position_size=0.5,
            max_portfolio_exposure=1.0,
            max_daily_loss=0.5,
            max_drawdown=0.5,
        )
    )
    service = ExecutionService(PaperExecutionBroker(), manager)
    portfolio = PortfolioState(cash=10_000, equity=10_000, peak_equity=10_000)

    result = asyncio.run(
        service.place_order(
            OrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0, price=100.0),
            market_price=100.0,
            portfolio_state=portfolio,
        )
    )

    assert result.filled == 1.0
    assert portfolio.positions["BTC/USDT"].quantity == 1.0
    assert portfolio.cash == pytest.approx(9_900.0)
    assert portfolio.equity == pytest.approx(10_000.0)


def test_execution_service_rejects_oversized_order() -> None:
    """Orders beyond the position limit should be rejected."""

    manager = RiskManager(
        RiskLimits(
            max_position_size=0.05,
            max_portfolio_exposure=1.0,
            max_daily_loss=0.5,
            max_drawdown=0.5,
        )
    )
    portfolio = PortfolioState(cash=10_000, equity=10_000, peak_equity=10_000)

    with pytest.raises(RiskLimitBreachError):
        manager.validate_order(
            OrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, amount=10.0, price=100.0),
            market_price=100.0,
            portfolio_state=portfolio,
        )


def test_risk_manager_allows_order_that_reduces_existing_position() -> None:
    """Projected risk should be based on the resulting position, not gross order size."""

    manager = RiskManager(
        RiskLimits(
            max_position_size=0.15,
            max_portfolio_exposure=0.30,
            max_daily_loss=0.5,
            max_drawdown=0.5,
        )
    )
    portfolio = PortfolioState(
        cash=7_000,
        equity=10_000,
        peak_equity=10_000,
        positions={
            "BTC/USDT": Position(
                symbol="BTC/USDT",
                quantity=30.0,
                average_price=100.0,
                market_price=100.0,
            )
        },
    )

    manager.validate_order(
        OrderRequest(symbol="BTC/USDT", side=OrderSide.SELL, amount=20.0, price=100.0),
        market_price=100.0,
        portfolio_state=portfolio,
    )

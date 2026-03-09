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
            target_volatility=0.20,
            kelly_fraction_cap=0.50,
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
            target_volatility=0.20,
            kelly_fraction_cap=0.50,
        )
    )
    service = ExecutionService(
        PaperExecutionBroker(slippage_bps=0.0, fee_rate=0.0, latency_ms=0),
        manager,
    )
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
            target_volatility=0.20,
            kelly_fraction_cap=0.50,
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
            target_volatility=0.20,
            kelly_fraction_cap=0.50,
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


def test_paper_execution_broker_applies_latency_and_costs() -> None:
    """Paper execution should simulate a non-zero transaction cost."""

    broker = PaperExecutionBroker(slippage_bps=10.0, fee_rate=0.001, latency_ms=0)

    result = asyncio.run(
        broker.place_order(
            OrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, amount=1.0, price=100.0)
        )
    )

    assert result.average_price == pytest.approx(100.10)
    assert float(result.raw["transaction_cost"]) > 0.0


def test_risk_manager_recommended_position_fraction_is_bounded() -> None:
    """Risk sizing should honor volatility targeting and Kelly caps."""

    manager = RiskManager(
        RiskLimits(
            max_position_size=0.20,
            max_portfolio_exposure=1.0,
            max_daily_loss=0.5,
            max_drawdown=0.5,
            target_volatility=0.10,
            kelly_fraction_cap=0.25,
        )
    )

    fraction = manager.recommended_position_fraction(
        base_fraction=0.50,
        realized_volatility=0.20,
        win_rate=0.60,
        profit_factor=1.50,
    )

    assert 0.0 < fraction <= 0.20

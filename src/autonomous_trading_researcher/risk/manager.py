"""Runtime risk management and trading halts."""

from __future__ import annotations

from datetime import UTC, date, datetime

from autonomous_trading_researcher.core.enums import OrderSide
from autonomous_trading_researcher.core.exceptions import RiskLimitBreachError
from autonomous_trading_researcher.core.models import OrderRequest, PortfolioState, RiskSnapshot
from autonomous_trading_researcher.risk.limits import RiskLimits


class RiskManager:
    """Enforce position, exposure, loss, and drawdown constraints."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.start_of_day_equity: float | None = None
        self.peak_equity: float | None = None
        self.session_date: date | None = None
        self.halted = False
        self.breach_reasons: list[str] = []

    def _roll_session_if_needed(self, equity: float, as_of: datetime | None = None) -> None:
        """Reset daily loss tracking when the UTC date changes."""

        session_date = (as_of or datetime.now(tz=UTC)).date()
        if self.session_date != session_date:
            self.session_date = session_date
            self.start_of_day_equity = equity

    def evaluate_portfolio(self, portfolio_state: PortfolioState) -> RiskSnapshot:
        """Evaluate current portfolio risk and update halt status."""

        equity = max(portfolio_state.equity, 1e-9)
        self._roll_session_if_needed(equity)
        self.start_of_day_equity = self.start_of_day_equity or equity
        self.peak_equity = max(self.peak_equity or equity, equity)
        current_exposure = (
            sum(abs(position.notional) for position in portfolio_state.positions.values()) / equity
        )
        daily_loss = max(0.0, (self.start_of_day_equity - equity) / self.start_of_day_equity)
        drawdown = max(0.0, (self.peak_equity - equity) / self.peak_equity)

        breach_reasons: list[str] = []
        if current_exposure > self.limits.max_portfolio_exposure:
            breach_reasons.append("max_portfolio_exposure")
        if daily_loss > self.limits.max_daily_loss:
            breach_reasons.append("max_daily_loss")
        if drawdown > self.limits.max_drawdown:
            breach_reasons.append("max_drawdown")

        self.halted = bool(breach_reasons)
        self.breach_reasons = breach_reasons
        return RiskSnapshot(
            current_exposure=current_exposure,
            daily_loss=daily_loss,
            drawdown=drawdown,
            halted=self.halted,
            breach_reasons=breach_reasons,
        )

    def _projected_position_ratio(
        self,
        request: OrderRequest,
        market_price: float,
        portfolio_state: PortfolioState,
    ) -> float:
        """Return projected absolute exposure for the order symbol."""

        equity = max(portfolio_state.equity, 1e-9)
        signed_delta = request.amount if request.side == OrderSide.BUY else -request.amount
        current_quantity = portfolio_state.positions.get(request.symbol, None)
        existing_quantity = current_quantity.quantity if current_quantity is not None else 0.0
        projected_quantity = existing_quantity + signed_delta
        return abs(projected_quantity * market_price) / equity

    def _projected_portfolio_exposure(
        self,
        request: OrderRequest,
        market_price: float,
        portfolio_state: PortfolioState,
    ) -> float:
        """Return projected portfolio exposure after applying an order fill."""

        equity = max(portfolio_state.equity, 1e-9)
        signed_delta = request.amount if request.side == OrderSide.BUY else -request.amount
        projected_notional = 0.0
        symbols = set(portfolio_state.positions) | {request.symbol}
        for symbol in symbols:
            position = portfolio_state.positions.get(symbol)
            quantity = position.quantity if position is not None else 0.0
            if symbol == request.symbol:
                quantity += signed_delta
                projected_notional += abs(quantity * market_price)
            elif position is not None:
                projected_notional += abs(position.notional)
        return projected_notional / equity

    def validate_order(
        self,
        request: OrderRequest,
        market_price: float,
        portfolio_state: PortfolioState,
    ) -> None:
        """Validate an order against current risk limits."""

        snapshot = self.evaluate_portfolio(portfolio_state)
        if snapshot.halted:
            raise RiskLimitBreachError(
                f"trading_halted:{','.join(snapshot.breach_reasons)}"
            )
        projected_position_ratio = self._projected_position_ratio(
            request,
            market_price,
            portfolio_state,
        )
        if projected_position_ratio > self.limits.max_position_size:
            raise RiskLimitBreachError("max_position_size")
        projected_exposure = self._projected_portfolio_exposure(
            request,
            market_price,
            portfolio_state,
        )
        if projected_exposure > self.limits.max_portfolio_exposure:
            raise RiskLimitBreachError("max_portfolio_exposure")

    def volatility_target_multiplier(self, realized_volatility: float) -> float:
        """Return a volatility-targeting multiplier."""

        if realized_volatility <= 0.0:
            return 1.0
        return min(1.0, self.limits.target_volatility / realized_volatility)

    def kelly_fraction(
        self,
        win_rate: float,
        profit_factor: float,
    ) -> float:
        """Estimate a bounded Kelly fraction from win rate and payoff quality."""

        if win_rate <= 0.0 or profit_factor <= 0.0:
            return 0.0
        loss_rate = max(1.0 - win_rate, 1e-9)
        payoff_ratio = max(profit_factor * loss_rate / max(win_rate, 1e-9), 1e-9)
        kelly = win_rate - (loss_rate / payoff_ratio)
        return max(0.0, min(self.limits.kelly_fraction_cap, kelly))

    def recommended_position_fraction(
        self,
        base_fraction: float,
        realized_volatility: float,
        win_rate: float,
        profit_factor: float,
    ) -> float:
        """Return a risk-shaped target fraction for a candidate deployment."""

        volatility_multiplier = self.volatility_target_multiplier(realized_volatility)
        kelly_multiplier = self.kelly_fraction(win_rate, profit_factor)
        if kelly_multiplier == 0.0:
            return 0.0
        target_fraction = base_fraction * volatility_multiplier * kelly_multiplier
        return min(self.limits.max_position_size, max(0.0, target_fraction))

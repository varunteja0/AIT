"""Risk limit models."""

from __future__ import annotations

from dataclasses import dataclass

from autonomous_trading_researcher.config import RiskConfig


@dataclass(slots=True)
class RiskLimits:
    """Portfolio-level risk thresholds."""

    max_position_size: float
    max_portfolio_exposure: float
    max_daily_loss: float
    max_drawdown: float

    @classmethod
    def from_config(cls, config: RiskConfig) -> RiskLimits:
        """Create risk limits from application configuration."""

        return cls(
            max_position_size=config.max_position_size,
            max_portfolio_exposure=config.max_portfolio_exposure,
            max_daily_loss=config.max_daily_loss,
            max_drawdown=config.max_drawdown,
        )

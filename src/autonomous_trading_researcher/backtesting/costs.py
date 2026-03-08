"""Transaction cost and slippage models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CostModel:
    """Simple transaction cost model using proportional fees and slippage."""

    fee_rate: float
    slippage_bps: float

    def execution_price(self, market_price: float, quantity_delta: float) -> float:
        """Apply directional slippage to a market execution price."""

        slippage_multiplier = self.slippage_bps / 10_000
        if quantity_delta > 0:
            return market_price * (1 + slippage_multiplier)
        if quantity_delta < 0:
            return market_price * (1 - slippage_multiplier)
        return market_price

    def transaction_cost(self, notional: float) -> float:
        """Calculate proportional exchange fees."""

        return abs(notional) * self.fee_rate


"""Momentum strategy implementation."""

from __future__ import annotations

import pandas as pd

from autonomous_trading_researcher.core.enums import SignalDirection
from autonomous_trading_researcher.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Trend-following strategy based on momentum and moving-average alignment."""

    name = "momentum"

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Emit directional signals from momentum conditions."""

        threshold = float(self.parameters.get("threshold", 0.002))
        signals = self.flat_signal_index(features)
        long_mask = (
            (features["momentum"] > threshold)
            & (features["fast_ma"] > features["slow_ma"])
            & (features["close"] > features["fast_ma"])
        )
        short_mask = (
            (features["momentum"] < -threshold)
            & (features["fast_ma"] < features["slow_ma"])
            & (features["close"] < features["fast_ma"])
        )
        signals.loc[long_mask] = SignalDirection.LONG.value
        signals.loc[short_mask] = SignalDirection.SHORT.value
        return signals


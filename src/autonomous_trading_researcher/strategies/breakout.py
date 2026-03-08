"""Breakout strategy implementation."""

from __future__ import annotations

import pandas as pd

from autonomous_trading_researcher.core.enums import SignalDirection
from autonomous_trading_researcher.strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Trade range breaks using rolling highs and lows."""

    name = "breakout"

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Emit signals based on breakout thresholds."""

        lookback = int(self.parameters.get("lookback", 20))
        rolling_high = features["high"].rolling(window=lookback, min_periods=lookback).max()
        rolling_low = features["low"].rolling(window=lookback, min_periods=lookback).min()
        signals = self.flat_signal_index(features)
        signals.loc[features["close"] > rolling_high.shift(1)] = SignalDirection.LONG.value
        signals.loc[features["close"] < rolling_low.shift(1)] = SignalDirection.SHORT.value
        return signals.fillna(SignalDirection.FLAT.value)


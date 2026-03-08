"""Mean reversion strategy implementation."""

from __future__ import annotations

import pandas as pd

from autonomous_trading_researcher.core.enums import SignalDirection
from autonomous_trading_researcher.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Fade price excursions away from the slower moving average."""

    name = "mean_reversion"

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Emit signals when price deviates significantly from trend."""

        threshold = float(self.parameters.get("z_score_threshold", 1.0))
        rolling_std = features["close"].rolling(window=20, min_periods=20).std()
        z_score = (features["close"] - features["slow_ma"]) / rolling_std.replace(0, pd.NA)
        signals = self.flat_signal_index(features)
        signals.loc[z_score < -threshold] = SignalDirection.LONG.value
        signals.loc[z_score > threshold] = SignalDirection.SHORT.value
        return signals.fillna(SignalDirection.FLAT.value)


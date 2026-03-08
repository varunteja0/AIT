"""Base strategy interface used by the research platform."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from autonomous_trading_researcher.core.enums import SignalDirection


class BaseStrategy(ABC):
    """Abstract base class for all research strategies."""

    name = "base"

    def __init__(self, **parameters: float | int | str) -> None:
        self.parameters = parameters
        self.name = str(parameters.get("strategy_id", self.name))

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate `LONG`, `SHORT`, or `FLAT` signals."""

    def target_exposure(self, features: pd.DataFrame) -> pd.Series:
        """Convert directional signals into numeric portfolio targets."""

        leverage = float(self.parameters.get("leverage", 1.0))
        signals = self.generate_signals(features)
        mapping = {
            SignalDirection.LONG.value: leverage,
            SignalDirection.SHORT.value: -leverage,
            SignalDirection.FLAT.value: 0.0,
        }
        return signals.map(mapping).fillna(0.0).astype(float)

    @staticmethod
    def flat_signal_index(features: pd.DataFrame) -> pd.Series:
        """Generate a default flat signal series."""

        return pd.Series(SignalDirection.FLAT.value, index=features.index)

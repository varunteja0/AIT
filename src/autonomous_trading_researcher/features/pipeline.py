"""Composable feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autonomous_trading_researcher.config import FeatureEngineeringConfig
from autonomous_trading_researcher.features.registry import (
    FeatureRegistry,
    default_feature_registry,
)


class FeaturePipeline:
    """Compute research-ready features from market datasets."""

    def __init__(
        self,
        config: FeatureEngineeringConfig,
        registry: FeatureRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or default_feature_registry()

    @property
    def registered_features(self) -> list[str]:
        """Return the configured feature builder names."""

        return self.registry.registered_names()

    def build(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute the configured feature set over a market dataframe."""

        if market_data.empty:
            return market_data.copy()
        frame = market_data.copy().sort_index()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp")
        frame = self.registry.apply(frame, self.config)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame.dropna().copy()

"""Composable feature engineering pipeline."""

from __future__ import annotations

from collections.abc import Mapping

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
        base_timeframe: str | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or default_feature_registry()
        self.base_timeframe = base_timeframe

    @property
    def registered_features(self) -> list[str]:
        """Return the configured feature builder names."""

        return self.registry.registered_names()

    @staticmethod
    def _normalize_timeframe_label(timeframe: str) -> str:
        """Convert pandas-style timeframes into stable suffix labels."""

        alias_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "60min": "1h",
        }
        return alias_map.get(timeframe, timeframe)

    def _normalize_frame(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize a market dataframe into a UTC time-indexed frame."""

        frame = market_data.copy().sort_index()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp")
        return frame

    def _build_single_frame(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute the registered feature set for one timeframe."""

        if market_data.empty:
            return market_data.copy()
        frame = self._normalize_frame(market_data)
        frame = self.registry.apply(frame, self.config)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame.dropna().copy()

    def _base_timeframe_label(self, timeframes: list[str]) -> str:
        """Select the anchor timeframe for unsuffixed features."""

        if self.base_timeframe is not None and self.base_timeframe in timeframes:
            return self.base_timeframe
        if self.config.timeframes:
            for configured in self.config.timeframes:
                if configured in timeframes:
                    return configured
        return min(timeframes, key=lambda timeframe: pd.Timedelta(timeframe).total_seconds())

    def build_multi_timeframe(
        self,
        market_frames: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute features across multiple timeframes on one anchor frame."""

        if not market_frames:
            return pd.DataFrame()

        normalized_frames = {
            timeframe: self._normalize_frame(frame)
            for timeframe, frame in market_frames.items()
            if not frame.empty
        }
        if not normalized_frames:
            return pd.DataFrame()

        anchor_timeframe = self._base_timeframe_label(list(normalized_frames.keys()))
        anchor_features = self._build_single_frame(normalized_frames[anchor_timeframe])
        if anchor_features.empty:
            return anchor_features

        output = anchor_features.copy()
        raw_columns = {"open", "high", "low", "close", "volume", "trade_count"}
        anchor_suffix = self._normalize_timeframe_label(anchor_timeframe)
        for column in output.columns:
            if column not in raw_columns:
                output[f"{column}_{anchor_suffix}"] = output[column]

        for timeframe, frame in normalized_frames.items():
            if timeframe == anchor_timeframe:
                continue
            timeframe_features = self._build_single_frame(frame)
            if timeframe_features.empty:
                continue
            derived_columns = [
                column for column in timeframe_features.columns if column not in raw_columns
            ]
            if not derived_columns:
                continue
            # Shift coarser features so only completed higher-timeframe bars are visible.
            lagged = timeframe_features[derived_columns].shift(1)
            aligned = lagged.reindex(output.index, method="ffill")
            suffix = self._normalize_timeframe_label(timeframe)
            output = output.join(
                aligned.rename(columns={column: f"{column}_{suffix}" for column in derived_columns})
            )

        output = output.replace([np.inf, -np.inf], np.nan)
        return output.dropna().copy()

    def build(self, market_data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute the configured feature set over one or many market dataframes."""

        if isinstance(market_data, Mapping):
            return self.build_multi_timeframe(market_data)
        return self._build_single_frame(market_data)

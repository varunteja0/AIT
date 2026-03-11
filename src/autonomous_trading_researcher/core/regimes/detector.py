"""Market regime detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RegimeDetection:
    """Detected regimes and associated metrics."""

    labels: list[str]
    metrics: dict[str, dict[str, float]]


class RegimeDetector:
    """Detect common market regimes from feature data."""

    def __init__(
        self,
        *,
        window: int = 50,
        trend_threshold: float = 0.6,
        mean_reversion_threshold: float = -0.15,
        volatility_expansion_threshold: float = 1.5,
        low_liquidity_quantile: float = 0.2,
    ) -> None:
        self.window = max(10, window)
        self.trend_threshold = trend_threshold
        self.mean_reversion_threshold = mean_reversion_threshold
        self.volatility_expansion_threshold = volatility_expansion_threshold
        self.low_liquidity_quantile = low_liquidity_quantile

    def detect(self, features: pd.DataFrame) -> RegimeDetection:
        """Return regime labels and diagnostic metrics."""

        if features.empty:
            return RegimeDetection(labels=[], metrics={})

        returns = (
            features["returns"]
            if "returns" in features.columns
            else features["close"].pct_change().fillna(0.0)
        )
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        window = min(self.window, len(returns))

        recent = returns.iloc[-window:]
        mean_return = float(recent.mean())
        vol = float(recent.std(ddof=0) or 1e-9)
        trend_score = abs(mean_return) / vol if vol else 0.0

        lagged = recent.shift(1).dropna()
        if len(lagged) >= 5:
            corr = float(recent.iloc[1:].corr(lagged))
        else:
            corr = 0.0

        short_window = max(5, window // 3)
        long_window = window
        recent_vol = float(returns.iloc[-short_window:].std(ddof=0) or 0.0)
        long_vol = float(returns.iloc[-long_window:].std(ddof=0) or 1e-9)
        vol_ratio = recent_vol / long_vol if long_vol else 0.0

        liquidity_series = features.get("liquidity_score")
        if liquidity_series is None:
            liquidity_series = features.get("volume", pd.Series(dtype="float"))
        liquidity_series = liquidity_series.replace([np.inf, -np.inf], np.nan).dropna()
        liquidity_threshold = (
            float(liquidity_series.quantile(self.low_liquidity_quantile))
            if not liquidity_series.empty
            else 0.0
        )
        latest_liquidity = (
            float(liquidity_series.iloc[-1]) if not liquidity_series.empty else 0.0
        )

        labels: list[str] = []
        metrics: dict[str, dict[str, float]] = {}

        if trend_score >= self.trend_threshold:
            labels.append("trend")
            metrics["trend"] = {"trend_score": trend_score}

        if corr <= self.mean_reversion_threshold:
            labels.append("mean_reversion")
            metrics["mean_reversion"] = {"autocorr": corr}

        if vol_ratio >= self.volatility_expansion_threshold:
            labels.append("volatility_expansion")
            metrics["volatility_expansion"] = {"vol_ratio": vol_ratio}

        if latest_liquidity <= liquidity_threshold:
            labels.append("low_liquidity")
            metrics["low_liquidity"] = {
                "liquidity": latest_liquidity,
                "threshold": liquidity_threshold,
            }

        return RegimeDetection(labels=labels, metrics=metrics)

    def detect_metadata(self, features: pd.DataFrame) -> dict[str, Any]:
        """Convenience wrapper returning serialisable metadata."""

        detection = self.detect(features)
        return {"labels": detection.labels, "metrics": detection.metrics}

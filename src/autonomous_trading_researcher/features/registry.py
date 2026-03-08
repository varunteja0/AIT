"""Registry of composable feature builders."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from autonomous_trading_researcher.config import FeatureEngineeringConfig
from autonomous_trading_researcher.features.calculators import (
    calculate_liquidity_score,
    calculate_log_returns,
    calculate_momentum,
    calculate_moving_average,
    calculate_order_book_imbalance,
    calculate_relative_spread,
    calculate_returns,
    calculate_volatility,
)
from autonomous_trading_researcher.features.microstructure import build_microstructure_features

type FeatureBuilder = Callable[[pd.DataFrame, FeatureEngineeringConfig], pd.Series | pd.DataFrame]


@dataclass(frozen=True, slots=True)
class RegisteredFeature:
    """Named feature builder entry."""

    name: str
    builder: FeatureBuilder


class FeatureRegistry:
    """Ordered registry of feature builders applied by the pipeline."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, RegisteredFeature] = OrderedDict()

    def register(self, name: str, builder: FeatureBuilder) -> None:
        """Register or replace a feature builder."""

        self._entries[name] = RegisteredFeature(name=name, builder=builder)

    def registered_names(self) -> list[str]:
        """Return registered feature identifiers."""

        return list(self._entries.keys())

    def apply(self, frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
        """Apply each registered builder in order."""

        enriched = frame.copy()
        for entry in self._entries.values():
            output = entry.builder(enriched, config)
            if isinstance(output, pd.DataFrame):
                for column in output.columns:
                    enriched[column] = output[column]
            else:
                enriched[str(output.name or entry.name)] = output
        return enriched


def _returns(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build arithmetic returns."""

    return calculate_returns(frame["close"], config.returns_window).rename("returns")


def _log_returns(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build log returns."""

    return calculate_log_returns(frame["close"], config.returns_window).rename("log_returns")


def _volatility(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build rolling volatility."""

    returns = (
        frame["returns"]
        if "returns" in frame.columns
        else calculate_returns(frame["close"], config.returns_window)
    )
    return calculate_volatility(returns, config.volatility_window).rename("volatility")


def _fast_ma(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build the fast moving average."""

    return calculate_moving_average(frame["close"], config.fast_ma_window).rename("fast_ma")


def _slow_ma(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build the slow moving average."""

    return calculate_moving_average(frame["close"], config.slow_ma_window).rename("slow_ma")


def _momentum(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build momentum."""

    return calculate_momentum(frame["close"], config.momentum_window).rename("momentum")


def _microstructure(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """Build microstructure signals from trade and order book fields."""

    del config
    return build_microstructure_features(frame)


def _order_book_imbalance(
    frame: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> pd.Series:
    """Build the canonical order book imbalance alias."""

    del config
    if "orderbook_imbalance" in frame.columns:
        return frame["orderbook_imbalance"].rename("order_book_imbalance")
    if {"bid_depth", "ask_depth"}.issubset(frame.columns):
        return calculate_order_book_imbalance(frame["bid_depth"], frame["ask_depth"]).rename(
            "order_book_imbalance"
        )
    return pd.Series(0.0, index=frame.index, name="order_book_imbalance")


def _relative_spread(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build relative spread."""

    del config
    if {"best_bid", "best_ask"}.issubset(frame.columns):
        return calculate_relative_spread(frame["best_bid"], frame["best_ask"]).rename(
            "relative_spread"
        )
    return pd.Series(0.0, index=frame.index, name="relative_spread")


def _liquidity_score(frame: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.Series:
    """Build a liquidity score."""

    spread_source = (
        frame["spread"]
        if "spread" in frame.columns
        else frame.get("relative_spread", pd.Series(0.0, index=frame.index)).replace(0.0, 1e-9)
    )
    volume_source = frame.get("volume", pd.Series(0.0, index=frame.index))
    return calculate_liquidity_score(
        spread_source,
        volume_source,
        config.liquidity_window,
    ).rename("liquidity_score")


def default_feature_registry() -> FeatureRegistry:
    """Create the default feature registry."""

    registry = FeatureRegistry()
    registry.register("returns", _returns)
    registry.register("log_returns", _log_returns)
    registry.register("volatility", _volatility)
    registry.register("fast_ma", _fast_ma)
    registry.register("slow_ma", _slow_ma)
    registry.register("momentum", _momentum)
    registry.register("microstructure", _microstructure)
    registry.register("order_book_imbalance", _order_book_imbalance)
    registry.register("relative_spread", _relative_spread)
    registry.register("liquidity_score", _liquidity_score)
    return registry

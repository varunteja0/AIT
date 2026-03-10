"""Large-scale random strategy generation built on feature combinations."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

import numpy as np
import pandas as pd

from autonomous_trading_researcher.core.enums import SignalDirection
from autonomous_trading_researcher.strategies.base import BaseStrategy

type StrategyParameter = float | int | str
type StrategyParameters = dict[str, StrategyParameter]

MICROSTRUCTURE_COLUMNS = {
    "orderbook_imbalance",
    "microprice",
    "trade_intensity",
    "volume_delta",
    "spread",
    "order_flow_imbalance",
    "order_book_slope",
    "liquidity_pressure",
    "vwap_distance",
}


def build_strategy_id(parameters: StrategyParameters) -> str:
    """Build a deterministic identifier for a generated strategy."""

    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"generated_{digest}"


class GeneratedStrategy(BaseStrategy):
    """Strategy class created dynamically from feature templates."""

    name = "generated"

    def _apply_holding_period(self, signals: pd.Series) -> pd.Series:
        """Extend non-flat signals for a fixed number of bars."""

        holding_period = int(self.parameters.get("holding_period") or 0)
        if holding_period <= 1:
            return signals.fillna(SignalDirection.FLAT.value)

        active_signal = SignalDirection.FLAT.value
        remaining_bars = 0
        stabilized: list[str] = []
        for signal in signals.fillna(SignalDirection.FLAT.value):
            current_signal = str(signal)
            if current_signal != SignalDirection.FLAT.value:
                active_signal = current_signal
                remaining_bars = holding_period - 1
                stabilized.append(current_signal)
                continue
            if remaining_bars > 0:
                stabilized.append(active_signal)
                remaining_bars -= 1
                continue
            active_signal = SignalDirection.FLAT.value
            stabilized.append(active_signal)
        return pd.Series(stabilized, index=signals.index)

    def _finalize_signals(self, signals: pd.Series) -> pd.Series:
        """Apply signal post-processing rules."""

        return self._apply_holding_period(signals)

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate signals from the configured strategy template."""

        template = str(self.parameters.get("template", "feature_gt_long"))
        signals = self.flat_signal_index(features)

        if template in {"feature_gt_long", "feature_gt_threshold"}:
            feature_name = str(self.parameters.get("primary_feature", "momentum"))
            threshold = float(self.parameters.get("threshold") or 0.0)
            signals.loc[features[feature_name] > threshold] = SignalDirection.LONG.value
            return self._finalize_signals(signals)

        if template in {"feature_lt_short", "feature_lt_threshold"}:
            feature_name = str(self.parameters.get("primary_feature", "momentum"))
            threshold = float(self.parameters.get("threshold") or 0.0)
            signals.loc[features[feature_name] < threshold] = SignalDirection.SHORT.value
            return self._finalize_signals(signals)

        if template == "feature_combo_long":
            primary_feature = str(self.parameters.get("primary_feature", "momentum"))
            secondary_feature = str(self.parameters.get("secondary_feature", "volatility"))
            primary_threshold = float(self.parameters.get("threshold") or 0.0)
            secondary_threshold = float(self.parameters.get("secondary_threshold") or 0.0)
            long_mask = (
                (features[primary_feature] > primary_threshold)
                & (features[secondary_feature] < secondary_threshold)
            )
            signals.loc[long_mask] = SignalDirection.LONG.value
            return self._finalize_signals(signals)

        if template == "feature_cross":
            primary_feature = str(self.parameters.get("primary_feature", "fast_ma"))
            secondary_feature = str(self.parameters.get("secondary_feature", "slow_ma"))
            spread = features[primary_feature] - features[secondary_feature]
            previous_spread = spread.shift(1)
            signals.loc[(spread > 0) & (previous_spread <= 0)] = SignalDirection.LONG.value
            signals.loc[(spread < 0) & (previous_spread >= 0)] = SignalDirection.SHORT.value
            return self._finalize_signals(signals)

        if template == "slope_combo_short":
            primary_feature = str(self.parameters.get("primary_feature", "momentum"))
            secondary_feature = str(self.parameters.get("secondary_feature", "volatility"))
            lookback = int(self.parameters.get("lookback") or 1)
            primary_rising = features[primary_feature].diff(lookback) > 0
            secondary_falling = features[secondary_feature].diff(lookback) < 0
            signals.loc[primary_rising & secondary_falling] = SignalDirection.SHORT.value
            return self._finalize_signals(signals)

        if template == "mean_reversion":
            feature_name = str(self.parameters.get("primary_feature", "returns"))
            lookback = int(self.parameters.get("lookback") or 20)
            threshold = float(self.parameters.get("threshold") or 1.0)
            rolling_mean = (
                features[feature_name]
                .rolling(window=lookback, min_periods=lookback)
                .mean()
            )
            rolling_std = (
                features[feature_name].rolling(window=lookback, min_periods=lookback).std()
            )
            z_score = (features[feature_name] - rolling_mean) / rolling_std.replace(0.0, np.nan)
            signals.loc[z_score < -threshold] = SignalDirection.LONG.value
            signals.loc[z_score > threshold] = SignalDirection.SHORT.value
            return self._finalize_signals(signals.fillna(SignalDirection.FLAT.value))

        if template == "momentum_breakout":
            lookback = int(self.parameters.get("lookback") or 20)
            threshold = float(self.parameters.get("threshold") or 0.0)
            momentum_feature = str(self.parameters.get("primary_feature", "momentum"))
            rolling_high = features["high"].rolling(window=lookback, min_periods=lookback).max()
            rolling_low = features["low"].rolling(window=lookback, min_periods=lookback).min()
            signals.loc[
                (features["close"] > rolling_high.shift(1))
                & (features[momentum_feature] > threshold)
            ] = SignalDirection.LONG.value
            signals.loc[
                (features["close"] < rolling_low.shift(1))
                & (features[momentum_feature] < -threshold)
            ] = SignalDirection.SHORT.value
            return self._finalize_signals(signals.fillna(SignalDirection.FLAT.value))

        if template == "microstructure_alignment":
            imbalance_threshold = float(self.parameters.get("threshold") or 0.0)
            flow_threshold = float(self.parameters.get("secondary_threshold") or 0.0)
            spread_threshold = float(self.parameters.get("spread_threshold") or np.inf)
            microprice_change = features["microprice"].diff().fillna(0.0)
            long_mask = (
                (features["orderbook_imbalance"] > imbalance_threshold)
                & (features["order_flow_imbalance"] > flow_threshold)
                & (features["volume_delta"] > 0)
                & (microprice_change > 0)
                & (features["spread"] < spread_threshold)
            )
            short_mask = (
                (features["orderbook_imbalance"] < -imbalance_threshold)
                & (features["order_flow_imbalance"] < -flow_threshold)
                & (features["volume_delta"] < 0)
                & (microprice_change < 0)
                & (features["spread"] < spread_threshold)
            )
            signals.loc[long_mask] = SignalDirection.LONG.value
            signals.loc[short_mask] = SignalDirection.SHORT.value
            return self._finalize_signals(signals)

        if template == "microstructure_reversal":
            imbalance_threshold = float(self.parameters.get("threshold") or 0.0)
            liquidity_threshold = float(self.parameters.get("secondary_threshold") or 0.0)
            microprice_change = features["microprice"].diff().fillna(0.0)
            long_mask = (
                (features["orderbook_imbalance"] < -imbalance_threshold)
                & (features["liquidity_pressure"] > liquidity_threshold)
                & (microprice_change > 0)
            )
            short_mask = (
                (features["orderbook_imbalance"] > imbalance_threshold)
                & (features["liquidity_pressure"] < -liquidity_threshold)
                & (microprice_change < 0)
            )
            signals.loc[long_mask] = SignalDirection.LONG.value
            signals.loc[short_mask] = SignalDirection.SHORT.value
            return self._finalize_signals(signals)

        return self._finalize_signals(signals)


class MassiveStrategyGenerator:
    """Randomly create large populations of feature-driven strategies."""

    TEMPLATES = (
        "feature_gt_threshold",
        "feature_lt_threshold",
        "feature_cross",
        "feature_combo_long",
        "mean_reversion",
        "momentum_breakout",
        "slope_combo_short",
        "microstructure_alignment",
        "microstructure_reversal",
    )

    def __init__(self, seed: int = 17) -> None:
        self.rng = random.Random(seed)  # noqa: S311

    @staticmethod
    def _candidate_features(frame: pd.DataFrame) -> list[str]:
        """Return numeric feature columns suitable for rule generation."""

        excluded = {"open", "high", "low", "close"}
        return [
            column
            for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        ]

    @staticmethod
    def _quantile_threshold(series: pd.Series, quantile: float) -> float:
        """Return a threshold from a series quantile, falling back to the median."""

        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        return float(clean.quantile(quantile))

    def _build_single_feature_strategy(
        self,
        features: pd.DataFrame,
        template: str,
        symbol: str,
    ) -> GeneratedStrategy:
        """Create a one-feature rule strategy."""

        feature_name = self.rng.choice(self._candidate_features(features))
        quantile = self.rng.choice([0.25, 0.4, 0.5, 0.6, 0.75])
        parameters: StrategyParameters = {
            "template": template,
            "primary_feature": feature_name,
            "threshold": self._quantile_threshold(features[feature_name], quantile),
            "leverage": self.rng.choice([0.5, 1.0, 1.5]),
            "holding_period": self.rng.choice([1, 2, 3, 5]),
            "stop_loss": self.rng.choice([0.005, 0.01, 0.02]),
            "take_profit": self.rng.choice([0.01, 0.02, 0.03]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def _build_combo_strategy(self, features: pd.DataFrame, symbol: str) -> GeneratedStrategy:
        """Create a two-feature conjunctive strategy."""

        primary_feature, secondary_feature = self.rng.sample(
            self._candidate_features(features),
            k=2,
        )
        parameters: StrategyParameters = {
            "template": "feature_combo_long",
            "primary_feature": primary_feature,
            "secondary_feature": secondary_feature,
            "threshold": self._quantile_threshold(features[primary_feature], 0.65),
            "secondary_threshold": self._quantile_threshold(features[secondary_feature], 0.35),
            "leverage": self.rng.choice([0.5, 1.0, 1.5]),
            "holding_period": self.rng.choice([1, 2, 3]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def _build_cross_strategy(self, features: pd.DataFrame, symbol: str) -> GeneratedStrategy:
        """Create a feature cross strategy."""

        primary_feature, secondary_feature = self.rng.sample(
            self._candidate_features(features),
            k=2,
        )
        parameters: StrategyParameters = {
            "template": "feature_cross",
            "primary_feature": primary_feature,
            "secondary_feature": secondary_feature,
            "leverage": self.rng.choice([0.5, 1.0, 1.5]),
            "holding_period": self.rng.choice([1, 2, 3]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def _build_slope_strategy(self, features: pd.DataFrame, symbol: str) -> GeneratedStrategy:
        """Create a trend-slope strategy."""

        primary_feature, secondary_feature = self.rng.sample(
            self._candidate_features(features),
            k=2,
        )
        parameters: StrategyParameters = {
            "template": "slope_combo_short",
            "primary_feature": primary_feature,
            "secondary_feature": secondary_feature,
            "lookback": self.rng.choice([1, 2, 3, 5]),
            "leverage": self.rng.choice([0.5, 1.0, 1.5]),
            "holding_period": self.rng.choice([1, 2, 3]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def _build_mean_reversion_strategy(
        self,
        features: pd.DataFrame,
        symbol: str,
    ) -> GeneratedStrategy:
        """Create a generated mean reversion strategy."""

        feature_name = self.rng.choice(self._candidate_features(features))
        parameters: StrategyParameters = {
            "template": "mean_reversion",
            "primary_feature": feature_name,
            "threshold": self.rng.choice([0.5, 1.0, 1.5]),
            "lookback": self.rng.choice([5, 10, 20]),
            "leverage": self.rng.choice([0.5, 1.0]),
            "holding_period": self.rng.choice([1, 2, 3, 5]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def _build_momentum_breakout_strategy(
        self,
        features: pd.DataFrame,
        symbol: str,
    ) -> GeneratedStrategy:
        """Create a momentum breakout strategy."""

        parameters: StrategyParameters = {
            "template": "momentum_breakout",
            "primary_feature": "momentum",
            "threshold": abs(self._quantile_threshold(features["momentum"], 0.65)),
            "lookback": self.rng.choice([5, 10, 20]),
            "leverage": self.rng.choice([0.5, 1.0, 1.5]),
            "holding_period": self.rng.choice([1, 2, 3]),
            "stop_loss": self.rng.choice([0.005, 0.01]),
            "take_profit": self.rng.choice([0.01, 0.02, 0.03]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def _build_microstructure_strategy(
        self,
        features: pd.DataFrame,
        symbol: str,
    ) -> GeneratedStrategy:
        """Create a microstructure-alignment momentum strategy."""

        spread_threshold = self._quantile_threshold(features["spread"], 0.4)
        parameters: StrategyParameters = {
            "template": "microstructure_alignment",
            "threshold": abs(self._quantile_threshold(features["orderbook_imbalance"], 0.65)),
            "secondary_threshold": abs(
                self._quantile_threshold(features["order_flow_imbalance"], 0.65)
            ),
            "spread_threshold": spread_threshold if spread_threshold > 0 else 1.0,
            "leverage": self.rng.choice([0.5, 1.0, 1.5]),
            "holding_period": self.rng.choice([1, 2, 3]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def _build_microstructure_reversal_strategy(
        self,
        features: pd.DataFrame,
        symbol: str,
    ) -> GeneratedStrategy:
        """Create a microstructure reversal strategy."""

        parameters: StrategyParameters = {
            "template": "microstructure_reversal",
            "threshold": abs(self._quantile_threshold(features["orderbook_imbalance"], 0.8)),
            "secondary_threshold": abs(
                self._quantile_threshold(features["liquidity_pressure"], 0.65)
            ),
            "leverage": self.rng.choice([0.5, 1.0]),
            "holding_period": self.rng.choice([1, 2, 3]),
            "stop_loss": self.rng.choice([0.005, 0.01]),
            "take_profit": self.rng.choice([0.01, 0.02]),
            "symbol": symbol,
        }
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def generate(
        self,
        features: pd.DataFrame,
        symbol: str,
        candidate_count: int,
    ) -> list[GeneratedStrategy]:
        """Generate a large, deduplicated strategy population."""

        generated: dict[str, GeneratedStrategy] = {}
        numeric_features = self._candidate_features(features)
        has_microstructure = MICROSTRUCTURE_COLUMNS.issubset(features.columns)
        if not numeric_features:
            return []

        attempts = 0
        max_attempts = max(candidate_count * 20, 100)
        while len(generated) < candidate_count and attempts < max_attempts:
            attempts += 1
            template = self.rng.choice(self.TEMPLATES)
            if template == "feature_gt_threshold":
                strategy = self._build_single_feature_strategy(features, template, symbol)
            elif template == "feature_lt_threshold":
                strategy = self._build_single_feature_strategy(features, template, symbol)
            elif template == "feature_cross":
                if len(numeric_features) < 2:
                    continue
                strategy = self._build_cross_strategy(features, symbol)
            elif template == "feature_combo_long":
                if len(numeric_features) < 2:
                    continue
                strategy = self._build_combo_strategy(features, symbol)
            elif template == "mean_reversion":
                strategy = self._build_mean_reversion_strategy(features, symbol)
            elif template == "momentum_breakout":
                strategy = self._build_momentum_breakout_strategy(features, symbol)
            elif template == "slope_combo_short":
                if len(numeric_features) < 2:
                    continue
                strategy = self._build_slope_strategy(features, symbol)
            elif template == "microstructure_alignment":
                if not has_microstructure:
                    continue
                strategy = self._build_microstructure_strategy(features, symbol)
            else:
                if not has_microstructure:
                    continue
                strategy = self._build_microstructure_reversal_strategy(features, symbol)
            generated[strategy.name] = strategy
        return list(generated.values())

    def mutate(self, strategy: GeneratedStrategy, features: pd.DataFrame) -> GeneratedStrategy:
        """Mutate a generated strategy."""

        parameters = dict(strategy.parameters)
        template = str(parameters.get("template"))
        if template in {
            "feature_gt_long",
            "feature_lt_short",
            "feature_gt_threshold",
            "feature_lt_threshold",
        }:
            feature_name = str(parameters["primary_feature"])
            parameters["threshold"] = self._quantile_threshold(
                features[feature_name],
                self.rng.choice([0.2, 0.35, 0.5, 0.65, 0.8]),
            )
        elif template == "feature_cross":
            parameters["holding_period"] = self.rng.choice([1, 2, 3])
        elif template == "feature_combo_long":
            parameters["threshold"] = self._quantile_threshold(
                features[str(parameters["primary_feature"])],
                self.rng.choice([0.55, 0.65, 0.75]),
            )
            parameters["secondary_threshold"] = self._quantile_threshold(
                features[str(parameters["secondary_feature"])],
                self.rng.choice([0.25, 0.35, 0.45]),
            )
        elif template == "slope_combo_short":
            parameters["lookback"] = self.rng.choice([1, 2, 3, 5])
        elif template == "mean_reversion":
            parameters["threshold"] = self.rng.choice([0.5, 1.0, 1.5])
            parameters["lookback"] = self.rng.choice([5, 10, 20])
        elif template == "momentum_breakout":
            parameters["threshold"] = abs(self._quantile_threshold(features["momentum"], 0.65))
            parameters["lookback"] = self.rng.choice([5, 10, 20])
        elif template == "microstructure_alignment":
            parameters["threshold"] = abs(
                self._quantile_threshold(features["orderbook_imbalance"], 0.65)
            )
            parameters["secondary_threshold"] = abs(
                self._quantile_threshold(features["order_flow_imbalance"], 0.65)
            )
        elif template == "microstructure_reversal":
            parameters["threshold"] = abs(
                self._quantile_threshold(features["orderbook_imbalance"], 0.8)
            )
            parameters["secondary_threshold"] = abs(
                self._quantile_threshold(features["liquidity_pressure"], 0.65)
            )
        parameters["leverage"] = self.rng.choice([0.5, 1.0, 1.5])
        parameters["holding_period"] = self.rng.choice([1, 2, 3, 5])
        parameters["stop_loss"] = self.rng.choice([0.005, 0.01, 0.02])
        parameters["take_profit"] = self.rng.choice([0.01, 0.02, 0.03])
        parameters["strategy_id"] = build_strategy_id(parameters)
        return GeneratedStrategy(**parameters)

    def crossover(
        self,
        left: GeneratedStrategy,
        right: GeneratedStrategy,
    ) -> GeneratedStrategy:
        """Combine two generated strategies into a child strategy."""

        left_parameters = dict(left.parameters)
        right_parameters = dict(right.parameters)
        template = (
            left_parameters.get("template")
            if self.rng.random() < 0.5
            else right_parameters.get("template")
        ) or left_parameters.get("template") or right_parameters.get("template")
        child: StrategyParameters = {
            "template": str(template),
            "symbol": str(left_parameters.get("symbol") or right_parameters.get("symbol") or ""),
            "leverage": self.rng.choice(
                [
                    float(left_parameters.get("leverage", 1.0)),
                    float(right_parameters.get("leverage", 1.0)),
                ]
            ),
        }

        def coalesce(key: str, default: StrategyParameter) -> StrategyParameter:
            for candidate in (
                left_parameters.get(key),
                right_parameters.get(key),
                default,
            ):
                if candidate is not None:
                    return candidate
            return default

        if template in {
            "feature_gt_long",
            "feature_lt_short",
            "feature_gt_threshold",
            "feature_lt_threshold",
        }:
            child["primary_feature"] = coalesce("primary_feature", "momentum")
            child["threshold"] = coalesce("threshold", 0.0)
        elif template == "feature_cross":
            child["primary_feature"] = coalesce("primary_feature", "fast_ma")
            child["secondary_feature"] = coalesce("secondary_feature", "slow_ma")
        elif template == "feature_combo_long":
            child["primary_feature"] = coalesce("primary_feature", "momentum")
            child["secondary_feature"] = coalesce("secondary_feature", "volatility")
            child["threshold"] = coalesce("threshold", 0.0)
            child["secondary_threshold"] = coalesce("secondary_threshold", 0.0)
        elif template == "slope_combo_short":
            child["primary_feature"] = coalesce("primary_feature", "momentum")
            child["secondary_feature"] = coalesce("secondary_feature", "volatility")
            child["lookback"] = coalesce("lookback", 1)
        elif template == "mean_reversion":
            child["primary_feature"] = coalesce("primary_feature", "returns")
            child["threshold"] = coalesce("threshold", 1.0)
            child["lookback"] = coalesce("lookback", 20)
        elif template == "momentum_breakout":
            child["primary_feature"] = coalesce("primary_feature", "momentum")
            child["threshold"] = coalesce("threshold", 0.0)
            child["lookback"] = coalesce("lookback", 20)
        elif template == "microstructure_alignment":
            child["threshold"] = coalesce("threshold", 0.0)
            child["secondary_threshold"] = coalesce("secondary_threshold", 0.0)
            child["spread_threshold"] = coalesce("spread_threshold", 1.0)
        elif template == "microstructure_reversal":
            child["threshold"] = coalesce("threshold", 0.0)
            child["secondary_threshold"] = coalesce("secondary_threshold", 0.0)
        child["holding_period"] = coalesce("holding_period", 1)
        child["stop_loss"] = coalesce("stop_loss", 0.01)
        child["take_profit"] = coalesce("take_profit", 0.02)
        child["strategy_id"] = build_strategy_id(child)
        return GeneratedStrategy(**child)

    def top_features(
        self,
        strategies: Sequence[GeneratedStrategy],
        top_n: int = 5,
    ) -> list[str]:
        """Return the most frequently used features across a strategy population."""

        counts: dict[str, int] = {}
        for strategy in strategies:
            for key in ("primary_feature", "secondary_feature"):
                feature_name = strategy.parameters.get(key)
                if isinstance(feature_name, str):
                    counts[feature_name] = counts.get(feature_name, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [feature_name for feature_name, _ in ranked[:top_n]]

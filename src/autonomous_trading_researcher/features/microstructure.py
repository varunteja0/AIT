"""Market microstructure feature calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def orderbook_imbalance(bid_volume: pd.Series, ask_volume: pd.Series) -> pd.Series:
    """Calculate order book imbalance."""

    denominator = (bid_volume + ask_volume).replace(0, np.nan)
    values = (bid_volume - ask_volume) / denominator
    return values.fillna(0.0).rename("orderbook_imbalance")


def microprice(
    bid_price: pd.Series,
    ask_price: pd.Series,
    bid_volume: pd.Series,
    ask_volume: pd.Series,
) -> pd.Series:
    """Calculate the queue-weighted microprice."""

    denominator = (bid_volume + ask_volume).replace(0, np.nan)
    values = ((ask_price * bid_volume) + (bid_price * ask_volume)) / denominator
    midpoint = (bid_price + ask_price) / 2
    return values.fillna(midpoint).rename("microprice")


def trade_intensity(
    trade_count: pd.Series,
    timestamps: pd.Index | pd.Series | None = None,
) -> pd.Series:
    """Calculate trades per second from resampled trade counts."""

    index = trade_count.index if timestamps is None else pd.Index(timestamps)
    if len(index) <= 1:
        seconds = 1.0
    else:
        diffs = pd.Series(index).diff().dropna().dt.total_seconds()
        seconds = float(diffs.median()) if not diffs.empty else 1.0
        seconds = max(seconds, 1.0)
    return (trade_count.fillna(0.0) / seconds).rename("trade_intensity")


def volume_delta(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    """Calculate signed traded volume."""

    return (buy_volume.fillna(0.0) - sell_volume.fillna(0.0)).rename("volume_delta")


def spread(bid_price: pd.Series, ask_price: pd.Series) -> pd.Series:
    """Calculate bid/ask spread."""

    return (ask_price - bid_price).fillna(0.0).rename("spread")


def order_flow_imbalance(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    """Calculate normalized order flow imbalance from trade direction."""

    denominator = (buy_volume + sell_volume).replace(0, np.nan)
    values = (buy_volume - sell_volume) / denominator
    return values.fillna(0.0).rename("order_flow_imbalance")


def build_microstructure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build all supported market microstructure features."""

    features = pd.DataFrame(index=frame.index)
    if {"bid_depth", "ask_depth"}.issubset(frame.columns):
        features["orderbook_imbalance"] = orderbook_imbalance(
            frame["bid_depth"],
            frame["ask_depth"],
        )
    if {"best_bid", "best_ask", "bid_depth", "ask_depth"}.issubset(frame.columns):
        features["microprice"] = microprice(
            frame["best_bid"],
            frame["best_ask"],
            frame["bid_depth"],
            frame["ask_depth"],
        )
        features["spread"] = spread(frame["best_bid"], frame["best_ask"])
    if {"trade_count"}.issubset(frame.columns):
        features["trade_intensity"] = trade_intensity(frame["trade_count"], frame.index)
    if {"buy_volume", "sell_volume"}.issubset(frame.columns):
        features["volume_delta"] = volume_delta(frame["buy_volume"], frame["sell_volume"])
        features["order_flow_imbalance"] = order_flow_imbalance(
            frame["buy_volume"],
            frame["sell_volume"],
        )
    return features


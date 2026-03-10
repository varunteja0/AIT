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


def vwap_distance(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate normalized distance from a rolling VWAP."""

    rolling_notional = (close * volume).rolling(window=window, min_periods=1).sum()
    rolling_volume = volume.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    rolling_vwap = rolling_notional / rolling_volume
    values = (close - rolling_vwap) / rolling_vwap
    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("vwap_distance")


def order_book_slope(
    bid_price: pd.Series,
    ask_price: pd.Series,
    bid_volume: pd.Series,
    ask_volume: pd.Series,
) -> pd.Series:
    """Estimate the slope of the top-of-book liquidity curve."""

    half_spread = ((ask_price - bid_price) / 2).replace(0, np.nan)
    bid_slope = bid_volume / half_spread
    ask_slope = ask_volume / half_spread
    denominator = (bid_slope.abs() + ask_slope.abs()).replace(0, np.nan)
    values = (bid_slope - ask_slope) / denominator
    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("order_book_slope")


def liquidity_pressure(
    bid_volume: pd.Series,
    ask_volume: pd.Series,
    buy_volume: pd.Series | None = None,
    sell_volume: pd.Series | None = None,
) -> pd.Series:
    """Measure net liquidity pressure across book depth and executed flow."""

    buy_flow = buy_volume if buy_volume is not None else pd.Series(0.0, index=bid_volume.index)
    sell_flow = (
        sell_volume if sell_volume is not None else pd.Series(0.0, index=bid_volume.index)
    )
    numerator = (bid_volume + buy_flow) - (ask_volume + sell_flow)
    denominator = (bid_volume + ask_volume + buy_flow + sell_flow).replace(0, np.nan)
    values = numerator / denominator
    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("liquidity_pressure")


def build_microstructure_features(frame: pd.DataFrame, vwap_window: int = 20) -> pd.DataFrame:
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
        features["order_book_slope"] = order_book_slope(
            frame["best_bid"],
            frame["best_ask"],
            frame["bid_depth"],
            frame["ask_depth"],
        )
    if {"trade_count"}.issubset(frame.columns):
        features["trade_intensity"] = trade_intensity(frame["trade_count"], frame.index)
    if {"buy_volume", "sell_volume"}.issubset(frame.columns):
        features["volume_delta"] = volume_delta(frame["buy_volume"], frame["sell_volume"])
        features["order_flow_imbalance"] = order_flow_imbalance(
            frame["buy_volume"],
            frame["sell_volume"],
        )
    if {"close", "volume"}.issubset(frame.columns):
        features["vwap_distance"] = vwap_distance(
            frame["close"],
            frame["volume"],
            window=vwap_window,
        )
    if {"bid_depth", "ask_depth"}.issubset(frame.columns):
        features["liquidity_pressure"] = liquidity_pressure(
            frame["bid_depth"],
            frame["ask_depth"],
            frame["buy_volume"] if "buy_volume" in frame.columns else None,
            frame["sell_volume"] if "sell_volume" in frame.columns else None,
        )
    return features

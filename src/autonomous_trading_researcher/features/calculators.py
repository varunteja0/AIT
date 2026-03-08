"""Feature calculator functions used by research pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate arithmetic returns."""

    return series.pct_change(periods=periods)


def calculate_log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns."""

    return np.log(series / series.shift(periods))


def calculate_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling volatility from returns."""

    return returns.rolling(window=window, min_periods=window).std()


def calculate_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate a rolling moving average."""

    return series.rolling(window=window, min_periods=window).mean()


def calculate_momentum(series: pd.Series, window: int) -> pd.Series:
    """Calculate price momentum over a rolling horizon."""

    return series / series.shift(window) - 1.0


def calculate_order_book_imbalance(
    bid_depth: pd.Series,
    ask_depth: pd.Series,
) -> pd.Series:
    """Compute order book depth imbalance."""

    denominator = bid_depth + ask_depth
    values = np.where(denominator == 0, 0.0, (bid_depth - ask_depth) / denominator)
    return pd.Series(values, index=bid_depth.index)


def calculate_relative_spread(
    best_bid: pd.Series,
    best_ask: pd.Series,
) -> pd.Series:
    """Compute relative bid/ask spread."""

    mid_price = (best_bid + best_ask) / 2
    values = np.where(mid_price == 0, 0.0, (best_ask - best_bid) / mid_price)
    return pd.Series(values, index=best_bid.index)


def calculate_liquidity_score(
    spread: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Estimate a simple liquidity score from spread and volume."""

    adjusted_spread = spread.replace(0, np.nan)
    rolling_volume = volume.rolling(window=window, min_periods=1).mean()
    score = rolling_volume / adjusted_spread
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

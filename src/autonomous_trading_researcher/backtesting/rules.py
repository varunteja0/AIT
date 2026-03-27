"""Position-management rules shared across backtesting engines."""

from __future__ import annotations

import pandas as pd


def apply_position_rules(
    features: pd.DataFrame,
    target_exposure: pd.Series,
    parameters: dict[str, float | int | str],
) -> pd.Series:
    """Apply holding-period and stop/target rules to a target exposure series."""

    if target_exposure.empty or "close" not in features.columns:
        return target_exposure

    holding_period = int(parameters.get("holding_period") or 0)
    stop_loss = float(parameters.get("stop_loss") or 0.0)
    take_profit = float(parameters.get("take_profit") or 0.0)
    if holding_period <= 0 and stop_loss <= 0.0 and take_profit <= 0.0:
        return target_exposure

    adjusted = target_exposure.copy().astype(float)
    current_exposure = 0.0
    entry_price = 0.0
    bars_held = 0

    for index, price in features["close"].items():
        desired_exposure = float(adjusted.loc[index])
        if current_exposure == 0.0 and desired_exposure != 0.0:
            current_exposure = desired_exposure
            entry_price = float(price)
            bars_held = 0
            adjusted.loc[index] = current_exposure
            continue

        if current_exposure != 0.0:
            bars_held += 1
            direction = 1.0 if current_exposure > 0 else -1.0
            pnl_ratio = direction * ((float(price) / entry_price) - 1.0)
            rule_exit = (
                (stop_loss > 0.0 and pnl_ratio <= -stop_loss)
                or (take_profit > 0.0 and pnl_ratio >= take_profit)
                or (holding_period > 0 and bars_held >= holding_period)
            )
            if rule_exit:
                current_exposure = 0.0
                entry_price = 0.0
                bars_held = 0
                adjusted.loc[index] = 0.0
                continue

        adjusted.loc[index] = desired_exposure
        if desired_exposure == 0.0:
            current_exposure = 0.0
            entry_price = 0.0
            bars_held = 0
        elif current_exposure == 0.0 or desired_exposure * current_exposure < 0.0:
            current_exposure = desired_exposure
            entry_price = float(price)
            bars_held = 0
        else:
            current_exposure = desired_exposure

    return adjusted

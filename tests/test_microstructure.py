"""Tests for microstructure feature generation."""

from __future__ import annotations

from autonomous_trading_researcher.features.microstructure import build_microstructure_features


def test_microstructure_features_align_to_timestamp(synthetic_market_data) -> None:
    """Microstructure features should align with the input index."""

    frame = synthetic_market_data.copy()
    frame["buy_volume"] = frame["volume"] * 0.55
    frame["sell_volume"] = frame["volume"] * 0.45
    micro = build_microstructure_features(frame)

    assert list(micro.index) == list(frame.index)
    assert {
        "orderbook_imbalance",
        "microprice",
        "trade_intensity",
        "volume_delta",
        "spread",
        "order_flow_imbalance",
    }.issubset(micro.columns)
    assert (micro["spread"] > 0).all()


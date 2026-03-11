"""Tests for feature engineering."""

from __future__ import annotations

from autonomous_trading_researcher.features.pipeline import FeaturePipeline


def test_feature_pipeline_builds_expected_columns(app_config, synthetic_market_data) -> None:
    """The feature pipeline should emit the required research columns."""

    pipeline = FeaturePipeline(app_config.feature_engineering)
    features = pipeline.build(synthetic_market_data)

    required_columns = {
        "returns",
        "log_returns",
        "volatility",
        "fast_ma",
        "slow_ma",
        "momentum",
        "order_book_imbalance",
        "relative_spread",
        "liquidity_score",
        "vwap_distance",
        "order_book_slope",
        "liquidity_pressure",
    }
    assert not features.empty
    assert required_columns.issubset(set(features.columns))
    assert features["order_book_imbalance"].between(-1, 1).all()
    assert (features["liquidity_score"] >= 0).all()


def test_feature_pipeline_uses_registry_defaults(app_config, synthetic_market_data) -> None:
    """The pipeline should expose the registered feature builder set."""

    pipeline = FeaturePipeline(app_config.feature_engineering)

    features = pipeline.build(synthetic_market_data)

    assert "microstructure" in pipeline.registered_features
    assert {
        "microprice",
        "trade_intensity",
        "orderbook_imbalance",
        "order_flow_imbalance",
        "liquidity_pressure",
    }.issubset(features.columns)


def test_feature_pipeline_builds_multi_timeframe_features(
    app_config,
    synthetic_market_data,
) -> None:
    """The pipeline should suffix derived features for multiple timeframes."""

    pipeline = FeaturePipeline(
        app_config.feature_engineering,
        base_timeframe="1min",
    )
    minute_frame = synthetic_market_data.resample("1min").ffill()
    features = pipeline.build({"1min": minute_frame, "5min": synthetic_market_data})

    assert {"momentum_1m", "volatility_5m", "vwap_distance_5m"}.issubset(features.columns)
    assert "momentum" in features.columns

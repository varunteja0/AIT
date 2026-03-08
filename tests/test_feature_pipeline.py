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
    assert {"microprice", "trade_intensity", "orderbook_imbalance"}.issubset(features.columns)

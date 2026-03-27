"""Shared pytest fixtures for the trading research platform."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from autonomous_trading_researcher.config import (
    AppConfig,
    BacktestingConfig,
    DataConfig,
    ExecutionConfig,
    FeatureEngineeringConfig,
    LoggingConfig,
    MonitoringConfig,
    ResearchConfig,
    RiskConfig,
    ValidationConfig,
)


@pytest.fixture()
def synthetic_market_data() -> pd.DataFrame:
    """Create a deterministic synthetic market dataset with price and order book fields."""

    index = pd.date_range("2024-01-01", periods=180, freq="5min", tz=UTC)
    trend = np.linspace(100, 125, len(index))
    seasonal = np.sin(np.arange(len(index)) / 7) * 1.5
    close = trend + seasonal
    volume = 100 + (np.cos(np.arange(len(index)) / 5) * 10)
    frame = pd.DataFrame(
        {
            "open": close - 0.15,
            "high": close + 0.45,
            "low": close - 0.45,
            "close": close,
            "volume": volume,
            "buy_volume": volume * 0.55,
            "sell_volume": volume * 0.45,
            "trade_count": np.full(len(index), 8),
            "best_bid": close - 0.05,
            "best_ask": close + 0.05,
            "bid_depth": 500 + (np.sin(np.arange(len(index)) / 4) * 40),
            "ask_depth": 480 + (np.cos(np.arange(len(index)) / 4) * 40),
            "spread": np.full(len(index), 0.10),
        },
        index=index,
    )
    return frame


@pytest.fixture()
def app_config(tmp_path) -> AppConfig:
    """Build a compact app config for tests."""

    return AppConfig(
        logging=LoggingConfig(level="INFO", json=False),
        data=DataConfig(
            exchange_id="binance",
            symbols=["BTC/USDT"],
            data_dir=str(tmp_path / "lake"),
            ohlcv_timeframe="5min",
        ),
        feature_engineering=FeatureEngineeringConfig(
            returns_window=1,
            volatility_window=10,
            fast_ma_window=5,
            slow_ma_window=12,
            momentum_window=6,
            liquidity_window=5,
            vwap_window=10,
            timeframes=["1s", "5s", "1min", "5min"],
        ),
        backtesting=BacktestingConfig(
            starting_cash=10_000.0,
            fee_rate=0.0002,
            slippage_bps=2.0,
            position_size=0.2,
            annualization_factor=252,
        ),
        research=ResearchConfig(
            enabled_strategies=["momentum", "mean_reversion", "breakout"],
            ranking_weights={
                "sharpe_ratio": 0.4,
                "sortino_ratio": 0.2,
                "profit_factor": 0.2,
                "win_rate": 0.1,
                "total_return": 0.1,
            },
            grid_search_limit=4,
            bayesian_trials=2,
            genetic_population=4,
            genetic_generations=1,
            generated_strategy_count=8,
            generated_strategy_seed=17,
            candidates=8,
            generations=1,
            max_parallel_workers=1,
            top_n_strategies=3,
            validate_top_n=1,
            generated_strategy_dir=str(tmp_path / "generated-strategies"),
            experiment_db_path=str(tmp_path / "experiments.db"),
            optuna_storage_path=str(tmp_path / "optuna_studies.db"),
            strategy_parameter_space={
                "momentum": {"threshold": [0.001, 0.002], "leverage": [0.5, 1.0]},
                "mean_reversion": {
                    "z_score_threshold": [0.75, 1.0],
                    "leverage": [0.5, 1.0],
                },
                "breakout": {"lookback": [8, 12], "leverage": [0.5, 1.0]},
            },
        ),
        risk=RiskConfig(
            max_position_size=0.25,
            max_portfolio_exposure=1.0,
            max_daily_loss=0.10,
            max_drawdown=0.15,
            target_volatility=0.20,
            kelly_fraction_cap=0.50,
        ),
        execution=ExecutionConfig(
            enabled=False,
            paper_trading=True,
            mode="paper",
            ensemble_size=3,
            paper_slippage_bps=2.0,
            paper_fee_rate=0.0002,
            paper_latency_ms=0,
        ),
        monitoring=MonitoringConfig(
            heartbeat_seconds=5,
            status_path=str(tmp_path / "system_status.json"),
            event_log_path=str(tmp_path / "system_events.jsonl"),
            retain_events=50,
        ),
        validation=ValidationConfig(
            min_sharpe=-5.0,
            min_sortino=-5.0,
            min_profit_factor=0.0,
            max_drawdown=1.0,
            min_alpha_t_stat=-10.0,
        ),
    )


@pytest.fixture()
def alternate_app_config(app_config: AppConfig, tmp_path) -> AppConfig:
    """Provide a second config with a separate storage path when needed."""

    return replace(
        app_config,
        data=replace(app_config.data, data_dir=str(tmp_path / "alternate-lake")),
    )

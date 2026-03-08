"""Configuration models and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LoggingConfig:
    """Logging-related settings."""

    level: str = "INFO"
    json: bool = True


@dataclass(slots=True)
class DataConfig:
    """Market data collection settings."""

    exchange_id: str
    symbols: list[str]
    data_dir: str = "data/lake"
    trade_poll_interval_seconds: int = 15
    order_book_poll_interval_seconds: int = 5
    order_book_depth: int = 10
    ohlcv_timeframe: str = "5min"
    trade_fetch_limit: int = 200


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """Feature window definitions."""

    returns_window: int = 1
    volatility_window: int = 20
    fast_ma_window: int = 10
    slow_ma_window: int = 30
    momentum_window: int = 12
    liquidity_window: int = 15


@dataclass(slots=True)
class BacktestingConfig:
    """Simulation settings for backtests."""

    starting_cash: float = 100_000.0
    fee_rate: float = 0.0005
    slippage_bps: float = 5.0
    position_size: float = 0.1
    annualization_factor: int = 252
    walk_forward_splits: int = 3
    walk_forward_train_size: int = 60
    walk_forward_test_size: int = 20


@dataclass(slots=True)
class ResearchConfig:
    """Automated strategy research settings."""

    enabled_strategies: list[str] = field(default_factory=lambda: ["momentum"])
    ranking_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sharpe_ratio": 0.5,
            "sortino_ratio": 0.2,
            "profit_factor": 0.2,
            "win_rate": 0.1,
        }
    )
    grid_search_limit: int = 32
    bayesian_trials: int = 20
    genetic_population: int = 20
    genetic_generations: int = 10
    generated_strategy_count: int = 1000
    generated_strategy_seed: int = 17
    max_parallel_workers: int = 4
    top_n_strategies: int = 20
    validate_top_n: int = 5
    generated_strategy_dir: str = "src/autonomous_trading_researcher/strategies/generated"
    experiment_db_path: str = "data/experiments.db"
    strategy_parameter_space: dict[str, dict[str, list[float | int | str]]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class RiskConfig:
    """Risk constraints for research and live execution."""

    max_position_size: float = 0.25
    max_portfolio_exposure: float = 1.0
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.12


@dataclass(slots=True)
class ExecutionConfig:
    """Execution-layer settings."""

    enabled: bool = False
    paper_trading: bool = True
    exchange_id: str = "binance"
    sandbox: bool = False
    api_key_env: str = "ATR_EXCHANGE_API_KEY"
    api_secret_env: str = "ATR_EXCHANGE_API_SECRET"


@dataclass(slots=True)
class MonitoringConfig:
    """Monitoring-related settings."""

    heartbeat_seconds: int = 30
    status_path: str = "data/system_status.json"
    event_log_path: str = "data/system_events.jsonl"
    retain_events: int = 250


@dataclass(slots=True)
class UIConfig:
    """Dashboard configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    refresh_seconds: int = 10


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    logging: LoggingConfig
    data: DataConfig
    feature_engineering: FeatureEngineeringConfig
    backtesting: BacktestingConfig
    research: ResearchConfig
    risk: RiskConfig
    execution: ExecutionConfig
    monitoring: MonitoringConfig
    ui: UIConfig = field(default_factory=UIConfig)


def _build_dataclass(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a raw dictionary payload."""

    raw_payload = payload or {}
    allowed_fields = {item.name for item in fields(dataclass_type)}
    filtered_payload = {
        key: value for key, value in raw_payload.items() if key in allowed_fields
    }
    return dataclass_type(**filtered_payload)


def load_config(path: str | Path) -> AppConfig:
    """Load application configuration from YAML."""

    config_path = Path(path)
    raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        logging=_build_dataclass(LoggingConfig, raw_payload.get("logging")),
        data=_build_dataclass(DataConfig, raw_payload.get("data")),
        feature_engineering=_build_dataclass(
            FeatureEngineeringConfig,
            raw_payload.get("feature_engineering"),
        ),
        backtesting=_build_dataclass(BacktestingConfig, raw_payload.get("backtesting")),
        research=_build_dataclass(ResearchConfig, raw_payload.get("research")),
        risk=_build_dataclass(RiskConfig, raw_payload.get("risk")),
        execution=_build_dataclass(ExecutionConfig, raw_payload.get("execution")),
        monitoring=_build_dataclass(MonitoringConfig, raw_payload.get("monitoring")),
        ui=_build_dataclass(UIConfig, raw_payload.get("ui")),
    )

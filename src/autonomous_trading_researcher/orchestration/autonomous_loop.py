"""Autonomous research loop that coordinates the full platform pipeline."""

from __future__ import annotations

import logging
import os
from collections import Counter

import pandas as pd

from autonomous_trading_researcher.backtesting.engine import EventDrivenBacktestEngine
from autonomous_trading_researcher.backtesting.vectorized import VectorizedBacktestEngine
from autonomous_trading_researcher.config import AppConfig
from autonomous_trading_researcher.core.enums import OrderSide, SignalDirection
from autonomous_trading_researcher.core.models import (
    OrderRequest,
    PortfolioState,
    StrategyCandidate,
)
from autonomous_trading_researcher.data.clients.ccxt_client import CCXTMarketDataClient
from autonomous_trading_researcher.data.datasets import HistoricalDatasetBuilder
from autonomous_trading_researcher.data.ingestion import MarketDataCollector
from autonomous_trading_researcher.data.storage import ParquetDataLake
from autonomous_trading_researcher.execution.ccxt_execution import CCXTExecutionBroker
from autonomous_trading_researcher.execution.order_manager import ExecutionService
from autonomous_trading_researcher.execution.paper import PaperExecutionBroker
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.monitoring.service import MonitoringService
from autonomous_trading_researcher.research.discovery import StrategyDiscoveryService
from autonomous_trading_researcher.risk.limits import RiskLimits
from autonomous_trading_researcher.risk.manager import RiskManager
from autonomous_trading_researcher.strategies.registry import get_strategy

LOGGER = logging.getLogger(__name__)


class AutonomousResearchLoop:
    """Coordinate data collection, research, ranking, deployment, and monitoring."""

    def __init__(
        self,
        config: AppConfig,
        storage: ParquetDataLake,
        collector: MarketDataCollector | None,
        dataset_builder: HistoricalDatasetBuilder,
        feature_pipeline: FeaturePipeline,
        discovery_service: StrategyDiscoveryService,
        risk_manager: RiskManager,
        monitoring_service: MonitoringService,
        execution_service: ExecutionService | None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.collector = collector
        self.dataset_builder = dataset_builder
        self.feature_pipeline = feature_pipeline
        self.discovery_service = discovery_service
        self.risk_manager = risk_manager
        self.monitoring_service = monitoring_service
        self.execution_service = execution_service
        self.portfolio_state = PortfolioState(
            cash=config.backtesting.starting_cash,
            equity=config.backtesting.starting_cash,
            peak_equity=config.backtesting.starting_cash,
        )
        self.deployed_candidate: StrategyCandidate | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> AutonomousResearchLoop:
        """Construct a fully wired orchestrator from application config."""

        storage = ParquetDataLake(config.data.data_dir)
        collector = MarketDataCollector(
            client=CCXTMarketDataClient(
                exchange_id=config.data.exchange_id,
                api_key=os.getenv(config.execution.api_key_env),
                api_secret=os.getenv(config.execution.api_secret_env),
                sandbox=config.execution.sandbox,
            ),
            storage=storage,
            symbols=config.data.symbols,
            trade_poll_interval_seconds=config.data.trade_poll_interval_seconds,
            order_book_poll_interval_seconds=config.data.order_book_poll_interval_seconds,
            order_book_depth=config.data.order_book_depth,
            trade_fetch_limit=config.data.trade_fetch_limit,
        )
        dataset_builder = HistoricalDatasetBuilder()
        feature_pipeline = FeaturePipeline(config.feature_engineering)
        discovery_service = StrategyDiscoveryService(
            config=config.research,
            vectorized_backtester=VectorizedBacktestEngine(config.backtesting),
            event_driven_backtester=EventDrivenBacktestEngine(config.backtesting),
        )
        risk_manager = RiskManager(RiskLimits.from_config(config.risk))
        monitoring_service = MonitoringService(
            status_path=config.monitoring.status_path,
            event_log_path=config.monitoring.event_log_path,
            retain_events=config.monitoring.retain_events,
        )

        execution_service: ExecutionService | None = None
        if config.execution.paper_trading:
            execution_service = ExecutionService(PaperExecutionBroker(), risk_manager)
        elif config.execution.enabled:
            execution_service = ExecutionService(
                CCXTExecutionBroker(
                    exchange_id=config.execution.exchange_id,
                    api_key=os.getenv(config.execution.api_key_env),
                    api_secret=os.getenv(config.execution.api_secret_env),
                    sandbox=config.execution.sandbox,
                ),
                risk_manager,
            )

        return cls(
            config=config,
            storage=storage,
            collector=collector,
            dataset_builder=dataset_builder,
            feature_pipeline=feature_pipeline,
            discovery_service=discovery_service,
            risk_manager=risk_manager,
            monitoring_service=monitoring_service,
            execution_service=execution_service,
        )

    async def _collect_data(self) -> None:
        """Attempt one ingestion cycle before research."""

        if self.collector is None:
            return
        self.monitoring_service.record_event("data_collection_started")
        try:
            await self.collector.collect_once()
            self.monitoring_service.record_event("data_collection_completed")
        except Exception:
            LOGGER.warning("data_collection_skipped", exc_info=True)
            self.monitoring_service.record_event(
                "data_collection_skipped",
                {"reason": "collector_failure"},
            )

    async def collect_data_once(self) -> dict[str, int]:
        """Collect data one time and return record counts."""

        if self.collector is None:
            return {"trades": 0, "order_books": 0}
        self.monitoring_service.record_event("collect_data_command")
        return await self.collector.collect_once()

    async def close(self) -> None:
        """Close any async resources owned by the orchestrator."""

        if self.collector is not None:
            await self.collector.client.close()
        if self.execution_service is not None:
            await self.execution_service.broker.close()

    def _load_features(self, symbol: str):
        """Load, aggregate, and feature-engineer market data for one symbol."""

        trades = self.storage.read_dataset("trades", self.config.data.exchange_id, symbol)
        if trades.empty:
            return trades
        bars = self.dataset_builder.build_ohlcv_from_trades(
            trades,
            timeframe=self.config.data.ohlcv_timeframe,
        )
        order_books = self.storage.read_dataset(
            "order_books",
            self.config.data.exchange_id,
            symbol,
        )
        dataset = self.dataset_builder.attach_order_book_features(
            bars,
            order_books,
            timeframe=self.config.data.ohlcv_timeframe,
        )
        return self.feature_pipeline.build(dataset)

    async def _deploy_candidate(self, candidate: StrategyCandidate, features) -> None:
        """Deploy the best candidate, safely defaulting to paper mode when configured."""

        self.deployed_candidate = candidate
        if self.execution_service is None or features.empty:
            return
        strategy = get_strategy(candidate.strategy_name, candidate.parameters)
        signal = strategy.generate_signals(features).iloc[-1]
        if signal == SignalDirection.FLAT.value:
            return
        side = OrderSide.BUY if signal == SignalDirection.LONG.value else OrderSide.SELL
        market_price = float(features["close"].iloc[-1])
        leverage = float(candidate.parameters.get("leverage", 1.0))
        target_fraction = min(
            self.config.backtesting.position_size * leverage,
            self.config.risk.max_position_size,
        )
        amount = (self.portfolio_state.equity * target_fraction) / market_price
        request = OrderRequest(
            symbol=candidate.symbol,
            side=side,
            amount=amount,
            price=market_price,
        )
        await self.execution_service.place_order(request, market_price, self.portfolio_state)
        LOGGER.info(
            "strategy_deployed",
            extra={
                "symbol": candidate.symbol,
                "strategy": candidate.strategy_name,
                "parameters": candidate.parameters,
                "signal": signal,
                "amount": amount,
            },
        )
        self.monitoring_service.record_event(
            "strategy_deployed",
            {
                "symbol": candidate.symbol,
                "strategy": candidate.strategy_name,
                "amount": amount,
            },
        )

    def _extract_top_features(
        self,
        candidates: list[StrategyCandidate],
        limit: int = 5,
    ) -> list[str]:
        """Extract the most common features from discovered candidates."""

        counts: Counter[str] = Counter()
        feature_aliases = {
            "momentum": ["momentum", "fast_ma", "slow_ma"],
            "mean_reversion": ["slow_ma", "volatility"],
            "breakout": ["high", "low"],
        }
        for candidate in candidates:
            counts.update(feature_aliases.get(candidate.strategy_name, []))
            for key in ("primary_feature", "secondary_feature"):
                feature_name = candidate.parameters.get(key)
                if isinstance(feature_name, str):
                    counts[feature_name] += 1
            if candidate.parameters.get("template") == "microstructure_alignment":
                counts.update(
                    [
                        "orderbook_imbalance",
                        "order_flow_imbalance",
                        "volume_delta",
                        "microprice",
                        "spread",
                    ]
                )
        return [feature_name for feature_name, _ in counts.most_common(limit)]

    def _feature_correlations(self, features: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Compute a small correlation matrix for dashboard visualizations."""

        columns = [
            column
            for column in [
                "returns",
                "volatility",
                "momentum",
                "orderbook_imbalance",
                "microprice",
                "trade_intensity",
                "volume_delta",
                "order_flow_imbalance",
                "liquidity_score",
            ]
            if column in features.columns
        ]
        if len(columns) < 2:
            return {}
        correlation = features[columns].corr().round(4).fillna(0.0)
        return correlation.to_dict()

    async def discover_only(self) -> dict[str, object]:
        """Run discovery without deployment."""

        return await self.run_cycle(emit_monitoring_only=True)

    def backtest_saved_strategy(
        self,
        strategy_id: str | None = None,
        symbol: str | None = None,
    ):
        """Backtest a saved strategy against the latest available feature set."""

        target_symbol = symbol or self.config.data.symbols[0]
        features = self._load_features(target_symbol)
        if features.empty:
            raise ValueError(f"no_features_available:{target_symbol}")
        if strategy_id is None:
            top_strategies = self.discovery_service.experiment_db.top_strategies(limit=1)
            if not top_strategies:
                raise ValueError("no_saved_strategies")
            strategy_id = str(top_strategies[0]["strategy_id"])
        strategy = get_strategy(strategy_id, {})
        self.monitoring_service.record_event(
            "backtest_saved_strategy",
            {"strategy_id": strategy_id, "symbol": target_symbol},
        )
        return self.discovery_service.event_driven_backtester.run(
            features,
            strategy,
            symbol=target_symbol,
        )

    async def run_cycle(self, emit_monitoring_only: bool = False) -> dict[str, object]:
        """Run one autonomous research cycle."""

        self.monitoring_service.record_event("research_cycle_started")
        await self._collect_data()
        all_candidates: list[StrategyCandidate] = []
        feature_frames: dict[str, pd.DataFrame] = {}

        for symbol in self.config.data.symbols:
            self.monitoring_service.record_event("feature_generation_started", {"symbol": symbol})
            features = self._load_features(symbol)
            if features.empty:
                LOGGER.warning("symbol_has_no_features", extra={"symbol": symbol})
                continue
            feature_frames[symbol] = features
            self.monitoring_service.record_event("strategy_discovery_started", {"symbol": symbol})
            candidates = self.discovery_service.discover_for_symbol(symbol, features)
            if candidates:
                all_candidates.extend(candidates)
                self.monitoring_service.record_event(
                    "strategy_discovery_completed",
                    {"symbol": symbol, "candidates": len(candidates)},
                )

        ranked_candidates = self.discovery_service.ranker.rank(all_candidates)
        best_candidate = ranked_candidates[0] if ranked_candidates else None
        if best_candidate and not emit_monitoring_only:
            await self._deploy_candidate(best_candidate, feature_frames[best_candidate.symbol])

        top_features = self._extract_top_features(ranked_candidates)
        feature_correlations = (
            self._feature_correlations(feature_frames[best_candidate.symbol])
            if best_candidate is not None
            else {}
        )
        risk_snapshot = self.risk_manager.evaluate_portfolio(self.portfolio_state)
        monitoring_snapshot = self.monitoring_service.build_snapshot(
            portfolio_state=self.portfolio_state,
            risk_snapshot=risk_snapshot,
            datasets_ready=len(feature_frames),
            deployed_strategy=(
                f"{self.deployed_candidate.strategy_name}:{self.deployed_candidate.symbol}"
                if self.deployed_candidate
                else None
            ),
            number_of_strategies_tested=len(ranked_candidates),
            best_strategy_score=best_candidate.score if best_candidate else 0.0,
            top_features=top_features,
            active_strategy=best_candidate.strategy_name if best_candidate else None,
            feature_correlations=feature_correlations,
        )
        self.monitoring_service.record_event(
            "research_cycle_completed",
            {
                "candidate_count": len(ranked_candidates),
                "best_strategy": best_candidate.strategy_name if best_candidate else None,
            },
        )
        return {
            "best_candidate": best_candidate,
            "candidate_count": len(ranked_candidates),
            "top_candidates": ranked_candidates[: self.config.research.top_n_strategies],
            "monitoring": monitoring_snapshot,
        }

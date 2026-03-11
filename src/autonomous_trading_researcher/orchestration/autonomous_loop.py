"""Autonomous research loop that coordinates the full platform pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

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
from autonomous_trading_researcher.core.portfolio import AllocationResult, PortfolioAllocator
from autonomous_trading_researcher.core.regimes import RegimeDetector
from autonomous_trading_researcher.data.clients.ccxt_client import CCXTMarketDataClient
from autonomous_trading_researcher.data.datasets import HistoricalDatasetBuilder
from autonomous_trading_researcher.data.ingestion import MarketDataCollector
from autonomous_trading_researcher.data.storage import ParquetDataLake
from autonomous_trading_researcher.data.versioning import DatasetVersionManager
from autonomous_trading_researcher.execution.ccxt_execution import CCXTExecutionBroker
from autonomous_trading_researcher.execution.ensemble import StrategyEnsembleEngine
from autonomous_trading_researcher.execution.order_manager import ExecutionService
from autonomous_trading_researcher.execution.paper_broker import PaperExecutionBroker
from autonomous_trading_researcher.features.pipeline import FeaturePipeline
from autonomous_trading_researcher.features.feature_sets import FeatureSetStore
from autonomous_trading_researcher.infra.distributed import build_backend
from autonomous_trading_researcher.monitoring.service import MonitoringService
from autonomous_trading_researcher.research.discovery import StrategyDiscoveryService
from autonomous_trading_researcher.research.experiment_db import Experiment
from autonomous_trading_researcher.risk.limits import RiskLimits
from autonomous_trading_researcher.risk.manager import RiskManager
from autonomous_trading_researcher.strategies.registry import get_strategy

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetContext:
    """Metadata produced when building a dataset and feature set."""

    dataset_version: str | None
    feature_set_id: str | None
    regime_labels: list[str]
    regime_metrics: dict[str, dict[str, float]]


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
        ensemble_engine: StrategyEnsembleEngine,
        dataset_versioner: DatasetVersionManager | None,
        feature_set_store: FeatureSetStore | None,
        regime_detector: RegimeDetector,
        portfolio_allocator: PortfolioAllocator,
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
        self.ensemble_engine = ensemble_engine
        self.dataset_versioner = dataset_versioner
        self.feature_set_store = feature_set_store
        self.regime_detector = regime_detector
        self.portfolio_allocator = portfolio_allocator
        self.portfolio_state = PortfolioState(
            cash=config.backtesting.starting_cash,
            equity=config.backtesting.starting_cash,
            peak_equity=config.backtesting.starting_cash,
        )
        self.deployed_candidate: StrategyCandidate | None = None
        self.deployed_candidates: list[StrategyCandidate] = []
        self.latest_allocation: AllocationResult | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> AutonomousResearchLoop:
        """Construct a fully wired orchestrator from application config."""

        storage = ParquetDataLake(config.data.data_dir)
        collector = MarketDataCollector(
            exchange_id=config.data.exchange_id,
            client=CCXTMarketDataClient(
                exchange_id=config.data.exchange_id,
                api_key=os.getenv(config.execution.api_key_env),
                api_secret=os.getenv(config.execution.api_secret_env),
                sandbox=config.execution.sandbox,
            ),
            storage=storage,
            symbols=config.data.symbols,
            trade_poll_interval_seconds=config.data.trade_poll_interval_seconds,
            order_book_poll_interval_seconds=(
                config.data.orderbook_interval_seconds
                or config.data.order_book_poll_interval_seconds
            ),
            order_book_depth=config.data.order_book_depth,
            trade_fetch_limit=config.data.trade_fetch_limit,
            max_trade_batches_per_cycle=config.data.max_trade_batches_per_cycle,
            checkpoint_path=config.data.checkpoint_path,
            resume_from_checkpoint=config.data.resume_from_checkpoint,
            backfill_start=config.data.backfill_start,
        )
        dataset_builder = HistoricalDatasetBuilder()
        feature_pipeline = FeaturePipeline(
            config.feature_engineering,
            base_timeframe=config.data.ohlcv_timeframe,
        )
        dataset_root = Path(config.data.data_dir).parent
        dataset_dir = (
            Path(config.data.dataset_dir)
            if config.data.dataset_dir is not None
            else dataset_root / "datasets"
        )
        feature_set_dir = (
            Path(config.feature_engineering.feature_set_dir)
            if config.feature_engineering.feature_set_dir is not None
            else dataset_root / "feature_sets"
        )
        dataset_versioner = (
            DatasetVersionManager(dataset_dir)
            if config.data.dataset_versioning_enabled
            else None
        )
        feature_set_store = (
            FeatureSetStore(feature_set_dir)
            if config.feature_engineering.feature_set_enabled
            else None
        )
        regime_detector = RegimeDetector(
            window=config.regimes.window,
            trend_threshold=config.regimes.trend_threshold,
            mean_reversion_threshold=config.regimes.mean_reversion_threshold,
            volatility_expansion_threshold=config.regimes.volatility_expansion_threshold,
            low_liquidity_quantile=config.regimes.low_liquidity_quantile,
        )
        portfolio_allocator = PortfolioAllocator(
            annualization_factor=config.backtesting.annualization_factor,
        )
        execution_backend = build_backend(
            config.research.execution_backend,
            max_workers=config.research.max_parallel_workers,
            ray_address=config.research.ray_address,
            ray_namespace=config.research.ray_namespace,
        )
        discovery_service = StrategyDiscoveryService(
            config=config.research,
            validation_config=config.validation,
            vectorized_backtester=VectorizedBacktestEngine(config.backtesting),
            event_driven_backtester=EventDrivenBacktestEngine(config.backtesting),
            execution_backend=execution_backend,
        )
        risk_manager = RiskManager(RiskLimits.from_config(config.risk))
        monitoring_service = MonitoringService(
            status_path=config.monitoring.status_path,
            event_log_path=config.monitoring.event_log_path,
            retain_events=config.monitoring.retain_events,
        )

        execution_service: ExecutionService | None = None
        execution_mode = config.execution.mode.lower()
        if execution_mode == "paper" or config.execution.paper_trading:
            execution_service = ExecutionService(
                PaperExecutionBroker(
                    slippage_bps=config.execution.paper_slippage_bps,
                    fee_rate=config.execution.paper_fee_rate,
                    latency_ms=config.execution.paper_latency_ms,
                ),
                risk_manager,
            )
        elif execution_mode == "live" or config.execution.enabled:
            execution_service = ExecutionService(
                CCXTExecutionBroker(
                    exchange_id=config.execution.exchange_id,
                    api_key=os.getenv(config.execution.api_key_env),
                    api_secret=os.getenv(config.execution.api_secret_env),
                    sandbox=config.execution.sandbox,
                ),
                risk_manager,
            )
        ensemble_engine = StrategyEnsembleEngine(config.execution.ensemble_size)

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
            ensemble_engine=ensemble_engine,
            dataset_versioner=dataset_versioner,
            feature_set_store=feature_set_store,
            regime_detector=regime_detector,
            portfolio_allocator=portfolio_allocator,
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

    def _load_features(self, symbol: str) -> tuple[pd.DataFrame, DatasetContext | None]:
        """Load, aggregate, and feature-engineer market data for one symbol."""

        trades = self.storage.read_dataset("trades", self.config.data.exchange_id, symbol)
        if trades.empty:
            return trades, None
        order_books = self.storage.read_dataset(
            "order_books",
            self.config.data.exchange_id,
            symbol,
        )
        timeframes = list(
            dict.fromkeys(
                [self.config.data.ohlcv_timeframe, *self.config.feature_engineering.timeframes]
            )
        )
        datasets = self.dataset_builder.build_multi_resolution_datasets(
            trades,
            order_books,
            timeframes=timeframes,
        )
        features = self.feature_pipeline.build(datasets)
        if features.empty:
            return features, None

        regime_detection = self.regime_detector.detect(features)
        dataset_version: str | None = None
        if self.dataset_versioner is not None and self.config.data.dataset_versioning_enabled:
            manifest = self.dataset_versioner.persist(
                exchange_id=self.config.data.exchange_id,
                symbol=symbol,
                datasets=datasets,
                regime_labels=regime_detection.labels,
                metadata={"timeframes": timeframes},
            )
            dataset_version = manifest.version.version_id

        feature_set_id: str | None = None
        if (
            dataset_version is not None
            and self.feature_set_store is not None
            and self.config.feature_engineering.feature_set_enabled
        ):
            feature_set = self.feature_set_store.create(features.columns, dataset_version)
            feature_set_id = feature_set.feature_set_id

        context = DatasetContext(
            dataset_version=dataset_version,
            feature_set_id=feature_set_id,
            regime_labels=regime_detection.labels,
            regime_metrics=regime_detection.metrics,
        )
        return features, context

    async def _deploy_ensemble(
        self,
        candidates: list[StrategyCandidate],
        feature_frames: dict[str, pd.DataFrame],
    ) -> AllocationResult:
        """Deploy the top ensemble of validated strategies."""

        selected = self.ensemble_engine.select(candidates)
        allocation = self.portfolio_allocator.allocate(selected)
        self.latest_allocation = allocation
        previous_ids = {
            str(candidate.parameters.get("strategy_id", candidate.strategy_name))
            for candidate in self.deployed_candidates
        }
        current_ids = {
            str(candidate.parameters.get("strategy_id", candidate.strategy_name))
            for candidate in selected
        }
        retired = sorted(previous_ids - current_ids)
        if retired:
            self.monitoring_service.record_event("strategies_retired", {"strategy_ids": retired})

        self.deployed_candidates = selected
        self.deployed_candidate = selected[0] if selected else None
        if self.execution_service is None or not selected:
            return allocation

        for symbol in sorted({candidate.symbol for candidate in selected}):
            symbol_candidates = [candidate for candidate in selected if candidate.symbol == symbol]
            features = feature_frames.get(symbol)
            if features is None or features.empty:
                continue
            weights = [
                allocation.weights.get(
                    str(candidate.parameters.get("strategy_id", candidate.strategy_name)),
                    0.0,
                )
                for candidate in symbol_candidates
            ]
            decision = self.ensemble_engine.aggregate_signal(
                symbol_candidates,
                features,
                weights=weights,
            )
            if decision.direction == SignalDirection.FLAT:
                continue
            side = (
                OrderSide.BUY if decision.direction == SignalDirection.LONG else OrderSide.SELL
            )
            market_price = float(features["close"].iloc[-1])
            realized_volatility = (
                float(features["volatility"].iloc[-1]) if "volatility" in features else 0.0
            )
            avg_win_rate = sum(
                candidate.backtest_result.metrics.win_rate for candidate in symbol_candidates
            ) / len(symbol_candidates)
            avg_profit_factor = sum(
                candidate.backtest_result.metrics.profit_factor for candidate in symbol_candidates
            ) / len(symbol_candidates)
            symbol_weight = allocation.symbol_weights.get(symbol, 0.0)
            base_fraction = (
                self.config.backtesting.position_size
                * decision.confidence
                * max(symbol_weight, 0.0)
            )
            target_fraction = self.risk_manager.recommended_position_fraction(
                base_fraction=base_fraction,
                realized_volatility=realized_volatility,
                win_rate=avg_win_rate,
                profit_factor=avg_profit_factor,
            )
            if target_fraction <= 0.0:
                continue
            amount = (self.portfolio_state.equity * target_fraction) / market_price
            request = OrderRequest(
                symbol=symbol,
                side=side,
                amount=amount,
                price=market_price,
            )
            await self.execution_service.place_order(request, market_price, self.portfolio_state)
            LOGGER.info(
                "ensemble_deployed",
                extra={
                    "symbol": symbol,
                    "members": decision.members,
                    "direction": decision.direction.value,
                    "confidence": decision.confidence,
                    "amount": amount,
                },
            )
            self.monitoring_service.record_event(
                "ensemble_deployed",
                {
                    "symbol": symbol,
                    "members": decision.members,
                    "direction": decision.direction.value,
                    "confidence": decision.confidence,
                    "amount": amount,
                },
            )
        return allocation

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
                        "order_book_slope",
                        "liquidity_pressure",
                        "vwap_distance",
                    ]
                )
            if candidate.parameters.get("template") == "microstructure_reversal":
                counts.update(
                    [
                        "orderbook_imbalance",
                        "liquidity_pressure",
                        "microprice",
                        "order_book_slope",
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
                "vwap_distance",
                "order_book_slope",
                "liquidity_pressure",
                "liquidity_score",
            ]
            if column in features.columns
        ]
        if len(columns) < 2:
            return {}
        correlation = features[columns].corr().round(4).fillna(0.0)
        return correlation.to_dict()

    @staticmethod
    def _sanitize_symbol(symbol: str) -> str:
        return symbol.replace("/", "-").replace(":", "-")

    def _start_experiment(
        self,
        symbol: str,
        context: DatasetContext | None,
    ) -> str:
        """Create a new experiment run record."""

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"exp-{timestamp}-{self._sanitize_symbol(symbol)}"
        experiment = Experiment(
            experiment_id=experiment_id,
            dataset_version=context.dataset_version if context else None,
            feature_set_id=context.feature_set_id if context else None,
            strategy_config={
                "enabled_strategies": list(self.config.research.enabled_strategies),
                "ranking_weights": dict(self.config.research.ranking_weights),
            },
            parameters={
                "exchange_id": self.config.data.exchange_id,
                "symbol": symbol,
                "timeframes": list(self.config.feature_engineering.timeframes),
                "ohlcv_timeframe": self.config.data.ohlcv_timeframe,
                "dataset_version": context.dataset_version if context else None,
                "feature_set_id": context.feature_set_id if context else None,
                "regime_labels": context.regime_labels if context else [],
                "regime_metrics": context.regime_metrics if context else {},
            },
            metrics=None,
            status="running",
            start_time=datetime.now(tz=UTC),
            end_time=None,
        )
        self.discovery_service.experiment_db.record_experiment_start(experiment)
        return experiment_id

    def _finalize_experiment(
        self,
        experiment_id: str,
        candidates: list[StrategyCandidate],
    ) -> None:
        """Persist experiment summary metrics."""

        if not candidates:
            metrics: dict[str, object] = {"candidate_count": 0}
        else:
            best = candidates[0]
            metrics = {
                "candidate_count": len(candidates),
                "best_score": best.score,
                "best_strategy": best.strategy_name,
                "metrics": asdict(best.backtest_result.metrics),
            }
        self.discovery_service.experiment_db.record_experiment_result(
            experiment_id,
            metrics=metrics,
            status="completed",
        )

    async def discover_only(self) -> dict[str, object]:
        """Run discovery without deployment."""

        return await self.run_cycle(emit_monitoring_only=True)

    async def run_forever(
        self,
        max_cycles: int | None = None,
        cycle_delay_seconds: int | None = None,
        emit_monitoring_only: bool = False,
    ) -> None:
        """Run continuous research and deployment cycles."""

        cycle = 0
        delay = cycle_delay_seconds or self.config.monitoring.heartbeat_seconds
        while max_cycles is None or cycle < max_cycles:
            await self.run_cycle(emit_monitoring_only=emit_monitoring_only)
            cycle += 1
            await asyncio.sleep(delay)

    def backtest_saved_strategy(
        self,
        strategy_id: str | None = None,
        symbol: str | None = None,
    ):
        """Backtest a saved strategy against the latest available feature set."""

        target_symbol = symbol or self.config.data.symbols[0]
        features, _ = self._load_features(target_symbol)
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
            features, context = self._load_features(symbol)
            if features.empty:
                LOGGER.warning("symbol_has_no_features", extra={"symbol": symbol})
                continue
            context = context or DatasetContext(
                dataset_version=None,
                feature_set_id=None,
                regime_labels=[],
                regime_metrics={},
            )
            feature_frames[symbol] = features
            experiment_id = self._start_experiment(symbol, context)
            self.monitoring_service.record_event("strategy_discovery_started", {"symbol": symbol})
            candidates = self.discovery_service.discover_for_symbol(
                symbol,
                features,
                dataset_version=context.dataset_version,
                feature_set_id=context.feature_set_id,
                experiment_id=experiment_id,
                regime_labels=context.regime_labels,
                regime_metrics=context.regime_metrics,
            )
            if candidates:
                all_candidates.extend(candidates)
                self.monitoring_service.record_event(
                    "strategy_discovery_completed",
                    {"symbol": symbol, "candidates": len(candidates)},
                )
            self._finalize_experiment(experiment_id, candidates)

        ranked_candidates = self.discovery_service.ranker.rank(all_candidates)
        deployable_candidates = [
            candidate
            for candidate in ranked_candidates
            if int(candidate.parameters.get("stat_validation_passed", 1)) == 1
        ]
        best_candidate = deployable_candidates[0] if deployable_candidates else (
            ranked_candidates[0] if ranked_candidates else None
        )
        allocation_result: AllocationResult | None = None
        if deployable_candidates and not emit_monitoring_only:
            allocation_result = await self._deploy_ensemble(
                deployable_candidates,
                feature_frames,
            )
        elif not deployable_candidates and not emit_monitoring_only:
            self.deployed_candidates = []
            self.deployed_candidate = None
            self.latest_allocation = None

        top_features = self._extract_top_features(ranked_candidates)
        feature_correlations = (
            self._feature_correlations(feature_frames[best_candidate.symbol])
            if best_candidate is not None
            else {}
        )
        equity_curve = best_candidate.backtest_result.equity_curve if best_candidate else []
        drawdown_curve = []
        if equity_curve:
            equity_series = pd.Series(equity_curve)
            drawdown_curve = (equity_series / equity_series.cummax() - 1.0).fillna(0.0).tolist()
        risk_snapshot = self.risk_manager.evaluate_portfolio(self.portfolio_state)
        monitoring_snapshot = self.monitoring_service.build_snapshot(
            portfolio_state=self.portfolio_state,
            risk_snapshot=risk_snapshot,
            datasets_ready=len(feature_frames),
            deployed_strategy=(
                ",".join(
                    f"{candidate.strategy_name}:{candidate.symbol}"
                    for candidate in self.deployed_candidates
                )
                if self.deployed_candidates
                else None
            ),
            number_of_strategies_tested=len(ranked_candidates),
            best_strategy_score=best_candidate.score if best_candidate else 0.0,
            top_features=top_features,
            active_strategy=(
                ",".join(candidate.strategy_name for candidate in self.deployed_candidates)
                if self.deployed_candidates
                else None
            ),
            feature_correlations=feature_correlations,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            portfolio_allocation=(
                allocation_result.symbol_weights if allocation_result is not None else {}
            ),
            sharpe_ratio=(
                best_candidate.backtest_result.metrics.sharpe_ratio if best_candidate else 0.0
            ),
            win_rate=best_candidate.backtest_result.metrics.win_rate if best_candidate else 0.0,
            trade_count=len(best_candidate.backtest_result.trade_log) if best_candidate else 0,
        )
        self.monitoring_service.record_event(
            "research_cycle_completed",
            {
                "candidate_count": len(ranked_candidates),
                "best_strategy": best_candidate.strategy_name if best_candidate else None,
                "ensemble_size": len(self.deployed_candidates),
            },
        )
        return {
            "best_candidate": best_candidate,
            "candidate_count": len(ranked_candidates),
            "top_candidates": ranked_candidates[: self.config.research.top_n_strategies],
            "deployed_candidates": self.deployed_candidates,
            "monitoring": monitoring_snapshot,
        }

"""Autonomous research loop that coordinates the full platform pipeline."""

from __future__ import annotations

import asyncio
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
from autonomous_trading_researcher.execution.ensemble import StrategyEnsembleEngine
from autonomous_trading_researcher.execution.order_manager import ExecutionService
from autonomous_trading_researcher.execution.paper_broker import PaperExecutionBroker
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
        ensemble_engine: StrategyEnsembleEngine,
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
        self.portfolio_state = PortfolioState(
            cash=config.backtesting.starting_cash,
            equity=config.backtesting.starting_cash,
            peak_equity=config.backtesting.starting_cash,
        )
        self.deployed_candidate: StrategyCandidate | None = None
        self.deployed_candidates: list[StrategyCandidate] = []

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
        discovery_service = StrategyDiscoveryService(
            config=config.research,
            validation_config=config.validation,
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

    def _load_features(self, symbol: str) -> pd.DataFrame:
        """Load, aggregate, and feature-engineer market data for one symbol."""

        trades = self.storage.read_dataset("trades", self.config.data.exchange_id, symbol)
        if trades.empty:
            return trades
        order_books = self.storage.read_dataset(
            "order_books",
            self.config.data.exchange_id,
            symbol,
        )
        timeframes = list(dict.fromkeys(
            [self.config.data.ohlcv_timeframe, *self.config.feature_engineering.timeframes]
        ))
        datasets = self.dataset_builder.build_multi_resolution_datasets(
            trades,
            order_books,
            timeframes=timeframes,
        )
        return self.feature_pipeline.build(datasets)

    async def _deploy_ensemble(
        self,
        candidates: list[StrategyCandidate],
        feature_frames: dict[str, pd.DataFrame],
    ) -> None:
        """Deploy the top ensemble of validated strategies."""

        selected = self.ensemble_engine.select(candidates)
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
            return

        for symbol in sorted({candidate.symbol for candidate in selected}):
            symbol_candidates = [candidate for candidate in selected if candidate.symbol == symbol]
            features = feature_frames.get(symbol)
            if features is None or features.empty:
                continue
            decision = self.ensemble_engine.aggregate_signal(symbol_candidates, features)
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
            base_fraction = self.config.backtesting.position_size * decision.confidence
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
        deployable_candidates = [
            candidate
            for candidate in ranked_candidates
            if int(candidate.parameters.get("stat_validation_passed", 1)) == 1
        ]
        best_candidate = deployable_candidates[0] if deployable_candidates else (
            ranked_candidates[0] if ranked_candidates else None
        )
        if deployable_candidates and not emit_monitoring_only:
            await self._deploy_ensemble(deployable_candidates, feature_frames)
        elif not deployable_candidates and not emit_monitoring_only:
            self.deployed_candidates = []
            self.deployed_candidate = None

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

"""Monitoring snapshot service."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autonomous_trading_researcher.core.models import (
    MonitoringSnapshot,
    PortfolioState,
    RiskSnapshot,
)
from autonomous_trading_researcher.monitoring.health import SystemHealthCheck

LOGGER = logging.getLogger(__name__)


class MonitoringService:
    """Build and emit monitoring snapshots."""

    def __init__(
        self,
        status_path: str | Path = "data/system_status.json",
        event_log_path: str | Path = "data/system_events.jsonl",
        retain_events: int = 250,
    ) -> None:
        self.health_check = SystemHealthCheck()
        self.status_path = Path(status_path)
        self.event_log_path = Path(event_log_path)
        self.retain_events = retain_events
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append an event to the monitoring event log."""

        event = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "event_type": event_type,
            "payload": payload or {},
        }
        existing = (
            self.recent_events(limit=self.retain_events - 1)
            if self.retain_events > 1
            else []
        )
        existing.append(event)
        self.event_log_path.write_text(
            "\n".join(json.dumps(entry, default=str) for entry in existing) + "\n",
            encoding="utf-8",
        )

    def recent_events(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return recent monitoring events."""

        if not self.event_log_path.exists():
            return []
        lines = self.event_log_path.read_text(encoding="utf-8").splitlines()
        payloads = [json.loads(line) for line in lines if line.strip()]
        return payloads[-limit:]

    def build_snapshot(
        self,
        portfolio_state: PortfolioState,
        risk_snapshot: RiskSnapshot,
        datasets_ready: int,
        deployed_strategy: str | None,
        number_of_strategies_tested: int = 0,
        best_strategy_score: float = 0.0,
        top_features: list[str] | None = None,
        active_strategy: str | None = None,
        feature_correlations: dict[str, dict[str, float]] | None = None,
        equity_curve: list[float] | None = None,
        drawdown_curve: list[float] | None = None,
        sharpe_ratio: float = 0.0,
        win_rate: float = 0.0,
        trade_count: int = 0,
    ) -> MonitoringSnapshot:
        """Create a monitoring payload for logs and external sinks."""

        snapshot = MonitoringSnapshot(
            timestamp=datetime.now(tz=UTC),
            pnl=portfolio_state.realized_pnl + portfolio_state.unrealized_pnl,
            equity=portfolio_state.equity,
            open_positions=sum(
                1
                for position in portfolio_state.positions.values()
                if position.quantity != 0
            ),
            risk_exposure=risk_snapshot.current_exposure,
            system_healthy=self.health_check.assess(datasets_ready, risk_snapshot),
            details={
                "risk_halted": risk_snapshot.halted,
                "breaches": risk_snapshot.breach_reasons,
                "deployed_strategy": deployed_strategy,
                "number_of_strategies_tested": number_of_strategies_tested,
                "best_strategy_score": best_strategy_score,
                "top_features": top_features or [],
                "active_strategy": active_strategy,
                "feature_correlations": feature_correlations or {},
                "equity_curve": equity_curve or [],
                "drawdown_curve": drawdown_curve or [],
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "trade_count": trade_count,
            },
        )
        self.status_path.write_text(json.dumps(asdict(snapshot), default=str), encoding="utf-8")
        self.record_event(
            "monitoring_snapshot",
            {
                "equity": snapshot.equity,
                "pnl": snapshot.pnl,
                "number_of_strategies_tested": number_of_strategies_tested,
                "best_strategy_score": best_strategy_score,
                "active_strategy": active_strategy,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "trade_count": trade_count,
            },
        )
        LOGGER.info(
            "monitoring_snapshot",
            extra={
                "equity": snapshot.equity,
                "pnl": snapshot.pnl,
                "open_positions": snapshot.open_positions,
                "risk_exposure": snapshot.risk_exposure,
                "system_healthy": snapshot.system_healthy,
                "deployed_strategy": deployed_strategy,
                "number_of_strategies_tested": number_of_strategies_tested,
                "best_strategy_score": best_strategy_score,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "trade_count": trade_count,
            },
        )
        return snapshot

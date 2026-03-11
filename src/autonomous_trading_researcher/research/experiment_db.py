"""SQLite-backed experiment tracking for strategy discovery.

This module is responsible for recording individual strategy experiments in a
backward-compatible schema that also supports richer experiment tracking for
AIT v2. Existing consumers (the dashboard and orchestration loop) continue to
work without modification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import json
import sqlite3

from autonomous_trading_researcher.core.models import StrategyCandidate
from autonomous_trading_researcher.research.knowledge_graph.ingestion import (
    ingest_strategy_candidates,
)
from autonomous_trading_researcher.research.knowledge_graph.store import (
    SqliteKnowledgeGraphStore,
)


@dataclass(slots=True)
class Experiment:
    """High-level experiment metadata used for reproducibility."""

    experiment_id: str
    dataset_version: str | None
    feature_set_id: str | None
    strategy_config: dict[str, Any]
    parameters: dict[str, Any]
    metrics: dict[str, Any] | None
    status: str
    start_time: datetime
    end_time: datetime | None = None


@dataclass(slots=True)
class ExperimentResult:
    """Per-strategy result bundle for an experiment."""

    experiment_id: str
    strategy_id: str
    metrics: dict[str, Any]
    equity_curve: list[float]
    trade_log: list[dict[str, Any]]


class ExperimentDatabase:
    """Persist strategy experiments for later analysis and UI consumption."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection."""

        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def _create_tables(self) -> None:
        """Create the experiment schema if missing."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    backtest_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    experiment_id TEXT,
                    dataset_version TEXT,
                    feature_set_id TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT
                )
                """
            )
            # Backward-compatible migrations for existing deployments.
            existing_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info('experiments')").fetchall()
            }
            for column_name in (
                "experiment_id",
                "dataset_version",
                "feature_set_id",
                "status",
                "start_time",
                "end_time",
            ):
                if column_name not in existing_columns:
                    connection.execute(f"ALTER TABLE experiments ADD COLUMN {column_name} TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiments_score
                ON experiments (score DESC, created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    experiment_id TEXT PRIMARY KEY,
                    dataset_version TEXT,
                    feature_set_id TEXT,
                    strategy_config_json TEXT,
                    parameters_json TEXT,
                    metrics_json TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    created_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiment_runs_created
                ON experiment_runs (created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_results (
                    strategy_id TEXT,
                    experiment_id TEXT,
                    sharpe_ratio REAL,
                    pnl REAL,
                    trades INTEGER,
                    win_rate REAL
                )
                """
            )
            connection.commit()

    def _initialize(self) -> None:
        """Backward-compatible alias for schema initialization."""

        self._create_tables()

    def record_candidate(self, candidate: StrategyCandidate) -> None:
        """Persist a single strategy experiment.

        This method is the primary integration point for the experiment engine,
        the dashboard, and the research intelligence graph. It stores a
        JSON-serialised payload that the existing FastAPI dashboard expects
        while also emitting metadata required for dataset and feature-set
        versioning.
        """

        metrics_payload = asdict(candidate.backtest_result.metrics)
        backtest_payload = asdict(candidate.backtest_result)
        strategy_id = str(candidate.parameters.get("strategy_id", candidate.strategy_name))
        experiment_id = str(
            candidate.parameters.get("experiment_id", f"exp-{datetime.now(tz=UTC).timestamp()}")
        )
        dataset_version = candidate.parameters.get("dataset_version")
        feature_set_id = candidate.parameters.get("feature_set_id")
        regime_metrics = candidate.parameters.get("regime_metrics")
        now = datetime.now(tz=UTC)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    strategy_id,
                    symbol,
                    strategy_name,
                    parameters_json,
                    metrics_json,
                    backtest_json,
                    score,
                    created_at,
                    experiment_id,
                    dataset_version,
                    feature_set_id,
                    status,
                    start_time,
                    end_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    candidate.symbol,
                    candidate.strategy_name,
                    json.dumps(candidate.parameters, default=str),
                    json.dumps(metrics_payload, default=str),
                    json.dumps(backtest_payload, default=str),
                    candidate.score,
                    now.isoformat(),
                    experiment_id,
                    dataset_version,
                    feature_set_id,
                    "completed",
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO experiment_results (
                    strategy_id,
                    experiment_id,
                    sharpe_ratio,
                    pnl,
                    trades,
                    win_rate
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    experiment_id,
                    float(metrics_payload.get("sharpe_ratio", 0.0)),
                    float(metrics_payload.get("total_return", 0.0)),
                    int(len(candidate.backtest_result.trade_log)),
                    float(metrics_payload.get("win_rate", 0.0)),
                ),
            )

        # Also record this candidate in the research knowledge graph. The graph
        # database is colocated with the experiment database by default so that
        # deployments do not need additional configuration.
        graph_path = self.path.with_name("knowledge_graph.db")
        graph_store = SqliteKnowledgeGraphStore(graph_path)
        ingest_strategy_candidates(
            graph_store,
            candidates=[candidate],
            dataset_version=str(dataset_version) if dataset_version is not None else None,
            feature_set_id=str(feature_set_id) if feature_set_id is not None else None,
            experiment_id=experiment_id,
            regime_metrics=(
                regime_metrics
                if isinstance(regime_metrics, dict)
                else None
            ),
        )

    def record_candidates(self, candidates: list[StrategyCandidate]) -> None:
        """Persist multiple strategy experiments."""

        for candidate in candidates:
            self.record_candidate(candidate)

    def record_experiment_start(self, experiment: Experiment) -> None:
        """Record the start of an experiment run."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiment_runs (
                    experiment_id,
                    dataset_version,
                    feature_set_id,
                    strategy_config_json,
                    parameters_json,
                    metrics_json,
                    status,
                    start_time,
                    end_time,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.dataset_version,
                    experiment.feature_set_id,
                    json.dumps(experiment.strategy_config, default=str),
                    json.dumps(experiment.parameters, default=str),
                    json.dumps(experiment.metrics or {}, default=str),
                    experiment.status,
                    experiment.start_time.isoformat(),
                    experiment.end_time.isoformat() if experiment.end_time else None,
                    experiment.start_time.isoformat(),
                ),
            )

    def record_experiment_result(
        self,
        experiment_id: str,
        *,
        metrics: dict[str, Any],
        status: str = "completed",
        end_time: datetime | None = None,
    ) -> None:
        """Update an experiment run with final metrics."""

        finished_at = end_time or datetime.now(tz=UTC)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET metrics_json = ?, status = ?, end_time = ?
                WHERE experiment_id = ?
                """,
                (
                    json.dumps(metrics, default=str),
                    status,
                    finished_at.isoformat(),
                    experiment_id,
                ),
            )

    def list_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent experiment runs."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM experiment_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            payloads.append(
                {
                    "experiment_id": row["experiment_id"],
                    "dataset_version": row["dataset_version"],
                    "feature_set_id": row["feature_set_id"],
                    "strategy_config": json.loads(row["strategy_config_json"] or "{}"),
                    "parameters": json.loads(row["parameters_json"] or "{}"),
                    "metrics": json.loads(row["metrics_json"] or "{}"),
                    "status": row["status"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "created_at": row["created_at"],
                }
            )
        return payloads

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a SQLite row into a JSON-friendly dictionary."""

        return {
            "id": row["id"],
            "strategy_id": row["strategy_id"],
            "symbol": row["symbol"],
            "strategy_name": row["strategy_name"],
            "parameters": json.loads(row["parameters_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "backtest_result": json.loads(row["backtest_json"]),
            "score": row["score"],
            "created_at": row["created_at"],
        }

    def list_strategies(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent strategy experiments."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def top_strategies(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return top strategies ranked by score."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM experiments
                ORDER BY score DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        """Return summary metrics for the experiment history."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS strategy_count,
                    MAX(score) AS best_score
                FROM experiments
                """
            ).fetchone()
        return {
            "strategy_count": int(row["strategy_count"] or 0),
            "best_score": float(row["best_score"] or 0.0),
        }

"""SQLite-backed experiment tracking for strategy discovery."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autonomous_trading_researcher.core.models import StrategyCandidate


class ExperimentDatabase:
    """Persist strategy experiments for later analysis and UI consumption."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection."""

        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
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
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiments_score
                ON experiments (score DESC, created_at DESC)
                """
            )

    def record_candidate(self, candidate: StrategyCandidate) -> None:
        """Persist a single strategy experiment."""

        metrics_payload = asdict(candidate.backtest_result.metrics)
        backtest_payload = asdict(candidate.backtest_result)
        strategy_id = str(candidate.parameters.get("strategy_id", candidate.strategy_name))
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
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    candidate.symbol,
                    candidate.strategy_name,
                    json.dumps(candidate.parameters, default=str),
                    json.dumps(metrics_payload, default=str),
                    json.dumps(backtest_payload, default=str),
                    candidate.score,
                    datetime.now(tz=UTC).isoformat(),
                ),
            )

    def record_candidates(self, candidates: list[StrategyCandidate]) -> None:
        """Persist multiple strategy experiments."""

        for candidate in candidates:
            self.record_candidate(candidate)

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
            connection.row_factory = sqlite3.Row
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
            connection.row_factory = sqlite3.Row
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
            connection.row_factory = sqlite3.Row
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

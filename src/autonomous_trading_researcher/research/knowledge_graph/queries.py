"""High-level query helpers for the Research Intelligence Graph.

These functions are thin analytical helpers over the underlying
``KnowledgeGraphStore``. They are intentionally conservative and rely on
aggregations that scale to thousands of experiments.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .store import SqliteKnowledgeGraphStore


def get_top_features_by_sharpe(
    store: SqliteKnowledgeGraphStore,
    *,
    regime: Optional[str] = None,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Return the top features ranked by Sharpe ratio.

    This implementation relies on metrics stored on
    ``STRATEGY_TESTED_IN_EXPERIMENT`` edges.
    """

    query = """
        SELECT
            f.id AS feature_id,
            AVG(CAST(json_extract(e2.metadata_json, '$.metrics.sharpe_ratio') AS REAL)) AS avg_sharpe,
            COUNT(*) AS experiment_count
        FROM edges e1
        JOIN edges e2 ON e1.target_id = e2.source_id
        JOIN nodes f ON f.id = e1.source_id
        WHERE e1.relation_type = 'FEATURE_USED_IN_STRATEGY'
          AND e2.relation_type = 'STRATEGY_TESTED_IN_EXPERIMENT'
          AND f.type = 'FEATURE'
        GROUP BY f.id
        HAVING experiment_count > 0
        ORDER BY avg_sharpe DESC
        LIMIT ?
    """
    # Regime-specific rankings can be layered in a later iteration once regime
    # metrics are persisted on dedicated edges.

    with store._connect() as connection:  # type: ignore[attr-defined]
        rows = connection.execute(query, (top_n,)).fetchall()
    return [
        {
            "feature_id": row["feature_id"],
            "avg_sharpe": float(row["avg_sharpe"] or 0.0),
            "experiment_count": int(row["experiment_count"] or 0),
        }
        for row in rows
    ]


def get_strategies_that_fail_in_regime(
    store: SqliteKnowledgeGraphStore,
    *,
    regime: str,
    sharpe_threshold: float,
) -> List[Dict[str, Any]]:
    """Return strategies that consistently underperform in a given regime."""

    query = """
        SELECT
            e.source_id AS strategy_node_id,
            AVG(CAST(json_extract(e.metadata_json, '$.metrics.sharpe_ratio') AS REAL)) AS avg_sharpe,
            COUNT(*) AS sample_size
        FROM edges e
        WHERE e.relation_type = 'STRATEGY_PERFORMS_POORLY_IN_REGIME'
          AND json_extract(e.metadata_json, '$.regime') = ?
        GROUP BY e.source_id
        HAVING avg_sharpe < ?
        ORDER BY avg_sharpe ASC
    """
    with store._connect() as connection:  # type: ignore[attr-defined]
        rows = connection.execute(query, (regime, sharpe_threshold)).fetchall()
    return [
        {
            "strategy_node_id": row["strategy_node_id"],
            "avg_sharpe": float(row["avg_sharpe"] or 0.0),
            "sample_size": int(row["sample_size"] or 0),
        }
        for row in rows
    ]


def get_feature_combinations_that_improve_returns(
    store: SqliteKnowledgeGraphStore,
    *,
    min_delta: float,
    min_support: int = 5,
) -> List[Dict[str, Any]]:
    """Return feature pairs that appear to improve returns when used together.

    This is a coarse approximation that looks at strategies using both
    features and compares average Sharpe against strategies using only one.
    """

    with store._connect() as connection:  # type: ignore[attr-defined]
        # Strategies per feature.
        feature_to_strategies: Dict[str, set[str]] = {}
        rows = connection.execute(
            """
            SELECT source_id AS feature_id, target_id AS strategy_node_id
            FROM edges
            WHERE relation_type = 'FEATURE_USED_IN_STRATEGY'
            """
        ).fetchall()
        for row in rows:
            feature_to_strategies.setdefault(row["feature_id"], set()).add(row["strategy_node_id"])

        feature_ids = list(feature_to_strategies.keys())
        results: List[Dict[str, Any]] = []

        def average_sharpe_for_strategies(strategy_node_ids: Iterable[str]) -> float:
            ids = tuple(strategy_node_ids)
            if not ids:
                return 0.0
            placeholders = ",".join("?" for _ in ids)
            metrics_rows = connection.execute(
                f"""
                SELECT AVG(CAST(json_extract(metadata_json, '$.metrics.sharpe_ratio') AS REAL)) AS avg_sharpe
                FROM edges
                WHERE relation_type = 'STRATEGY_TESTED_IN_EXPERIMENT'
                  AND source_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
            if not metrics_rows:
                return 0.0
            value = metrics_rows[0]["avg_sharpe"]
            return float(value or 0.0)

        for i, fa in enumerate(feature_ids):
            for fb in feature_ids[i + 1 :]:
                sa = feature_to_strategies[fa]
                sb = feature_to_strategies[fb]
                both = sa & sb
                if len(both) < min_support:
                    continue
                only_a = sa - sb
                only_b = sb - sa
                base_strategies = only_a | only_b
                combined_sharpe = average_sharpe_for_strategies(both)
                base_sharpe = average_sharpe_for_strategies(base_strategies)
                delta = combined_sharpe - base_sharpe
                if delta >= min_delta:
                    results.append(
                        {
                            "feature_a": fa,
                            "feature_b": fb,
                            "delta_sharpe": delta,
                            "support": len(both),
                        }
                    )

    # Sort by improvement descending.
    return sorted(results, key=lambda item: item["delta_sharpe"], reverse=True)

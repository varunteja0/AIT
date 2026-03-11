"""Storage layer for the Research Intelligence Graph.

The initial implementation is a lightweight SQLite-backed graph store that
persists nodes and edges in relational tables. Higher-level components can
choose to materialise an in-memory graph (for example using NetworkX) on top
of this store when needed, without changing the on-disk schema.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

from .models import BaseNode, Edge, NodeType, RelationType


class KnowledgeGraphStore(Protocol):
    """Abstract interface for reading and writing the research knowledge graph."""

    def upsert_node(self, node: BaseNode) -> None: ...

    def upsert_nodes(self, nodes: Iterable[BaseNode]) -> None: ...

    def upsert_edge(self, edge: Edge) -> None: ...

    def upsert_edges(self, edges: Iterable[Edge]) -> None: ...


class SqliteKnowledgeGraphStore:
    """SQLite-backed implementation of :class:`KnowledgeGraphStore`.

    The schema is deliberately simple:

    - ``nodes``: one row per logical node (feature, strategy, dataset, regime, experiment).
    - ``edges``: one row per logical relationship, with a uniqueness constraint on
      (source_id, target_id, relation_type).

    This is sufficient for thousands of experiments while keeping dependencies
    minimal and avoiding any impact on the existing research pipeline.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    labels_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (source_id, target_id, relation_type)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_edges_source
                ON edges (source_id, relation_type)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_edges_target
                ON edges (target_id, relation_type)
                """
            )

    def upsert_node(self, node: BaseNode) -> None:
        self.upsert_nodes([node])

    def upsert_nodes(self, nodes: Iterable[BaseNode]) -> None:
        now = datetime.now(tz=UTC).isoformat()
        with self._connect() as connection:
            for node in nodes:
                record = node.to_record()
                # Ensure timestamps are set even if caller left them at defaults.
                created_at = record.get("created_at") or now
                updated_at = record.get("updated_at") or now
                connection.execute(
                    """
                    INSERT INTO nodes (id, type, labels_json, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        type = excluded.type,
                        labels_json = excluded.labels_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        record["id"],
                        record["type"],
                        json.dumps(record["labels"], default=str),
                        json.dumps(record["metadata"], default=str),
                        created_at,
                        updated_at,
                    ),
                )

    def upsert_edge(self, edge: Edge) -> None:
        self.upsert_edges([edge])

    def upsert_edges(self, edges: Iterable[Edge]) -> None:
        now = datetime.now(tz=UTC).isoformat()
        with self._connect() as connection:
            for edge in edges:
                record = edge.to_record()
                created_at = record.get("created_at") or now
                updated_at = record.get("updated_at") or now
                connection.execute(
                    """
                    INSERT INTO edges (
                        source_id,
                        target_id,
                        relation_type,
                        weight,
                        metadata_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                        weight = excluded.weight,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        record["source_id"],
                        record["target_id"],
                        record["relation_type"],
                        record["weight"],
                        json.dumps(record["metadata"], default=str),
                        created_at,
                        updated_at,
                    ),
                )

    def list_nodes(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return recent nodes for visualization."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM nodes
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            payloads.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "labels": json.loads(row["labels_json"] or "[]"),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return payloads

    def list_edges(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Return recent edges for visualization."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM edges
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            payloads.append(
                {
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "relation_type": row["relation_type"],
                    "weight": row["weight"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return payloads

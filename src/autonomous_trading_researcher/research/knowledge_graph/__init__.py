"""Research Intelligence Graph integration for AIT v2.

This package defines:

- data models for nodes and edges,
- a lightweight SQLite-backed graph store,
- ingestion helpers for experiment results,
- high-level analytical queries.

The initial integration is read-only from the perspective of the rest of the
system: existing pipelines continue to function without modification, while
new components (agents, dashboards) can start depending on this package.
"""

from .models import (
    BaseNode,
    DatasetNode,
    Edge,
    ExperimentNode,
    FeatureNode,
    NodeType,
    RegimeNode,
    RelationType,
    StrategyNode,
)
from .store import KnowledgeGraphStore, SqliteKnowledgeGraphStore

__all__ = [
    "BaseNode",
    "DatasetNode",
    "Edge",
    "ExperimentNode",
    "FeatureNode",
    "NodeType",
    "RegimeNode",
    "RelationType",
    "StrategyNode",
    "KnowledgeGraphStore",
    "SqliteKnowledgeGraphStore",
]


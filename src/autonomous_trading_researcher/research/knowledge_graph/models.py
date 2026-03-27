"""Data models for the Research Intelligence Graph.

These models intentionally sit in the `research` layer so they can be shared by
both the experiment engine and the dashboard without creating new core
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Any, Dict, List, Mapping, MutableMapping, Optional


class NodeType(StrEnum):
    """Logical type of a node in the research knowledge graph."""

    FEATURE = "FEATURE"
    STRATEGY = "STRATEGY"
    DATASET = "DATASET"
    REGIME = "REGIME"
    EXPERIMENT = "EXPERIMENT"


class RelationType(StrEnum):
    """Relationship types between graph nodes."""

    FEATURE_USED_IN_STRATEGY = "FEATURE_USED_IN_STRATEGY"
    STRATEGY_TESTED_IN_EXPERIMENT = "STRATEGY_TESTED_IN_EXPERIMENT"
    EXPERIMENT_CONDUCTED_ON_DATASET = "EXPERIMENT_CONDUCTED_ON_DATASET"
    STRATEGY_PERFORMS_WELL_IN_REGIME = "STRATEGY_PERFORMS_WELL_IN_REGIME"
    STRATEGY_PERFORMS_POORLY_IN_REGIME = "STRATEGY_PERFORMS_POORLY_IN_REGIME"
    FEATURE_COMBINATION_IMPROVES_METRIC = "FEATURE_COMBINATION_IMPROVES_METRIC"
    EXPERIMENT_GENERATED_STRATEGY = "EXPERIMENT_GENERATED_STRATEGY"
    FEATURE_EFFECTIVE_IN_REGIME = "FEATURE_EFFECTIVE_IN_REGIME"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class BaseNode:
    """Base node type shared by all knowledge graph nodes."""

    id: str
    type: NodeType
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def to_record(self) -> Dict[str, Any]:
        """Convert node into a JSON‑serialisable record."""

        return {
            "id": self.id,
            "type": self.type.value,
            "labels": list(self.labels),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class FeatureNode(BaseNode):
    """Represents a single feature or feature family."""

    name: str = ""
    family: Optional[str] = None
    timeframe: Optional[str] = None
    source: Optional[str] = None  # e.g. "handcrafted", "auto_generated", "template_based"


@dataclass(slots=True)
class StrategyNode(BaseNode):
    """Represents a logical strategy configuration."""

    strategy_id: str = ""
    strategy_type: Optional[str] = None  # e.g. "rule", "generated", "ml", "hybrid"
    family: Optional[str] = None  # e.g. "momentum", "mean_reversion"
    feature_names: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetNode(BaseNode):
    """Represents a dataset version."""

    dataset_version: str = ""
    exchanges: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class RegimeNode(BaseNode):
    """Represents a market regime (trend, mean-reversion, etc.)."""

    name: str = ""


@dataclass(slots=True)
class ExperimentNode(BaseNode):
    """Represents a single experiment run."""

    experiment_id: str = ""
    dataset_version: Optional[str] = None
    feature_set_id: Optional[str] = None
    strategy_ids: List[str] = field(default_factory=list)
    status: Optional[str] = None


@dataclass(slots=True)
class Edge:
    """Relationship between two nodes in the knowledge graph."""

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def to_record(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


NodeLike = BaseNode
MetadataMapping = Mapping[str, Any] | MutableMapping[str, Any] | Dict[str, Any]


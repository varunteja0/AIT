"""Experiment result ingestion into the Research Intelligence Graph.

This module is intentionally conservative: it does not modify existing
behaviour, but provides helper functions that the experiment engine and
orchestrator can call when they are ready to start feeding the knowledge
graph.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from autonomous_trading_researcher.core.models import StrategyCandidate

from .models import (
    BaseNode,
    DatasetNode,
    Edge,
    ExperimentNode,
    FeatureNode,
    NodeType,
    RelationType,
    RegimeNode,
    StrategyNode,
)
from .store import KnowledgeGraphStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _feature_node_from_name(feature_name: str) -> FeatureNode:
    return FeatureNode(
        id=f"feature:{feature_name}",
        type=NodeType.FEATURE,
        name=feature_name,
    )


def _strategy_node_from_candidate(candidate: StrategyCandidate) -> StrategyNode:
    strategy_id = str(candidate.parameters.get("strategy_id", candidate.strategy_name))
    # Features are optional and may be provided via parameters. We avoid relying
    # on attributes that are not part of the BacktestResult interface to keep
    # this helper robust against upstream changes.
    raw_features = candidate.parameters.get("features", [])
    if isinstance(raw_features, str):
        feature_names = [raw_features]
    elif isinstance(raw_features, list):
        feature_names = [name for name in raw_features if isinstance(name, str)]
    else:
        feature_names = []
    return StrategyNode(
        id=f"strategy:{strategy_id}",
        type=NodeType.STRATEGY,
        strategy_id=strategy_id,
        strategy_type=str(candidate.parameters.get("strategy_type", "")) or None,
        family=str(candidate.parameters.get("strategy_family", "")) or None,
        feature_names=feature_names,
    )


def ingest_strategy_candidates(
    store: KnowledgeGraphStore,
    *,
    candidates: Iterable[StrategyCandidate],
    dataset_version: Optional[str] = None,
    feature_set_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    regime_metrics: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    """Ingest one or more :class:`StrategyCandidate` objects into the graph.

    Parameters
    ----------
    store:
        Graph store instance.
    candidates:
        Strategy candidates produced by discovery / optimisation.
    dataset_version:
        Optional dataset version identifier used for these candidates.
    feature_set_id:
        Optional feature-set identifier used for these candidates.
    experiment_id:
        Optional experiment identifier; when provided, an :class:`ExperimentNode`
        is created and linked to strategies and dataset.
    regime_metrics:
        Optional mapping of regime name to metric dictionaries; if present,
        regime relationships will be recorded per strategy.
    """

    now = _now()

    experiment_node: Optional[ExperimentNode] = None
    if experiment_id is not None:
        experiment_node = ExperimentNode(
            id=f"experiment:{experiment_id}",
            type=NodeType.EXPERIMENT,
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            feature_set_id=feature_set_id,
            created_at=now,
            updated_at=now,
        )
        store.upsert_node(experiment_node)

    dataset_node: Optional[DatasetNode] = None
    if dataset_version is not None:
        dataset_node = DatasetNode(
            id=f"dataset:{dataset_version}",
            type=NodeType.DATASET,
            dataset_version=dataset_version,
            created_at=now,
            updated_at=now,
        )
        store.upsert_node(dataset_node)

    for candidate in candidates:
        strategy_node = _strategy_node_from_candidate(candidate)
        store.upsert_node(strategy_node)

        feature_nodes = [_feature_node_from_name(name) for name in strategy_node.feature_names]
        if feature_nodes:
            store.upsert_nodes(feature_nodes)
            feature_edges = [
                Edge(
                    source_id=feature_node.id,
                    target_id=strategy_node.id,
                    relation_type=RelationType.FEATURE_USED_IN_STRATEGY,
                    metadata={"role": "signal"},
                )
                for feature_node in feature_nodes
            ]
            store.upsert_edges(feature_edges)

        primary_feature = candidate.parameters.get("primary_feature")
        secondary_feature = candidate.parameters.get("secondary_feature")
        if isinstance(primary_feature, str) and isinstance(secondary_feature, str):
            combo_edge = Edge(
                source_id=f"feature:{primary_feature}",
                target_id=f"feature:{secondary_feature}",
                relation_type=RelationType.FEATURE_COMBINATION_IMPROVES_METRIC,
                weight=float(candidate.backtest_result.metrics.sharpe_ratio),
                metadata={
                    "strategy_id": strategy_node.strategy_id,
                    "metrics": asdict(candidate.backtest_result.metrics),
                },
            )
            store.upsert_edge(combo_edge)

        if experiment_node is not None:
            edge = Edge(
                source_id=strategy_node.id,
                target_id=experiment_node.id,
                relation_type=RelationType.STRATEGY_TESTED_IN_EXPERIMENT,
                metadata={
                    "symbol": candidate.symbol,
                    "metrics": asdict(candidate.backtest_result.metrics),
                },
            )
            store.upsert_edge(edge)

        if dataset_node is not None and experiment_node is not None:
            dataset_edge = Edge(
                source_id=experiment_node.id,
                target_id=dataset_node.id,
                relation_type=RelationType.EXPERIMENT_CONDUCTED_ON_DATASET,
                metadata={"feature_set_id": feature_set_id},
            )
            store.upsert_edge(dataset_edge)

        if regime_metrics:
            for regime_name, metrics in regime_metrics.items():
                regime_node = RegimeNode(
                    id=f"regime:{regime_name}",
                    type=NodeType.REGIME,
                    name=regime_name,
                    created_at=now,
                    updated_at=now,
                )
                store.upsert_node(regime_node)
                sharpe = float(candidate.backtest_result.metrics.sharpe_ratio)
                relation = (
                    RelationType.STRATEGY_PERFORMS_WELL_IN_REGIME
                    if sharpe >= 0.0
                    else RelationType.STRATEGY_PERFORMS_POORLY_IN_REGIME
                )
                store.upsert_edge(
                    Edge(
                        source_id=strategy_node.id,
                        target_id=regime_node.id,
                        relation_type=relation,
                        weight=sharpe,
                        metadata={"regime": regime_name, **metrics},
                    )
                )
                if feature_nodes:
                    for feature_node in feature_nodes:
                        store.upsert_edge(
                            Edge(
                                source_id=feature_node.id,
                                target_id=regime_node.id,
                                relation_type=RelationType.FEATURE_EFFECTIVE_IN_REGIME,
                                weight=sharpe,
                                metadata={"regime": regime_name},
                            )
                        )

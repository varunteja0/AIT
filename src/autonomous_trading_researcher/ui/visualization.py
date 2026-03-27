"""Plotly visualizations for the dashboard API."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def _empty_figure(title: str) -> dict[str, Any]:
    """Return an empty figure payload."""

    return go.Figure(layout={"title": title}).to_plotly_json()


def equity_curve_figure(top_strategies: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an equity curve figure for the best strategy."""

    if not top_strategies:
        return _empty_figure("Equity Curve")
    best = top_strategies[0]
    curve = best["backtest_result"].get("equity_curve", [])
    figure = go.Figure()
    figure.add_trace(go.Scatter(y=curve, mode="lines", name=best["strategy_id"]))
    figure.update_layout(title="Equity Curve", xaxis_title="Step", yaxis_title="Equity")
    return figure.to_plotly_json()


def drawdown_figure(top_strategies: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a drawdown figure for the best strategy."""

    if not top_strategies:
        return _empty_figure("Drawdown")
    curve = pd.Series(top_strategies[0]["backtest_result"].get("equity_curve", []), dtype=float)
    if curve.empty:
        return _empty_figure("Drawdown")
    drawdown = curve / curve.cummax() - 1.0
    figure = go.Figure()
    figure.add_trace(go.Scatter(y=drawdown.tolist(), mode="lines", name="drawdown"))
    figure.update_layout(title="Drawdown", xaxis_title="Step", yaxis_title="Drawdown")
    return figure.to_plotly_json()


def strategy_performance_figure(top_strategies: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a bar chart of strategy scores."""

    if not top_strategies:
        return _empty_figure("Strategy Performance")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[strategy["strategy_id"] for strategy in top_strategies[:10]],
            y=[strategy["score"] for strategy in top_strategies[:10]],
            text=[strategy["metrics"].get("sharpe_ratio", 0.0) for strategy in top_strategies[:10]],
            name="score",
        )
    )
    figure.update_layout(title="Strategy Performance", xaxis_title="Strategy", yaxis_title="Score")
    return figure.to_plotly_json()


def feature_importance_figure(top_strategies: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a feature usage chart from saved strategy definitions."""

    if not top_strategies:
        return _empty_figure("Feature Importance")
    counts: Counter[str] = Counter()
    for strategy in top_strategies:
        parameters = strategy["parameters"]
        for key in ("primary_feature", "secondary_feature"):
            value = parameters.get(key)
            if isinstance(value, str):
                counts[value] += 1
    if not counts:
        return _empty_figure("Feature Importance")
    figure = go.Figure()
    figure.add_trace(go.Bar(x=list(counts.keys())[:10], y=list(counts.values())[:10]))
    figure.update_layout(title="Feature Importance", xaxis_title="Feature", yaxis_title="Usage")
    return figure.to_plotly_json()


def feature_correlation_figure(
    feature_correlations: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build a heatmap from feature correlation payloads."""

    if not feature_correlations:
        return _empty_figure("Feature Correlations")
    frame = pd.DataFrame(feature_correlations).fillna(0.0)
    figure = go.Figure(
        data=go.Heatmap(
            z=frame.values,
            x=frame.columns.tolist(),
            y=frame.index.tolist(),
            colorscale="RdBu",
            zmid=0.0,
        )
    )
    figure.update_layout(title="Feature Correlations")
    return figure.to_plotly_json()


def optimization_results_figure(top_strategies: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a scatter chart summarizing optimization outcomes."""

    if not top_strategies:
        return _empty_figure("Optimization Results")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[strategy["metrics"].get("total_return", 0.0) for strategy in top_strategies[:50]],
            y=[strategy["metrics"].get("sharpe_ratio", 0.0) for strategy in top_strategies[:50]],
            mode="markers",
            text=[strategy["strategy_id"] for strategy in top_strategies[:50]],
            marker={
                "size": [
                    max(8.0, abs(strategy["score"]) * 10)
                    for strategy in top_strategies[:50]
                ],
                "color": [
                    strategy["metrics"].get("max_drawdown", 0.0)
                    for strategy in top_strategies[:50]
                ],
                "colorscale": "Viridis",
                "showscale": True,
            },
        )
    )
    figure.update_layout(
        title="Optimization Results",
        xaxis_title="Total Return",
        yaxis_title="Sharpe Ratio",
    )
    return figure.to_plotly_json()


def experiment_performance_figure(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """Plot experiment scores over time."""

    if not experiments:
        return _empty_figure("Experiment Performance")
    sorted_experiments = sorted(experiments, key=lambda item: item.get("start_time") or "")
    times = [item.get("start_time") for item in sorted_experiments]
    scores = [
        float(item.get("metrics", {}).get("best_score", 0.0)) for item in sorted_experiments
    ]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=times, y=scores, mode="lines+markers", name="best_score"))
    figure.update_layout(title="Experiment Performance", xaxis_title="Start Time")
    return figure.to_plotly_json()


def strategy_network_figure(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Render a lightweight strategy network visualization."""

    if not nodes:
        return _empty_figure("Strategy Network")
    type_levels = {
        "FEATURE": 0,
        "STRATEGY": 1,
        "EXPERIMENT": 2,
        "DATASET": 3,
        "REGIME": 4,
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        grouped.setdefault(node["type"], []).append(node)
    positions: dict[str, tuple[float, float]] = {}
    for node_type, items in grouped.items():
        y = type_levels.get(node_type, 0)
        for index, node in enumerate(items):
            positions[node["id"]] = (float(index), float(y))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for edge in edges:
        source = positions.get(edge["source_id"])
        target = positions.get(edge["target_id"])
        if source is None or target is None:
            continue
        edge_x.extend([source[0], target[0], None])
        edge_y.extend([source[1], target[1], None])

    node_x = [positions[node["id"]][0] for node in nodes if node["id"] in positions]
    node_y = [positions[node["id"]][1] for node in nodes if node["id"] in positions]
    node_labels = [node["id"] for node in nodes if node["id"] in positions]

    figure = go.Figure()
    if edge_x:
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 1, "color": "#94a3b8"},
                hoverinfo="none",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={"size": 10, "color": "#0f766e"},
            text=node_labels,
            hoverinfo="text",
            showlegend=False,
        )
    )
    figure.update_layout(title="Strategy Network", xaxis_visible=False, yaxis_visible=False)
    return figure.to_plotly_json()


def regime_heatmap_figure(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """Visualize average experiment scores by detected regime."""

    if not experiments:
        return _empty_figure("Regime Heatmap")
    regime_scores: dict[str, list[float]] = {}
    for experiment in experiments:
        labels = experiment.get("parameters", {}).get("regime_labels", []) or []
        score = float(experiment.get("metrics", {}).get("best_score", 0.0))
        for label in labels:
            regime_scores.setdefault(label, []).append(score)
    if not regime_scores:
        return _empty_figure("Regime Heatmap")
    regimes = sorted(regime_scores.keys())
    values = [sum(regime_scores[label]) / len(regime_scores[label]) for label in regimes]
    figure = go.Figure(
        data=go.Heatmap(
            z=[values],
            x=regimes,
            y=["avg_score"],
            colorscale="YlOrRd",
        )
    )
    figure.update_layout(title="Regime Heatmap")
    return figure.to_plotly_json()


def portfolio_allocation_figure(allocation: dict[str, float]) -> dict[str, Any]:
    """Render current portfolio allocation weights."""

    if not allocation:
        return _empty_figure("Portfolio Allocation")
    symbols = list(allocation.keys())
    weights = [allocation[symbol] for symbol in symbols]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=symbols, y=weights))
    figure.update_layout(title="Portfolio Allocation", xaxis_title="Symbol", yaxis_title="Weight")
    return figure.to_plotly_json()

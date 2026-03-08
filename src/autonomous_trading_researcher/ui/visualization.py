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


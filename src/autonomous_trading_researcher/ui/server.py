"""FastAPI dashboard for monitoring autonomous research."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from autonomous_trading_researcher.config import load_config
from autonomous_trading_researcher.research.experiment_db import ExperimentDatabase
from autonomous_trading_researcher.research.knowledge_graph.queries import (
    get_strategies_that_fail_in_regime,
    get_top_features_by_sharpe,
)
from autonomous_trading_researcher.research.knowledge_graph.store import (
    SqliteKnowledgeGraphStore,
)
from autonomous_trading_researcher.ui.visualization import (
    drawdown_figure,
    equity_curve_figure,
    experiment_performance_figure,
    feature_correlation_figure,
    feature_importance_figure,
    portfolio_allocation_figure,
    optimization_results_figure,
    regime_heatmap_figure,
    strategy_network_figure,
    strategy_performance_figure,
)


def _read_status_file(path: Path) -> dict[str, Any]:
    """Read the persisted system status payload."""

    if not path.exists():
        return {
            "system_healthy": False,
            "details": {
                "number_of_strategies_tested": 0,
                "best_strategy_score": 0.0,
                "top_features": [],
                "active_strategy": None,
                "feature_correlations": {},
            },
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _read_recent_events(path: Path, limit: int = 30) -> list[dict[str, Any]]:
    """Read recent event-log entries."""

    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]


def create_app(config_path: str = "config/default.yaml") -> FastAPI:
    """Create the FastAPI dashboard application."""

    config = load_config(config_path)
    app = FastAPI(title="Autonomous Trading Researcher Dashboard")
    app.state.config = config
    app.state.database = ExperimentDatabase(config.research.experiment_db_path)
    graph_path = Path(config.research.experiment_db_path).with_name("knowledge_graph.db")
    app.state.graph_store = SqliteKnowledgeGraphStore(graph_path)
    app.state.status_path = Path(config.monitoring.status_path)
    app.state.event_log_path = Path(config.monitoring.event_log_path)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        """Serve the dashboard UI."""

        refresh_ms = config.ui.refresh_seconds * 1000
        return HTMLResponse(
            f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <title>Autonomous Trading Researcher</title>
              <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
              <style>
                body {{ font-family: 'IBM Plex Sans', sans-serif; margin: 0; background: #f6f4ee; color: #1f2933; }}
                header {{ padding: 24px 32px; background: linear-gradient(135deg, #0f766e, #f59e0b); color: white; }}
                main {{ padding: 24px 32px; display: grid; gap: 24px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
                .card {{ background: white; border-radius: 18px; padding: 18px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08); }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ text-align: left; padding: 8px 0; border-bottom: 1px solid #e5e7eb; font-size: 14px; }}
                .chart {{ min-height: 320px; }}
                pre {{ white-space: pre-wrap; font-size: 12px; }}
              </style>
            </head>
            <body>
              <header>
                <h1>Autonomous Trading Researcher</h1>
                <p>Monitoring the autonomous quantitative research laboratory.</p>
              </header>
              <main>
                <section class="grid">
                  <div class="card"><h3>System Status</h3><div id="system-status"></div></div>
                  <div class="card"><h3>Top Features</h3><div id="top-features"></div></div>
                  <div class="card"><h3>Live Logs</h3><pre id="live-logs"></pre></div>
                </section>
                <section class="card">
                  <h3>Top Strategies</h3>
                  <div id="top-strategies"></div>
                </section>
                <section class="grid">
                  <div class="card chart" id="equity-curve"></div>
                  <div class="card chart" id="drawdown"></div>
                  <div class="card chart" id="strategy-performance"></div>
                  <div class="card chart" id="feature-importance"></div>
                  <div class="card chart" id="feature-correlations"></div>
                  <div class="card chart" id="optimization-results"></div>
                  <div class="card chart" id="experiment-performance"></div>
                  <div class="card chart" id="strategy-network"></div>
                  <div class="card chart" id="regime-heatmap"></div>
                  <div class="card chart" id="portfolio-allocation"></div>
                </section>
              </main>
              <script>
                async function loadDashboard() {{
                  const [statusResp, topResp, metricsResp] = await Promise.all([
                    fetch('/api/system_status'),
                    fetch('/api/top_strategies'),
                    fetch('/api/metrics'),
                  ]);
                  const status = await statusResp.json();
                  const top = await topResp.json();
                  const metrics = await metricsResp.json();

                  document.getElementById('system-status').innerHTML = `
                    <div><strong>Healthy:</strong> ${{status.system_healthy}}</div>
                    <div><strong>Strategies Tested:</strong> ${{status.details.number_of_strategies_tested ?? 0}}</div>
                    <div><strong>Best Score:</strong> ${{(status.details.best_strategy_score ?? 0).toFixed(4)}}</div>
                    <div><strong>Active Strategy:</strong> ${{status.details.active_strategy ?? 'n/a'}}</div>
                  `;
                  document.getElementById('top-features').innerHTML =
                    (status.details.top_features ?? []).map((feature) => `<div>${{feature}}</div>`).join('');
                  document.getElementById('live-logs').textContent =
                    (status.live_logs ?? []).map((entry) => `${{entry.timestamp}} ${{entry.event_type}} ${{JSON.stringify(entry.payload)}}`).join('\\n');

                  const rows = (top.strategies ?? []).slice(0, 10).map((strategy) => `
                    <tr>
                      <td>${{strategy.strategy_id}}</td>
                      <td>${{strategy.symbol}}</td>
                      <td>${{strategy.score.toFixed(4)}}</td>
                      <td>${{(strategy.metrics.sharpe_ratio ?? 0).toFixed(3)}}</td>
                      <td>${{(strategy.metrics.total_return ?? 0).toFixed(3)}}</td>
                    </tr>
                  `).join('');
                  document.getElementById('top-strategies').innerHTML = `
                    <table>
                      <thead><tr><th>Strategy</th><th>Symbol</th><th>Score</th><th>Sharpe</th><th>Return</th></tr></thead>
                      <tbody>${{rows}}</tbody>
                    </table>
                  `;

                  Plotly.react('equity-curve', metrics.equity_curve.data, metrics.equity_curve.layout);
                  Plotly.react('drawdown', metrics.drawdown.data, metrics.drawdown.layout);
                  Plotly.react('strategy-performance', metrics.strategy_performance.data, metrics.strategy_performance.layout);
                  Plotly.react('feature-importance', metrics.feature_importance.data, metrics.feature_importance.layout);
                  Plotly.react('feature-correlations', metrics.feature_correlations.data, metrics.feature_correlations.layout);
                  Plotly.react('optimization-results', metrics.optimization_results.data, metrics.optimization_results.layout);
                  Plotly.react('experiment-performance', metrics.experiment_performance.data, metrics.experiment_performance.layout);
                  Plotly.react('strategy-network', metrics.strategy_network.data, metrics.strategy_network.layout);
                  Plotly.react('regime-heatmap', metrics.regime_heatmap.data, metrics.regime_heatmap.layout);
                  Plotly.react('portfolio-allocation', metrics.portfolio_allocation.data, metrics.portfolio_allocation.layout);
                }}

                loadDashboard();
                setInterval(loadDashboard, {refresh_ms});
              </script>
            </body>
            </html>
            """
        )

    @app.get("/api/strategies")
    async def strategies(limit: int = 100) -> dict[str, Any]:
        """Return recent strategy experiments."""

        return {"strategies": app.state.database.list_strategies(limit=limit)}

    @app.get("/api/top_strategies")
    async def top_strategies(limit: int = 20) -> dict[str, Any]:
        """Return the best historical strategies."""

        return {"strategies": app.state.database.top_strategies(limit=limit)}

    @app.get("/api/system_status")
    async def system_status() -> dict[str, Any]:
        """Return current monitoring state."""

        payload = _read_status_file(app.state.status_path)
        payload["live_logs"] = _read_recent_events(app.state.event_log_path)
        payload["experiment_summary"] = app.state.database.summary()
        return payload

    @app.get("/api/experiments")
    async def experiments(limit: int = 100) -> dict[str, Any]:
        """Return recent experiment runs."""

        return {"experiments": app.state.database.list_experiments(limit=limit)}

    @app.get("/api/knowledge/features")
    async def knowledge_features(top_n: int = 20) -> dict[str, Any]:
        """Return top features from the knowledge graph."""

        return {
            "features": get_top_features_by_sharpe(
                app.state.graph_store,
                top_n=top_n,
            )
        }

    @app.get("/api/knowledge/strategies")
    async def knowledge_strategies(
        regime: str | None = None,
        sharpe_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Return strategy intelligence from the knowledge graph."""

        if regime:
            strategies = get_strategies_that_fail_in_regime(
                app.state.graph_store,
                regime=regime,
                sharpe_threshold=sharpe_threshold,
            )
        else:
            strategies = app.state.database.top_strategies(limit=20)
        return {"strategies": strategies}

    @app.get("/api/knowledge/graph")
    async def knowledge_graph(
        node_limit: int = 500,
        edge_limit: int = 1000,
    ) -> dict[str, Any]:
        """Return graph nodes and edges for visualization."""

        return {
            "nodes": app.state.graph_store.list_nodes(limit=node_limit),
            "edges": app.state.graph_store.list_edges(limit=edge_limit),
        }

    @app.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        """Return dashboard-ready Plotly figures."""

        top_strategies_payload = app.state.database.top_strategies(limit=20)
        status_payload = _read_status_file(app.state.status_path)
        feature_correlations = status_payload.get("details", {}).get("feature_correlations", {})
        experiments_payload = app.state.database.list_experiments(limit=50)
        graph_payload = {
            "nodes": app.state.graph_store.list_nodes(limit=300),
            "edges": app.state.graph_store.list_edges(limit=600),
        }
        allocation_payload = status_payload.get("details", {}).get("portfolio_allocation", {})
        return {
            "equity_curve": equity_curve_figure(top_strategies_payload),
            "drawdown": drawdown_figure(top_strategies_payload),
            "strategy_performance": strategy_performance_figure(top_strategies_payload),
            "feature_importance": feature_importance_figure(top_strategies_payload),
            "feature_correlations": feature_correlation_figure(feature_correlations),
            "optimization_results": optimization_results_figure(top_strategies_payload),
            "experiment_performance": experiment_performance_figure(experiments_payload),
            "strategy_network": strategy_network_figure(
                graph_payload["nodes"],
                graph_payload["edges"],
            ),
            "regime_heatmap": regime_heatmap_figure(experiments_payload),
            "portfolio_allocation": portfolio_allocation_figure(allocation_payload),
        }

    return app


def run_dashboard(config_path: str, host: str, port: int) -> None:
    """Run the dashboard server."""

    uvicorn.run(create_app(config_path), host=host, port=port)

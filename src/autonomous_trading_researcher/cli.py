"""CLI entry points for the trading research platform."""

from __future__ import annotations

import argparse
import asyncio

from autonomous_trading_researcher.config import load_config
from autonomous_trading_researcher.logging_utils import configure_logging
from autonomous_trading_researcher.orchestration.autonomous_loop import (
    AutonomousResearchLoop,
)
from autonomous_trading_researcher.ui.server import run_dashboard


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Autonomous AI Trading Researcher")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    research_parser = subparsers.add_parser("research", help="Run autonomous research cycles.")
    run_parser = subparsers.add_parser("run", help="Run research cycles and deploy an ensemble.")
    discover_parser = subparsers.add_parser("discover", help="Run discovery without deployment.")
    subparsers.add_parser("monitor", help="Emit a monitoring snapshot after research.")
    collect_parser = subparsers.add_parser(
        "collect-data",
        help="Collect one market data cycle.",
    )
    collect_parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of collection cycles.",
    )
    backtest_parser = subparsers.add_parser("backtest", help="Backtest the best saved strategy.")
    backtest_parser.add_argument("--strategy-id", default=None, help="Saved strategy identifier.")
    backtest_parser.add_argument("--symbol", default=None, help="Symbol override for backtest.")
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the monitoring dashboard.")
    dashboard_parser.add_argument("--host", default=None, help="Dashboard bind host.")
    dashboard_parser.add_argument("--port", type=int, default=None, help="Dashboard port.")

    for parser_with_mode in (research_parser, run_parser, discover_parser):
        parser_with_mode.add_argument(
            "--mode",
            choices=("paper", "live"),
            default=None,
            help="Execution mode override.",
        )
        parser_with_mode.add_argument(
            "--cycles",
            type=int,
            default=1,
            help="Number of autonomous cycles to run.",
        )
        parser_with_mode.add_argument(
            "--continuous",
            action="store_true",
            help="Run continuously until interrupted.",
        )
    return parser


async def _run_async(args: argparse.Namespace) -> None:
    """Execute an asynchronous CLI command."""

    config = load_config(args.config)
    if getattr(args, "mode", None) == "paper":
        config.execution.mode = "paper"
        config.execution.paper_trading = True
        config.execution.enabled = False
    elif getattr(args, "mode", None) == "live":
        config.execution.mode = "live"
        config.execution.paper_trading = False
        config.execution.enabled = True
    configure_logging(config.logging.level, config.logging.json)
    orchestrator = AutonomousResearchLoop.from_config(config)
    try:
        if args.command == "monitor":
            await orchestrator.run_cycle(emit_monitoring_only=True)
            return
        if args.command == "collect-data":
            for _ in range(args.cycles):
                await orchestrator.collect_data_once()
            return
        if args.command == "discover":
            if args.continuous:
                await orchestrator.run_forever(max_cycles=None, emit_monitoring_only=True)
            else:
                for _ in range(args.cycles):
                    await orchestrator.discover_only()
            return
        if args.command == "backtest":
            orchestrator.backtest_saved_strategy(
                strategy_id=args.strategy_id,
                symbol=args.symbol,
            )
            return
        if args.continuous:
            await orchestrator.run_forever(max_cycles=None)
            return
        for _ in range(args.cycles):
            await orchestrator.run_cycle()
    finally:
        await orchestrator.close()


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "dashboard":
        config = load_config(args.config)
        configure_logging(config.logging.level, config.logging.json)
        run_dashboard(
            config_path=args.config,
            host=args.host or config.ui.host,
            port=args.port or config.ui.port,
        )
        return
    asyncio.run(_run_async(args))

"""End-to-end test for the autonomous research loop."""

from __future__ import annotations

import asyncio

from autonomous_trading_researcher.core.models import OrderBookSnapshot, Trade
from autonomous_trading_researcher.data.storage import ParquetDataLake
from autonomous_trading_researcher.orchestration.autonomous_loop import AutonomousResearchLoop


def test_autonomous_research_loop_runs_end_to_end(app_config, synthetic_market_data) -> None:
    """The orchestrator should discover and deploy a candidate from stored data."""

    loop = AutonomousResearchLoop.from_config(app_config)
    loop.collector = None
    storage = ParquetDataLake(app_config.data.data_dir)

    trades = [
        Trade(
            exchange_id=app_config.data.exchange_id,
            symbol="BTC/USDT",
            timestamp=timestamp.to_pydatetime(),
            price=float(row.close),
            amount=float(row.volume / 10),
            side="buy",
            trade_id=str(index),
        )
        for index, (timestamp, row) in enumerate(synthetic_market_data.iterrows())
    ]
    books = [
        OrderBookSnapshot(
            exchange_id=app_config.data.exchange_id,
            symbol="BTC/USDT",
            timestamp=timestamp.to_pydatetime(),
            bids=[(float(row.best_bid), float(row.bid_depth))],
            asks=[(float(row.best_ask), float(row.ask_depth))],
        )
        for timestamp, row in synthetic_market_data.iterrows()
    ]
    storage.write_trades(trades)
    storage.write_order_books(books)

    result = asyncio.run(loop.run_cycle())

    assert result["candidate_count"] > 0
    assert result["best_candidate"] is not None
    assert loop.deployed_candidate is not None
    assert len(loop.deployed_candidates) >= 1
    assert result["monitoring"].system_healthy is True

"""Tests for scalable market data ingestion behavior."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from autonomous_trading_researcher.core.models import OrderBookSnapshot, Trade
from autonomous_trading_researcher.data.clients.base import MarketDataClient
from autonomous_trading_researcher.data.ingestion import MarketDataCollector
from autonomous_trading_researcher.data.storage import ParquetDataLake


class FakeMarketDataClient(MarketDataClient):
    """Deterministic market data client for ingestion tests."""

    def __init__(self, trades: list[Trade]) -> None:
        self._trades = trades

    async def connect(self) -> None:
        return None

    async def fetch_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 200,
    ) -> list[Trade]:
        filtered = [
            trade
            for trade in self._trades
            if trade.symbol == symbol
            and (since is None or int(trade.timestamp.timestamp() * 1000) >= since)
        ]
        return filtered[:limit]

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 10,
    ) -> OrderBookSnapshot:
        del limit
        return OrderBookSnapshot(
            exchange_id="binance",
            symbol=symbol,
            timestamp=datetime.now(tz=UTC),
            bids=[(100.0, 10.0)],
            asks=[(100.1, 9.0)],
        )

    async def close(self) -> None:
        return None


def test_market_data_collector_paginates_and_persists_checkpoints(tmp_path) -> None:
    """The collector should paginate historical trades and resume from checkpoints."""

    start = datetime(2024, 1, 1, tzinfo=UTC)
    trades = [
        Trade(
            exchange_id="binance",
            symbol="BTC/USDT",
            timestamp=start + timedelta(seconds=index),
            price=100.0 + index,
            amount=1.0,
            side="buy",
            trade_id=str(index),
        )
        for index in range(6)
    ]
    storage = ParquetDataLake(tmp_path / "lake")
    checkpoint = tmp_path / "collector.json"
    collector = MarketDataCollector(
        exchange_id="binance",
        client=FakeMarketDataClient(trades),
        storage=storage,
        symbols=["BTC/USDT"],
        trade_fetch_limit=2,
        max_trade_batches_per_cycle=10,
        checkpoint_path=checkpoint,
    )

    collected = asyncio.run(collector.collect_trades_once())

    resumed = MarketDataCollector(
        exchange_id="binance",
        client=FakeMarketDataClient(trades),
        storage=storage,
        symbols=["BTC/USDT"],
        trade_fetch_limit=2,
        max_trade_batches_per_cycle=10,
        checkpoint_path=checkpoint,
    )

    assert collected == 6
    assert checkpoint.exists()
    assert resumed.state.last_trade_timestamp_ms["BTC/USDT"] > int(
        trades[-1].timestamp.timestamp() * 1000
    )
    assert len(storage.read_dataset("trades", "binance", "BTC/USDT")) == 6

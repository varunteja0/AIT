"""Tests for parquet storage and dataset building."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from autonomous_trading_researcher.core.models import OrderBookSnapshot, Trade
from autonomous_trading_researcher.data.datasets import HistoricalDatasetBuilder
from autonomous_trading_researcher.data.storage import ParquetDataLake


def test_storage_round_trip_and_dataset_build(tmp_path) -> None:
    """Trades and order books should round-trip through parquet storage."""

    storage = ParquetDataLake(tmp_path / "lake")
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    trades = [
        Trade(
            exchange_id="binance",
            symbol="BTC/USDT",
            timestamp=(timestamp + timedelta(minutes=offset)).to_pydatetime(),
            price=100 + offset,
            amount=1.0,
            side="buy",
            trade_id=str(offset),
        )
        for offset in range(12)
    ]
    books = [
        OrderBookSnapshot(
            exchange_id="binance",
            symbol="BTC/USDT",
            timestamp=(timestamp + timedelta(minutes=offset)).to_pydatetime(),
            bids=[(99.9 + offset, 5.0)],
            asks=[(100.1 + offset, 4.0)],
        )
        for offset in range(12)
    ]

    storage.write_trades(trades)
    storage.write_order_books(books)

    trade_frame = storage.read_dataset("trades", "binance", "BTC/USDT")
    book_frame = storage.read_dataset("order_books", "binance", "BTC/USDT")
    builder = HistoricalDatasetBuilder()
    bars = builder.build_ohlcv_from_trades(trade_frame, timeframe="5min")
    enriched = builder.attach_order_book_features(bars, book_frame, timeframe="5min")

    assert len(trade_frame) == 12
    assert len(book_frame) == 12
    assert not bars.empty
    assert {"best_bid", "best_ask", "bid_depth", "ask_depth"}.issubset(enriched.columns)


def test_storage_deduplicates_and_filters_by_time_range(tmp_path) -> None:
    """Storage reads should be deduplicated and support incremental time filters."""

    storage = ParquetDataLake(tmp_path / "lake")
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    duplicated_trades = [
        Trade(
            exchange_id="binance",
            symbol="BTC/USDT",
            timestamp=(timestamp + timedelta(minutes=offset)).to_pydatetime(),
            price=100 + offset,
            amount=1.0,
            side="buy",
            trade_id=str(offset),
        )
        for offset in range(6)
    ]
    duplicated_trades.append(duplicated_trades[-1])

    storage.write_trades(duplicated_trades)

    filtered = storage.read_dataset(
        "trades",
        "binance",
        "BTC/USDT",
        start="2024-01-01T00:02:00Z",
        end="2024-01-01T00:04:00Z",
    )
    integrity = storage.integrity_report("trades", "binance", "BTC/USDT")

    assert len(storage.read_dataset("trades", "binance", "BTC/USDT")) == 6
    assert len(filtered) == 3
    assert integrity["duplicate_rows"] == 0


def test_dataset_builder_creates_multi_resolution_bars(tmp_path) -> None:
    """The dataset builder should emit aligned datasets across configured timeframes."""

    storage = ParquetDataLake(tmp_path / "lake")
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    trades = [
        Trade(
            exchange_id="binance",
            symbol="BTC/USDT",
            timestamp=(timestamp + timedelta(seconds=offset)).to_pydatetime(),
            price=100 + (offset * 0.1),
            amount=1.0 + (offset * 0.01),
            side="buy" if offset % 2 == 0 else "sell",
            trade_id=str(offset),
        )
        for offset in range(30)
    ]
    books = [
        OrderBookSnapshot(
            exchange_id="binance",
            symbol="BTC/USDT",
            timestamp=(timestamp + timedelta(seconds=offset)).to_pydatetime(),
            bids=[(99.9 + offset * 0.1, 5.0 + offset)],
            asks=[(100.1 + offset * 0.1, 4.0 + offset)],
        )
        for offset in range(30)
    ]
    storage.write_trades(trades)
    storage.write_order_books(books)

    builder = HistoricalDatasetBuilder()
    datasets = builder.build_multi_resolution_datasets(
        storage.read_dataset("trades", "binance", "BTC/USDT"),
        storage.read_dataset("order_books", "binance", "BTC/USDT"),
        timeframes=["1s", "5s", "1min"],
    )

    assert set(datasets) == {"1s", "5s", "1min"}
    assert datasets["1s"].index.is_monotonic_increasing
    assert {"open", "close", "best_bid", "best_ask"}.issubset(datasets["5s"].columns)

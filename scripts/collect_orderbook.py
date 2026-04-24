"""
collect_orderbook.py — 24/7 orderbook snapshot collector.

Captures Binance BTC/USDT depth20 snapshots every 5 seconds.
Required for OBI and depth_ratio feature reconstruction in training.

Output: data/raw/orderbook_snapshots/ (parquet files, daily rotation)

Usage:
  python scripts/collect_orderbook.py
  python scripts/collect_orderbook.py --interval 5
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import click
import structlog
import websockets

logger = structlog.get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "orderbook_snapshots"
WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"


class OrderbookCollector:
    """Collect orderbook snapshots for training data."""

    def __init__(self, interval_seconds: int = 5) -> None:
        self._interval = interval_seconds
        self._buffer: List[Dict] = []
        self._buffer_limit = 1000  # Flush every 1000 snapshots
        self._current_date = ""
        self._snapshot_count = 0
        self._running = False
        self._latest_ob = None

    async def start(self) -> None:
        """Start WebSocket collection."""
        self._running = True
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        while self._running:
            try:
                await self._connect_and_collect()
            except Exception as e:
                logger.error("ob_collector_error", error=str(e))
                await asyncio.sleep(5.0)

    async def _connect_and_collect(self) -> None:
        """Connect to WS and sample at specified interval."""
        async with websockets.connect(WS_URL, ping_interval=20) as ws:
            logger.info("ob_collector_connected")
            last_sample_time = 0.0

            # Start a sampling task
            sample_task = asyncio.create_task(self._sample_loop())

            async for raw_msg in ws:
                if not self._running:
                    break

                try:
                    data = json.loads(raw_msg)
                    self._latest_ob = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "timestamp_ms": int(time.time() * 1000),
                        "bids": data.get("bids", [])[:20],
                        "asks": data.get("asks", [])[:20],
                    }
                except json.JSONDecodeError:
                    continue

            sample_task.cancel()

    async def _sample_loop(self) -> None:
        """Sample latest orderbook at fixed intervals."""
        while self._running:
            await asyncio.sleep(self._interval)

            if self._latest_ob is None:
                continue

            # Flatten for storage
            snapshot = {
                "timestamp": self._latest_ob["timestamp"],
                "timestamp_ms": self._latest_ob["timestamp_ms"],
            }

            # Flatten bid/ask levels
            for i, (price, qty) in enumerate(self._latest_ob.get("bids", [])[:20]):
                snapshot[f"bid_price_{i}"] = float(price)
                snapshot[f"bid_qty_{i}"] = float(qty)

            for i, (price, qty) in enumerate(self._latest_ob.get("asks", [])[:20]):
                snapshot[f"ask_price_{i}"] = float(price)
                snapshot[f"ask_qty_{i}"] = float(qty)

            self._buffer.append(snapshot)
            self._snapshot_count += 1

            # Progress logging
            if self._snapshot_count % 100 == 0:
                logger.info(
                    "ob_snapshots_collected",
                    count=self._snapshot_count,
                    buffer_size=len(self._buffer),
                )

            # Flush buffer
            if len(self._buffer) >= self._buffer_limit:
                self._flush_buffer()

            # Check date rotation
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != self._current_date:
                if self._buffer:
                    self._flush_buffer()
                self._current_date = today

    def _flush_buffer(self) -> None:
        """Write buffer to parquet file."""
        if not self._buffer:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(self._buffer)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            ts = datetime.now(timezone.utc).strftime("%H%M%S")
            filename = f"ob_{today}_{ts}.parquet"
            filepath = OUTPUT_DIR / filename

            df.to_parquet(filepath, index=False)
            logger.info(
                "ob_buffer_flushed",
                file=str(filepath),
                rows=len(df),
            )
            self._buffer.clear()

        except Exception as e:
            logger.error("ob_flush_error", error=str(e))

    def stop(self) -> None:
        """Stop collection and flush remaining data."""
        self._running = False
        self._flush_buffer()
        logger.info("ob_collector_stopped", total_snapshots=self._snapshot_count)


@click.command()
@click.option("--interval", default=5, help="Snapshot interval in seconds")
def main(interval: int) -> None:
    """Run 24/7 orderbook snapshot collector."""
    click.echo(f"📸 Starting orderbook collector (interval: {interval}s)")
    click.echo(f"   Output: {OUTPUT_DIR}")
    click.echo("   Press Ctrl+C to stop\n")

    collector = OrderbookCollector(interval_seconds=interval)

    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        collector.stop()
        click.echo("\nCollector stopped.")


if __name__ == "__main__":
    main()

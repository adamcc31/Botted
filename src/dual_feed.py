"""
dual_feed.py — Dual-feed price monitor via Polymarket RTDS WebSocket.

Subscribes to two topics from wss://ws-live-data.polymarket.com:
  1. crypto_prices        → Binance BTC/USDT price (relayed by Polymarket)
  2. crypto_prices_chainlink → Chainlink BTC/USD oracle price

Produces DualFeedSnapshot with rolling 60-second window for spread analysis.
Used as the settlement-aligned reference price for fair probability computation.

NO SILENT FALLBACK: If oracle feed is unavailable or stale, callers MUST NOT
silently fall back to Binance price. The spread filter will return SKIP.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Optional, Tuple

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging
import websockets
from websockets.exceptions import ConnectionClosed

from src.config_manager import ConfigManager
from src.schemas import DualFeedSnapshot

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class DualFeed:
    """
    Real-time dual-feed price monitor via Polymarket RTDS.

    Provides:
    - Chainlink oracle price (settlement-aligned reference)
    - Binance price via RTDS relay
    - Spread tracking with rolling window
    - Staleness detection — no silent fallback policy
    """

    RTDS_URL = "wss://ws-live-data.polymarket.com"

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._rtds_url = config.get("dual_feed.rtds_ws_url", self.RTDS_URL)
        self._rolling_window_s = config.get("dual_feed.rolling_window_seconds", 60)
        self._stale_threshold_s = config.get("dual_feed.stale_threshold_seconds", 10)

        # Reconnection parameters
        self._max_retries = config.get("dual_feed.reconnect_max_retries", 10)
        self._initial_delay_s = config.get("dual_feed.reconnect_initial_delay_s", 1)
        self._max_delay_s = config.get("dual_feed.reconnect_max_delay_s", 30)
        self._backoff_multiplier = 2

        # Latest prices
        self._binance_price: Optional[float] = None
        self._chainlink_price: Optional[float] = None
        self._binance_ts: float = 0.0
        self._chainlink_ts: float = 0.0

        # Rolling window: deque of (timestamp, binance_price, chainlink_price)
        self._snapshot_history: Deque[Tuple[float, float, float]] = deque(
            maxlen=max(60, self._rolling_window_s) * 10  # ~10 ticks/sec headroom
        )

        # Connection state
        self._ws_connection = None
        self._running = False
        self._retry_count = 0
        self._messages_received = 0

    # ── Public Properties ─────────────────────────────────────

    @property
    def chainlink_price(self) -> Optional[float]:
        """Current Chainlink oracle BTC/USD price. None if unavailable."""
        return self._chainlink_price

    @property
    def binance_price_rtds(self) -> Optional[float]:
        """Binance BTC/USDT price as relayed by RTDS. None if unavailable."""
        return self._binance_price

    @property
    def is_chainlink_stale(self) -> bool:
        """True if Chainlink price is missing or hasn't updated within threshold."""
        if self._chainlink_price is None or self._chainlink_ts == 0.0:
            return True
        return (time.time() - self._chainlink_ts) > self._stale_threshold_s

    @property
    def is_binance_rtds_stale(self) -> bool:
        """True if RTDS Binance price is missing or hasn't updated within threshold."""
        if self._binance_price is None or self._binance_ts == 0.0:
            return True
        return (time.time() - self._binance_ts) > self._stale_threshold_s

    @property
    def is_available(self) -> bool:
        """True if both feeds have recent data."""
        return not self.is_chainlink_stale and not self.is_binance_rtds_stale

    @property
    def messages_received(self) -> int:
        return self._messages_received

    # ── Snapshot ──────────────────────────────────────────────

    def get_snapshot(self) -> Optional[DualFeedSnapshot]:
        """
        Get current dual-feed snapshot.

        Returns None if either feed is unavailable or stale.
        NO SILENT FALLBACK — caller must handle None explicitly.
        """
        if self._chainlink_price is None or self._binance_price is None:
            return None
        if self.is_chainlink_stale or self.is_binance_rtds_stale:
            return None

        spread = self._binance_price - self._chainlink_price
        spread_pct = abs(spread / self._chainlink_price) * 100.0

        if abs(spread_pct) < 0.001:
            direction = "CONVERGED"
        elif spread > 0:
            direction = "BINANCE_ABOVE"
        else:
            direction = "CHAINLINK_ABOVE"

        return DualFeedSnapshot(
            timestamp=datetime.now(timezone.utc),
            binance_price=self._binance_price,
            chainlink_price=self._chainlink_price,
            spread=spread,
            spread_pct=spread_pct,
            spread_direction=direction,
        )

    def get_oracle_price(self) -> Optional[float]:
        """
        Get Chainlink oracle price for use as settlement-aligned BTC reference.

        Returns None if Chainlink is stale or unavailable.
        NO SILENT FALLBACK — callers MUST NOT substitute Binance price.
        """
        if self.is_chainlink_stale:
            return None
        return self._chainlink_price

    def get_rolling_spread_stats(self) -> dict:
        """
        Compute spread statistics over the rolling window.

        Returns dict with: mean_spread_pct, max_spread_pct, min_spread_pct,
        tick_count, window_seconds.
        """
        now = time.time()
        cutoff = now - self._rolling_window_s
        recent = [(ts, bp, cp) for ts, bp, cp in self._snapshot_history if ts >= cutoff]

        if len(recent) < 2:
            return {
                "mean_spread_pct": 0.0,
                "max_spread_pct": 0.0,
                "min_spread_pct": 0.0,
                "tick_count": len(recent),
                "window_seconds": self._rolling_window_s,
            }

        spreads_pct = [
            abs((bp - cp) / cp) * 100.0
            for _, bp, cp in recent
            if cp > 0
        ]

        if not spreads_pct:
            return {
                "mean_spread_pct": 0.0,
                "max_spread_pct": 0.0,
                "min_spread_pct": 0.0,
                "tick_count": 0,
                "window_seconds": self._rolling_window_s,
            }

        return {
            "mean_spread_pct": sum(spreads_pct) / len(spreads_pct),
            "max_spread_pct": max(spreads_pct),
            "min_spread_pct": min(spreads_pct),
            "tick_count": len(spreads_pct),
            "window_seconds": self._rolling_window_s,
        }

    # ── WebSocket Connection ──────────────────────────────────

    async def start(self) -> None:
        """Start RTDS WebSocket connection with reconnection logic."""
        self._running = True
        self._retry_count = 0

        while self._running:
            try:
                await self._connect_and_listen()
            except (ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                delay = self._compute_backoff_delay()
                logger.warning(
                    "rtds_ws_disconnected",
                    error=str(e),
                    retry_count=self._retry_count,
                    reconnect_delay_s=delay,
                )
                await asyncio.sleep(delay)
                self._retry_count += 1

                if self._retry_count > self._max_retries:
                    logger.error(
                        "rtds_ws_max_retries_exceeded",
                        max_retries=self._max_retries,
                    )
                    self._retry_count = 0
                    await asyncio.sleep(self._max_delay_s)
            except Exception as e:
                logger.error("rtds_ws_unexpected_error", error=str(e))
                if not self._running:
                    break
                await asyncio.sleep(5.0)

    async def stop(self) -> None:
        """Stop RTDS WebSocket gracefully."""
        self._running = False
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
        logger.info("dual_feed_stopped")

    async def _connect_and_listen(self) -> None:
        """Establish RTDS WebSocket connection, subscribe, and process messages."""
        async with websockets.connect(
            self._rtds_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws_connection = ws
            self._retry_count = 0
            logger.info("rtds_ws_connected", url=self._rtds_url)

            # Subscribe to both feeds
            # Feed 1: Binance price via RTDS
            binance_sub = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": "btcusdt",
                }],
            }
            await ws.send(json.dumps(binance_sub))

            # Feed 2: Chainlink oracle price
            chainlink_sub = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": "",
                }],
            }
            await ws.send(json.dumps(chainlink_sub))

            logger.info("rtds_subscriptions_sent", feeds=["crypto_prices", "crypto_prices_chainlink"])

            async for raw_msg in ws:
                if not self._running:
                    break
                self._messages_received += 1

                try:
                    msg = json.loads(raw_msg)
                    self._handle_message(msg)
                except json.JSONDecodeError:
                    logger.warning("rtds_invalid_json")
                except Exception as e:
                    logger.error("rtds_message_error", error=str(e))

    # ── Message Handler ───────────────────────────────────────

    def _handle_message(self, msg: dict) -> None:
        """Parse RTDS message and update price state."""
        topic = msg.get("topic") or msg.get("channel") or ""
        payload = msg.get("payload") or msg.get("data") or {}

        if not topic or not payload:
            return

        # Extract price value (handle multiple payload formats)
        price_val = (
            payload.get("value")
            or payload.get("price")
            or payload.get("last")
            or payload.get("close")
            or payload.get("mark")
        )

        if price_val is None:
            return

        try:
            price = float(price_val)
        except (ValueError, TypeError):
            return

        if price <= 0:
            return

        ts = time.time()
        symbol = str(
            payload.get("symbol") or payload.get("pair") or payload.get("name") or ""
        ).lower()

        if topic == "crypto_prices_chainlink":
            # Chainlink oracle feed — accept btc/usd or any BTC symbol
            if "btc" in symbol or not symbol:
                self._chainlink_price = price
                self._chainlink_ts = ts
                self._record_snapshot()

                logger.debug(
                    "chainlink_tick",
                    price=round(price, 2),
                    symbol=symbol,
                )

        elif topic == "crypto_prices":
            # Binance feed via RTDS
            if "btcusdt" in symbol or "btc" in symbol or not symbol:
                self._binance_price = price
                self._binance_ts = ts
                self._record_snapshot()

                logger.debug(
                    "rtds_binance_tick",
                    price=round(price, 2),
                    symbol=symbol,
                )

    def _record_snapshot(self) -> None:
        """Record snapshot if both prices are available."""
        if self._binance_price is not None and self._chainlink_price is not None:
            self._snapshot_history.append(
                (time.time(), self._binance_price, self._chainlink_price)
            )

    def _compute_backoff_delay(self) -> float:
        """Compute exponential backoff delay."""
        delay = self._initial_delay_s * (self._backoff_multiplier ** self._retry_count)
        return min(delay, self._max_delay_s)

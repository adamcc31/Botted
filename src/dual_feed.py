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
from src.binance_feed import BinanceFeed

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

    def __init__(self, config: ConfigManager, binance_feed: BinanceFeed) -> None:
        self._config = config
        self._binance_feed = binance_feed
        self._rtds_url = config.get("dual_feed.rtds_ws_url", self.RTDS_URL)
        self._rolling_window_s = config.get("dual_feed.rolling_window_seconds", 60)
        import os
        self._stale_threshold_s = int(os.getenv("RTDS_STALE_THRESHOLD_SECONDS", config.get("dual_feed.stale_threshold_seconds", 10)))

        # Reconnection parameters
        self._max_retries = config.get("dual_feed.reconnect_max_retries", 10)
        self._initial_delay_s = config.get("dual_feed.reconnect_initial_delay_s", 1)
        self._max_delay_s = config.get("dual_feed.reconnect_max_delay_s", 30)
        self._backoff_multiplier = 2

        # Latest prices
        self._chainlink_price: Optional[float] = None
        self._chainlink_ts: float = 0.0

        # Rolling window: deque of (timestamp, binance_price, chainlink_price)
        self._snapshot_history: Deque[Tuple[float, float, float]] = deque(
            maxlen=max(60, self._rolling_window_s) * 10  # ~10 ticks/sec headroom
        )

        # Precise Chainlink history for strike price resolution (epoch boundary lookup)
        # 10 ticks/sec * 600s (10 min) = 6000 entries max for lookback safety
        self._chainlink_history: Deque[Tuple[float, float]] = deque(maxlen=6000)

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
        """Binance BTC/USDT price from direct Binance WS feed."""
        return self._binance_feed.latest_price

    @property
    def is_chainlink_stale(self) -> bool:
        """True if Chainlink price is missing or hasn't updated within threshold."""
        if self._chainlink_price is None or self._chainlink_ts == 0.0:
            return True
        return (time.time() - self._chainlink_ts) > self._stale_threshold_s

    @property
    def is_binance_rtds_stale(self) -> bool:
        """True if direct Binance WS feed is stale."""
        return self._binance_feed.is_stale

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

        Returns None if either feed is unavailable or stale (>60s for Chainlink).
        Degraded Mode: Allows Chainlink price if stale < 60s.
        """
        bp = self._binance_feed.latest_price
        if self._chainlink_price is None or bp is None:
            return None
        
        # Binance must be live
        if self._binance_feed.is_stale:
            return None

        age = time.time() - self._chainlink_ts
        if age >= 60.0:
            return None

        source = "CHAINLINK_LIVE" if age <= self._stale_threshold_s else "CHAINLINK_CACHED"

        spread = bp - self._chainlink_price
        spread_pct = abs(spread) / self._chainlink_price * 100.0

        if abs(spread_pct) < 0.001:
            direction = "CONVERGED"
        elif spread > 0:
            direction = "BINANCE_ABOVE"
        else:
            direction = "CHAINLINK_ABOVE"

        return DualFeedSnapshot(
            timestamp=datetime.now(timezone.utc),
            binance_price=bp,
            chainlink_price=self._chainlink_price,
            spread=spread,
            spread_pct=spread_pct,
            spread_direction=direction,
            oracle_source=source
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

    def get_oracle_price_with_source(self) -> Tuple[Optional[float], str]:
        """
        Get oracle price and its source (LIVE, CACHED, or UNAVAILABLE).
        
        Degraded Mode: If stale < 60s, use cached price.
        """
        if self._chainlink_price is None or self._chainlink_ts == 0.0:
            return None, "UNAVAILABLE"
            
        age = time.time() - self._chainlink_ts
        
        if age <= self._stale_threshold_s:
            return self._chainlink_price, "CHAINLINK_LIVE"
            
        if age < 60.0:
            logger.warning("using_cached_chainlink", age_seconds=round(age, 2))
            return self._chainlink_price, "CHAINLINK_CACHED"
            
        return None, "UNAVAILABLE"

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

        # Start heartbeat monitor
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor(), name="rtds_heartbeat")

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
        
        heartbeat_task.cancel()

    async def _heartbeat_monitor(self) -> None:
        """Monitor for silent hangs (WebSocket connected but no data)."""
        heartbeat_interval = 15  # seconds
        while self._running:
            await asyncio.sleep(heartbeat_interval)
            if self._chainlink_ts == 0.0:
                continue
            
            age = time.time() - self._chainlink_ts
            # Stale lebih dari 3x threshold = silent hang
            if age > self._stale_threshold_s * 3:
                logger.warning(
                    "rtds_silent_hang_detected",
                    age_seconds=round(age, 2),
                    threshold_seconds=self._stale_threshold_s,
                    action="forcing_reconnect",
                )
                if self._ws_connection:
                    await self._ws_connection.close()
                    # Reconnect will be handled by the loop in start()

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
            ping_interval=30,  # Heartbeat every 30s
            ping_timeout=10,   # Reconnect if pong not received in 10s
            close_timeout=5,
        ) as ws:
            self._ws_connection = ws
            self._retry_count = 0
            logger.info("rtds_ws_connected", url=self._rtds_url)

            # Subscribe to Chainlink oracle price
            chainlink_sub = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": "",
                }],
            }
            await ws.send(json.dumps(chainlink_sub))

            logger.info("rtds_subscriptions_sent", feeds=["crypto_prices_chainlink"])

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

                # Record precise history for epoch-aligned strike lookups
                self._chainlink_history.append((ts, price))

                logger.debug(
                    "chainlink_tick",
                    price=round(price, 2),
                    symbol=symbol,
                )

    def _record_snapshot(self) -> None:
        """Record snapshot if both prices are available."""
        bp = self._binance_feed.latest_price
        if bp is not None and self._chainlink_price is not None:
            self._snapshot_history.append(
                (time.time(), bp, self._chainlink_price)
            )

    def _compute_backoff_delay(self) -> float:
        """Compute exponential backoff delay."""
        delay = self._initial_delay_s * (self._backoff_multiplier ** self._retry_count)
        return min(delay, self._max_delay_s)

    def get_chainlink_at_epoch(self, epoch_ts: int) -> float | None:
        """
        Retrieve the Chainlink price closest to a specific Unix epoch timestamp.
        Used to resolve official Polymarket strike prices without third-party oracles.
        """
        return self._get_value_n_seconds_ago_at(
            self._chainlink_history,
            target_ts=float(epoch_ts)
        )

    def _get_value_n_seconds_ago_at(
        self,
        history: Deque[Tuple[float, float]],
        target_ts: float,
        tolerance_seconds: int = 30
    ) -> float | None:
        """
        Helper to find the value in history closest to target_ts within tolerance.
        """
        if not history:
            return None
            
        # Linear search from newest to oldest since we usually look back 0-300s
        closest = None
        min_diff = float('inf')
        
        for ts, val in reversed(history):
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                closest = val
            if diff > tolerance_seconds and ts < target_ts:
                # Optimized: history is sorted by ts, so we can stop if we're past tolerance
                break
                
        if closest is not None and min_diff <= tolerance_seconds:
            return closest
            
        return None

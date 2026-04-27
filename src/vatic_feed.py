"""
vatic_feed.py — Real-time strike price feed via Vatic Trading WebSocket.

Connects to wss://api.vatic.trading/ws and subscribes to BTC 5-minute
window_open events. Each event carries the exact strike price ("Price to Beat")
that Polymarket uses for market settlement.

This module is intentionally ISOLATED from dual_feed.py:
  - DualFeed handles tick-by-tick Binance/Chainlink for probability calculations
  - VaticFeed handles epoch boundary strike prices (SOT for market settlement)

Architecture:
  VaticFeed → on_strike_price callback → MarketDiscovery._epoch_strike_cache

Reconnection:
  - Exponential backoff with 60-second cap
  - Health check before each WS connection attempt
  - Graceful handling of Vatic downtime (bot continues with Chainlink fallback)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class VaticFeed:
    """
    Real-time strike price feed via Vatic Trading WebSocket.

    Connects to wss://api.vatic.trading/ws and subscribes to
    window_open events for BTC 5-minute markets.

    The `on_strike_price` callback is invoked with (epoch_ts, price, source)
    for every window_open event, including initial snapshots.
    """

    VATIC_WS_URL = "wss://api.vatic.trading/ws"
    VATIC_HEALTH_URL = "https://api.vatic.trading/health"

    # Reconnection parameters
    MAX_BACKOFF_SECONDS = 60       # Hard cap on retry delay
    INITIAL_BACKOFF_SECONDS = 2    # First retry delay
    HEALTH_CHECK_TIMEOUT = 5       # HTTP health check timeout

    def __init__(
        self,
        on_strike_price: Callable[[int, float, str], None],
        asset: str = "btc",
        market_types: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            on_strike_price: Callback(epoch_ts, price, source).
                             Called for every window_open event.
            asset: Asset to subscribe to (default: "btc").
            market_types: Market types to subscribe to (default: ["5min"]).
        """
        self._on_strike_price = on_strike_price
        self._asset = asset
        self._market_types = market_types or ["5min"]

        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._connected = False
        self._retry_count = 0
        self._messages_received = 0
        self._last_window_open_ts: float = 0.0

    # ── Public Properties ─────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is active."""
        return self._connected

    @property
    def messages_received(self) -> int:
        """Total messages received since start."""
        return self._messages_received

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self) -> None:
        """
        Main loop: connect → subscribe → listen → reconnect on failure.

        Runs indefinitely with exponential backoff (capped at 60s).
        Performs health check before each connection attempt.
        """
        self._running = True
        logger.info("vatic_feed_starting",
                     ws_url=self.VATIC_WS_URL,
                     asset=self._asset,
                     market_types=self._market_types)

        while self._running:
            try:
                # --- Health check before connecting ---
                if not await self._health_check():
                    delay = self._compute_backoff()
                    logger.warning("vatic_health_check_failed",
                                   retry_in_seconds=delay,
                                   attempt=self._retry_count)
                    self._retry_count += 1
                    await asyncio.sleep(delay)
                    continue

                # --- WebSocket connection ---
                await self._connect_and_listen()

            except (ConnectionClosed, ConnectionRefusedError, OSError) as e:
                self._connected = False
                delay = self._compute_backoff()
                logger.warning("vatic_ws_disconnected",
                               error=str(e),
                               error_type=type(e).__name__,
                               retry_in_seconds=delay,
                               attempt=self._retry_count)
                self._retry_count += 1
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                logger.info("vatic_feed_cancelled")
                break

            except Exception as e:
                self._connected = False
                delay = self._compute_backoff()
                logger.error("vatic_ws_unexpected_error",
                             error=str(e),
                             error_type=type(e).__name__,
                             retry_in_seconds=delay,
                             attempt=self._retry_count)
                self._retry_count += 1
                await asyncio.sleep(delay)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("vatic_feed_stopped")

    # ── Internal: Health Check ────────────────────────────────

    async def _health_check(self) -> bool:
        """
        Check Vatic API health before attempting WebSocket connection.

        Returns True if healthy (HTTP 200 + {"ok": true}), False otherwise.
        Prevents wasting connection attempts when Vatic is known-down.
        """
        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_TIMEOUT) as client:
                resp = await client.get(self.VATIC_HEALTH_URL)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok") is True:
                        logger.debug("vatic_health_ok",
                                     service=data.get("service"),
                                     server_time=data.get("now"))
                        return True
                logger.debug("vatic_health_not_ok",
                             status=resp.status_code,
                             body=resp.text[:200])
                return False
        except Exception as e:
            logger.debug("vatic_health_unreachable",
                         error=str(e))
            return False

    # ── Internal: WebSocket ───────────────────────────────────

    async def _connect_and_listen(self) -> None:
        """Connect to Vatic WS, subscribe, and process messages."""
        async with websockets.connect(
            self.VATIC_WS_URL,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
            open_timeout=10,
        ) as ws:
            self._ws = ws
            self._connected = True
            self._retry_count = 0  # Reset on successful connection
            logger.info("vatic_ws_connected", url=self.VATIC_WS_URL)

            async for raw_msg in ws:
                if not self._running:
                    break
                self._messages_received += 1

                try:
                    msg = json.loads(raw_msg)
                    await self._handle_message(msg)
                except json.JSONDecodeError:
                    logger.warning("vatic_ws_invalid_json")
                except Exception as e:
                    logger.error("vatic_ws_message_error",
                                 error=str(e),
                                 error_type=type(e).__name__)

        # Connection closed normally
        self._connected = False
        logger.info("vatic_ws_connection_closed")

    async def _handle_message(self, msg: dict) -> None:
        """Route incoming Vatic WebSocket messages."""
        event = msg.get("event", "")

        if event == "connected":
            # Server ready — send subscription
            supported_assets = msg.get("supportedAssets", [])
            supported_types = msg.get("supportedMarketTypes", [])
            logger.info("vatic_ws_server_hello",
                        supported_assets=supported_assets,
                        supported_types=supported_types)

            sub = {
                "type": "subscribe",
                "asset": self._asset,
                "marketTypes": self._market_types,
            }
            await self._ws.send(json.dumps(sub))
            logger.info("vatic_ws_subscribed",
                        asset=self._asset,
                        market_types=self._market_types)

        elif event == "subscribed":
            logger.info("vatic_ws_subscription_confirmed",
                        asset=msg.get("asset"),
                        market_types=msg.get("marketTypes"))

        elif event == "window_open":
            self._handle_window_open(msg)

        elif event == "window_open_error":
            logger.warning("vatic_window_open_error",
                           asset=msg.get("asset"),
                           market_type=msg.get("marketType"),
                           window_start=msg.get("windowStart"),
                           error=msg.get("error"))

        elif event == "error":
            logger.error("vatic_ws_error_event",
                         message=msg.get("message", str(msg)))

        elif event == "unsubscribed":
            logger.info("vatic_ws_unsubscribed")

        else:
            logger.debug("vatic_ws_unknown_event", event=event)

    def _handle_window_open(self, msg: dict) -> None:
        """
        Process a window_open event — this IS the strike price.

        Event format:
        {
            "event": "window_open",
            "asset": "btc",
            "marketType": "5min",
            "windowStart": 1744027500,
            "windowStartIso": "2025-04-07T12:05:00.000Z",
            "price": 82450.75,
            "source": "chainlink",
            "snapshot": true  (optional, present on initial subscription)
        }
        """
        epoch_ts = msg.get("windowStart")
        price = msg.get("price")
        source = msg.get("source", "chainlink")
        is_snapshot = msg.get("snapshot", False)
        market_type = msg.get("marketType", "")
        window_iso = msg.get("windowStartIso", "")

        if epoch_ts is None or price is None:
            logger.warning("vatic_window_open_missing_fields",
                           raw=str(msg)[:200])
            return

        epoch_ts = int(epoch_ts)
        price = float(price)

        label = "SNAPSHOT" if is_snapshot else "LIVE"
        vatic_source = f"VATIC_WS_{label}"

        logger.info("vatic_strike_received",
                     epoch=epoch_ts,
                     price=price,
                     source=source,
                     type=label,
                     market_type=market_type,
                     window_iso=window_iso)

        self._last_window_open_ts = time.time()

        # Invoke callback — this injects directly into MarketDiscovery cache
        try:
            self._on_strike_price(epoch_ts, price, vatic_source)
        except Exception as e:
            logger.error("vatic_callback_error",
                         epoch=epoch_ts,
                         price=price,
                         error=str(e))

    # ── Internal: Backoff ─────────────────────────────────────

    def _compute_backoff(self) -> float:
        """
        Exponential backoff with hard cap at MAX_BACKOFF_SECONDS (60s).

        Sequence: 2, 4, 8, 16, 32, 60, 60, 60, ...
        """
        delay = self.INITIAL_BACKOFF_SECONDS * (2 ** self._retry_count)
        return min(delay, self.MAX_BACKOFF_SECONDS)

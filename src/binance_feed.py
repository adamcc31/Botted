"""
binance_feed.py — Binance WebSocket + REST data feed.

Subscribes to: btcusdt@depth20@100ms, btcusdt@aggTrade, btcusdt@kline_15m.
Maintains circular buffer of 500 bars + OB snapshots.
Implements exponential backoff reconnection and data validation.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import httpx
try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging
import websockets
from websockets.exceptions import ConnectionClosed

from src.config_manager import ConfigManager
from src.schemas import WSHealthMetrics

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class BinanceFeed:
    """
    Real-time Binance data feed via WebSocket with REST fallback.

    Provides:
    - OHLCV 15-minute bars (circular buffer of 500)
    - Orderbook snapshots (top 20 levels)
    - Aggregated trade data for TFM calculation
    - Connection health metrics
    """

    STREAMS = [
        "btcusdt@depth20@100ms",
        "btcusdt@aggTrade",
        "btcusdt@kline_15m",
        "btcusdt@kline_1m",
    ]

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._ws_base_url = config.get("binance.ws_base_url", "wss://stream.binance.com:9443/stream")
        self._rest_base_url = config.get("binance.rest_base_url", "https://api.binance.com")
        self._ws_base_urls = config.get(
            "binance.ws_base_urls",
            [
                self._ws_base_url,
                "wss://stream.binance.com:443/stream",
                "wss://stream.binance.us:9443/stream",
            ],
        )
        self._rest_base_urls = config.get(
            "binance.rest_base_urls",
            [
                self._rest_base_url,
                "https://api1.binance.com",
                "https://api-gcp.binance.com",
                "https://api.binance.us",
            ],
        )
        if not isinstance(self._ws_base_urls, list):
            self._ws_base_urls = [self._ws_base_url]
        if not isinstance(self._rest_base_urls, list):
            self._rest_base_urls = [self._rest_base_url]
        self._ws_url_index = 0
        self._rest_url_index = 0
        self._buffer_capacity = config.get("binance.buffer_capacity", 500)
        self._buffer_capacity_1m = config.get("binance.buffer_capacity_1m", 3000)
        self._stale_threshold_s = config.get("binance.stale_threshold_s", 30)

        # Reconnection parameters
        self._max_retries = config.get("binance.max_retries", 10)
        self._initial_delay_s = config.get("binance.initial_delay_s", 1)
        self._backoff_multiplier = config.get("binance.backoff_multiplier", 2)
        self._max_delay_s = config.get("binance.max_delay_s", 60)

        # Circular buffers — thread-safe deques
        self._ohlcv_buffer: Deque[Dict[str, Any]] = deque(maxlen=self._buffer_capacity)
        self._ohlcv_1m_buffer: Deque[Dict[str, Any]] = deque(
            maxlen=self._buffer_capacity_1m
        )
        self._ob_buffer: Deque[Dict[str, Any]] = deque(maxlen=self._buffer_capacity)

        # Trade flow accumulator for TFM
        self._trade_buffer: Deque[Dict[str, Any]] = deque(maxlen=10_000)

        # Latest state
        self._latest_ob: Optional[Dict[str, Any]] = None
        self._latest_price: Optional[float] = None
        self._last_message_time: float = 0.0
        self._last_bar_close_time: Optional[datetime] = None

        # Callbacks
        self._on_bar_close: Optional[Callable] = None

        # Health metrics
        self._health = WSHealthMetrics()
        self._ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._retry_count = 0

    # ── Public Properties ─────────────────────────────────────

    @property
    def latest_price(self) -> Optional[float]:
        """Current BTC/USDT price from last aggTrade."""
        return self._latest_price

    @property
    def latest_orderbook(self) -> Optional[Dict[str, Any]]:
        """Latest orderbook snapshot (top 20 levels)."""
        return self._latest_ob

    @property
    def ohlcv_buffer(self) -> List[Dict[str, Any]]:
        """OHLCV bar history as list (newest last)."""
        return list(self._ohlcv_buffer)

    @property
    def ohlcv_1m_buffer(self) -> List[Dict[str, Any]]:
        """1-minute OHLCV bar history as list (newest last)."""
        return list(self._ohlcv_1m_buffer)

    @property
    def ob_buffer(self) -> List[Dict[str, Any]]:
        """Orderbook snapshot history."""
        return list(self._ob_buffer)

    @property
    def trade_buffer(self) -> List[Dict[str, Any]]:
        """Recent trade data for TFM calculation."""
        return list(self._trade_buffer)

    @property
    def health(self) -> WSHealthMetrics:
        return self._health

    @property
    def is_stale(self) -> bool:
        """Check if data is stale (no messages for > threshold)."""
        if self._last_message_time == 0.0:
            return True
        return (time.time() - self._last_message_time) > self._stale_threshold_s

    def set_on_bar_close(self, callback: Callable) -> None:
        """Register callback for bar close events."""
        self._on_bar_close = callback

    # ── REST Bootstrap ────────────────────────────────────────

    async def bootstrap_historical(self, limit: int = 500) -> int:
        """
        Bootstrap OHLCV buffer with historical data from REST API.
        Returns number of bars loaded.
        """
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": limit}
        try:
            resp = await self._rest_get("/api/v3/klines", params=params)
            klines = resp.json()

            for k in klines:
                bar = self._parse_rest_kline(k)
                if bar and self._validate_bar(bar):
                    self._ohlcv_buffer.append(bar)

            logger.info(
                "binance_bootstrap_complete",
                bars_loaded=len(self._ohlcv_buffer),
            )
            return len(self._ohlcv_buffer)

        except httpx.HTTPError as e:
            logger.error("binance_bootstrap_failed", error=str(e))
            return 0

    async def fetch_rest_klines(
        self, limit: int = 500, start_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch historical klines from REST API (for gap-fill)."""
        params: Dict[str, Any] = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time

        resp = await self._rest_get("/api/v3/klines", params=params)
        klines = resp.json()
        return [
            bar
            for k in klines
            if (bar := self._parse_rest_kline(k)) and self._validate_bar(bar)
        ]

    async def get_1m_settlement_price(
        self,
        resolution_time: datetime,
        price_type: str = "close",
        symbol: str = "BTCUSDT",
    ) -> Optional[float]:
        """
        Return the Binance 1-minute candle price aligned to `resolution_time`.

        Settlement correctness requirement:
        - We choose the 1m candle where: candle_open_time <= resolution_time < candle_close_time
        - For now, only `close` is supported. If `price_type` is unknown, we fall back to `close`.
        """
        if resolution_time.tzinfo is None:
            resolution_time = resolution_time.replace(tzinfo=timezone.utc)

        res_ms = int(resolution_time.timestamp() * 1000)
        candle_open_ms = (res_ms // 60000) * 60000
        candle_close_ms = candle_open_ms + 60000

        # Simple per-minute cache to avoid duplicate REST calls.
        if (
            hasattr(self, "_last_settlement_candle_open_ms")
            and getattr(self, "_last_settlement_candle_open_ms") == candle_open_ms
        ):
            return getattr(self, "_last_settlement_price", None)

        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": candle_open_ms,
            "limit": 2,
        }

        resp = await self._rest_get("/api/v3/klines", params=params)
        klines = resp.json()

        bars: list[Dict[str, Any]] = []
        for k in klines:
            bar = self._parse_rest_kline(k)
            if bar and self._validate_bar(bar):
                bars.append(bar)

        chosen: Optional[Dict[str, Any]] = None
        for bar in bars:
            if bar["open_time"] <= res_ms < bar["close_time"]:
                chosen = bar
                break

        if chosen is None and bars:
            # Fallback: use first bar if we couldn't match the time window.
            chosen = bars[0]

        if chosen is None:
            return None

        # Cache
        setattr(self, "_last_settlement_candle_open_ms", candle_open_ms)
        setattr(self, "_last_settlement_price", chosen.get("close"))

        if price_type == "close" or price_type == "vwap" or price_type == "unknown":
            # Binance REST kline doesn't provide VWAP per candle in this endpoint.
            return float(chosen.get("close"))

        return float(chosen.get("close"))

    # ── WebSocket Connection ──────────────────────────────────

    async def start(self) -> None:
        """Start WebSocket connection with reconnection logic."""
        self._running = True
        self._retry_count = 0

        while self._running:
            try:
                await self._connect_and_listen()
            except (ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                self._health.reconnect_count += 1
                self._ws_url_index = (self._ws_url_index + 1) % max(1, len(self._ws_base_urls))
                delay = self._compute_backoff_delay()
                logger.warning(
                    "binance_ws_disconnected",
                    error=str(e),
                    retry_count=self._retry_count,
                    reconnect_delay_s=delay,
                )
                await asyncio.sleep(delay)
                self._retry_count += 1

                if self._retry_count > self._max_retries:
                    logger.error(
                        "binance_ws_max_retries_exceeded",
                        max_retries=self._max_retries,
                    )
                    self._retry_count = 0
                    await asyncio.sleep(self._max_delay_s)
            except Exception as e:
                logger.error("binance_ws_unexpected_error", error=str(e))
                if not self._running:
                    break
                self._ws_url_index = (self._ws_url_index + 1) % max(1, len(self._ws_base_urls))
                await asyncio.sleep(5.0)

    async def stop(self) -> None:
        """Stop WebSocket gracefully."""
        self._running = False
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
        logger.info("binance_feed_stopped")

    async def _connect_and_listen(self) -> None:
        """Establish WS connection and process messages."""
        streams_param = "/".join(self.STREAMS)
        ws_base = self._ws_base_urls[self._ws_url_index % len(self._ws_base_urls)]
        url = f"{ws_base}?streams={streams_param}"

        async with websockets.connect(
            url, ping_interval=20, ping_timeout=10, close_timeout=5
        ) as ws:
            self._ws_connection = ws
            self._retry_count = 0  # Reset on successful connect
            logger.info("binance_ws_connected", streams=self.STREAMS)

            async for raw_msg in ws:
                if not self._running:
                    break
                self._last_message_time = time.time()
                self._health.messages_received += 1

                try:
                    msg = json.loads(raw_msg)
                    stream = msg.get("stream", "")
                    data = msg.get("data", {})

                    if "depth20" in stream:
                        self._handle_depth(data)
                    elif "aggTrade" in stream:
                        self._handle_agg_trade(data)
                    elif "kline" in stream:
                        await self._handle_kline(stream, data)

                except json.JSONDecodeError:
                    logger.warning("binance_ws_invalid_json")
                except Exception as e:
                    logger.error("binance_ws_message_error", error=str(e))

    # ── Message Handlers ──────────────────────────────────────

    def _handle_depth(self, data: Dict[str, Any]) -> None:
        """Process depth20 orderbook snapshot."""
        try:
            bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

            snapshot = {
                "timestamp": datetime.now(timezone.utc),
                "bids": bids,
                "asks": asks,
            }
            self._latest_ob = snapshot
            self._ob_buffer.append(snapshot)

            # Update latest price from mid
            if bids and asks:
                self._latest_price = (bids[0][0] + asks[0][0]) / 2.0

        except (ValueError, IndexError) as e:
            logger.warning("binance_depth_parse_error", error=str(e))

    def _handle_agg_trade(self, data: Dict[str, Any]) -> None:
        """Process aggregated trade for TFM calculation."""
        try:
            trade = {
                "timestamp": datetime.fromtimestamp(
                    data["T"] / 1000.0, tz=timezone.utc
                ),
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "is_buyer_maker": data["m"],  # True = taker sell, False = taker buy
            }
            self._trade_buffer.append(trade)
            self._latest_price = trade["price"]

        except (KeyError, ValueError) as e:
            logger.warning("binance_agg_trade_parse_error", error=str(e))

    async def _handle_kline(self, stream: str, data: Dict[str, Any]) -> None:
        """Process kline (candlestick) data for supported intervals."""
        try:
            kline = data.get("k", {})
            is_closed = kline.get("x", False)

            if is_closed:
                interval = "unknown"
                if "@kline_" in stream:
                    interval = stream.split("@kline_", 1)[1]

                bar = {
                    "open_time": kline["t"],
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                    "close_time": kline["T"],
                    "interval": interval,
                }

                if self._validate_bar(bar):
                    if interval == "15m":
                        self._ohlcv_buffer.append(bar)
                        self._last_bar_close_time = datetime.fromtimestamp(
                            bar["close_time"] / 1000.0, tz=timezone.utc
                        )
                    elif interval == "1m":
                        self._ohlcv_1m_buffer.append(bar)
                        # Keep 15m bar-close callback semantics unchanged:
                        # we only trigger _on_bar_close from 15m candles.
                    else:
                        # Store nothing for unknown intervals.
                        pass
                    logger.info(
                        "binance_bar_closed",
                        close=bar["close"],
                        volume=bar["volume"],
                        buffer_size=len(self._ohlcv_buffer)
                        if interval == "15m"
                        else len(self._ohlcv_1m_buffer),
                    )

                    # Trigger bar close callback
                    if self._on_bar_close and interval == "15m":
                        await self._on_bar_close(bar)
                else:
                    logger.warning("binance_bar_rejected", bar=bar)

        except (KeyError, ValueError) as e:
            logger.warning("binance_kline_parse_error", error=str(e))

    # ── Orderbook Helpers ─────────────────────────────────────

    def get_top_n_ob(self, n: int = 5) -> Optional[Dict[str, Any]]:
        """Get top N levels of latest orderbook."""
        if not self._latest_ob:
            return None
        return {
            "timestamp": self._latest_ob["timestamp"],
            "bids": self._latest_ob["bids"][:n],
            "asks": self._latest_ob["asks"][:n],
        }

    def get_ob_imbalance(self, levels: int = 5) -> Optional[float]:
        """Calculate OBI from latest orderbook snapshot."""
        ob = self.get_top_n_ob(levels)
        if not ob or not ob["bids"] or not ob["asks"]:
            return None

        bid_qty = sum(q for _, q in ob["bids"])
        ask_qty = sum(q for _, q in ob["asks"])
        total = bid_qty + ask_qty

        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total

    def get_depth_ratio(self, levels: int = 3) -> Optional[float]:
        """Calculate depth ratio (bid/ask) for top N levels."""
        ob = self.get_top_n_ob(levels)
        if not ob or not ob["bids"] or not ob["asks"]:
            return None

        bid_size = sum(q for _, q in ob["bids"])
        ask_size = sum(q for _, q in ob["asks"])
        return bid_size / (ask_size + 1e-8)

    def get_binance_spread_bps(self) -> Optional[float]:
        """Calculate spread in bps from best bid/ask."""
        if not self._latest_ob or not self._latest_ob["bids"] or not self._latest_ob["asks"]:
            return None

        best_bid = self._latest_ob["bids"][0][0]
        best_ask = self._latest_ob["asks"][0][0]
        mid = (best_bid + best_ask) / 2.0

        if mid == 0:
            return None
        return (best_ask - best_bid) / mid * 10000.0

    def get_top5_bid_btc(self) -> float:
        """Get total BTC quantity in top 5 bid levels."""
        if not self._latest_ob or not self._latest_ob["bids"]:
            return 0.0
        return sum(q for _, q in self._latest_ob["bids"][:5])

    def get_trade_flow_data(self, window_seconds: int = 60) -> Tuple[float, float]:
        """
        Get taker buy/sell volumes within a time window.
        Returns (taker_buy_volume, taker_sell_volume).
        """
        now = datetime.now(timezone.utc)
        buy_vol = 0.0
        sell_vol = 0.0

        for trade in reversed(list(self._trade_buffer)):
            delta = (now - trade["timestamp"]).total_seconds()
            if delta > window_seconds:
                break
            qty = trade["quantity"]
            if trade["is_buyer_maker"]:
                sell_vol += qty  # is_buyer_maker=True → taker sell
            else:
                buy_vol += qty   # is_buyer_maker=False → taker buy

        return buy_vol, sell_vol

    # ── Validation ────────────────────────────────────────────

    @staticmethod
    def _validate_bar(bar: Dict[str, Any]) -> bool:
        """Validate OHLCV bar data integrity."""
        try:
            o, h, l, c, v = (
                bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]
            )
            if v <= 0:
                return False
            if h < l:
                return False
            if o > h or o < l:
                return False
            if c > h or c < l:
                return False
            if bar["open_time"] >= bar["close_time"]:
                return False
            return True
        except (KeyError, TypeError):
            return False

    @staticmethod
    def _parse_rest_kline(k: list) -> Optional[Dict[str, Any]]:
        """Parse a REST API kline array into bar dict."""
        try:
            return {
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
            }
        except (IndexError, ValueError, TypeError):
            return None

    def _compute_backoff_delay(self) -> float:
        """Compute exponential backoff delay."""
        delay = self._initial_delay_s * (self._backoff_multiplier ** self._retry_count)
        return min(delay, self._max_delay_s)

    async def _rest_get(self, path: str, params: Dict[str, Any]) -> httpx.Response:
        """Try multiple Binance REST bases, rotate on failures like HTTP 451."""
        last_error: Exception | None = None
        n = len(self._rest_base_urls)
        for i in range(n):
            idx = (self._rest_url_index + i) % n
            base = self._rest_base_urls[idx]
            url = f"{base}{path}"
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                self._rest_url_index = idx
                return resp
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code if e.response else None
                logger.warning("binance_rest_failed", base=base, status=status)
                continue
            except Exception as e:
                last_error = e
                logger.warning("binance_rest_failed", base=base, error=str(e))
                continue
        if last_error:
            raise last_error
        raise RuntimeError("All Binance REST endpoints failed")

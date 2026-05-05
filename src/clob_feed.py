"""
clob_feed.py — Polymarket CLOB orderbook polling + cache + stale detection.

REST polling every 5 seconds with retry + exponential backoff.
Constructs CLOBState with liquidity metrics and vig calculation.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict, Deque

import httpx
import structlog
import websockets
from websockets.exceptions import ConnectionClosed

from src.config_manager import ConfigManager
from src.schemas import ActiveMarket, CLOBState

logger = structlog.get_logger(__name__)


class CLOBFeed:
    """
    Polymarket CLOB data feed via REST polling.

    Maintains cached CLOBState with staleness detection.
    """

    CLOB_BASE_URL = "https://clob.polymarket.com"
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._poll_interval = config.get("clob.poll_interval_seconds", 5)
        self._stale_timeout = config.get("clob.stale_threshold_seconds", 30)
        self._min_depth_usd = config.get("clob.min_depth_usd", 10.0)
        self._max_vig = config.get("clob.max_market_vig", 0.07)

        # Retry config
        self._max_retries = 3
        self._backoff_delays = [1, 2, 4]

        # Circuit breaker
        self._max_consecutive_404 = config.get("clob.max_consecutive_404", 3)
        self._consecutive_404_count: int = 0
        self._circuit_breaker_tripped: bool = False

        # State
        self._cached_state: Optional[CLOBState] = None
        self._last_fetch_time: float = 0.0
        self._stale_event_count: int = 0
        self._running = False
        
        # WebSocket / Cache State
        self._cached_books: Dict[str, dict] = {}
        self._last_fetch_time_per_token: Dict[str, float] = {}
        self._rest_rate_limit_s = 2.0
        self._active_tokens: set[str] = set()
        self._ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_task: Optional[asyncio.Task] = None

        # Velocity tracking: history of books per token
        self._history_maxlen = 100
        self._clob_history: Dict[str, Deque[dict]] = {}

    # ── Public Properties ─────────────────────────────────────

    @property
    def clob_state(self) -> Optional[CLOBState]:
        """Current CLOB state (may be cached)."""
        if self._cached_state and self._is_stale():
            return self._cached_state.model_copy(update={"is_stale": True})
        return self._cached_state

    @property
    def stale_event_count(self) -> int:
        return self._stale_event_count

    @property
    def circuit_breaker_tripped(self) -> bool:
        """True when consecutive 404s have reached max_consecutive_404 threshold."""
        return self._circuit_breaker_tripped

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after force_rediscover is triggered."""
        self._consecutive_404_count = 0
        self._circuit_breaker_tripped = False
        logger.info("clob_circuit_breaker_reset")

    # ── Polling Loop ──────────────────────────────────────────

    async def start(self, market: Optional[ActiveMarket] = None) -> None:
        """Start WebSocket loop."""
        self._running = True
        logger.info("clob_ws_started", url=self.WS_URL)
        
        if not self._ws_task or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop())

    async def stop(self) -> None:
        self._running = False
        if self._ws_connection:
            await self._ws_connection.close()
        logger.info("clob_ws_stopped")

    async def _ws_loop(self) -> None:
        """Background loop to receive push data from CLOB WebSocket."""
        while self._running:
            if not self._active_tokens:
                await asyncio.sleep(1)
                continue
                
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws_connection = ws
                    logger.info("clob_ws_connected", tokens=list(self._active_tokens))
                    
                    # V2_MIGRATION: Defensive WS subscription (assets_ids vs market_ids)
                    sub_msg = {
                        "auth": {}, 
                        "type": "MARKET", 
                        "assets_ids": list(self._active_tokens),
                        "market_ids": list(self._active_tokens),
                        "assets": list(self._active_tokens)
                    }
                    await ws.send(json.dumps(sub_msg))
                    
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        
                        try:
                            raw = json.loads(raw_msg)
                            events = raw if isinstance(raw, list) else [raw]
                            
                            for event in events:
                                if not isinstance(event, dict):
                                    continue
                                event_type = event.get("event_type") or event.get("type")
                                if event_type == "book":
                                    token_id = event.get("asset_id")
                                    if token_id and token_id in self._active_tokens:
                                        self._cached_books[token_id] = event
                                        self._last_fetch_time = time.time()
                                        
                                        # Record history for velocity features
                                        if token_id not in self._clob_history:
                                            self._clob_history[token_id] = deque(maxlen=self._history_maxlen)
                                        
                                        self._clob_history[token_id].append({
                                            "timestamp": datetime.now(timezone.utc),
                                            "book": event
                                        })
                                        
                                        logger.info("clob_book_updated", source="websocket", token_id=token_id[:16])
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error("clob_ws_message_error", error=str(e))
                            
            except (ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning("clob_ws_disconnected", error=str(e))
            except Exception as e:
                logger.error("clob_ws_unexpected_error", error=str(e))
            
            self._ws_connection = None
            if self._running:
                await asyncio.sleep(2)

    # ── Snapshot Fetch ────────────────────────────────────────

    async def fetch_clob_snapshot(
        self, market: ActiveMarket
    ) -> Optional[CLOBState]:
        """
        Fetch CLOB orderbook for YES and NO tokens.
        Returns CLOBState with best bid/ask, depth, vig, and liquidity flag.
        """
        yes_token = market.clob_token_ids.get("YES", "")
        no_token = market.clob_token_ids.get("NO", "")

        if not yes_token or not no_token:
            logger.warning("clob_missing_token_ids", market_id=market.market_id)
            return None

        new_tokens = {yes_token, no_token}
        if self._active_tokens != new_tokens:
            logger.info("clob_rotating_market_subscription", old=list(self._active_tokens), new=list(new_tokens))
            self._active_tokens = new_tokens
            self._cached_books.clear()
            if self._ws_connection:
                # Force reconnect to subscribe to new tokens
                await self._ws_connection.close()
                
        # Get from WS cache first
        yes_book = self._cached_books.get(yes_token)
        no_book = self._cached_books.get(no_token)
        
        # Fallback to REST initial fetch
        if not yes_book:
            yes_book = await self._fetch_book(yes_token)
            if yes_book:
                self._cached_books[yes_token] = yes_book
                
        if not no_book:
            no_book = await self._fetch_book(no_token)
            if no_book:
                self._cached_books[no_token] = no_book

        if not yes_book or not no_book:
            # Use cached state if available, flag as potentially stale
            if self._cached_state:
                self._cached_state = self._cached_state.model_copy(
                    update={"is_stale": True}
                )
                if self._is_stale():
                    self._stale_event_count += 1
                    logger.error(
                        "clob_stale",
                        last_fetch_age_s=round(time.time() - self._last_fetch_time, 1),
                    )
                return self._cached_state
            return None

        # Extract best bid/ask
        yes_ask = self._best_ask(yes_book)
        yes_bid = self._best_bid(yes_book)
        no_ask = self._best_ask(no_book)
        no_bid = self._best_bid(no_book)

        # Calculate depth within 3% of ask
        yes_depth = self._calc_depth_near_ask(yes_book, yes_ask, pct=0.03)
        no_depth = self._calc_depth_near_ask(no_book, no_ask, pct=0.03)

        # Market vig
        vig = yes_ask + no_ask - 1.0

        # Liquidity check
        is_liquid = (
            yes_depth >= self._min_depth_usd
            and no_depth >= self._min_depth_usd
            and vig <= self._max_vig
        )

        state = CLOBState(
            market_id=market.market_id,
            timestamp=datetime.now(timezone.utc),
            yes_ask=yes_ask,
            yes_bid=yes_bid,
            no_ask=no_ask,
            no_bid=no_bid,
            yes_depth_usd=yes_depth,
            no_depth_usd=no_depth,
            market_vig=vig,
            is_liquid=is_liquid,
            is_stale=False,
        )

        return state

    async def _fetch_book(self, token_id: str) -> Optional[dict]:
        """
        Fetch orderbook for a single token with retry.

        404 responses are treated as a hard signal that the market has expired:
          - No retry is performed (retrying a dead market wastes time).
          - Consecutive 404 counter is incremented.
          - When counter reaches max_consecutive_404, circuit_breaker_tripped is set,
            signalling main.py to call force_rediscover().
        All other HTTP or connection errors use the existing exponential backoff retry.
        """
        # Rate Limiting (2 seconds per token)
        last_fetch = self._last_fetch_time_per_token.get(token_id, 0.0)
        if time.time() - last_fetch < self._rest_rate_limit_s:
            # We skip the fetch if called too recently to avoid 429
            return None
            
        self._last_fetch_time_per_token[token_id] = time.time()
        
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{self.CLOB_BASE_URL}/book",
                        params={"token_id": token_id},
                    )

                    # ── 404: market expired, no point retrying ────────────
                    if resp.status_code == 404:
                        self._consecutive_404_count += 1
                        logger.warning(
                            "clob_book_not_found",
                            token_id=token_id[:16],
                            consecutive_404s=self._consecutive_404_count,
                        )
                        if self._consecutive_404_count >= self._max_consecutive_404:
                            self._circuit_breaker_tripped = True
                            logger.error(
                                "clob_circuit_breaker_tripped",
                                consecutive_404s=self._consecutive_404_count,
                                threshold=self._max_consecutive_404,
                            )
                        return None  # No retry for 404

                    resp.raise_for_status()

                    # Successful fetch — reset 404 counter
                    self._consecutive_404_count = 0
                    return resp.json()

            except httpx.HTTPStatusError as e:
                # Non-404 HTTP error — retry with backoff
                delay = self._backoff_delays[min(attempt, len(self._backoff_delays) - 1)]
                logger.warning(
                    "clob_fetch_retry",
                    attempt=attempt + 1,
                    token_id=token_id[:16],
                    error=str(e),
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
            except httpx.HTTPError as e:
                # Connection / timeout error — retry with backoff
                delay = self._backoff_delays[min(attempt, len(self._backoff_delays) - 1)]
                logger.warning(
                    "clob_fetch_retry",
                    attempt=attempt + 1,
                    token_id=token_id[:16],
                    error=str(e),
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error("clob_fetch_unexpected", error=str(e))
                break

        logger.error("clob_fetch_exhausted", token_id=token_id[:16])
        return None

    # ── Orderbook Parsing ─────────────────────────────────────

    @staticmethod
    def _best_ask(book: dict) -> float:
        """Extract best (lowest) ask price."""
        # V2_MIGRATION: asks are sorted DESC, best ask is at the end
        asks = book.get("asks") or book.get("ask") or book.get("data", {}).get("asks") or []
        if not asks:
            return 1.0  # No asks → max price
        try:
            best_ask_obj = asks[-1]
            return float(best_ask_obj.get("price", 1.0))
        except (ValueError, TypeError, IndexError, AttributeError):
            return 1.0

    @staticmethod
    def _best_bid(book: dict) -> float:
        """Extract best (highest) bid price."""
        # V2_MIGRATION: bids are sorted DESC, best bid is at the beginning
        bids = book.get("bids") or book.get("bid") or book.get("data", {}).get("bids") or []
        if not bids:
            return 0.0  # No bids → min price
        try:
            best_bid_obj = bids[0]
            return float(best_bid_obj.get("price", 0.0))
        except (ValueError, TypeError, IndexError, AttributeError):
            return 0.0

    @staticmethod
    def _calc_depth_near_ask(book: dict, best_ask: float, pct: float = 0.03) -> float:
        """Calculate total USDC depth within pct% of best ask."""
        # V2_MIGRATION: field name may have changed
        asks = book.get("asks") or book.get("ask") or book.get("data", {}).get("asks") or []
        total_depth = 0.0
        upper_bound = best_ask * (1.0 + pct)

        for a in asks:
            try:
                price = float(a.get("price", 0))
                size = float(a.get("size", 0))
                if price <= upper_bound:
                    total_depth += price * size  # USDC value
            except (ValueError, TypeError):
                continue

        return total_depth

    def _is_stale(self) -> bool:
        """Check if CLOB data is stale."""
        if self._last_fetch_time == 0.0:
            return True
        return (time.time() - self._last_fetch_time) > self._stale_timeout

    def get_historical_book(self, token_id: str, seconds_ago: float) -> Optional[dict]:
        """
        Retrieve orderbook snapshot from approximately N seconds ago.
        Returns the snapshot book or None if history is insufficient.
        """
        history = self._clob_history.get(token_id)
        if not history or len(history) < 2:
            return None

        target_time = datetime.now(timezone.utc).timestamp() - seconds_ago
        
        # Search from newest to oldest for the first snapshot older than target_time
        # Since it's a deque and WebSocket updates are frequent, we just look back
        for item in reversed(history):
            if item["timestamp"].timestamp() <= target_time:
                return item["book"]
        
        # If all items are newer than target_time, return the oldest available
        return history[0]["book"]

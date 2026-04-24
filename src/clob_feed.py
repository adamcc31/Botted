"""
clob_feed.py — Polymarket CLOB orderbook polling + cache + stale detection.

REST polling every 5 seconds with retry + exponential backoff.
Constructs CLOBState with liquidity metrics and vig calculation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from src.config_manager import ConfigManager
from src.schemas import ActiveMarket, CLOBState

logger = structlog.get_logger(__name__)


class CLOBFeed:
    """
    Polymarket CLOB data feed via REST polling.

    Maintains cached CLOBState with staleness detection.
    """

    CLOB_BASE_URL = "https://clob.polymarket.com"

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

    async def start(self, market: ActiveMarket) -> None:
        """Start polling loop for given market."""
        self._running = True
        logger.info(
            "clob_feed_started",
            market_id=market.market_id,
            poll_interval=self._poll_interval,
        )

        while self._running:
            try:
                state = await self.fetch_clob_snapshot(market)
                if state:
                    self._cached_state = state
                    self._last_fetch_time = time.time()
            except Exception as e:
                logger.error("clob_feed_loop_error", error=str(e))

            await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("clob_feed_stopped")

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

        yes_book = await self._fetch_book(yes_token)
        no_book = await self._fetch_book(no_token)

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
        asks = book.get("asks", [])
        if not asks:
            return 1.0  # No asks → max price
        try:
            prices = [float(a.get("price", 1.0)) for a in asks]
            return min(prices) if prices else 1.0
        except (ValueError, TypeError):
            return 1.0

    @staticmethod
    def _best_bid(book: dict) -> float:
        """Extract best (highest) bid price."""
        bids = book.get("bids", [])
        if not bids:
            return 0.0  # No bids → min price
        try:
            prices = [float(b.get("price", 0.0)) for b in bids]
            return max(prices) if prices else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _calc_depth_near_ask(book: dict, best_ask: float, pct: float = 0.03) -> float:
        """Calculate total USDC depth within pct% of best ask."""
        asks = book.get("asks", [])
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
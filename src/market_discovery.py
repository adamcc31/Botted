"""
market_discovery.py — Polymarket market state machine.

States: SEARCHING → ACTIVE → WAITING → SEARCHING
Polls Polymarket Gamma API to discover "Bitcoin Up or Down" 15-minute markets.

CRITICAL DESIGN NOTE (from validation):
- Strike price is STATIC and hardcoded by market creator, NOT Binance price at T_open.
- Must parse strike price from market question text or Gamma API metadata.
- Market intervals are NOT guaranteed to be exactly 15 minutes.
- Resolution oracle varies (Pyth, Coinbase, CoinGecko via UMA).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Literal, Any
import os

import httpx
import structlog

from src.config_manager import ConfigManager
from src.schemas import ActiveMarket

logger = structlog.get_logger(__name__)

SLUG_PREFIX = os.getenv("POLYMARKET_SLUG_PREFIX", "btc-updown-5m")


class DiscoveryState(str, Enum):
    SEARCHING = "SEARCHING"
    ACTIVE = "ACTIVE"
    WAITING = "WAITING"


class MarketDiscovery:
    """
    Polymarket market discovery with state machine.

    Polls Gamma API for active "Bitcoin Up or Down" markets.
    Extracts strike price from market metadata (question text parsing).
    """

    # Base URLs
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"

    # Market identification patterns
    MARKET_PATTERNS = [
        r"bitcoin.*up.*or.*down",
        r"btc.*up.*or.*down",
        r"bitcoin.*above.*\$",
        r"btc.*above.*\$",
        r"bitcoin.*reach.*\$",
        r"btc.*reach.*\$",
        r"bitcoin.*dip.*\$",
        r"btc.*dip.*\$",
        r"btc.*5.*minute.*up.*or.*down",
        r"bitcoin.*5.*min",
        r"btc.*5min",
    ]

    # Strike price extraction patterns
    STRIKE_PATTERNS = [
        r"\$([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",  # $66,500.00 or $66500
        r"above\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
        r"below\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
        r"reach\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
        r"dip(?:\s+to)?\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
        r"up\s+or\s+down\s+from\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    ]

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._state = DiscoveryState.SEARCHING
        self._active_market: Optional[ActiveMarket] = None
        self._active_since: Optional[datetime] = None
        self._last_trade_at: Optional[datetime] = None
        self._poll_interval = config.get("market_discovery.poll_interval_s", 30)
        self._waiting_poll = config.get("market_discovery.waiting_poll_s", 60)
        self._min_ttr = config.get("market_discovery.min_ttr_to_discover", 5.0)
        self._late_ttr = config.get("market_discovery.late_ttr_minutes", 3.0)
        self._candidate_pool_size = config.get("market_discovery.candidate_pool_size", 5)
        self._rotation_score_buffer = config.get("market_discovery.rotation_score_buffer", 0.03)
        self._target_yes_prob = config.get("market_discovery.target_yes_probability", 0.5)
        self._target_ttr_minutes = config.get("market_discovery.target_ttr_minutes", 5.0)
        self._running = False
        self._last_log_time: float = 0.0
        self._candidate_pool: list[dict[str, Any]] = []
        self._vatic_cache: dict[str, Any] = {"epoch": None, "price": None}

    # ── Public Properties ─────────────────────────────────────

    @property
    def state(self) -> DiscoveryState:
        return self._state

    @property
    def active_market(self) -> Optional[ActiveMarket]:
        return self._active_market

    @property
    def is_market_active(self) -> bool:
        return self._state == DiscoveryState.ACTIVE and self._active_market is not None

    # ── State Machine ─────────────────────────────────────────

    async def start(self) -> None:
        """Start the discovery state machine loop."""
        self._running = True
        logger.info("market_discovery_started", state=self._state.value)

        while self._running:
            try:
                if self._state == DiscoveryState.SEARCHING:
                    await self._handle_searching()
                elif self._state == DiscoveryState.ACTIVE:
                    await self._handle_active()
                elif self._state == DiscoveryState.WAITING:
                    await self._handle_waiting()
            except Exception as e:
                logger.error("market_discovery_error", error=str(e), state=self._state.value)
                await asyncio.sleep(10.0)

    async def stop(self) -> None:
        self._running = False
        logger.info("market_discovery_stopped")

    async def _sleep_epoch_synchronized(self, base_interval: float) -> None:
        """
        Epoch Tracking & Latency Mitigation:
        Polymarket 5-minute markets refresh exactly on the 5-minute bounds (e.g., :00, :05, :10).
        This method predicts the sub-market transition and tightens the polling rate
        to 500ms loops during the critical transition window to beat API latency.
        """
        now = datetime.now(timezone.utc)
        minutes_past = now.minute % 5
        seconds_past = minutes_past * 60 + now.second + (now.microsecond / 1000000.0)
        seconds_to_epoch = (5 * 60) - seconds_past

        # 1. Active Transition Window latency mitigation (0 to 3 seconds past epoch)
        if seconds_past < 3.0:
            await asyncio.sleep(0.5)
            return

        # 2. VATIC PRE-FETCH WINDOW (Wake up EXACTLY at T-30s)
        # Jika waktu menuju epoch lebih dari 30s, dan sisa waktunya menuju T-30s lebih kecil dari interval normal
        if seconds_to_epoch > 30.0 and (seconds_to_epoch - 30.0) <= base_interval:
            await asyncio.sleep((seconds_to_epoch - 30.0) + 0.05)
            return

        # 3. Precision Timer for exactly T-0
        if seconds_to_epoch <= base_interval:
            await asyncio.sleep(seconds_to_epoch + 0.05)
            return

        # 4. Default Polling
        await asyncio.sleep(base_interval)

    async def _handle_searching(self) -> None:
        """Poll for active markets."""
        market = await self._find_active_market()
        if market:
            self._active_market = market
            self._active_since = datetime.now(timezone.utc)
            self._state = DiscoveryState.ACTIVE
            logger.info(
                "market_discovered",
                market_id=market.market_id,
                strike_price=market.strike_price,
                TTR_minutes=market.TTR_minutes,
                resolution_source=market.resolution_source,
            )
        else:
            # Cegah transisi ke WAITING jika kita menjalankan strategi 5m dynamic
            has_dynamic_targets = bool(self._config.get("market_discovery.dynamic_5m_event_slugs", []))
            if has_dynamic_targets:
                # Tetap di SEARCHING mode, stalking oracle price
                logger.debug("dynamic_market_stalking", msg="Staying in SEARCHING state to monitor Oracle")
            else:
                self._state = DiscoveryState.WAITING
                logger.info("no_active_market_found", transitioning_to="WAITING")
        await self._sleep_epoch_synchronized(self._poll_interval)

    async def _handle_active(self) -> None:
        """Monitor active market TTR and validity."""
        if not self._active_market:
            self._state = DiscoveryState.SEARCHING
            return

        now = datetime.now(timezone.utc)
        ttr_seconds = (self._active_market.T_resolution - now).total_seconds()
        ttr_minutes = ttr_seconds / 60.0

        if ttr_seconds <= 0:
            # Market has resolved
            logger.info(
                "market_resolved",
                market_id=self._active_market.market_id,
            )
            self._active_market = None
            self._state = DiscoveryState.SEARCHING
        elif ttr_minutes < self._late_ttr:
            # Mark as LATE — no new entries allowed
            logger.info(
                "market_late_phase",
                market_id=self._active_market.market_id,
                TTR_minutes=round(ttr_minutes, 2),
            )
            # Update TTR on market
            self._active_market = self._active_market.model_copy(
                update={"TTR_minutes": ttr_minutes}
            )
        else:
            # Update TTR
            self._active_market = self._active_market.model_copy(
                update={"TTR_minutes": ttr_minutes}
            )

        await self._sleep_epoch_synchronized(self._poll_interval)

    async def _handle_waiting(self) -> None:
        """Wait mode — poll less frequently."""
        import time as _time

        now = _time.time()
        if now - self._last_log_time > 300:  # Log every 5 minutes
            logger.info("market_discovery_waiting_mode")
            self._last_log_time = now

        market = await self._find_active_market()
        if market:
            self._active_market = market
            self._active_since = datetime.now(timezone.utc)
            self._state = DiscoveryState.ACTIVE
            logger.info(
                "market_found_from_waiting",
                market_id=market.market_id,
                strike_price=market.strike_price,
                TTR_minutes=market.TTR_minutes,
            )
        await self._sleep_epoch_synchronized(self._waiting_poll)

    def force_rediscover(self) -> None:
        """
        Force immediate reset to SEARCHING state.
        Called by main.py when CLOBFeed circuit breaker trips (consecutive 404s).
        Safe to call from any state.
        """
        logger.warning(
            "force_rediscover_triggered",
            previous_state=self._state.value,
            previous_market=self._active_market.market_id if self._active_market else None,
        )
        self._active_market = None
        self._active_since = None
        self._state = DiscoveryState.SEARCHING

    def mark_trade_executed(self) -> None:
        """Called by orchestrator when a trade is actually opened."""
        self._last_trade_at = datetime.now(timezone.utc)

    async def check_and_rotate(self) -> bool:
        """
        Hybrid rotation check — called on every 15-minute bar close.

        Rotation criteria (two-stage):
          1. TTR gate: candidate.TTR > current.TTR + rotation_ttr_buffer_minutes
             (prevents excessive switching between markets with similar lifetimes)
          2. Liquidity tiebreaker: among qualifying candidates, prefer highest
             Gamma volume (avoids switching to a longer-lived but illiquid market)

        Returns True if rotation occurred, False otherwise.
        By being called at bar close (not on an independent timer), this ensures
        market switches never interrupt a Z-score computation mid-window.
        """
        if not self._active_market:
            return False

        now = datetime.now(timezone.utc)
        # Rotation lock 1: minimum dwell time on active market
        min_dwell = float(self._config.get("rotation.min_dwell_minutes", 20.0))
        if self._active_since is not None:
            dwell_minutes = (now - self._active_since).total_seconds() / 60.0
            if dwell_minutes < min_dwell:
                logger.info(
                    "rotation_locked_dwell",
                    dwell_minutes=round(dwell_minutes, 2),
                    min_dwell_minutes=min_dwell,
                    active_market_id=self._active_market.market_id,
                )
                return False

        # Rotation lock 2: freeze while current market is still in valid entry window
        freeze_entry_window = bool(
            self._config.get("rotation.freeze_when_in_entry_window", True)
        )
        if freeze_entry_window:
            ttr_min, ttr_max = self._resolve_signal_ttr_window(self._active_market)
            cur_ttr = float(self._active_market.TTR_minutes)
            if ttr_min <= cur_ttr <= ttr_max:
                logger.info(
                    "rotation_locked_entry_window",
                    market_id=self._active_market.market_id,
                    TTR_minutes=round(cur_ttr, 2),
                    ttr_min=ttr_min,
                    ttr_max=ttr_max,
                )
                return False

        # Rotation lock 3: cooldown after a real trade
        cooldown = float(self._config.get("rotation.cooldown_after_trade_minutes", 0.0))
        if cooldown > 0 and self._last_trade_at is not None:
            since_trade = (now - self._last_trade_at).total_seconds() / 60.0
            if since_trade < cooldown:
                logger.info(
                    "rotation_locked_trade_cooldown",
                    minutes_since_trade=round(since_trade, 2),
                    cooldown_minutes=cooldown,
                    active_market_id=self._active_market.market_id,
                )
                return False

        candidates = await self._query_candidates()
        if not candidates:
            return False

        current = next(
            (c for c in candidates if c["market"].market_id == self._active_market.market_id),
            None,
        )
        current_score = current["score"] if current else -1.0

        better = [
            c
            for c in candidates
            if c["market"].market_id != self._active_market.market_id
            and c["score"] > current_score + self._rotation_score_buffer
        ]

        if not better:
            return False

        best = max(better, key=lambda c: c["score"])
        new_market = best["market"]

        logger.info(
            "market_rotated",
            old_market_id=self._active_market.market_id,
            old_ttr_minutes=round(self._active_market.TTR_minutes, 2),
            new_market_id=new_market.market_id,
            new_ttr_minutes=round(new_market.TTR_minutes, 2),
            new_volume_usd=round(best["volume"], 2),
            old_score=round(current_score, 4),
            new_score=round(best["score"], 4),
        )

        self._active_market = new_market
        self._active_since = datetime.now(timezone.utc)
        return True

    # ── Market Discovery Logic ────────────────────────────────

    async def _find_active_market(self) -> Optional[ActiveMarket]:
        """
        Query Gamma API and return the single best candidate.
        Thin wrapper around _query_candidates for use by the state machine.
        """
        candidates = await self._query_candidates()
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c["score"])
        self._candidate_pool = candidates[: self._candidate_pool_size]
        logger.info(
            "candidate_pool_ranked",
            pool_size=len(self._candidate_pool),
            top_market_id=best["market"].market_id,
            top_score=round(best["score"], 4),
            top_volume=round(best["volume"], 2),
        )
        return best["market"]

    # ── Dynamic 5-Minute Market Discovery ────────────────────────

    async def _query_dynamic_5m_markets(self, spot_price: Optional[float]) -> list[dict]:
        """
        Targeted discovery for dynamic short-interval markets (e.g. BTC Up/Down 5-min).

        WHY THIS EXISTS:
        - Dynamic 5-min markets have only ~5 min of lifespan, so they NEVER accumulate
          enough volume24hr to rank in a generic volume-sorted scan (top-200).
        - The Gamma /markets?order=volume24hr endpoint will never surface them.
        - Fix: query the known parent event by slug via /events, then extract the
          currently active sub-market directly.

        ALSO FIXES:
        - `startDateIso` on sub-markets is the *event* creation date (days ago),
          not the current 5-min window open time.  We synthesize a correct T_open
          as T_resolution − 5 min so that the lifespan check in _parse_market passes.
        """
        event_slugs: list[str] = self._config.get(
            "market_discovery.dynamic_5m_event_slugs",
            [SLUG_PREFIX],
        )
        candidates: list[dict] = []

        async with httpx.AsyncClient(timeout=15.0) as client:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            for base_slug in event_slugs:
                window_seconds = 300
                if "15m" in base_slug:
                    window_seconds = 900
                elif "1h" in base_slug:
                    window_seconds = 3600

                current_window_ts = (now_ts // window_seconds) * window_seconds

                for offset in (0, 1):
                    window_ts = current_window_ts + (offset * window_seconds)
                    slug = f"{base_slug.strip()}-{window_ts}"

                    try:
                        # 1. Path-based lookup
                        resp = await client.get(
                            f"{self.GAMMA_API_BASE}/events/slug/{slug}",
                            headers={"Accept": "application/json"}
                        )
                        
                        if resp.is_success:
                            payload = resp.json()
                            if payload and isinstance(payload, dict) and payload.get("slug"):
                                event = payload
                            else:
                                continue
                        else:
                            # 2. Query fallback lookup
                            resp = await client.get(
                                f"{self.GAMMA_API_BASE}/events",
                                params={"slug": slug, "limit": 1},
                                headers={"Accept": "application/json"}
                            )
                            if not resp.is_success:
                                continue
                            
                            payload = resp.json()
                            if isinstance(payload, dict):
                                payload = payload.get("data", [payload])
                            if not isinstance(payload, list) or not payload:
                                continue
                            event = payload[0]

                        sub_markets: list[dict] = event.get("markets", [])

                        if not sub_markets:
                            logger.warning("dynamic_5m_event_has_no_markets", slug=slug,
                                           event_id=event.get("id"))
                            continue
    
                        for m in sub_markets:
                            # Basic activity check
                            if not m.get("active") or m.get("closed"):
                                continue
                            if not m.get("enableOrderBook"):
                                continue
    
                            # --- Patch T_open: use T_resolution − 5 min ---
                            # The event-level startDate is the series creation date,
                            # NOT the current 5-min window open time. Fix it here so
                            # _parse_market's lifespan check does not reject the market.
                            end_date_str = (
                                m.get("end_date_iso") or m.get("endDateIso") or m.get("endDate", "")
                            )
                            T_res = self._parse_timestamp(end_date_str)
                            m_patched = dict(m)
                            if T_res is not None:
                                synthetic_open = T_res - timedelta(minutes=5)
                                iso_open = synthetic_open.isoformat()
                                # Overwrite all possible startDate field names
                                m_patched["startDateIso"] = iso_open
                                m_patched["startDate"] = iso_open
                                m_patched["createdAt"] = iso_open

                            # --- VATIC ORACLE HYDRATION ---
                            # window_ts adalah tepat waktu awal epoch (timestamp)
                            vatic_strike = await self._fetch_vatic_strike(window_ts)
                            if vatic_strike:
                                m_patched["groupItemThreshold"] = str(vatic_strike)
    
                            parsed = self._parse_market(m_patched)
                            if parsed is None:
                                logger.info(
                                    "dynamic_5m_parse_returned_none",
                                    slug=slug,
                                    market_id=m.get("conditionId") or m.get("id"),
                                    question=m.get("question", "")[:80],
                                )
                                continue

                            # --- FILTER TIME-TO-RESOLUTION (TTR) KETAT ---
                            # Jangan pertimbangkan epoch yang sisa waktunya di bawah 4 menit
                            if parsed.TTR_minutes < 4.0:
                                logger.debug(
                                    "dynamic_5m_skipped_too_late", 
                                    market_id=parsed.market_id, 
                                    TTR=round(parsed.TTR_minutes, 2)
                                )
                                continue
    
                            if not parsed.clob_token_ids.get("YES") or not parsed.clob_token_ids.get("NO"):
                                continue
    
                            # 5-min markets have negligible 24hr volume by design;
                            # use total volume (all-time) as the liquidity signal instead.
                            volume = float(
                                m.get("volume") or m.get("volumeNum", 0.0) or
                                m.get("volume24hr", 0.0) or 0.0
                            )
                            yes_prob = self._extract_yes_probability(m)
                            score_components = self._score_candidate(
                                market=parsed,
                                volume_24h=volume,
                                yes_prob=yes_prob,
                                spot_price=spot_price,
                            )
                            candidates.append(
                                {
                                    "market": parsed,
                                    "volume": volume,
                                    "yes_prob": yes_prob,
                                    "score": score_components["score_total"],
                                    "score_components": score_components,
                                    "source": f"event:{slug}",
                                }
                            )
                            logger.info(
                                "dynamic_5m_candidate_found",
                                market_id=parsed.market_id,
                                question=parsed.question[:80],
                                TTR_minutes=round(parsed.TTR_minutes, 2),
                                strike_price=parsed.strike_price,
                                yes_prob=yes_prob,
                                slug=slug,
                            )
                    except httpx.HTTPError as e:
                        logger.warning("dynamic_5m_event_http_error", slug=slug, error=str(e))
                    except Exception as e:
                        logger.warning("dynamic_5m_event_parse_error", slug=slug, error=str(e))

        return candidates

    async def _query_candidates(self) -> list[dict]:
        """
        Query Gamma API via /markets endpoint for active price-action targets.
        Uses a high limit (200) to find daily markets even if buried in volume.

        ALSO merges results from _query_dynamic_5m_markets() — a targeted event-based
        lookup for short-interval markets that never surface via volume-sorted scans.
        """
        min_volume = self._config.get("market_discovery.min_volume_24hr", 1000.0)
        spot_price = await self._fetch_binance_spot()

        try:
            # ── Path A: targeted event-based lookup for 5-min dynamic markets ──
            dynamic_candidates = await self._query_dynamic_5m_markets(spot_price)

            # ── Path B: generic volume-sorted scan for daily/hourly markets ──
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self.GAMMA_API_BASE}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "order": "volume24hr",
                        "ascending": "false",
                        "limit": 200
                    },
                )
                resp.raise_for_status()
                markets = resp.json()

                if not isinstance(markets, list):
                    markets = markets.get("data", []) if isinstance(markets, dict) else []

                candidates = list(dynamic_candidates)  # seed with dynamic results
                seen_ids = {c["market"].market_id for c in candidates}
                skipped_reasons = []  # Log top 5 skipped for transparency

                for m in markets:
                    q = m.get("question", "")

                    # 1. Basic Technical Filter
                    if not m.get("active") or m.get("closed") or not m.get("enableOrderBook"):
                        continue

                    # 2. Design Pattern Check (Strict Above/Up/Down)
                    if not self._is_btc_up_down_market(m):
                        if len(skipped_reasons) < 5 and ("bitcoin" in q.lower() or "btc" in q.lower()):
                            skipped_reasons.append(f"PA-Mismatch: '{q[:50]}...'")
                        continue

                    # 3. Volume Check — skip for 5-min dynamic markets (handled in Path A)
                    volume = float(m.get("volume24hr", 0.0) or m.get("volume", 0.0) or 0.0)
                    is_dynamic = "up or down - 5 minute" in q.lower() or "up or down - 5 min" in q.lower()
                    if not is_dynamic and volume < min_volume:
                        if len(skipped_reasons) < 5:
                            skipped_reasons.append(f"Low-Volume: '${volume:,.0f}' for '{q[:50]}...'")
                        continue

                    parsed = self._parse_market(m)
                    if not parsed:
                        if len(skipped_reasons) < 5 and ("bitcoin" in q.lower() or "btc" in q.lower()):
                            skipped_reasons.append(f"ParseFailed: '{q[:50]}...'")
                        continue

                    # Deduplicate against dynamic results
                    if parsed.market_id in seen_ids:
                        continue
                    seen_ids.add(parsed.market_id)

                    # Bypass strict TTR minimum for 5-minute dynamic markets
                    is_5m_target = "5 minute" in parsed.question.lower()
                    if is_5m_target or parsed.TTR_minutes >= self._min_ttr:
                        # Ensure CLOB tradability: must have YES/NO token IDs.
                        if not parsed.clob_token_ids.get("YES") or not parsed.clob_token_ids.get("NO"):
                            if len(skipped_reasons) < 5:
                                skipped_reasons.append(f"Missing-Token-IDs: '{q[:50]}...'")
                            continue

                    # Basis-risk policy: only hard-skip if explicitly configured.
                    non_binance_policy = self._config.get(
                        "settlement.non_binance_policy", "uncertainty_inflate"
                    )
                    non_binance_is_mismatch = not (
                        parsed.settlement_exchange == "BINANCE"
                        and parsed.settlement_granularity == "1m"
                    )
                    if non_binance_policy == "abstain" and non_binance_is_mismatch:
                        continue

                    yes_prob = self._extract_yes_probability(m)
                    score_components = self._score_candidate(
                        market=parsed,
                        volume_24h=volume,
                        yes_prob=yes_prob,
                        spot_price=spot_price,
                    )
                    candidates.append(
                        {
                            "market": parsed,
                            "volume": volume,
                            "yes_prob": yes_prob,
                            "score": score_components["score_total"],
                            "score_components": score_components,
                            "source": "volume_scan",
                        }
                    )

                if not candidates and skipped_reasons:
                    logger.info(
                        "discovery_skipped_candidates",
                        top_reasons=skipped_reasons,
                        total_markets_scanned=len(markets),
                        dynamic_candidates_found=len(dynamic_candidates),
                    )

                candidates.sort(key=lambda c: c["score"], reverse=True)
                return candidates

        except httpx.HTTPError as e:
            logger.error("gamma_api_error_markets", error=str(e))
            return []
        except Exception as e:
            logger.error("market_discovery_parse_error_markets", error=str(e))
            return []

    def _is_btc_up_down_market(self, market_data: dict) -> bool:
        """Check if market matches Bitcoin Up/Down patterns."""
        question = market_data.get("question", "").lower()
        description = market_data.get("description", "").lower()
        text = f"{question} {description}"

        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.MARKET_PATTERNS
        )

    def _parse_market(self, market_data: dict) -> Optional[ActiveMarket]:
        """
        Parse market JSON into ActiveMarket schema.
        CRITICAL: Strike price extracted from question text / metadata.
        """
        try:
            market_id = (
                market_data.get("conditionId")
                or market_data.get("condition_id")
                or market_data.get("id", "")
            )
            slug = market_data.get("slug", "")
            question = market_data.get("question", "")
            group_item = market_data.get("groupItemTitle", "")

            # ----------------------------------------------------
            # DYNAMIC 5-MIN DETECTION
            # ----------------------------------------------------
            # Be permissive: catch any variant of "up or down" with "5 min"
            # in either the question or groupItemTitle field.
            _detect_text = f"{question} {group_item}".lower()
            is_dynamic_5m = (
                ("up or down" in _detect_text and "5 min" in _detect_text)
                or "up or down - 5 minutes" in _detect_text
                or SLUG_PREFIX in str(market_data.get("slug", "")).lower()
            )

            # Parse timestamps early to enforce horizon limits
            end_date_str = market_data.get("end_date_iso") or market_data.get("endDate", "")
            created_str = market_data.get("startDateIso") or market_data.get("startDate") or market_data.get("createdAt", "")

            if not end_date_str or not created_str:
                return None

            T_resolution = self._parse_timestamp(end_date_str)
            T_open = self._parse_timestamp(created_str)

            if T_resolution is None or T_open is None:
                return None

            lifespan_minutes = (T_resolution - T_open).total_seconds() / 60.0
            
            # Enforce strictly short-horizon markets, BUT BYPASS NO-STRIKE 5-MINUTE MARKETS!
            # Polymarket mints dynamic markets early, so their lifespan_minutes > 15!
            if not is_dynamic_5m:
                target_horizons = self._config.get("market_discovery.target_horizons_minutes", [5.0])
                max_horizon = max(target_horizons) * 3.0  # Max 15 minutes for a 5min target
                if lifespan_minutes > max_horizon:
                    # logger.debug("skipped_long_horizon", market_id=market_id, lifespan=lifespan_minutes)
                    return None  # Silently skip long-horizon markets

            now = datetime.now(timezone.utc)
            ttr_minutes = (T_resolution - now).total_seconds() / 60.0

            if ttr_minutes <= 0:
                return None

            strike_price = None

            if is_dynamic_5m:
                # API Payload Extraction: grab the Price To Beat from raw JSON values
                raw_target = (
                    market_data.get("groupItemThreshold") or 
                    market_data.get("initial_price") or 
                    market_data.get("strike_price")
                )
                if raw_target is not None:
                    try:
                        extracted = float(raw_target)
                        # Ensure it's populated and not exactly 0 (which may happen precisely at 00:00 before oracle update)
                        if extracted > 1000.0: 
                            strike_price = extracted
                    except (ValueError, TypeError):
                        pass

            # ----------------------------------------------------
            # TEXT REGEX EXTRACTION (Fallback / Standard)
            # ----------------------------------------------------
            if strike_price is None:
                strike_price = self._extract_strike_price(question)
            if strike_price is None:
                strike_price = self._extract_strike_price(group_item)
            if strike_price is None:
                desc = market_data.get("description", "")
                strike_price = self._extract_strike_price(desc)
                
            if strike_price is None:
                question_l = question.lower()
                if is_dynamic_5m:
                    # Not an unsupported market, just waiting for the API to lock the price!
                    logger.info("dynamic_strike_pending", market_id=market_id, msg="Waiting for oracle price to beat")
                elif "up or down" in question_l and "from $" not in question_l:
                    logger.info(
                        "unsupported_market_type_no_strike",
                        market_id=market_id,
                        question=question,
                        lifespan=lifespan_minutes
                    )
                else:
                    logger.warning(
                        "strike_price_not_found",
                        market_id=market_id,
                        question=question,
                    )

            if strike_price is None:
                return None

            # Extract CLOB token IDs
            tokens = market_data.get("tokens", [])
            clob_token_ids = self._extract_token_ids(tokens, market_data)

            (
                settlement_exchange,
                settlement_instrument,
                settlement_granularity,
                settlement_price_type,
                resolution_source,
            ) = self._extract_settlement_descriptor(market_data)

            # Explicit override for dynamic 5m markets where we KNOW the source
            # but Polymarket metadata might be empty/generic.
            if is_dynamic_5m:
                settlement_exchange = "BINANCE"
                settlement_granularity = "1m"
                settlement_instrument = "BTCUSDT"
                settlement_price_type = "close"
                resolution_source = "Binance"

            return ActiveMarket(
                market_id=market_id,
                slug=slug,
                question=question,
                strike_price=strike_price,
                T_open=T_open,
                T_resolution=T_resolution,
                TTR_minutes=ttr_minutes,
                clob_token_ids=clob_token_ids,
                settlement_exchange=settlement_exchange,
                settlement_instrument=settlement_instrument,
                settlement_granularity=settlement_granularity,
                settlement_price_type=settlement_price_type,
                resolution_source=resolution_source,
            )

        except Exception as e:
            logger.warning(
                "market_parse_failed",
                error=str(e),
                market_id=market_data.get("condition_id", "unknown"),
            )
            return None

    def _extract_strike_price(self, text: str) -> Optional[float]:
        """
        Extract strike price from market question/description text.

        DESIGN NOTE: Strike price is static and set by market creator.
        Examples:
          "Will Bitcoin be above $66,500.00 at 4:15 PM ET?" → 66500.00
          "Bitcoin Up or Down from $98,450?" → 98450.00
        """
        for pattern in self.STRIKE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(",", "")
                try:
                    price = float(price_str)
                    # Sanity check: BTC price should be reasonable
                    if 1_000 < price < 1_000_000:
                        return price
                except ValueError:
                    continue
        return None

    def _extract_yes_probability(self, market_data: dict) -> Optional[float]:
        """Best-effort parse YES implied probability from Gamma payload."""
        outcomes_raw = market_data.get("outcomes")
        prices_raw = market_data.get("outcomePrices")
        try:
            outcomes = outcomes_raw if isinstance(outcomes_raw, list) else json.loads(outcomes_raw or "[]")
            prices = prices_raw if isinstance(prices_raw, list) else json.loads(prices_raw or "[]")
            if not isinstance(outcomes, list) or not isinstance(prices, list):
                return None
            idx = None
            for i, name in enumerate(outcomes):
                if str(name).strip().upper() == "YES":
                    idx = i
                    break
            if idx is None or idx >= len(prices):
                return None
            p = float(prices[idx])
            return p if 0.0 <= p <= 1.0 else None
        except Exception:
            return None

    def _score_candidate(
        self,
        market: ActiveMarket,
        volume_24h: float,
        yes_prob: Optional[float],
        spot_price: Optional[float],
    ) -> dict[str, float]:
        """
        Rank multiple tradable markets:
        - liquidity signal from log(volume)
        - probability proximity to configurable target (default 0.5)
        - TTR proximity to target duration
        - strike rationality vs spot
        - horizon alignment fit
        """
        volume_score = max(0.0, min(1.0, math.log1p(max(0.0, volume_24h)) / 12.0))

        if yes_prob is None:
            prob_score = 0.5
        else:
            prob_score = max(0.0, 1.0 - abs(yes_prob - self._target_yes_prob) * 2.0)

        ttr_delta = abs(market.TTR_minutes - self._target_ttr_minutes)
        ttr_score = max(0.0, 1.0 - (ttr_delta / max(30.0, self._target_ttr_minutes)))

        # Additional rationality score: strike should not be absurdly far from spot
        strike_score = 0.5
        horizon_score = 0.5
        hard_penalty = 0.0
        if spot_price is not None and spot_price > 0:
            strike_dist_pct = abs(market.strike_price - spot_price) / spot_price
            strike_soft_cap = float(
                self._config.get("market_discovery.strike_distance_soft_cap_pct", 0.20)
            )
            strike_hard_cap = float(
                self._config.get("market_discovery.strike_distance_hard_cap_pct", 0.50)
            )
            strike_score = max(0.0, 1.0 - (strike_dist_pct / max(1e-6, strike_soft_cap)))
            if strike_dist_pct > strike_hard_cap:
                hard_penalty = float(
                    self._config.get("market_discovery.hard_penalty_absurd_strike", 0.30)
                )

        target_horizons = self._config.get(
            "market_discovery.target_horizons_minutes", [60.0, 240.0, 480.0, 720.0]
        )
        if isinstance(target_horizons, list) and target_horizons:
            try:
                targets = [float(v) for v in target_horizons if float(v) > 0]
            except Exception:
                targets = [60.0, 240.0, 480.0, 720.0]
            nearest = min(abs(market.TTR_minutes - t) for t in targets) if targets else 0.0
            denom = max(30.0, min(targets) if targets else 60.0)
            horizon_score = max(0.0, 1.0 - (nearest / denom))

        w_volume = float(self._config.get("market_discovery.weight_volume", 0.25))
        w_prob = float(self._config.get("market_discovery.weight_prob", 0.20))
        w_ttr = float(self._config.get("market_discovery.weight_ttr", 0.15))
        w_strike = float(self._config.get("market_discovery.weight_strike", 0.25))
        w_horizon = float(self._config.get("market_discovery.weight_horizon", 0.15))

        weighted = (
            (w_volume * volume_score)
            + (w_prob * prob_score)
            + (w_ttr * ttr_score)
            + (w_strike * strike_score)
            + (w_horizon * horizon_score)
        )
        score_total = max(0.0, weighted - hard_penalty)
        return {
            "score_total": score_total,
            "volume_score": volume_score,
            "prob_score": prob_score,
            "ttr_score": ttr_score,
            "strike_score": strike_score,
            "horizon_score": horizon_score,
            "hard_penalty": hard_penalty,
        }

    async def _fetch_binance_spot(self) -> Optional[float]:
        """Low-latency spot reference for market rationality checks."""
        url = "https://api.binance.com/api/v3/ticker/price"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params={"symbol": "BTCUSDT"})
                resp.raise_for_status()
                payload = resp.json()
                price = float(payload.get("price", 0.0))
                return price if price > 0 else None
        except Exception:
            return None

    async def _fetch_vatic_strike(self, epoch_ts: int) -> Optional[float]:
        """Fetch precise strike price from Vatic Oracle with epoch caching."""
        if self._vatic_cache["epoch"] == epoch_ts and self._vatic_cache["price"] is not None:
            return self._vatic_cache["price"]

        url = "https://api.vatic.trading/api/v1/targets/timestamp"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    url, 
                    params={"asset": "btc", "type": "5min", "timestamp": epoch_ts}
                )
                if resp.is_success:
                    data = resp.json()
                    # Per the documentation, they provide strike price absolute based on epoch timestamp.
                    price = data.get("target_price") or data.get("target") or data.get("price")
                    if price:
                        price_float = float(price)
                        # Simpan ke cache
                        self._vatic_cache = {"epoch": epoch_ts, "price": price_float}
                        logger.info("vatic_oracle_strike_acquired", epoch=epoch_ts, strike=price_float)
                        return price_float
        except Exception as e:
            logger.debug("vatic_api_fetch_failed", epoch=epoch_ts, error=str(e))
            
        return None

    def _resolve_signal_ttr_window(self, market: ActiveMarket) -> tuple[float, float]:
        """
        Mirror signal-generator dynamic TTR policy at discovery layer,
        so rotation lock can honor valid entry windows.
        """
        dyn_enabled = bool(self._config.get("signal.dynamic_ttr_enabled", True))
        if not dyn_enabled:
            ttr_min = float(self._config.get("signal.ttr_min_minutes", 5.0))
            ttr_max = float(self._config.get("signal.ttr_max_minutes", 12.0))
            return ttr_min, ttr_max

        lifespan_h = max(
            0.0,
            (market.T_resolution - market.T_open).total_seconds() / 3600.0,
        )
        lifespan_min = lifespan_h * 60.0

        # Ultra-short market (≤ 10 minutes lifespan)
        if lifespan_min <= 10.0:
            entry_open_pct = float(self._config.get("signal.ultrashort_entry_open_pct", 0.80))
            entry_close_pct = float(self._config.get("signal.ultrashort_entry_close_pct", 0.10))
            return (
                lifespan_min * entry_close_pct,
                lifespan_min * entry_open_pct,
            )

        if lifespan_h <= 2.0:
            return (
                float(self._config.get("signal.entry_window_short_min_minutes", 5.0)),
                float(self._config.get("signal.entry_window_short_max_minutes", 45.0)),
            )
        if lifespan_h <= 8.0:
            return (
                float(self._config.get("signal.entry_window_medium_min_minutes", 30.0)),
                float(self._config.get("signal.entry_window_medium_max_minutes", 240.0)),
            )
        return (
            float(self._config.get("signal.entry_window_long_min_minutes", 60.0)),
            float(self._config.get("signal.entry_window_long_max_minutes", 720.0)),
        )

    @staticmethod
    def _extract_token_ids(tokens: list, market_data: dict) -> dict:
        """Extract YES/NO CLOB token IDs from market data."""
        token_ids = {"YES": "", "NO": ""}

        if isinstance(tokens, list):
            for token in tokens:
                outcome = str(token.get("outcome", "")).upper()
                token_id = (
                    token.get("token_id")
                    or token.get("tokenId")
                    or token.get("clobTokenId")
                    or ""
                )
                if outcome in ("YES", "NO") and token_id:
                    token_ids[outcome] = token_id

        # Fallback: try clobTokenIds field
        if not token_ids.get("YES"):
            clob_ids = market_data.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                # Some responses return stringified JSON list.
                try:
                    import json
                    clob_ids = json.loads(clob_ids)
                except Exception:
                    clob_ids = []
            if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                token_ids["YES"] = clob_ids[0]
                token_ids["NO"] = clob_ids[1]

        return token_ids

    @staticmethod
    def _extract_settlement_descriptor(
        market_data: dict,
    ) -> tuple[
        str,
        Optional[str],
        Literal["1m", "unknown"],
        Literal["close", "vwap", "unknown"],
        Optional[str],
    ]:
        """
        Extract structured settlement descriptor from market rules/description.

        This is a basis-risk gate input. If we cannot prove the exchange/granularity,
        we mark it as `unknown` so downstream logic can widen uncertainty or abstain.
        """
        rules = (
            market_data.get("description", "")
            + " "
            + market_data.get("resolution_source", "")
            + " "
            + market_data.get("resolutionSource", "")
            + " "
            + str(market_data.get("uma_resolution_rules", ""))
        ).lower()

        # Defaults
        settlement_exchange: str = "UNKNOWN"
        settlement_instrument: Optional[str] = "BTCUSDT"
        settlement_granularity: Literal["1m", "unknown"] = "unknown"
        settlement_price_type: Literal["close", "vwap", "unknown"] = "unknown"
        resolution_source: Optional[str] = None

        if "binance" in rules:
            settlement_exchange = "BINANCE"
            resolution_source = "Binance"
            settlement_instrument = "BTCUSDT"

            if "1m" in rules or "1 minute" in rules or "1-minute" in rules:
                settlement_granularity = "1m"

            settlement_price_type = "vwap" if "vwap" in rules else "close"
            return (
                settlement_exchange,
                settlement_instrument,
                settlement_granularity,
                settlement_price_type,
                resolution_source,
            )

        if "pyth" in rules:
            settlement_exchange = "PYTH"
            resolution_source = "Pyth"
        elif "coinbase" in rules:
            settlement_exchange = "COINBASE"
            resolution_source = "Coinbase"
        elif "coingecko" in rules:
            settlement_exchange = "COINBASE"
            resolution_source = "CoinGecko"
        elif "uma" in rules:
            settlement_exchange = "UMA"
            resolution_source = "UMA"

        return (
            settlement_exchange,
            settlement_instrument,
            settlement_granularity,
            settlement_price_type,
            resolution_source,
        )

    @staticmethod
    def _extract_resolution_source(market_data: dict) -> Optional[str]:
        """Backwards-compatible wrapper: return only the resolution oracle name."""
        *_rest, resolution_source = MarketDiscovery._extract_settlement_descriptor(market_data)
        return resolution_source

    @staticmethod
    def _parse_timestamp(ts: str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime (UTC)."""
        if not ts:
            return None
        try:
            # Handle various ISO formats
            ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    # ── TTR Phase Logic ───────────────────────────────────────

    def get_ttr_phase(self) -> str:
        """
        Get current TTR phase for signal gating.
        Returns: EARLY, ENTRY_WINDOW, or LATE.
        """
        if not self._active_market:
            return "LATE"

        ttr = self._active_market.TTR_minutes
        ttr_min = self._config.get("signal.ttr_min_minutes", 5.0)
        ttr_max = self._config.get("signal.ttr_max_minutes", 12.0)

        if ttr > ttr_max:
            return "EARLY"
        elif ttr >= ttr_min:
            return "ENTRY_WINDOW"
        else:
            return "LATE"

    async def refresh_ttr(self) -> Optional[float]:
        """Refresh TTR on active market."""
        if not self._active_market:
            return None
        now = datetime.now(timezone.utc)
        ttr_seconds = (self._active_market.T_resolution - now).total_seconds()
        ttr_minutes = max(0.0, ttr_seconds / 60.0)
        self._active_market = self._active_market.model_copy(
            update={"TTR_minutes": ttr_minutes}
        )
        return ttr_minutes
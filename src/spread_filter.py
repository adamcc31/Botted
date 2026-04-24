"""
spread_filter.py — Pre-entry gate: Binance–Chainlink spread filter.

Prevents entries when the Binance–Chainlink price spread is elevated,
indicating oracle lag or market dislocation.

NO SILENT FALLBACK POLICY:
  - If DualFeed is unavailable or stale → SKIP + WARNING log
  - Never silently use Binance price when oracle is missing

Thresholds (configurable via config.json):
  - Normal:   spread ≤ 0.03%  → PROCEED
  - Elevated: 0.03% < spread ≤ 0.08% → WAIT
  - Wide:     spread > 0.08%  → SKIP
"""

from __future__ import annotations

from typing import Optional

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging

from src.config_manager import ConfigManager
from src.dual_feed import DualFeed
from src.schemas import DualFeedSnapshot, SpreadFilterResult

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class SpreadFilter:
    """
    Pre-entry gate that checks Binance–Chainlink spread.

    Critical rule: if DualFeed is unavailable or stale,
    the filter returns SKIP (never silently falls back).
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    def check(self, dual_feed: DualFeed) -> SpreadFilterResult:
        """
        Evaluate current dual-feed state against spread thresholds.

        Returns SpreadFilterResult with PROCEED/WAIT/SKIP recommendation.

        If dual_feed snapshot is None (unavailable/stale), this function
        returns SKIP with an explicit WARNING — no silent fallback.
        """
        # Threshold config
        normal_threshold = float(
            self._config.get("dual_feed.spread_threshold_normal_pct", 0.03)
        )
        elevated_threshold = float(
            self._config.get("dual_feed.spread_threshold_elevated_pct", 0.08)
        )

        # ── Feed Availability Check ───────────────────────────
        snapshot = dual_feed.get_snapshot()

        if snapshot is None:
            # Determine the specific reason for unavailability
            if dual_feed.chainlink_price is None and dual_feed.binance_price_rtds is None:
                reason = "DUAL_FEED_UNAVAILABLE: No data from RTDS (both feeds missing)"
            elif dual_feed.is_chainlink_stale:
                reason = (
                    f"CHAINLINK_STALE: Last update >{dual_feed._stale_threshold_s}s ago "
                    f"(price={dual_feed.chainlink_price})"
                )
            elif dual_feed.is_binance_rtds_stale:
                reason = (
                    f"BINANCE_RTDS_STALE: Last update >{dual_feed._stale_threshold_s}s ago "
                    f"(price={dual_feed.binance_price_rtds})"
                )
            else:
                reason = "DUAL_FEED_UNAVAILABLE: Unknown state"

            logger.warning(
                "spread_filter_skip_feed_unavailable",
                reason=reason,
                chainlink_price=dual_feed.chainlink_price,
                binance_price_rtds=dual_feed.binance_price_rtds,
                is_chainlink_stale=dual_feed.is_chainlink_stale,
                is_binance_rtds_stale=dual_feed.is_binance_rtds_stale,
            )

            return SpreadFilterResult(
                passed=False,
                reason=reason,
                spread_pct=0.0,
                recommendation="SKIP",
            )

        # ── Spread Check ──────────────────────────────────────
        spread_pct = snapshot.spread_pct

        if spread_pct <= normal_threshold:
            return SpreadFilterResult(
                passed=True,
                reason=f"SPREAD_NORMAL: {spread_pct:.4f}% <= {normal_threshold}%",
                spread_pct=spread_pct,
                recommendation="PROCEED",
            )

        if spread_pct <= elevated_threshold:
            logger.info(
                "spread_filter_elevated",
                spread_pct=round(spread_pct, 4),
                threshold=normal_threshold,
                direction=snapshot.spread_direction,
            )
            return SpreadFilterResult(
                passed=False,
                reason=(
                    f"SPREAD_ELEVATED: {spread_pct:.4f}% > {normal_threshold}% "
                    f"(direction={snapshot.spread_direction})"
                ),
                spread_pct=spread_pct,
                recommendation="WAIT",
            )

        # spread_pct > elevated_threshold → SKIP unconditionally
        logger.warning(
            "spread_filter_skip_wide_spread",
            spread_pct=round(spread_pct, 4),
            threshold=elevated_threshold,
            direction=snapshot.spread_direction,
            binance_price=round(snapshot.binance_price, 2),
            chainlink_price=round(snapshot.chainlink_price, 2),
        )
        return SpreadFilterResult(
            passed=False,
            reason=(
                f"SPREAD_TOO_WIDE: {spread_pct:.4f}% > {elevated_threshold}% "
                f"(BN={snapshot.binance_price:.2f}, CL={snapshot.chainlink_price:.2f})"
            ),
            spread_pct=spread_pct,
            recommendation="SKIP",
        )

    @staticmethod
    def check_from_snapshot(
        snapshot: Optional[DualFeedSnapshot],
        normal_threshold: float = 0.03,
        elevated_threshold: float = 0.08,
    ) -> SpreadFilterResult:
        """
        Static convenience method for testing with pre-built snapshots.

        Same logic as check() but accepts a DualFeedSnapshot directly.
        """
        if snapshot is None:
            return SpreadFilterResult(
                passed=False,
                reason="SNAPSHOT_IS_NONE",
                spread_pct=0.0,
                recommendation="SKIP",
            )

        spread_pct = snapshot.spread_pct

        if spread_pct <= normal_threshold:
            return SpreadFilterResult(
                passed=True,
                reason=f"SPREAD_NORMAL: {spread_pct:.4f}%",
                spread_pct=spread_pct,
                recommendation="PROCEED",
            )

        if spread_pct <= elevated_threshold:
            return SpreadFilterResult(
                passed=False,
                reason=f"SPREAD_ELEVATED: {spread_pct:.4f}%",
                spread_pct=spread_pct,
                recommendation="WAIT",
            )

        return SpreadFilterResult(
            passed=False,
            reason=f"SPREAD_TOO_WIDE: {spread_pct:.4f}%",
            spread_pct=spread_pct,
            recommendation="SKIP",
        )

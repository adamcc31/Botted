"""
risk_manager.py — Half-Kelly position sizing with CLOB odds and hard limits.

Implements:
  - Decimal odds from CLOB prices
  - Full Kelly → Half-Kelly with dynamic multiplier
  - Consecutive loss decay (down to 25% floor)
  - Hard limits: daily loss, session loss, max positions, capital floor
  - Atomic position tracking via asyncio.Lock
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Optional, Union

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging

from src.config_manager import ConfigManager
from src.database import DatabaseManager
from src.schemas import ApprovedBet, RejectedBet, SignalResult
from src.zone_matrix import classify_zone
from sqlalchemy import text

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class RiskManager:
    """
    Position sizing and risk gate enforcement.

    Uses Half-Kelly fraction with CLOB-derived decimal odds.
    Dynamic multiplier reduces exposure after consecutive losses.
    """

    def __init__(self, config: ConfigManager, db: DatabaseManager) -> None:
        self._config = config
        self._db = db
        self._position_lock = asyncio.Lock()
        self._daily_pnl: float = 0.0
        self._session_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trade_history: List[dict] = []
        # Memory-based pending tracking to bridge the gap between approval and DB insertion
        self._pending_approvals: List[dict] = []

    # ── Public Properties ─────────────────────────────────────

    @property
    def open_positions(self) -> int:
        return len(self._pending_approvals)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def session_pnl(self) -> float:
        return self._session_pnl

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    # ── Database Sync ─────────────────────────────────────────

    async def _get_db_exposure(self) -> tuple[int, float]:
        """
        Query the database for trades still marked as PENDING.
        We also clear any stale memory approvals that are older than 30 seconds
        to prevent memory leaks if a DB insertion failed silently.
        """
        now = datetime.utcnow()
        # Clean up stale memory approvals (> 30 seconds old)
        self._pending_approvals = [
            p for p in self._pending_approvals 
            if (now - p["approved_at"]).total_seconds() < 30.0
        ]
        
        async with self._db.get_session() as session:
            # Time-lock query: count actual pending trades.
            # Even if outcome='PENDING', if it's older than max TTR (~1 hour), ignore it to prevent deadlocks.
            query = text("""
                SELECT COUNT(trade_id), COALESCE(SUM(bet_size_usd), 0.0)
                FROM trades
                WHERE outcome = 'PENDING'
                  AND timestamp_entry >= datetime('now', '-60 minutes')
            """)
            result = await session.execute(query)
            row = result.fetchone()
            db_count = row[0] if row else 0
            db_exposure = row[1] if row else 0.0
            
            return db_count, db_exposure

    # ── Trade Approval ────────────────────────────────────────

    async def approve(
        self,
        signal: SignalResult,
        capital: float,
    ) -> Union[ApprovedBet, RejectedBet]:
        """
        Evaluate signal against risk constraints and determine bet size.
        Uses asyncio.Lock to prevent race conditions on position tracking.
        """
        async with self._position_lock:
            # 1. Real-time Database Synchronization
            db_open, db_exposure = await self._get_db_exposure()
            
            # Combine DB state with uncommitted memory state
            total_open = db_open + len(self._pending_approvals)
            mem_exposure = sum(p["bet_size"] for p in self._pending_approvals)
            total_exposure = db_exposure + mem_exposure

            # ── Hard Limit Checks ─────────────────────────────
            daily_loss_limit = self._config.get("risk.daily_loss_limit_pct", 0.05)
            session_loss_limit = self._config.get("risk.session_loss_limit_pct", 0.03)
            max_positions = int(self._config.get("risk.max_positions", 3))
            max_exposure_pct = float(self._config.get("risk.max_exposure_pct", 0.60))
            min_capital_floor = 5.0

            if self._daily_pnl < -(daily_loss_limit * capital):
                return RejectedBet(signal=signal, reason="DAILY_LOSS_LIMIT_HIT")

            if self._session_pnl < -(session_loss_limit * capital):
                return RejectedBet(signal=signal, reason="SESSION_LOSS_LIMIT_HIT")

            if total_open >= max_positions:
                return RejectedBet(signal=signal, reason="MAX_POSITIONS_REACHED")
                
            available_exposure = max(0.0, (capital * max_exposure_pct) - total_exposure)
            min_bet = self._config.get("risk.min_bet_usd", 1.00)
            
            if available_exposure < min_bet:
                return RejectedBet(signal=signal, reason="MAX_EXPOSURE_REACHED")

            if capital < min_capital_floor:
                return RejectedBet(signal=signal, reason="CAPITAL_BELOW_FLOOR")

            # ── Bet Sizing ────────────────────────────────────
            bet_size, kelly_fraction, kelly_multiplier = self._compute_bet_size(
                signal, capital, available_exposure
            )

            if bet_size < min_bet:
                return RejectedBet(signal=signal, reason="BET_BELOW_MINIMUM")

            # ── Approve ───────────────────────────────────────
            # Register in memory until DB insertion catches up
            self._pending_approvals.append({
                "signal_id": getattr(signal, "signal_id", signal.market_id),
                "bet_size": bet_size,
                "approved_at": datetime.utcnow()
            })

            logger.info(
                "trade_approved",
                signal=signal.signal,
                zone_id=signal.zone_id,
                bet_size=round(bet_size, 2),
                kelly_fraction=round(kelly_fraction, 4),
                open_positions=total_open + 1,
                total_exposure=round(total_exposure + bet_size, 2),
            )

            return ApprovedBet(
                signal=signal,
                bet_size=bet_size,
                kelly_fraction=kelly_fraction,
                kelly_multiplier=kelly_multiplier,
            )

    # ── Position Lifecycle ────────────────────────────────────

    async def on_trade_resolved(self, pnl: float) -> None:
        """Update state after trade resolution."""
        async with self._position_lock:
            self._open_positions = max(0, self._open_positions - 1)
            self._daily_pnl += pnl
            self._session_pnl += pnl

            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            self._trade_history.append({"pnl": pnl})

            logger.info(
                "trade_resolved_risk_update",
                pnl=round(pnl, 2),
                daily_pnl=round(self._daily_pnl, 2),
                session_pnl=round(self._session_pnl, 2),
                consecutive_losses=self._consecutive_losses,
                open_positions=self._open_positions,
            )

    def reset_session(self) -> None:
        """Reset session-level counters (keep daily)."""
        self._session_pnl = 0.0
        self._open_positions = 0
        logger.info("risk_session_reset")

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._daily_pnl = 0.0
        self._session_pnl = 0.0
        self._open_positions = 0
        self._consecutive_losses = 0
        self._trade_history.clear()
        logger.info("risk_daily_reset")

    # ── Kelly Computation ─────────────────────────────────────

    def _compute_bet_size(
        self, signal: SignalResult, capital: float, available_exposure: float
    ) -> tuple[float, float, float]:
        """
        Compute bet size using empirical Zone-Aware Kelly V4 with dynamic multiplier.
        Returns: (bet_size_usd, kelly_fraction, kelly_multiplier)
        """
        from src.zone_matrix import classify_zone

        min_bet = self._config.get("risk.min_bet_usd", 1.00)
        use_flat_bet = self._config.get("risk.use_flat_bet", True)
        flat_bet_size = self._config.get("risk.flat_bet_size_usd", 1.00)
        
        # ── Zone-Aware Kelly V4 ──────────────────────────────
        # We re-classify to get the V4 properties (fraction and cap)
        ttr = getattr(signal, "TTR_minutes", 0.0)
        
        curr_price = getattr(signal, "current_price", None)
        strike_price = getattr(signal, "strike_price", None)
        
        if curr_price is None or strike_price is None:
            logger.warning("risk_calc_missing_prices", market_id=signal.market_id)
            return (0.0, 0.0, 0.0)
            
        dist = abs(curr_price - strike_price)
        odds = getattr(signal, "entry_odds", 0.5)
        
        zone = classify_zone(ttr, dist, odds)
        
        if zone.zone_type != "ALPHA" or zone.kelly_fraction <= 0:
            return (0.0, 0.0, 0.0)

        if use_flat_bet:
            # Flat bet mode: return constant size if within available exposure
            bet_size = min(flat_bet_size, available_exposure)
            return bet_size, 0.0, 1.0

        # ── Dynamic Kelly Sizing ──────────────────────────────
        multiplier_decay = self._config.get("risk.consecutive_loss_multiplier", 0.10)
        kelly_floor = self._config.get("risk.kelly_floor_multiplier", 0.50)

        # Dynamic Multiplier (consecutive loss decay)
        kelly_multiplier = max(
            kelly_floor,
            1.0 - self._consecutive_losses * multiplier_decay,
        )

        # Signal edge (ML/FairProb derived) - used for Kelly formula base
        # If signal.live_edge is not available, we could use a conservative edge based on zone.empirical_ev
        # but the standard approach is to use the model's edge if available.
        edge = getattr(signal, "live_edge", zone.empirical_ev)
        if edge <= 0:
            edge = 0.01 # minimal edge placeholder if somehow missing
            
        # raw_kelly = edge / (1 - edge) is for odds=1 (Binary). 
        # For odds b:1, Kelly is (bp - q) / b. 
        # Here p = win_prob, q = 1-p. entry_odds = 1 / (b+1).
        # However, user code specifically said: raw_kelly = edge / (1 - edge)
        # We will use the user's simplified formula or the zone's empirical edge.
        
        raw_kelly = edge / (1 - edge) if edge < 1 else 0.5
        fractional_kelly = raw_kelly * zone.kelly_fraction * kelly_multiplier
        
        bet_size = min(
            fractional_kelly * capital,
            zone.kelly_cap * capital,
            available_exposure
        )

        if bet_size < min_bet:
            return (0.0, 0.0, 0.0)
            
        # Integer rounding for cleaner execution
        int_part = int(bet_size)
        dec_part = bet_size - int_part
        if dec_part <= 0.5:
            bet_size = max(min_bet, float(int_part))
        else:
            bet_size = max(min_bet, float(int_part + 1))

        return bet_size, zone.kelly_fraction, kelly_multiplier

    # ── Utility ───────────────────────────────────────────────

    def get_recent_trade_pnls(self, n: int = 30) -> List[float]:
        """Get PnL for last N trades."""
        return [t["pnl"] for t in self._trade_history[-n:]]

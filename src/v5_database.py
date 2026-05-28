"""
src/v5_database.py — Slingger V5 native SQLite database manager.

Independent write path for V5 trade and session state.
Completely decoupled from DryRunEngine and the legacy V1 database stack.

Tables:
  - v5_trades: Individual trade lifecycle (PENDING → WIN/LOSS/EMERGENCY_EXIT)
  - v5_session: Session-level aggregated state (Single Source of Truth)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

try:
    import structlog  # type: ignore
except ModuleNotFoundError:
    structlog = None

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


class V5DatabaseManager:
    """
    Async SQLite manager for Slingger V5 native state persistence.

    Tables:
      - v5_trades: Individual trade lifecycle (PENDING → WIN/LOSS/EMERGENCY_EXIT)
      - v5_session: Session-level aggregated state (Single Source of Truth)

    All write operations use `await` — never fire-and-forget.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path or str(_DATA_DIR / "v5_trading.db")
        self._session_id: Optional[str] = None

    # ── Schema Initialization ────────────────────────────────────

    async def init_db(self) -> None:
        """Create v5_trades and v5_session tables if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS v5_trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id        TEXT UNIQUE NOT NULL,
                    market_id       TEXT NOT NULL,
                    token_side      TEXT NOT NULL,
                    entry_odds      REAL NOT NULL,
                    target_odds     REAL NOT NULL DEFAULT 0.80,
                    entry_spread    REAL NOT NULL,
                    stake_usd       REAL NOT NULL,
                    shares          REAL NOT NULL,
                    status          TEXT NOT NULL,
                    exit_odds       REAL,
                    pnl_usd         REAL,
                    entry_ts        INTEGER NOT NULL,
                    exit_ts         INTEGER,
                    duration_sec    INTEGER
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS v5_session (
                    session_id      TEXT PRIMARY KEY,
                    start_ts        INTEGER NOT NULL,
                    capital_start   REAL NOT NULL,
                    capital_current REAL NOT NULL,
                    trades_executed INTEGER NOT NULL DEFAULT 0,
                    trades_win      INTEGER NOT NULL DEFAULT 0,
                    trades_loss     INTEGER NOT NULL DEFAULT 0,
                    total_pnl_usd   REAL NOT NULL DEFAULT 0.0,
                    last_updated    INTEGER NOT NULL
                )
            """)
            await db.commit()
        logger.info("v5_database_initialized", db_path=self._db_path)

    # ── Session Management ───────────────────────────────────────

    async def init_session(self, session_id: str, capital_start: float) -> None:
        """Insert or resume the active session record.

        If session already exists (e.g. bot restarted), state is preserved.
        A new record is only created for fresh sessions.
        """
        self._session_id = session_id
        now_ms = int(time.time() * 1000)
        async with aiosqlite.connect(self._db_path) as db:
            # Preserve existing session state across restarts
            cursor = await db.execute(
                "SELECT capital_current, trades_executed FROM v5_session "
                "WHERE session_id = ?", (session_id,)
            )
            existing = await cursor.fetchone()
            if existing:
                logger.info("v5_session_resumed",
                            session_id=session_id,
                            capital=existing[0],
                            trades=existing[1])
                return

            await db.execute("""
                INSERT INTO v5_session
                    (session_id, start_ts, capital_start, capital_current,
                     trades_executed, trades_win, trades_loss, total_pnl_usd,
                     last_updated)
                VALUES (?, ?, ?, ?, 0, 0, 0, 0.0, ?)
            """, (session_id, now_ms, capital_start, capital_start, now_ms))
            await db.commit()
        logger.info("v5_session_initialized",
                     session_id=session_id, capital=capital_start)

    # ── Trade Lifecycle ──────────────────────────────────────────

    async def record_new_trade(
        self,
        trade_id: str,
        market_id: str,
        token_side: str,
        entry_odds: float,
        target_odds: float,
        entry_spread: float,
        stake_usd: float,
        shares: float,
    ) -> None:
        """INSERT a new PENDING trade into v5_trades.

        Capital is NOT deducted here — V5 computes available capital
        at runtime by subtracting locked stakes from PENDING trades.
        """
        now_ms = int(time.time() * 1000)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                INSERT INTO v5_trades
                    (trade_id, market_id, token_side, entry_odds, target_odds,
                     entry_spread, stake_usd, shares, status, entry_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """, (trade_id, market_id, token_side, entry_odds, target_odds,
                  entry_spread, stake_usd, shares, now_ms))
            await db.commit()
        logger.info("v5_trade_recorded",
                     trade_id=trade_id[:8], stake=stake_usd,
                     entry_odds=round(entry_odds, 4))

    async def close_trade(
        self,
        trade_id: str,
        status: str,
        exit_odds: Optional[float],
        pnl_usd: float,
        exit_ts: Optional[int] = None,
    ) -> None:
        """Close a trade: UPDATE v5_trades and v5_session atomically.

        MUST be called with ``await`` — never fire-and-forget.
        This is the ONLY path that permanently updates PnL and trade counts.

        Args:
            trade_id: Unique trade identifier.
            status: One of WIN, LOSS, EMERGENCY_EXIT.
            exit_odds: Exit price (None for HOLD_TO_MATURITY).
            pnl_usd: Net profit/loss for this trade.
            exit_ts: Exit timestamp in ms (default: now).
        """
        now_ms = exit_ts or int(time.time() * 1000)
        async with aiosqlite.connect(self._db_path) as db:
            # Get entry_ts for duration calculation
            cursor = await db.execute(
                "SELECT entry_ts FROM v5_trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()
            if not row:
                logger.error("v5_close_trade_not_found", trade_id=trade_id[:8])
                return

            entry_ts = row[0]
            duration_sec = int((now_ms - entry_ts) / 1000)

            is_win = 1 if status == "WIN" else 0
            is_loss = 1 if status in ("LOSS", "EMERGENCY_EXIT") else 0

            # Update trade record
            await db.execute("""
                UPDATE v5_trades
                SET status = ?, exit_odds = ?, pnl_usd = ?,
                    exit_ts = ?, duration_sec = ?
                WHERE trade_id = ?
            """, (status, exit_odds, pnl_usd, now_ms, duration_sec, trade_id))

            # Update session: apply PnL to capital
            await db.execute("""
                UPDATE v5_session
                SET capital_current = capital_current + ?,
                    trades_executed = trades_executed + 1,
                    trades_win = trades_win + ?,
                    trades_loss = trades_loss + ?,
                    total_pnl_usd = total_pnl_usd + ?,
                    last_updated = ?
                WHERE session_id = ?
            """, (pnl_usd, is_win, is_loss, pnl_usd, now_ms, self._session_id))

            await db.commit()
        logger.info("v5_trade_closed",
                     trade_id=trade_id[:8], status=status,
                     pnl=round(pnl_usd, 4), duration_sec=duration_sec)

    # ── Query ────────────────────────────────────────────────────

    async def get_session_state(self, session_id: Optional[str] = None) -> Optional[dict]:
        """SELECT session state — Single Source of Truth for Trigger C reporting.

        Returns dict with keys: session_id, start_ts, capital_start,
        capital_current, trades_executed, trades_win, trades_loss,
        total_pnl_usd, last_updated.
        """
        sid = session_id or self._session_id
        if not sid:
            return None
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM v5_session WHERE session_id = ?", (sid,)
            )
            row = await cursor.fetchone()
            if row:
                return dict(row)
        return None

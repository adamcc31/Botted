"""
exporter.py — CSV + JSON export per session.

Exports follow the immutable schema contract from TRD Appendix B.
Changes require versioning.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import structlog
from sqlalchemy import text

from src.schemas import PaperTrade, SessionMetrics, CLOBState

logger = structlog.get_logger(__name__)

_EXPORTS_DIR = Path(__file__).parent.parent / "data" / "exports"


class Exporter:
    """Export session data to CSV and JSON per TRD Appendix B schema."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._session_dir = _EXPORTS_DIR / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._clob_log: List[dict] = []

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    # ── Trades Export ─────────────────────────────────────────

    def export_trades(self, trades: List[PaperTrade]) -> Path:
        """Export resolved trades to CSV per Appendix B trades.csv schema."""
        if not trades:
            logger.info("no_trades_to_export")
            return self._session_dir / "trades.csv"

        records = []
        for t in trades:
            records.append({
                "trade_id": t.trade_id,
                "session_id": t.session_id,
                "market_id": t.market_id,
                "timestamp_signal": t.timestamp_signal.isoformat() if t.timestamp_signal else "",
                "trigger_timestamp": t.trigger_timestamp.isoformat() if t.trigger_timestamp else "",
                "timestamp_resolution": t.timestamp_resolution.isoformat() if t.timestamp_resolution else "",
                "signal_type": t.signal_type,
                "P_model": t.P_model,
                "synthetic_edge": t.synthetic_edge,
                "live_edge": t.live_edge,
                "entry_price_usdc": t.entry_price,
                "bet_size_usd": t.bet_size,
                "kelly_fraction": t.kelly_fraction,
                "kelly_multiplier": t.kelly_multiplier,
                "strike_price": t.strike_price,
                "btc_price_at_trigger": t.btc_price_at_trigger,
                "btc_distance_to_strike": t.btc_distance_to_strike,
                "TTR_minutes": t.TTR_at_entry,
                "btc_at_resolution": t.btc_at_resolution,
                "outcome": t.outcome,
                "pnl_usd": t.pnl_usd,
                "pnl_pct_capital": t.pnl_pct_capital,
                "capital_before": t.capital_before,
                "capital_after": t.capital_after,
                "mode": "DRY",
            })

        df = pd.DataFrame(records)
        path = self._session_dir / "trades.csv"
        df.to_csv(path, index=False)
        logger.info("trades_exported", path=str(path), count=len(records))
        return path

    # ── Performance Export ────────────────────────────────────

    def export_performance(self, metrics: SessionMetrics) -> Path:
        """Export session performance to JSON."""
        path = self._session_dir / "performance.json"
        data = metrics.model_dump()
        # Convert datetime objects to strings
        for k, v in data.items():
            if isinstance(v, datetime):
                data[k] = v.isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("performance_exported", path=str(path))
        return path

    # ── CLOB Log ──────────────────────────────────────────────

    def record_clob_snapshot(self, clob_state: CLOBState, ttr_minutes: float) -> None:
        """Record CLOB snapshot for clob_log.csv."""
        self._clob_log.append({
            "timestamp": clob_state.timestamp.isoformat(),
            "market_id": clob_state.market_id,
            "TTR_minutes": round(ttr_minutes, 2),
            "yes_ask": clob_state.yes_ask,
            "yes_bid": clob_state.yes_bid,
            "no_ask": clob_state.no_ask,
            "no_bid": clob_state.no_bid,
            "yes_depth_usd": round(clob_state.yes_depth_usd, 2),
            "no_depth_usd": round(clob_state.no_depth_usd, 2),
            "market_vig": round(clob_state.market_vig, 4),
            "is_liquid": clob_state.is_liquid,
        })

    def export_clob_log(self) -> Path:
        """Export accumulated CLOB log to CSV."""
        path = self._session_dir / "clob_log.csv"
        if self._clob_log:
            df = pd.DataFrame(self._clob_log)
            df.to_csv(path, index=False)
            logger.info("clob_log_exported", path=str(path), snapshots=len(self._clob_log))
        return path

    # ── Equity Curve ──────────────────────────────────────────

    def export_equity_curve(
        self, trades: List[PaperTrade], initial_capital: float
    ) -> Path:
        """Export equity curve to CSV."""
        path = self._session_dir / "equity_curve.csv"
        if not trades:
            return path

        equity = initial_capital
        records = [{"trade_num": 0, "equity": equity, "pnl": 0.0}]

        for i, t in enumerate(trades, 1):
            equity += (t.pnl_usd or 0)
            records.append({
                "trade_num": i,
                "equity": round(equity, 2),
                "pnl": round(t.pnl_usd or 0, 4),
                "outcome": t.outcome,
                "timestamp": t.timestamp_signal.isoformat() if t.timestamp_signal else "",
            })

        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        logger.info("equity_curve_exported", path=str(path))
        return path

    # ── Signal Log (SQLite to CSV) ────────────────────────────

    async def export_signals(self, db_manager) -> Path:
        """Export all signals for this session from SQLite to CSV."""
        path = self._session_dir / f"dry_run_{self._session_id}.csv"
        
        try:
            async with db_manager.engine.connect() as conn:
                result = await conn.execute(text("""
                    SELECT 
                        timestamp_utc as timestamp,
                        market_id,
                        slug,
                        ttr_minutes * 60 as ttr_seconds,
                        strike_price,
                        clob_yes_ask as odds_yes,
                        clob_no_ask as odds_no,
                        signal_type as signal_direction,
                        p_model as confidence_score,
                        binance_price,
                        chainlink_price,
                        spread_pct,
                        spread_filter_passed,
                        spread_filter_reason,
                        entry_odds,
                        1.0 as theoretical_exit_odds,
                        theoretical_pnl,
                        signal_correct,
                        actual_outcome,
                        obi_value,
                        tfm_norm as tfm_value,
                        rv_value,
                        vol_percentile,
                        depth_ratio,
                        strike_distance_pct,
                        contest_urgency,
                        odds_yes_60s_ago,
                        odds_delta_60s,
                        btc_return_1m,
                        confidence_bucket,
                        entry_odds_source
                    FROM signals 
                    WHERE session_id = :session_id
                    ORDER BY timestamp_utc ASC
                """), {"session_id": self._session_id})
                
                rows = result.fetchall()
                if rows:
                    columns = result.keys()
                    df = pd.DataFrame(rows, columns=columns)
                    df.to_csv(path, index=False)
                    logger.info("signals_exported", path=str(path), count=len(rows))
                else:
                    logger.info("no_signals_to_export")
        except Exception as e:
            logger.error("export_signals_failed", error=str(e))
            
        return path

    # ── Full Session Export ───────────────────────────────────

    def export_session(
        self,
        trades: List[PaperTrade],
        metrics: SessionMetrics,
        initial_capital: float,
    ) -> Path:
        """Export all session data."""
        self.export_trades(trades)
        self.export_performance(metrics)
        self.export_clob_log()
        self.export_equity_curve(trades, initial_capital)

        logger.info("full_session_exported", session_dir=str(self._session_dir))
        return self._session_dir

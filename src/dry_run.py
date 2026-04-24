"""
dry_run.py — Paper trading engine with PASS/FAIL evaluation.

Simulates trades using real-time data without placing real orders.
Tracks all metrics required for go-live decision.
Implements composite scoring and abort conditions.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging

from src.binance_feed import BinanceFeed
from src.config_manager import ConfigManager
from src.database import DatabaseManager
from sqlalchemy import text
from src.schemas import (
    ActiveMarket,
    ApprovedBet,
    PaperTrade,
    SessionMetrics,
    SignalResult,
)

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class DryRunEngine:
    """
    Paper trading engine for validation and go-live evaluation.

    Modal simulasi: $100.00 USDC (virtual).
    Uses real-time Binance + CLOB data.
    Evaluates PASS/FAIL per session and tracks go-live criteria.
    """

    def __init__(self, config: ConfigManager, db: DatabaseManager, initial_capital: float = 100.0) -> None:
        self._config = config
        self._db = db
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._session_id = self._generate_session_id()

        # Trade tracking
        self._pending_trades: List[PaperTrade] = []
        self._resolved_trades: List[PaperTrade] = []
        self._all_predictions: List[Dict] = []  # For Brier score

        # Signal counters
        self._signals_evaluated: int = 0
        self._signals_abstained: int = 0
        self._abstain_reasons: Dict[str, int] = {
            "REGIME_BLOCK": 0,
            "LIQUIDITY_BLOCK": 0,
            "TTR_PHASE": 0,
            "NO_EDGE": 0,
            "NO_TRADE_ZONE": 0,
            "BASIS_RISK_BLOCK": 0,
            "BASIS_RISK_INFLATED": 0,
        }
        self._bars_processed: int = 0
        self._consecutive_losses: int = 0

        # Session timing
        self._start_time = datetime.now(timezone.utc)

    # ── Public Properties ─────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def trades_executed(self) -> int:
        return len(self._resolved_trades) + len(self._pending_trades)

    # ── Trade Simulation ──────────────────────────────────────

    async def record_signal(self, signal: SignalResult) -> None:
        """Record every signal evaluation (including ABSTAIN)."""
        self._signals_evaluated += 1
        if signal.signal == "ABSTAIN":
            self._signals_abstained += 1
            reason = signal.abstain_reason or "NO_EDGE"
            if reason in self._abstain_reasons:
                self._abstain_reasons[reason] += 1
            else:
                self._abstain_reasons[reason] = self._abstain_reasons.get(reason, 0) + 1

        # Track for Brier score
        self._all_predictions.append({
            "P_model": signal.P_model,
            "market_id": signal.market_id,
            "timestamp": signal.timestamp,
        })

        summary_every = int(self._config.get("dry_run.abstain_summary_every_signals", 25))
        if summary_every > 0 and self._signals_evaluated % summary_every == 0:
            logger.info(
                "abstain_reason_summary",
                signals_evaluated=self._signals_evaluated,
                signals_abstained=self._signals_abstained,
                abstain_rate=round(
                    self._signals_abstained / max(1, self._signals_evaluated), 4
                ),
                abstain_reasons=dict(self._abstain_reasons),
            )
            
        # SQLite Persistence
        signal_type_str = signal.signal
        if "SPREAD_FILTER_SKIP" in str(signal.abstain_reason):
            signal_type_str = "SKIP"
            
        entry_odds = signal.clob_yes_ask if signal_type_str == "BUY_YES" else signal.clob_no_ask
            
        async with self._db.engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO signals (
                    signal_id, session_id, market_id, timestamp_utc, signal_type,
                    abstain_reason, p_model, clob_yes_ask, clob_no_ask,
                    ttr_minutes, slug, strike_price, binance_price, chainlink_price,
                    spread_pct, spread_filter_passed, spread_filter_reason,
                    entry_odds, theoretical_pnl, signal_correct, actual_outcome, mode
                ) VALUES (
                    :signal_id, :session_id, :market_id, :timestamp_utc, :signal_type,
                    :abstain_reason, :p_model, :clob_yes_ask, :clob_no_ask,
                    :ttr_minutes, :slug, :strike_price, :binance_price, :chainlink_price,
                    :spread_pct, :spread_filter_passed, :spread_filter_reason,
                    :entry_odds, :theoretical_pnl, 'PENDING', 'PENDING', 'DRY'
                )
            """), {
                "signal_id": str(uuid.uuid4()),
                "session_id": self._session_id,
                "market_id": signal.market_id,
                "timestamp_utc": signal.timestamp,
                "signal_type": signal_type_str,
                "abstain_reason": signal.abstain_reason,
                "p_model": signal.P_model,
                "clob_yes_ask": signal.clob_yes_ask,
                "clob_no_ask": signal.clob_no_ask,
                "ttr_minutes": signal.TTR_minutes,
                "slug": None, # market_id is logged, we can backfill slug if needed, but the user requested slug. Let's pass it if we have it. Wait, active_market is not passed here.
                "strike_price": signal.strike_price,
                "binance_price": getattr(signal, "binance_price_at_signal", None),
                "chainlink_price": getattr(signal, "chainlink_price_at_signal", None),
                "spread_pct": getattr(signal, "spread_pct_at_signal", None),
                "spread_filter_passed": getattr(signal, "spread_filter_passed", None),
                "spread_filter_reason": getattr(signal, "spread_filter_reason", None),
                "entry_odds": entry_odds,
                "theoretical_pnl": 0.0, # Will be calculated during resolve
            })

    def simulate_trade(
        self,
        signal: SignalResult,
        approved_bet: ApprovedBet,
        active_market: ActiveMarket,
        entry_price_override: Optional[float] = None,
        bet_size_override: Optional[float] = None,
    ) -> PaperTrade:
        """
        Create paper trade record. Resolution happens asynchronously.
        """
        entry_price = (
            entry_price_override
            if entry_price_override is not None
            else (
                signal.clob_yes_ask
                if signal.signal == "BUY_YES"
                else signal.clob_no_ask
            )
        )
        bet_size = (
            float(bet_size_override)
            if bet_size_override is not None
            else float(approved_bet.bet_size)
        )

        btc_price_at_trigger = signal.current_price
        strike_price = active_market.strike_price
        dist_to_strike = btc_price_at_trigger - strike_price

        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            session_id=self._session_id,
            market_id=active_market.market_id,
            signal_type=signal.signal,
            entry_price=entry_price,
            bet_size=bet_size,
            strike_price=strike_price,
            T_resolution=active_market.T_resolution,
            TTR_at_entry=signal.TTR_minutes,
            P_model=signal.P_model,
            edge_yes=signal.edge_yes,
            edge_no=signal.edge_no,
            kelly_fraction=approved_bet.kelly_fraction,
            kelly_multiplier=approved_bet.kelly_multiplier,
            capital_before=self._capital,
            synthetic_edge=signal.synthetic_edge,
            live_edge=signal.live_edge,
            btc_price_at_trigger=btc_price_at_trigger,
            btc_distance_to_strike=dist_to_strike,
            trigger_timestamp=datetime.now(timezone.utc),
            timestamp_signal=signal.timestamp,
        )

        self._pending_trades.append(trade)

        logger.info(
            "paper_trade_opened",
            trade_id=trade.trade_id[:8],
            signal=trade.signal_type,
            bet_size=round(trade.bet_size, 2),
            entry_price=round(entry_price, 4),
            P_model=round(signal.P_model, 4),
        )

        return trade

    async def resolve_trade(
        self,
        trade: PaperTrade,
        btc_at_resolution: float,
    ) -> PaperTrade:
        """
        Resolve a pending trade with actual BTC price at resolution.

        Returns updated PaperTrade with outcome and PnL.
        """
        # Determine outcome
        won = (
            (trade.signal_type == "BUY_YES" and btc_at_resolution > trade.strike_price)
            or
            (trade.signal_type == "BUY_NO" and btc_at_resolution <= trade.strike_price)
        )

        outcome = "WIN" if won else "LOSS"

        # Compute PnL
        if won:
            # Win: payout = bet_size / entry_price (shares × $1) - bet_size
            pnl = trade.bet_size * ((1.0 / trade.entry_price) - 1.0)
        else:
            # Loss: lose entire bet
            pnl = -trade.bet_size

        # Update capital
        self._capital += pnl
        capital_after = self._capital

        # Track consecutive losses
        if outcome == "LOSS":
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Update trade
        resolved = trade.model_copy(update={
            "btc_at_resolution": btc_at_resolution,
            "outcome": outcome,
            "pnl_usd": round(pnl, 4),
            "pnl_pct_capital": round(pnl / (trade.capital_before + 1e-8) * 100, 2),
            "capital_after": round(capital_after, 2),
            "timestamp_resolution": datetime.now(timezone.utc),
        })

        # Move from pending to resolved
        self._pending_trades = [t for t in self._pending_trades if t.trade_id != trade.trade_id]
        self._resolved_trades.append(resolved)

        # Track prediction for Brier score
        actual_outcome = 1.0 if btc_at_resolution > trade.strike_price else 0.0
        self._all_predictions.append({
            "P_model": trade.P_model,
            "actual_outcome": actual_outcome,
            "market_id": trade.market_id,
        })

        logger.info(
            "paper_trade_resolved",
            trade_id=resolved.trade_id[:8],
            outcome=outcome,
            pnl_usd=round(pnl, 4),
            capital=round(capital_after, 2),
            btc_at_resolution=round(btc_at_resolution, 2),
            consecutive_losses=self._consecutive_losses,
        )

        return resolved

    def increment_bars(self) -> None:
        """Count a processed bar."""
        self._bars_processed += 1

    # ── Session Metrics ───────────────────────────────────────

    def compute_session_metrics(self, model_version: str = "") -> SessionMetrics:
        """Compute all session-level metrics for PASS/FAIL evaluation."""
        now = datetime.now(timezone.utc)
        duration = (now - self._start_time).total_seconds() / 3600.0

        trades = self._resolved_trades
        win_count = sum(1 for t in trades if t.outcome == "WIN")
        loss_count = sum(1 for t in trades if t.outcome == "LOSS")
        total = len(trades)

        win_rate = win_count / total if total > 0 else 0.0
        total_pnl = sum(t.pnl_usd or 0 for t in trades)

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in trades if (t.pnl_usd or 0) > 0)
        gross_loss = abs(sum(t.pnl_usd for t in trades if (t.pnl_usd or 0) < 0))
        profit_factor = gross_profit / (gross_loss + 1e-8)

        # Max drawdown
        max_dd = self._compute_max_drawdown(trades)

        # Sharpe and Sortino
        sharpe, sortino = self._compute_risk_ratios(trades)

        # Brier score
        brier = self._compute_brier_score()

        # Mean edges
        yes_trades = [t for t in trades if t.signal_type == "BUY_YES"]
        no_trades = [t for t in trades if t.signal_type == "BUY_NO"]
        mean_edge_yes = np.mean([t.edge_yes for t in yes_trades]) if yes_trades else 0.0
        mean_edge_no = np.mean([t.edge_no for t in no_trades]) if no_trades else 0.0
        mean_ttr = np.mean([t.TTR_at_entry for t in trades]) if trades else 0.0
        mean_strike_dist = np.mean([
            abs((t.pnl_usd or 0) / (t.bet_size + 1e-8))
            for t in trades
        ]) if trades else 0.0

        # Dry run composite score
        dry_run_score = self._compute_dry_run_score(
            win_rate, total_pnl / (self._initial_capital + 1e-8),
            sharpe, max_dd
        )

        # PASS/FAIL
        pass_fail = self._evaluate_pass_fail(
            total, win_rate, max_dd, profit_factor, dry_run_score
        )

        return SessionMetrics(
            session_id=self._session_id,
            date_utc=self._start_time.strftime("%Y-%m-%d"),
            start_time=self._start_time,
            end_time=now,
            duration_hours=round(duration, 2),
            mode="DRY",
            total_bars_processed=self._bars_processed,
            total_signals_evaluated=self._signals_evaluated,
            signals_abstained=self._signals_abstained,
            abstain_regime=self._abstain_reasons.get("REGIME_BLOCK", 0),
            abstain_liquidity=self._abstain_reasons.get("LIQUIDITY_BLOCK", 0),
            abstain_ttr=self._abstain_reasons.get("TTR_PHASE", 0),
            abstain_no_edge=self._abstain_reasons.get("NO_EDGE", 0),
            trades_executed=total,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            total_pnl_usd=round(total_pnl, 2),
            total_pnl_pct_capital=round(total_pnl / (self._initial_capital + 1e-8) * 100, 2),
            max_drawdown=round(max_dd, 4),
            sharpe_rolling=round(sharpe, 4),
            sortino_rolling=round(sortino, 4),
            brier_score=round(brier, 4),
            mean_edge_yes_traded=round(float(mean_edge_yes), 4),
            mean_edge_no_traded=round(float(mean_edge_no), 4),
            mean_ttr_at_signal=round(float(mean_ttr), 2),
            mean_strike_distance=round(float(mean_strike_dist), 4),
            capital_start=self._initial_capital,
            capital_end=round(self._capital, 2),
            dry_run_score=round(dry_run_score, 4),
            pass_fail=pass_fail,
            model_version=model_version,
            consecutive_losses=self._consecutive_losses,
        )

    # ── Abort Conditions ──────────────────────────────────────

    def check_abort_conditions(self) -> Optional[str]:
        """Check if session should be aborted."""
        abort_consec_losses = self._config.get("dry_run.abort_consecutive_losses", 6)

        if self._consecutive_losses >= abort_consec_losses:
            return f"CONSECUTIVE_LOSSES_{self._consecutive_losses}"

        # Check cumulative win rate (after 50+ trades)
        trades = self._resolved_trades
        if len(trades) >= 50:
            win_rate = sum(1 for t in trades if t.outcome == "WIN") / len(trades)
            threshold = self._config.get("dry_run.abort_win_rate_threshold", 0.48)
            if win_rate < threshold:
                return f"WIN_RATE_BELOW_{threshold}"

        return None

    # ── Private Helpers ───────────────────────────────────────

    def _compute_max_drawdown(self, trades: List[PaperTrade]) -> float:
        """Compute max drawdown from equity curve."""
        if not trades:
            return 0.0

        pnls = [t.pnl_usd or 0 for t in trades]
        equity = np.cumsum(pnls) + self._initial_capital
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak

        return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    def _compute_risk_ratios(self, trades: List[PaperTrade]) -> tuple[float, float]:
        """Compute Sharpe and Sortino ratios (per-trade basis)."""
        if len(trades) < 2:
            return 0.0, 0.0

        returns = [(t.pnl_usd or 0) / (t.bet_size + 1e-8) for t in trades]
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))

        # Annualization factor estimate
        n_annual = 96 * 252 * 0.3  # ~30% signal rate assumption

        sharpe = (mean_ret / (std_ret + 1e-8)) * math.sqrt(n_annual) if std_ret > 0 else 0.0

        downside = [r for r in returns if r < 0]
        downside_std = float(np.std(downside)) if downside else 1e-8
        sortino = (mean_ret / (downside_std + 1e-8)) * math.sqrt(n_annual)

        return sharpe, sortino

    def _compute_brier_score(self) -> float:
        """Compute Brier score from all predictions with known outcomes."""
        scored = [
            p for p in self._all_predictions
            if "actual_outcome" in p
        ]
        if not scored:
            return 0.25  # Random baseline

        brier = np.mean([
            (p["P_model"] - p["actual_outcome"]) ** 2
            for p in scored
        ])
        return float(brier)

    def _compute_dry_run_score(
        self, win_rate: float, expectancy: float, sharpe: float, max_dd: float
    ) -> float:
        """
        Composite dry run score per TRD Section 10.4.
        Score >= 0.70 → PASS.
        """
        def normalize(x: float, lower: float, upper: float) -> float:
            return max(0.0, min(1.0, (x - lower) / (upper - lower + 1e-8)))

        score = (
            0.35 * normalize(win_rate, lower=0.50, upper=0.62)
            + 0.25 * normalize(expectancy, lower=-0.01, upper=0.05)
            + 0.20 * normalize(sharpe, lower=0.0, upper=2.0)
            + 0.20 * normalize(1.0 + max_dd, lower=0.80, upper=1.0)
        )
        return score

    def _evaluate_pass_fail(
        self, total_trades: int, win_rate: float, max_dd: float,
        profit_factor: float, dry_run_score: float,
    ) -> str:
        """Evaluate session PASS/FAIL against hard gates."""
        min_trades = self._config.get("dry_run.min_trades_per_session", 10)
        pass_wr = self._config.get("dry_run.pass_win_rate", 0.53)
        pass_pf = self._config.get("dry_run.pass_profit_factor", 1.10)
        pass_dd = self._config.get("dry_run.pass_max_drawdown", -0.15)
        pass_score = self._config.get("dry_run.pass_dry_run_score", 0.70)

        # Hard gates — any failure = FAIL
        if total_trades < min_trades:
            return "FAIL"
        if win_rate < pass_wr:
            return "FAIL"
        if max_dd < pass_dd:
            return "FAIL"
        if profit_factor < pass_pf:
            return "FAIL"
        if dry_run_score < pass_score:
            return "FAIL"

        return "PASS"

    @staticmethod
    def _generate_session_id() -> str:
        """Generate session ID: YYYY-MM-DD_NNN."""
        now = datetime.now(timezone.utc)
        return f"{now.strftime('%Y-%m-%d')}_{now.strftime('%H%M%S')}"

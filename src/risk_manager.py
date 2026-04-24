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
from typing import List, Optional, Union

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging

from src.config_manager import ConfigManager
from src.schemas import ApprovedBet, RejectedBet, SignalResult

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class RiskManager:
    """
    Position sizing and risk gate enforcement.

    Uses Half-Kelly fraction with CLOB-derived decimal odds.
    Dynamic multiplier reduces exposure after consecutive losses.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._position_lock = asyncio.Lock()
        self._open_positions: int = 0
        self._daily_pnl: float = 0.0
        self._session_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trade_history: List[dict] = []

    # ── Public Properties ─────────────────────────────────────

    @property
    def open_positions(self) -> int:
        return self._open_positions

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def session_pnl(self) -> float:
        return self._session_pnl

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

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
            # ── Hard Limit Checks ─────────────────────────────

            daily_loss_limit = self._config.get("risk.daily_loss_limit_pct", 0.05)
            session_loss_limit = self._config.get("risk.session_loss_limit_pct", 0.03)
            max_positions = 1
            min_capital_floor = 5.0

            if self._daily_pnl < -(daily_loss_limit * capital):
                logger.warning(
                    "daily_loss_limit_hit",
                    daily_pnl=round(self._daily_pnl, 2),
                    limit=round(-daily_loss_limit * capital, 2),
                )
                return RejectedBet(signal=signal, reason="DAILY_LOSS_LIMIT_HIT")

            if self._session_pnl < -(session_loss_limit * capital):
                logger.warning(
                    "session_loss_limit_hit",
                    session_pnl=round(self._session_pnl, 2),
                    limit=round(-session_loss_limit * capital, 2),
                )
                return RejectedBet(signal=signal, reason="SESSION_LOSS_LIMIT_HIT")

            if self._open_positions >= max_positions:
                return RejectedBet(signal=signal, reason="MAX_POSITIONS_REACHED")

            if capital < min_capital_floor:
                logger.critical(
                    "capital_below_floor",
                    capital=round(capital, 2),
                    floor=min_capital_floor,
                )
                return RejectedBet(signal=signal, reason="CAPITAL_BELOW_FLOOR")

            # ── Bet Sizing ────────────────────────────────────

            bet_size, kelly_fraction, kelly_multiplier = self._compute_bet_size(
                signal, capital
            )

            min_bet = self._config.get("risk.min_bet_usd", 1.00)
            if bet_size < min_bet:
                return RejectedBet(signal=signal, reason="BET_BELOW_MINIMUM")

            # ── Approve ───────────────────────────────────────

            self._open_positions += 1

            logger.info(
                "trade_approved",
                signal=signal.signal,
                bet_size=round(bet_size, 2),
                kelly_fraction=round(kelly_fraction, 4),
                kelly_multiplier=round(kelly_multiplier, 2),
                capital=round(capital, 2),
                consecutive_losses=self._consecutive_losses,
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
        self, signal: SignalResult, capital: float
    ) -> tuple[float, float, float]:
        """
        Compute bet size using Half-Kelly with dynamic multiplier.

        Returns: (bet_size_usd, kelly_fraction, kelly_multiplier)
        """
        kelly_divisor = self._config.get("risk.kelly_divisor", 2)
        max_bet_frac = self._config.get("risk.max_bet_fraction", 0.10)
        min_bet = self._config.get("risk.min_bet_usd", 1.00)
        multiplier_decay = self._config.get("risk.consecutive_loss_multiplier", 0.15)
        kelly_floor = self._config.get("risk.kelly_floor_multiplier", 0.25)

        # ── Decimal odds from CLOB ────────────────────────────
        if signal.signal == "BUY_YES":
            clob_ask = signal.clob_yes_ask
            # Conservative probability: subtract uncertainty from the event we bet on.
            p_win = max(0.0, min(1.0, signal.P_model - signal.uncertainty_u))
        else:  # BUY_NO
            clob_ask = signal.clob_no_ask
            p_win = max(0.0, min(1.0, (1.0 - signal.P_model) - signal.uncertainty_u))

        # b = decimal odds: profit per unit wagered if win
        # For binary: buy at clob_ask, payout $1 → profit = 1/clob_ask - 1
        if clob_ask <= 0 or clob_ask >= 1:
            return (min_bet, 0.0, 0.0)

        b = (1.0 - clob_ask) / clob_ask

        # ── Full Kelly Fraction ───────────────────────────────
        # f* = (p * b - (1 - p)) / b
        full_kelly = (p_win * b - (1.0 - p_win)) / b

        if full_kelly <= 0:
            return (min_bet, 0.0, 0.0)

        # ── Half-Kelly ────────────────────────────────────────
        half_kelly = max(0.0, full_kelly / kelly_divisor)

        # ── Dynamic Multiplier (consecutive loss decay) ───────
        kelly_multiplier = max(
            kelly_floor,
            1.0 - self._consecutive_losses * multiplier_decay,
        )

        kelly_fraction = half_kelly

        # ── Final Bet Size ────────────────────────────────────
        raw_bet = capital * half_kelly * kelly_multiplier

        # Time-aware shrink factor: reduce exposure as resolution approaches.
        ttr_scale = float(self._config.get("risk.ttr_size_scale_minutes", 10.0))
        time_mult = min(1.0, max(0.25, float(signal.TTR_minutes) / (ttr_scale + 1e-8)))

        # Liquidity proxy: scale down in more aggressive books (higher vig).
        vig = float(signal.clob_yes_ask + signal.clob_no_ask - 1.0)
        max_vig = float(self._config.get("risk.max_vig_for_sizing", 0.07))
        vig_mult = min(1.0, max(0.25, 1.0 - (max(0.0, vig) / (max_vig + 1e-8))))

        bet_size = raw_bet * time_mult * vig_mult
        bet_size = max(bet_size, min_bet)
        bet_size = min(bet_size, capital * max_bet_frac)
        bet_size = round(bet_size, 2)  # USDC 2-decimal precision

        return bet_size, kelly_fraction, kelly_multiplier

    # ── Utility ───────────────────────────────────────────────

    def get_recent_trade_pnls(self, n: int = 30) -> List[float]:
        """Get PnL for last N trades."""
        return [t["pnl"] for t in self._trade_history[-n:]]

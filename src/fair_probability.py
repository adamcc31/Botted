"""
fair_probability.py — Settlement-aligned fair probability engine.

Production intent:
- Compute q_fair = P(S_T >= K) under a digital-option style approximation.
- S is derived from Binance (single source of truth).
- Volatility is estimated from Binance 1-minute returns.
- Output includes an uncertainty buffer (probability-point slack) used by
  the signal and risk layers.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.binance_feed import BinanceFeed
from src.config_manager import ConfigManager
from src.schemas import ActiveMarket, CLOBState


class FairProbResult(BaseModel):
    q_fair: float = Field(..., ge=0.0, le=1.0, description="Fair probability YES at resolution")
    sigma_used_ann: float = Field(..., ge=0.0, description="Annualized volatility used in the formula")
    uncertainty_u: float = Field(
        ...,
        ge=0.0,
        description="Probability-point uncertainty buffer (additive slack, not sigma)",
    )
    tau_seconds: float = Field(..., ge=0.0)
    diagnostics: Dict[str, float] = Field(default_factory=dict)


def _phi(x: float) -> float:
    """Standard normal CDF without scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class FairProbabilityEngine:
    def __init__(self, config: ConfigManager) -> None:
        self._config = config

        # Realized-vol windows in minutes (1-minute bar returns).
        self._fast_window_min = int(self._config.get("fair_prob.sigma_fast_window_min", 120))
        self._slow_window_min = int(self._config.get("fair_prob.sigma_slow_window_min", 1440))
        self._sigma_fast_weight = float(self._config.get("fair_prob.sigma_fast_weight", 0.6))

        # Numerical safety
        self._q_clip = float(self._config.get("fair_prob.q_clip", 1e-6))

        # Uncertainty buffer (probability points)
        self._base_u = float(self._config.get("fair_prob.base_uncertainty_p", 0.03))
        self._near_expiry_u_boost = float(self._config.get("fair_prob.near_expiry_u_boost", 0.10))
        self._min_u = float(self._config.get("fair_prob.min_uncertainty_p", 0.01))
        self._max_u = float(self._config.get("fair_prob.max_uncertainty_p", 0.25))

        # Volatility floors
        self._sigma_floor_ann = float(self._config.get("fair_prob.sigma_floor_ann", 0.20))
        self._sigma_floor_boost_near_expiry = float(
            self._config.get("fair_prob.sigma_floor_boost_near_expiry", 0.35)
        )

    def compute(
        self,
        binance_feed: BinanceFeed,
        active_market: ActiveMarket,
        clob_state: Optional[CLOBState] = None,
        sigma_ann_override: Optional[float] = None,
        data_confidence_override: Optional[float] = None,
        as_of_time: Optional[datetime] = None,
        oracle_price: Optional[float] = None,
    ) -> FairProbResult:
        now = as_of_time if as_of_time is not None else datetime.now(timezone.utc)
        if active_market.T_resolution <= now:
            return FairProbResult(
                q_fair=0.5,
                sigma_used_ann=0.0,
                uncertainty_u=1.0,
                tau_seconds=0.0,
                diagnostics={"reason": 0},
            )

        # ── BUG-2 FIX: Use oracle (Chainlink) price as spot S ──
        # The settlement oracle is Chainlink, NOT Binance. Using Binance
        # mid-price here caused systematic edge miscalculation because
        # q_fair = P(S_T >= K) was computed against the wrong reference.
        #
        # Binance feed is still needed for vol estimation (1m closes),
        # but the spot price S MUST come from the oracle.
        #
        # NO SILENT FALLBACK: if oracle_price is None, we raise.
        # The caller (main.py) is responsible for running the spread
        # filter first and blocking entry when oracle is unavailable.
        if oracle_price is not None and oracle_price > 0:
            s = oracle_price
        else:
            raise ValueError(
                "oracle_price is required for q_fair computation. "
                "Chainlink oracle must be available via DualFeed. "
                "NO SILENT FALLBACK to Binance allowed."
            )

        k = active_market.strike_price
        tau_seconds = max(0.0, (active_market.T_resolution - now).total_seconds())
        tau_years = tau_seconds / 31557600.0  # 365.25 days

        # Vol estimation from 1-minute closes (live) or override (backtest).
        if sigma_ann_override is not None:
            sigma_ann = float(sigma_ann_override)
            data_conf = (
                float(data_confidence_override)
                if data_confidence_override is not None
                else 1.0
            )
            closes_1m = None
        else:
            closes_1m = self._extract_recent_1m_closes(binance_feed)
            sigma_ann = self._estimate_sigma_ann(closes_1m)
            data_conf = self._data_confidence_from_closes(closes_1m)

        # Vol floor: inflate near expiry (digital option sensitivity is high).
        tau_min = tau_seconds / 60.0

        # Detect ultra-short market: if total lifespan ≤ 10 minutes,
        # disable near-expiry bias to avoid over-penalization.
        market_lifespan_min = max(
            1.0,
            (active_market.T_resolution - active_market.T_open).total_seconds() / 60.0
        )
        is_ultrashort = market_lifespan_min <= 10.0

        # Near-expiry weight in [0,1] (0 when far, 1 when <= 5 minutes).
        if is_ultrashort:
            near_w = 0.0
        else:
            near_w = max(0.0, min(1.0, (5.0 - tau_min) / 5.0))
        sigma_floor = self._sigma_floor_ann * (1.0 + self._sigma_floor_boost_near_expiry * near_w)
        sigma_used_ann = max(sigma_ann, sigma_floor)

        # Drift mu: short horizon => default 0.
        mu = float(self._config.get("fair_prob.drift_mu", 0.0))

        # Black-Scholes-like digital approximation.
        # q = P(S_T >= K) = Phi(d2)
        # d2 = (ln(S/K) + (mu - 0.5*sigma^2)*T) / (sigma*sqrt(T))
        if sigma_used_ann <= 0.0 or tau_years <= 0.0:
            q_fair = 0.5
            d2 = 0.0
        else:
            sigma_sqrt_t = sigma_used_ann * math.sqrt(tau_years)
            d2 = (math.log(s / k) + (mu - 0.5 * sigma_used_ann**2) * tau_years) / (sigma_sqrt_t + 1e-12)
            q_fair = _phi(d2)

        # Additive uncertainty buffer (probability points).
        # Base grows when:
        # - tau shrinks (digital sensitivity)
        # - we have less 1m data for the vol estimate
        conf_term = 1.0 + (1.0 - data_conf)  # more missing => larger u
        tau_term = 1.0 + self._near_expiry_u_boost * near_w
        clob_term = 1.0
        if clob_state is not None and hasattr(clob_state, "market_vig"):
            # High vig means spreads/fees are likely worse (execution + model mismatch).
            clob_term = 1.0 + min(1.0, max(0.0, float(clob_state.market_vig) / 0.07))

        uncertainty_u = self._base_u * conf_term * tau_term * clob_term
        uncertainty_u = max(self._min_u, min(self._max_u, uncertainty_u))

        q_fair = max(self._q_clip, min(1.0 - self._q_clip, q_fair))

        return FairProbResult(
            q_fair=q_fair,
            sigma_used_ann=sigma_used_ann,
            uncertainty_u=uncertainty_u,
            tau_seconds=tau_seconds,
            diagnostics={
                "tau_years": tau_years,
                "d2": d2,
                "sigma_ann_raw": sigma_ann,
                "sigma_floor_ann": sigma_floor,
                "near_w": near_w,
                "data_conf": data_conf,
            },
        )

    @staticmethod
    def _extract_recent_1m_closes(binance_feed: BinanceFeed) -> list[float]:
        closes_1m = getattr(binance_feed, "ohlcv_1m_buffer", None)
        if closes_1m is None:
            raise ValueError(
                "BinanceFeed must provide `ohlcv_1m_buffer` for realized-vol estimation."
            )
        return [bar["close"] for bar in closes_1m if bar.get("close") is not None]

    def _estimate_sigma_ann(self, closes_1m: list[float]) -> float:
        if len(closes_1m) < 10:
            return float(self._config.get("fair_prob.sigma_default_ann", 0.30))

        fast_n = min(self._fast_window_min, len(closes_1m) - 1)
        slow_n = min(self._slow_window_min, len(closes_1m) - 1)

        sigma_fast = self._realized_sigma_ann_from_closes(closes_1m, fast_n)
        sigma_slow = self._realized_sigma_ann_from_closes(closes_1m, slow_n)

        w = self._sigma_fast_weight
        # If slow window is too short, effectively rely on fast.
        if slow_n < (self._slow_window_min * 0.6):
            return sigma_fast
        return w * sigma_fast + (1.0 - w) * sigma_slow

    @staticmethod
    def _realized_sigma_ann_from_closes(closes: list[float], window_n: int) -> float:
        """
        Estimate sigma from log returns over the last `window_n` 1-minute returns.
        """
        if window_n <= 2:
            return 0.0

        recent = closes[-(window_n + 1) :]
        if len(recent) < 3:
            return 0.0

        rets = []
        for i in range(1, len(recent)):
            a = recent[i - 1]
            b = recent[i]
            if a is None or b is None or a <= 0 or b <= 0:
                continue
            rets.append(math.log(b / a))

        if len(rets) < 2:
            return 0.0

        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / max(1, (len(rets) - 1))
        std = math.sqrt(max(0.0, var))

        # Annualize: 365.25 days * 24h * 60min
        minutes_per_year = 365.25 * 24.0 * 60.0
        return std * math.sqrt(minutes_per_year)

    @staticmethod
    def _data_confidence_from_closes(closes_1m: list[float]) -> float:
        # Confidence in [0,1] based on availability of at least fast window.
        if len(closes_1m) < 10:
            return 0.0
        return max(0.0, min(1.0, (len(closes_1m) - 1) / 120.0))


"""
signal_generator.py — 6-step signal generation with hard gates.

Decision Flow:
  STEP 1: TTR Gate (ENTRY_WINDOW only)
  STEP 2: Regime Filter (vol percentile + spread + depth)
  STEP 3: Liquidity Filter (CLOB depth + vig + staleness)
  STEP 4: Mispricing Calculation (edge_yes, edge_no)
  STEP 5: Margin of Safety Check (edge > threshold)
  STEP 6: Final Signal Selection (BUY_UP / BUY_DOWN / ABSTAIN)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None
import logging

from src.config_manager import ConfigManager
from src.schemas import (
    ActiveMarket,
    CLOBState,
    FeatureMetadata,
    FeatureVector,
    SignalResult,
)
from src.zone_matrix import classify_zone

logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)


class SignalGenerator:
    """
    Evaluates model output against hard gates to produce trading signals.

    Expected abstain rate: ~60-75%. High selectivity is the edge, not a weakness.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    def _resolve_ttr_window(self, active_market: ActiveMarket) -> tuple[float, float]:
        """Dynamic TTR entry window aligned with market horizon."""
        dyn_enabled = bool(self._config.get("signal.dynamic_ttr_enabled", True))
        if not dyn_enabled:
            return (
                float(self._config.get("signal.ttr_min_minutes", 5.0)),
                float(self._config.get("signal.ttr_max_minutes", 12.0)),
            )

        lifespan_h = max(
            0.0,
            (active_market.T_resolution - active_market.T_open).total_seconds() / 3600.0,
        )
        lifespan_min = lifespan_h * 60.0

        # Ultra-short market (≤ 10 minutes lifespan)
        is_ultrashort = "5m" in active_market.slug or lifespan_min <= 10.0
        if is_ultrashort:
            actual_lifespan = 5.0 if "5m" in active_market.slug else lifespan_min
            entry_open_pct = float(self._config.get("signal.ultrashort_entry_open_pct", 0.80))
            entry_close_pct = float(self._config.get("signal.ultrashort_entry_close_pct", 0.10))
            return (
                actual_lifespan * entry_close_pct,
                actual_lifespan * entry_open_pct,
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

    def evaluate(
        self,
        P_model: float,
        uncertainty_u: float,
        clob_state: CLOBState,
        active_market: ActiveMarket,
        feature_vector: FeatureVector,
    ) -> SignalResult:
        """
        Run full 6-step signal evaluation pipeline.

        Returns SignalResult with signal direction + full diagnostics.
        """
        now = datetime.now(timezone.utc)
        metadata = feature_vector.metadata

        # Build feature dict for logging
        features_dict = dict(zip(feature_vector.feature_names, feature_vector.values))

        # Common fields
        base = {
            "clob_yes_ask": clob_state.yes_ask,
            "clob_yes_bid": clob_state.yes_bid,
            "clob_no_ask": clob_state.no_ask,
            "clob_no_bid": clob_state.no_bid,
            "entry_odds": None, # Placeholder, will be updated for active signals
            "TTR_minutes": metadata.TTR_minutes,
            "strike_price": active_market.strike_price,
            "current_price": metadata.current_btc_price,
            "strike_distance": features_dict.get("strike_distance_pct", 0.0),
            "market_id": active_market.market_id,
            "timestamp": now,
            "features": features_dict,
        }

        # Basis-risk adjustment:
        # If settlement source is not Binance 1m close, probability accuracy
        # degrades. We either abstain near resolution or inflate u.
        basis_mismatch = not (
            active_market.settlement_exchange == "BINANCE"
            and active_market.settlement_granularity == "1m"
        )
        non_binance_policy = self._config.get(
            "settlement.non_binance_policy", "uncertainty_inflate"
        )
        non_binance_abstain_ttr_min = float(
            self._config.get("settlement.non_binance_abstain_ttr_min", 6.0)
        )
        u_used = uncertainty_u
        if basis_mismatch:
            if non_binance_policy == "abstain":
                # If configured to hard-abstain, stop immediately regardless of TTR.
                edge_yes = P_model - clob_state.yes_ask
                edge_no = (1.0 - P_model) - clob_state.no_ask
                logger.debug(
                    "signal_abstain_basis_risk",
                    policy=non_binance_policy,
                    settlement_exchange=active_market.settlement_exchange,
                    settlement_granularity=active_market.settlement_granularity,
                )
                return SignalResult(
                    signal="ABSTAIN",
                    abstain_reason="BASIS_RISK_BLOCK",
                    P_model=P_model,
                    uncertainty_u=uncertainty_u,
                    edge_yes=edge_yes,
                    edge_no=edge_no,
                    clob_yes_ask=clob_state.yes_ask,
                    clob_yes_bid=clob_state.yes_bid,
                    clob_no_ask=clob_state.no_ask,
                    clob_no_bid=clob_state.no_bid,
                    TTR_minutes=metadata.TTR_minutes,
                    strike_price=active_market.strike_price,
                    current_price=metadata.current_btc_price,
                    strike_distance=features_dict.get("strike_distance_pct", 0.0),
                    market_id=active_market.market_id,
                    timestamp=now,
                    features=features_dict,
                )

            # Allow trading but with uncertainty inflation. If we are too close to
            # resolution, halt to avoid binary flip risk.
            if metadata.TTR_minutes < non_binance_abstain_ttr_min:
                edge_yes = P_model - clob_state.yes_ask
                edge_no = (1.0 - P_model) - clob_state.no_ask
                logger.debug(
                    "signal_abstain_basis_risk",
                    policy=non_binance_policy,
                    TTR_minutes=round(metadata.TTR_minutes, 2),
                    min_ttr=non_binance_abstain_ttr_min,
                    settlement_exchange=active_market.settlement_exchange,
                    settlement_granularity=active_market.settlement_granularity,
                )
                return SignalResult(
                    signal="ABSTAIN",
                    abstain_reason="BASIS_RISK_BLOCK",
                    P_model=P_model,
                    uncertainty_u=uncertainty_u,
                    edge_yes=edge_yes,
                    edge_no=edge_no,
                    clob_yes_ask=clob_state.yes_ask,
                    clob_yes_bid=clob_state.yes_bid,
                    clob_no_ask=clob_state.no_ask,
                    clob_no_bid=clob_state.no_bid,
                    TTR_minutes=metadata.TTR_minutes,
                    strike_price=active_market.strike_price,
                    current_price=metadata.current_btc_price,
                    strike_distance=features_dict.get("strike_distance_pct", 0.0),
                    market_id=active_market.market_id,
                    timestamp=now,
                    features=features_dict,
                )

            non_binance_u_mult = float(
                self._config.get("settlement.non_binance_u_multiplier", 2.0)
            )
            u_used = uncertainty_u * non_binance_u_mult

        # ── STEP 1: TTR GATE ─────────────────────────────────
        ttr_min, ttr_max = self._resolve_ttr_window(active_market)
        ttr = metadata.TTR_minutes

        if ttr < ttr_min or ttr > ttr_max:
            edge_yes = P_model - clob_state.yes_ask
            edge_no = (1.0 - P_model) - clob_state.no_ask
            logger.debug(
                "signal_abstain_ttr",
                TTR_minutes=round(ttr, 2),
                phase=metadata.TTR_phase,
                ttr_min=ttr_min,
                ttr_max=ttr_max,
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="TTR_PHASE",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 2: REGIME FILTER ────────────────────────────
        vol_lower = self._config.get("regime.vol_lower_threshold", 0.15)
        vol_upper = self._config.get("regime.vol_upper_threshold", 0.80)
        spread_max = self._config.get("regime.binance_spread_max_bps", 5.0)
        depth_min = self._config.get("regime.binance_min_depth_btc", 0.5)

        vol_pct = features_dict.get("vol_percentile", 0.5)
        spread_bps = features_dict.get("binance_spread_bps", 2.0)
        # Depth check uses feature_engine's depth_ratio as proxy
        depth_ratio = features_dict.get("depth_ratio", 1.0)

        regime_vol_ok = vol_lower < vol_pct < vol_upper
        spread_ok = spread_bps < spread_max
        # Depth check: simplified — top5 bid depth as ratio proxy
        depth_ok = True  # Will be properly checked via binance_feed in orchestrator

        regime_go = regime_vol_ok and spread_ok and depth_ok

        if not regime_go:
            edge_yes = P_model - clob_state.yes_ask
            edge_no = (1.0 - P_model) - clob_state.no_ask
            logger.debug(
                "signal_abstain_regime",
                vol_pct=round(vol_pct, 3),
                spread_bps=round(spread_bps, 2),
                regime_vol_ok=regime_vol_ok,
                spread_ok=spread_ok,
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="REGIME_BLOCK",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 3: LIQUIDITY FILTER ─────────────────────────
        if not clob_state.is_liquid or clob_state.is_stale:
            edge_yes = P_model - clob_state.yes_ask
            edge_no = (1.0 - P_model) - clob_state.no_ask
            logger.debug(
                "signal_abstain_liquidity",
                is_liquid=clob_state.is_liquid,
                is_stale=clob_state.is_stale,
                yes_depth=round(clob_state.yes_depth_usd, 2),
                no_depth=round(clob_state.no_depth_usd, 2),
                vig=round(clob_state.market_vig, 4),
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="LIQUIDITY_BLOCK",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 4: MISPRICING CALCULATION ───────────────────
        edge_yes_raw = P_model - clob_state.yes_ask
        edge_no_raw = (1.0 - P_model) - clob_state.no_ask

        # Conservative edges: subtract uncertainty buffer.
        edge_yes = edge_yes_raw - u_used
        edge_no = edge_no_raw - u_used

        # ── No-trade zone around fair (prevents churn) ─────────
        no_trade_zone_p = float(self._config.get("signal.no_trade_deadband", 0.02))
        mid_yes = (clob_state.yes_bid + clob_state.yes_ask) / 2.0
        
        # Bypass deadband for ultra-short markets – we WANT to trade at the money
        res_time = getattr(active_market, "T_resolution", None)
        open_time = getattr(active_market, "T_open", None)
        if res_time is None or open_time is None:
             is_ultrashort = "5m" in active_market.slug
        else:
             is_ultrashort = "5m" in active_market.slug or (res_time - open_time).total_seconds() / 60.0 <= 10.0
        
        # Guard clause for deadband calculation
        mid_yes = None
        if clob_state.yes_bid is not None and clob_state.yes_ask is not None:
            mid_yes = (clob_state.yes_bid + clob_state.yes_ask) / 2.0
        
        if mid_yes is None:
            logger.warning("mid_yes_null_abstain", market_id=active_market.market_id)
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="LIQUIDITY_BLOCK",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        if not is_ultrashort and mid_yes is not None and P_model is not None and u_used is not None:
            if abs(P_model - mid_yes) <= (no_trade_zone_p + u_used):
                logger.debug(
                    "signal_abstain_no_trade_zone",
                    fair_prob=round(P_model, 4),
                    mid_yes=round(mid_yes, 4),
                    deadband=round(no_trade_zone_p, 4),
                    uncertainty_u=round(u_used, 4),
                )
                return SignalResult(
                    signal="ABSTAIN",
                    abstain_reason="NO_TRADE_ZONE",
                    P_model=P_model,
                    uncertainty_u=u_used,
                    edge_yes=edge_yes,
                    edge_no=edge_no,
                    **base,
                )

        # ── STEP 5: MARGIN OF SAFETY CHECK ───────────────────
        margin = self._config.get("signal.margin_of_safety", 0.05)

        if max(edge_yes, edge_no) <= margin:
            logger.debug(
                "signal_abstain_no_edge",
                edge_yes=round(edge_yes, 4),
                edge_no=round(edge_no, 4),
                margin_of_safety=margin,
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="NO_EDGE",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        # ── STEP 6: FINAL SIGNAL SELECTION ───────────────────
        if edge_yes > margin and edge_no > margin:
            # Both edges positive — pick larger
            signal = "BUY_UP" if edge_yes >= edge_no else "BUY_DOWN"
        elif edge_yes > margin:
            signal = "BUY_UP"
        else:
            signal = "BUY_DOWN"

        # ── STEP 7: PREDATOR ZONE GATING ─────────────────────
        entry_odds = clob_state.yes_ask if signal == "BUY_UP" else clob_state.no_ask
        base["entry_odds"] = entry_odds
        
        curr_price = metadata.current_btc_price
        strike_price = active_market.strike_price
        
        if curr_price is None or strike_price is None:
            logger.warning("signal_gating_missing_prices", market_id=active_market.market_id)
            # Fallback to neutral or abstain
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="ORACLE_UNAVAILABLE",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )
            
        distance_usd = abs(curr_price - strike_price)
        zone = classify_zone(ttr, distance_usd, entry_odds)
        base["zone_id"] = zone.zone_id

        if zone.zone_type == "DEATH":
            logger.debug(
                "signal_abstain_death_zone",
                zone_id=zone.zone_id,
                ttr=round(ttr, 2),
                distance=round(distance_usd, 2),
                odds=round(entry_odds, 3),
            )
            # Cannot use DEAD_ZONE_BLOCK explicitly yet if it's not in the Literal schema, we just use a generic or string.
            # But the schema actually has no DEAD_ZONE_BLOCK in abstain_reason. Let's add it or use NO_TRADE_ZONE
            # Wait, the schema has strict literals. I will update schemas.py first or just use NO_TRADE_ZONE
            # Let's check schema.py: it doesn't have "DEATH_ZONE_BLOCK" or "NEUTRAL_ZONE_BLOCK". I will use "NO_TRADE_ZONE" and log it.
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="NO_TRADE_ZONE",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        if zone.zone_type == "NEUTRAL":
            logger.debug(
                "signal_abstain_neutral_zone",
                zone_id=zone.zone_id,
                ttr=round(ttr, 2),
                distance=round(distance_usd, 2),
                odds=round(entry_odds, 3),
            )
            return SignalResult(
                signal="ABSTAIN",
                abstain_reason="NO_TRADE_ZONE",
                P_model=P_model,
                uncertainty_u=u_used,
                edge_yes=edge_yes,
                edge_no=edge_no,
                **base,
            )

        logger.info(
            "signal_generated",
            signal=signal,
            zone_id=zone.zone_id,
            P_model=round(P_model, 4),
            edge_yes=round(edge_yes, 4),
            edge_no=round(edge_no, 4),
            TTR_minutes=round(ttr, 2),
            market_id=active_market.market_id,
        )

        return SignalResult(
            signal=signal,
            abstain_reason=None,
            P_model=P_model,
            uncertainty_u=u_used,
            edge_yes=edge_yes,
            edge_no=edge_no,
            **base,
        )

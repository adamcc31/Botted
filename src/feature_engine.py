"""
feature_engine.py — 24-feature computation with strict anti-lookahead.

ALL rolling computations use shift(1) before window calculation.
Value at time t only uses data from [t-window, t-1].

Features:
  01-08: Microstructure (OBI, TFM, VAM, RV, vol_percentile, depth_ratio, price_vs_ema20, spread_bps)
  09-12: Temporal cyclical (hour_sin/cos, dow_sin/cos)
  13-15: TTR contextual (TTR_normalized, TTR_sin, TTR_cos)
  16-17: Strike contextual (strike_distance_pct, contest_urgency)
  18-20: Interaction (ttr_x_obi, ttr_x_tfm, ttr_x_strike)
  21-24: CLOB (clob_yes_mid, clob_yes_spread, clob_no_spread, market_vig)
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from src.binance_feed import BinanceFeed
from src.config_manager import ConfigManager
from src.schemas import ActiveMarket, CLOBState, FeatureMetadata, FeatureVector

logger = structlog.get_logger(__name__)

EPSILON = 1e-8
FEATURE_LIST_PATH = Path(__file__).parent.parent / "config" / "feature_list.json"


def load_feature_list() -> List[str]:
    """Load immutable feature ordering from config."""
    with open(FEATURE_LIST_PATH, "r") as f:
        return json.load(f)


FEATURE_NAMES = load_feature_list()
assert len(FEATURE_NAMES) == 24, f"Expected 24 features, got {len(FEATURE_NAMES)}"


def z_score_safe(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Z-score normalization safe from lookahead bias.
    CRITICAL: shift(1) ensures rolling uses only data before t.
    """
    shifted = series.shift(1)
    rolling_mean = shifted.rolling(window=window, min_periods=20).mean()
    rolling_std = shifted.rolling(window=window, min_periods=20).std()
    return (series - rolling_mean) / (rolling_std + EPSILON)


class FeatureEngine:
    """
    Computes 24-feature vector for model inference.

    Anti-lookahead guarantee: all rolling windows use shift(1).
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    def compute(
        self,
        binance_feed: BinanceFeed,
        active_market: ActiveMarket,
        clob_state: CLOBState,
        oracle_price: Optional[float] = None,
    ) -> Optional[FeatureVector]:
        """
        Compute full 24-feature vector from current state.

        Args:
            binance_feed: Live Binance data feed (used ONLY for order flow features)
            active_market: Current market metadata
            clob_state: Current CLOB state
            oracle_price: Chainlink oracle BTC price from DualFeed.
                          Used for strike_distance_pct, contest_urgency, and metadata.
                          If None, raises ValueError — NO silent fallback to Binance.

        Returns:
            FeatureVector with 24 features + metadata, or None if insufficient data.
        """
        start_time = time.time()

        ohlcv = binance_feed.ohlcv_buffer
        if len(ohlcv) < 21:  # Need at least 21 bars for EMA20
            logger.warning("insufficient_ohlcv_data", bars=len(ohlcv))
            return None

        # Binance price — used ONLY for order flow features (OBI, TFM, depth, spread)
        binance_price = binance_feed.latest_price
        if binance_price is None:
            logger.warning("no_binance_price_for_order_flow")
            return None

        # ── BUG-1 FIX: Oracle price for strike-relative features ──
        # strike_distance_pct, contest_urgency, and current_btc_price
        # must use the settlement-aligned oracle price, not Binance.
        # NO SILENT FALLBACK: caller must provide oracle_price.
        if oracle_price is None or oracle_price <= 0:
            raise ValueError(
                "oracle_price is required for feature computation. "
                "Chainlink oracle must be available via DualFeed. "
                "NO SILENT FALLBACK to Binance allowed."
            )
        reference_price = oracle_price

        now = datetime.now(timezone.utc)

        # Build feature values
        try:
            features = {}

            # ── 01: OBI ───────────────────────────────────────
            obi = binance_feed.get_ob_imbalance(levels=5)
            features["OBI"] = obi if obi is not None else 0.0

            # ── 02: TFM_normalized ────────────────────────────
            features["TFM_normalized"] = self._compute_tfm(binance_feed)

            # ── 03: VAM ───────────────────────────────────────
            features["VAM"] = self._compute_vam(ohlcv)

            # ── 04: RV ────────────────────────────────────────
            features["RV"] = self._compute_rv(ohlcv)

            # ── 05: vol_percentile ────────────────────────────
            features["vol_percentile"] = self._compute_vol_percentile(ohlcv)

            # ── 06: depth_ratio ───────────────────────────────
            dr = binance_feed.get_depth_ratio(levels=3)
            features["depth_ratio"] = dr if dr is not None else 1.0

            # ── 07: price_vs_ema20 ────────────────────────────
            features["price_vs_ema20"] = self._compute_price_vs_ema20(ohlcv)

            # ── 08: binance_spread_bps ────────────────────────
            spread = binance_feed.get_binance_spread_bps()
            features["binance_spread_bps"] = spread if spread is not None else 2.0

            # ── 09-12: Temporal cyclical ──────────────────────
            utc_hour = now.hour + now.minute / 60.0
            dow = now.weekday()  # 0=Monday, 6=Sunday

            features["hour_sin"] = math.sin(2 * math.pi * utc_hour / 24.0)
            features["hour_cos"] = math.cos(2 * math.pi * utc_hour / 24.0)
            features["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
            features["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)

            # ── 13-15: TTR contextual ─────────────────────────
            ttr_seconds = (active_market.T_resolution - now).total_seconds()
            ttr_minutes = max(0.0, ttr_seconds / 60.0)
            
            # Adaptive normalization
            # If market lifespan > 2h, assume Daily (normalize by 24h)
            # Otherwise assume 15m
            lifespan_h = (active_market.T_resolution - active_market.T_open).total_seconds() / 3600.0
            ttr_norm_base = 1440.0 if lifespan_h > 2.0 else 15.0
            ttr_normalized = max(0.0, min(1.0, ttr_minutes / ttr_norm_base))

            features["TTR_normalized"] = ttr_normalized
            features["TTR_sin"] = math.sin(math.pi * ttr_normalized)
            features["TTR_cos"] = math.cos(math.pi * ttr_normalized)

            # ── 16: Strike distance ───────────────────────────
            # BUG-1 FIX: use oracle reference_price, NOT Binance
            strike = active_market.strike_price
            if strike is None or reference_price is None or strike == 0:
                strike_distance_pct = 0.0
            else:
                strike_distance_pct = (reference_price - strike) / strike * 100.0
            
            features["strike_distance_pct"] = strike_distance_pct
 
            # ── 17: Contest urgency ───────────────────────────
            features["contest_urgency"] = abs(strike_distance_pct) * (1.0 - ttr_normalized) if strike_distance_pct is not None else 0.0

            # ── 18-20: Interaction features ───────────────────
            features["ttr_x_obi"] = ttr_normalized * features["OBI"]
            features["ttr_x_tfm"] = ttr_normalized * features["TFM_normalized"]
            features["ttr_x_strike"] = ttr_normalized * strike_distance_pct

            # ── 21-24: CLOB features ─────────────────────────
            features["clob_yes_mid"] = (clob_state.yes_ask + clob_state.yes_bid) / 2.0
            features["clob_yes_spread"] = clob_state.yes_ask - clob_state.yes_bid
            features["clob_no_spread"] = clob_state.no_ask - clob_state.no_bid
            features["market_vig"] = clob_state.market_vig

            # ── Assemble in canonical order ───────────────────
            values = [features[name] for name in FEATURE_NAMES]

            # Replace NaN/Inf with 0
            values = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in values]

            compute_lag_ms = (time.time() - start_time) * 1000.0

            # Determine TTR phase (supports dynamic policy by market horizon)
            dyn_enabled = bool(self._config.get("signal.dynamic_ttr_enabled", True))
            if dyn_enabled:
                if lifespan_h <= 2.0:
                    ttr_min = float(
                        self._config.get("signal.entry_window_short_min_minutes", 5.0)
                    )
                    ttr_max = float(
                        self._config.get("signal.entry_window_short_max_minutes", 45.0)
                    )
                elif lifespan_h <= 8.0:
                    ttr_min = float(
                        self._config.get("signal.entry_window_medium_min_minutes", 30.0)
                    )
                    ttr_max = float(
                        self._config.get("signal.entry_window_medium_max_minutes", 240.0)
                    )
                else:
                    ttr_min = float(
                        self._config.get("signal.entry_window_long_min_minutes", 60.0)
                    )
                    ttr_max = float(
                        self._config.get("signal.entry_window_long_max_minutes", 720.0)
                    )
            else:
                ttr_min = float(self._config.get("signal.ttr_min_minutes", 5.0))
                ttr_max = float(self._config.get("signal.ttr_max_minutes", 12.0))
            if ttr_minutes > ttr_max:
                ttr_phase = "EARLY"
            elif ttr_minutes >= ttr_min:
                ttr_phase = "ENTRY_WINDOW"
            else:
                ttr_phase = "LATE"

            metadata = FeatureMetadata(
                timestamp=now,
                bar_close_time=datetime.fromtimestamp(
                    ohlcv[-1]["close_time"] / 1000.0, tz=timezone.utc
                ),
                market_id=active_market.market_id,
                strike_price=strike,
                current_btc_price=reference_price,  # BUG-1 FIX: oracle price
                TTR_minutes=ttr_minutes,
                TTR_phase=ttr_phase,
                compute_lag_ms=compute_lag_ms,
            )

            fv = FeatureVector(
                values=values,
                feature_names=FEATURE_NAMES,
                metadata=metadata,
            )

            if compute_lag_ms > 500:
                logger.warning("feature_compute_slow", lag_ms=round(compute_lag_ms, 1))

            return fv

        except Exception as e:
            logger.error("feature_compute_error", error=str(e))
            return None

    # ── Private Computation Methods ───────────────────────────

    def _compute_tfm(self, binance_feed: BinanceFeed) -> float:
        """
        Trade Flow Momentum (normalized).
        TFM_raw = Σ taker_buy_vol[60s] - Σ taker_sell_vol[60s]
        TFM_normalized = TFM_raw / (std(TFM_raw, window=20) + ε)
        """
        buy_vol, sell_vol = binance_feed.get_trade_flow_data(window_seconds=60)
        tfm_raw = buy_vol - sell_vol

        # For normalization, we'd need historical TFM values
        # In live mode, use a simple approximation based on current magnitude
        total_vol = buy_vol + sell_vol
        if total_vol < EPSILON:
            return 0.0

        # Simple normalization by total volume
        return tfm_raw / (total_vol + EPSILON)

    def _compute_vam(self, ohlcv: list) -> float:
        """
        Volatility-Adjusted Momentum.
        close_returns = ln(close[t] / close[t-1])
        realized_vol = std(close_returns, window=12)
        VAM = close_returns / (realized_vol + ε)
        """
        if len(ohlcv) < 13:
            return 0.0

        closes = [bar["close"] for bar in ohlcv]
        # Anti-lookahead: use [t-12, t-1] for vol, current return uses t
        if len(closes) < 2:
            return 0.0

        current_return = math.log(closes[-1] / (closes[-2] + EPSILON))

        # Realized vol from prior 12 bars (shift(1) equivalent)
        prior_closes = closes[-(13):-1]  # [t-12 to t-1]
        if len(prior_closes) < 2:
            return 0.0

        returns = [
            math.log(prior_closes[i] / (prior_closes[i - 1] + EPSILON))
            for i in range(1, len(prior_closes))
        ]
        if not returns:
            return 0.0

        std_returns = float(np.std(returns))
        return current_return / (std_returns + EPSILON)

    def _compute_rv(self, ohlcv: list) -> float:
        """
        Realized Volatility (annualized).
        RV = std(close_returns, window=12) × √(252 × 96)
        """
        if len(ohlcv) < 13:
            return 0.0

        closes = [bar["close"] for bar in ohlcv]
        # Anti-lookahead: window from [t-12, t-1]
        prior_closes = closes[-(13):-1]
        if len(prior_closes) < 2:
            return 0.0

        returns = [
            math.log(prior_closes[i] / (prior_closes[i - 1] + EPSILON))
            for i in range(1, len(prior_closes))
        ]
        if not returns:
            return 0.0

        std_returns = float(np.std(returns))
        annualizer = math.sqrt(252 * 96)  # 96 bars per day, 252 trading days
        return std_returns * annualizer

    def _compute_vol_percentile(self, ohlcv: list) -> float:
        """
        Volatility Percentile.
        vol_percentile = rolling_rank(RV[t], window=500) / 500
        Anti-lookahead: rank from [t-500, t-1].
        """
        window = min(len(ohlcv) - 1, 500)
        if window < 20:
            return 0.5  # Default neutral

        # Compute RV for each bar in window using [bar-12, bar-1]
        closes = [bar["close"] for bar in ohlcv]
        rvs = []

        for i in range(max(13, len(closes) - window), len(closes)):
            prior = closes[max(0, i - 12):i]
            if len(prior) < 2:
                continue
            rets = [
                math.log(prior[j] / (prior[j - 1] + EPSILON))
                for j in range(1, len(prior))
            ]
            if rets:
                rvs.append(float(np.std(rets)))

        if not rvs:
            return 0.5

        current_rv = rvs[-1] if rvs else 0.0
        rank = sum(1 for rv in rvs[:-1] if rv <= current_rv)
        return rank / (len(rvs) - 1 + EPSILON)

    def _compute_price_vs_ema20(self, ohlcv: list) -> float:
        """
        Price vs EMA20.
        (close[t] - EMA(close, 20)) / (close[t] + ε)
        Anti-lookahead: EMA from [t-20, t-1].
        """
        if len(ohlcv) < 21:
            return 0.0

        closes = pd.Series([bar["close"] for bar in ohlcv])
        # EMA calculated from shifted data (anti-lookahead)
        ema = closes.shift(1).ewm(span=20, adjust=False).mean()
        current_close = closes.iloc[-1]
        ema_value = ema.iloc[-1]

        if pd.isna(ema_value):
            return 0.0

        return (current_close - ema_value) / (current_close + EPSILON)

    # ── Batch Feature Computation (for training) ──────────────

    def compute_batch(
        self,
        ohlcv_df: pd.DataFrame,
        strike_price: float,
        T_resolution: datetime,
        clob_yes_ask: float = 0.5,
        clob_yes_bid: float = 0.48,
        clob_no_ask: float = 0.5,
        clob_no_bid: float = 0.48,
    ) -> pd.DataFrame:
        """
        Compute all features for a batch of historical data (training).

        Uses vectorized pandas operations with strict anti-lookahead.
        """
        df = ohlcv_df.copy()

        # Log returns with shift(1) for anti-lookahead
        df["close_return"] = np.log(df["close"] / df["close"].shift(1))

        # Feature 03: VAM
        shifted_returns = df["close_return"].shift(1)
        rv_window = shifted_returns.rolling(window=12, min_periods=2).std()
        df["VAM"] = df["close_return"] / (rv_window + EPSILON)

        # Feature 04: RV (annualized)
        df["RV"] = rv_window * math.sqrt(252 * 96)

        # Feature 05: Vol percentile
        df["vol_percentile"] = df["RV"].shift(1).rolling(500, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Feature 07: Price vs EMA20
        ema20 = df["close"].shift(1).ewm(span=20, adjust=False).mean()
        df["price_vs_ema20"] = (df["close"] - ema20) / (df["close"] + EPSILON)

        # Features 09-12: Time cyclical
        if "close_time" in df.columns:
            timestamps = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        else:
            timestamps = pd.to_datetime(df.index)

        hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
        dows = timestamps.dt.weekday

        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)

        # Feature 16: Strike distance
        df["strike_distance_pct"] = (df["close"] - strike_price) / strike_price * 100.0

        # CLOB features (static for training — will need actual CLOB data)
        df["clob_yes_mid"] = (clob_yes_ask + clob_yes_bid) / 2.0
        df["clob_yes_spread"] = clob_yes_ask - clob_yes_bid
        df["clob_no_spread"] = clob_no_ask - clob_no_bid
        df["market_vig"] = clob_yes_ask + clob_no_ask - 1.0

        return df

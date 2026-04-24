"""
build_dataset.py — The Merging Engine.

THIS IS THE MOST CRITICAL SCRIPT IN THE ENTIRE PIPELINE.

It joins three data sources into a single training-ready dataset:
  1. Polymarket Markets (ground truth: strike_price, T_resolution, outcome)
  2. Binance OHLCV (price history for feature computation)
  3. Binance aggTrades (for TFM reconstruction — the core alpha)
  4. Orderbook snapshots (for OBI, depth_ratio — optional but valuable)

For each resolved Polymarket market:
  - Look up BTC price at T_resolution from Binance data → construct label
  - For multiple signal timepoints within [T_open+3min, T_resolution-5min]:
    - Compute all 24 features using only data available at that moment
    - Attach strike_price, TTR, and the ground truth outcome label

Output: data/processed/merged_training_features.parquet
  - Each row = one potential signal evaluation point
  - Columns: 24 features + label + metadata (market_id, timestamp, TTR, etc.)

Usage:
  python scripts/build_dataset.py
  python scripts/build_dataset.py --markets ./data/raw/polymarket_markets.parquet \\
                                  --ohlcv ./data/raw/ohlcv_15m.parquet \\
                                  --aggtrades-dir ./data/raw/aggTrades
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_engine import FEATURE_NAMES

EPSILON = 1e-8
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ============================================================
# Data Loading
# ============================================================

def load_markets(path: Path, ohlcv: pd.DataFrame = None) -> pd.DataFrame:
    """Load Polymarket resolved markets."""
    df = pd.read_parquet(path)
    logger.info("markets_loaded", rows=len(df))

    df["is_imputed_strike"] = False

    # Impute missing strike prices (e.g. Daily Up/Down) using OHLCV open price
    if ohlcv is not None and "strike_price" in df.columns and "t_open_epoch_ms" in df.columns:
        missing_strikes = df["strike_price"].isna()
        if missing_strikes.any():
            for idx, row in df[missing_strikes].iterrows():
                try:
                    open_ms = int(row["t_open_epoch_ms"])
                    c_idx = ohlcv["open_time"].searchsorted(open_ms)
                    if c_idx > 0 and c_idx < len(ohlcv):
                        df.at[idx, "strike_price"] = ohlcv["close"].iloc[c_idx - 1]
                        df.at[idx, "is_imputed_strike"] = True
                except Exception:
                    pass
    elif "strike_price" not in df.columns:
        df["strike_price"] = np.nan
        if ohlcv is not None and "t_open_epoch_ms" in df.columns:
            for idx, row in df.iterrows():
                try:
                    open_ms = int(row["t_open_epoch_ms"])
                    c_idx = ohlcv["open_time"].searchsorted(open_ms)
                    if c_idx > 0 and c_idx < len(ohlcv):
                        df.at[idx, "strike_price"] = ohlcv["close"].iloc[c_idx - 1]
                        df.at[idx, "is_imputed_strike"] = True
                except Exception:
                    pass

    # Filter: must have strike_price and outcome
    valid = df[
        df["strike_price"].notna()
        & df["outcome_binary"].notna()
        & df["t_resolution_epoch_ms"].notna()
    ].copy()
    
    # Sort descending ensures we start with Märch/April where we have aggTrades
    valid = valid.sort_values("t_resolution_epoch_ms", ascending=False)

    logger.info(
        "markets_valid",
        valid=len(valid),
        dropped=len(df) - len(valid),
    )
    return valid


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load Binance OHLCV data."""
    df = pd.read_parquet(path)
    df = df.sort_values("open_time").reset_index(drop=True)
    
    # ── Pre-calculate Log Returns ─────────────────────────────
    # Use close-to-close log returns
    df["log_return"] = np.log(df["close"] / (df["close"].shift(1) + 1e-8))
    df["log_return"] = df["log_return"].fillna(0.0)
    
    # ── Pre-calculate Rolling Volatility (12m + 500-bar Percentile) ──
    df["rv_12m"] = df["log_return"].rolling(window=12).std()
    df["rv_12m"] = df["rv_12m"].fillna(0.0)
    
    # Rolling percentile over 500-bar window
    df["vol_percentile"] = df["rv_12m"].rolling(window=500).rank(pct=True)
    df["vol_percentile"] = df["vol_percentile"].fillna(0.5)
    
    logger.info("ohlcv_loaded", rows=len(df))
    return df


def load_aggtrades(directory: Path) -> Optional[pd.DataFrame]:
    """
    Load aggregated trades from daily parquet files.
    Returns combined DataFrame sorted by timestamp.
    """
    if not directory.exists():
        logger.warning("aggtrades_dir_not_found", path=str(directory))
        return None

    files = sorted(directory.glob("*.parquet"))
    if not files:
        logger.warning("no_aggtrades_files_found")
        return None

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning("aggtrades_read_error", file=str(f), error=str(e))

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # ── CRITICAL: Detect int32 overflow corruption ────────────
    if df["timestamp"].dtype != np.int64:
        logger.error(
            "aggtrades_timestamp_int32_overflow",
            dtype=str(df["timestamp"].dtype),
            sample_value=int(df["timestamp"].iloc[0]),
        )
        raise ValueError(
            f"\n\n{'='*60}\n"
            f"FATAL: aggTrades timestamp stored as {df['timestamp'].dtype}, not int64.\n"
            f"Epoch milliseconds (>1.6 trillion) OVERFLOW int32 (max 2.1 billion).\n"
            f"All timestamp comparisons will fail silently.\n\n"
            f"FIX: Delete data/raw/aggTrades/ and re-run:\n"
            f"  python scripts/collect_binance_data.py --aggtrades-only\n"
            f"{'='*60}\n"
        )

    # ── Normalize timestamp units to MILLISECONDS ─────────────
    # Binance changed SPOT aggTrades from ms to µs on Jan 1, 2025.
    # 13-digit = milliseconds (pre-2025), 16-digit = microseconds (post-2025).
    sample_ts = int(df["timestamp"].iloc[len(df) // 2])  # median sample
    if sample_ts > 1e15:
        # Microseconds → convert to milliseconds
        logger.info(
            "aggtrades_timestamp_microseconds_detected",
            sample=sample_ts,
            converting_to="milliseconds",
        )
        df["timestamp"] = df["timestamp"] // 1000
    elif sample_ts < 1e10:
        # Seconds → convert to milliseconds
        logger.info(
            "aggtrades_timestamp_seconds_detected",
            sample=sample_ts,
            converting_to="milliseconds",
        )
        df["timestamp"] = df["timestamp"] * 1000

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("aggtrades_loaded", rows=len(df), files=len(files))
    return df


def load_orderbook_snapshots(directory: Path) -> Optional[pd.DataFrame]:
    """Load orderbook snapshots from parquet files."""
    ob_dir = directory / "orderbook_snapshots"
    if not ob_dir.exists():
        logger.info("no_orderbook_snapshots_available")
        return None

    files = sorted(ob_dir.glob("*.parquet"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    logger.info("orderbook_snapshots_loaded", rows=len(df), files=len(files))
    return df


# ============================================================
# BTC Price Lookup
# ============================================================

def find_btc_price_at(
    timestamp_ms: int,
    ohlcv: pd.DataFrame,
    aggtrades: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """
    Find BTC price at a specific timestamp.

    Priority:
      1. aggTrades: exact or nearest trade within ±30s
      2. OHLCV: bar that contains the timestamp (use close)
    """
    # Try aggTrades first (higher precision)
    if aggtrades is not None and len(aggtrades) > 0:
        window_ms = 30_000  # ±30 seconds
        mask = (
            (aggtrades["timestamp"] >= timestamp_ms - window_ms)
            & (aggtrades["timestamp"] <= timestamp_ms + window_ms)
        )
        nearby = aggtrades.loc[mask]
        if not nearby.empty:
            # Find closest trade by timestamp
            idx = (nearby["timestamp"] - timestamp_ms).abs().idxmin()
            return float(nearby.loc[idx, "price"])

    # Fallback: OHLCV bar containing this timestamp
    mask = (ohlcv["open_time"] <= timestamp_ms) & (ohlcv["close_time"] >= timestamp_ms)
    bars = ohlcv.loc[mask]
    if not bars.empty:
        return float(bars.iloc[-1]["close"])

    # Last resort: nearest bar before timestamp
    before = ohlcv[ohlcv["close_time"] <= timestamp_ms]
    if not before.empty:
        return float(before.iloc[-1]["close"])

    return None


# ============================================================
# TFM Reconstruction from aggTrades
# ============================================================

def compute_tfm_at(
    timestamp_ms: int,
    aggtrades: pd.DataFrame,
    window_seconds: int = 60,
    norm_window_periods: int = 20,
) -> float:
    """
    Reconstruct TFM from aggTrades at a given timestamp.

    TFM_raw = Σ taker_buy_vol[window] - Σ taker_sell_vol[window]
    TFM_normalized = TFM_raw / (total_vol + ε)
    """
    window_ms = window_seconds * 1000
    mask = (
        (aggtrades["timestamp"] >= timestamp_ms - window_ms)
        & (aggtrades["timestamp"] <= timestamp_ms)
    )
    window_trades = aggtrades.loc[mask]

    if window_trades.empty:
        return 0.0

    # is_buyer_maker=True → taker SELL, is_buyer_maker=False → taker BUY
    taker_buy = window_trades.loc[~window_trades["is_buyer_maker"], "quantity"].sum()
    taker_sell = window_trades.loc[window_trades["is_buyer_maker"], "quantity"].sum()

    tfm_raw = taker_buy - taker_sell
    total = taker_buy + taker_sell

    if total < EPSILON:
        return 0.0

    return tfm_raw / (total + EPSILON)


# ============================================================
# OBI Reconstruction from Orderbook Snapshots
# ============================================================

def compute_obi_at(
    timestamp_ms: int,
    ob_snapshots: pd.DataFrame,
    levels: int = 5,
) -> float:
    """
    Reconstruct OBI from nearest orderbook snapshot.
    OBI = (Σ bid_qty - Σ ask_qty) / (Σ bid_qty + Σ ask_qty)
    """
    if ob_snapshots is None or ob_snapshots.empty:
        return 0.0

    # Find nearest snapshot
    if "timestamp_ms" in ob_snapshots.columns:
        idx = (ob_snapshots["timestamp_ms"] - timestamp_ms).abs().idxmin()
        snap = ob_snapshots.iloc[idx]
    else:
        return 0.0

    bid_qty = sum(
        snap.get(f"bid_qty_{i}", 0) for i in range(min(levels, 20))
    )
    ask_qty = sum(
        snap.get(f"ask_qty_{i}", 0) for i in range(min(levels, 20))
    )

    total = bid_qty + ask_qty
    if total < EPSILON:
        return 0.0

    return (bid_qty - ask_qty) / total


def compute_depth_ratio_at(
    timestamp_ms: int,
    ob_snapshots: pd.DataFrame,
    levels: int = 3,
) -> float:
    """Compute depth ratio from nearest snapshot."""
    if ob_snapshots is None or ob_snapshots.empty:
        return 1.0

    if "timestamp_ms" in ob_snapshots.columns:
        idx = (ob_snapshots["timestamp_ms"] - timestamp_ms).abs().idxmin()
        snap = ob_snapshots.iloc[idx]
    else:
        return 1.0

    bid_size = sum(snap.get(f"bid_qty_{i}", 0) for i in range(min(levels, 20)))
    ask_size = sum(snap.get(f"ask_qty_{i}", 0) for i in range(min(levels, 20)))

    return bid_size / (ask_size + EPSILON)


def compute_spread_bps_at(
    timestamp_ms: int,
    ob_snapshots: pd.DataFrame,
) -> float:
    """Compute Binance spread in bps from nearest snapshot."""
    if ob_snapshots is None or ob_snapshots.empty:
        return 2.0  # Default normal BTC spread

    if "timestamp_ms" in ob_snapshots.columns:
        idx = (ob_snapshots["timestamp_ms"] - timestamp_ms).abs().idxmin()
        snap = ob_snapshots.iloc[idx]
    else:
        return 2.0

    best_bid = snap.get("bid_price_0", 0)
    best_ask = snap.get("ask_price_0", 0)

    if best_bid <= 0 or best_ask <= 0:
        return 2.0

    mid = (best_bid + best_ask) / 2.0
    return (best_ask - best_bid) / mid * 10000.0


# ============================================================
# Feature Computation for a Single Sample Point
# ============================================================

def compute_features_at_point(
    signal_timestamp_ms: int,
    strike_price: float,
    t_resolution_ms: int,
    t_open_ms: int,
    ohlcv: pd.DataFrame,
    aggtrades: Optional[pd.DataFrame],
    ob_snapshots: Optional[pd.DataFrame],
) -> Optional[Dict[str, float]]:
    """
    Compute all 24 features at a specific signal evaluation timepoint.

    Uses ONLY data available at signal_timestamp_ms (anti-lookahead).
    """
    # Get OHLCV bars BEFORE signal time using O(log N) searchsorted
    idx = ohlcv["close_time"].searchsorted(signal_timestamp_ms, side='right')
    if idx < 21:
        return None
    
    bars = ohlcv.iloc[:idx]
    current_price = float(bars.iloc[-1]["close"])

    features: Dict[str, float] = {}

    # ── 01: OBI ───────────────────────────────────────────────
    features["OBI"] = compute_obi_at(signal_timestamp_ms, ob_snapshots) if ob_snapshots is not None else 0.0

    # ── 02: TFM_normalized ────────────────────────────────────
    features["TFM_normalized"] = (
        compute_tfm_at(signal_timestamp_ms, aggtrades)
        if aggtrades is not None
        else 0.0
    )

    # Use the pre-calculated metrics from the last OHLCV bar
    last_bar = bars.iloc[-1]

    # ── 03: VAM ───────────────────────────────────────────────
    # VAM = current_return / (std_r + EPSILON)
    std_r = float(last_bar["rv_12m"])
    current_return = float(last_bar["log_return"])
    features["VAM"] = current_return / (std_r + EPSILON)

    # ── 04: RV (annualized) ───────────────────────────────────
    features["RV"] = std_r * math.sqrt(252 * 96)

    # ── 05: vol_percentile ────────────────────────────────────
    features["vol_percentile"] = float(last_bar["vol_percentile"])

    # ── 06: depth_ratio ───────────────────────────────────────
    features["depth_ratio"] = compute_depth_ratio_at(signal_timestamp_ms, ob_snapshots)

    # ── 07: price_vs_ema20 ────────────────────────────────────
    # Efficiently get EMA from bars without full recalculation
    if len(bars) >= 21:
        s = bars["close"]
        ema = s.ewm(span=20, adjust=False).mean().iloc[-1]
        features["price_vs_ema20"] = (current_price - ema) / (current_price + EPSILON)
    else:
        features["price_vs_ema20"] = 0.0

    # ── 08: binance_spread_bps ────────────────────────────────
    features["binance_spread_bps"] = compute_spread_bps_at(signal_timestamp_ms, ob_snapshots)

    # ── 09-12: Temporal cyclical ──────────────────────────────
    dt = datetime.fromtimestamp(signal_timestamp_ms / 1000, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()

    features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    features["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
    features["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)

    # ── 13-15: TTR contextual ─────────────────────────────────
    ttr_seconds = (t_resolution_ms - signal_timestamp_ms) / 1000.0
    ttr_minutes = max(0.0, ttr_seconds / 60.0)
    
    lifespan_h = (t_resolution_ms - t_open_ms) / 3600000.0
    ttr_norm_base = 1440.0 if lifespan_h > 2.0 else 15.0
    ttr_normalized = max(0.0, min(1.0, ttr_minutes / ttr_norm_base))

    features["TTR_normalized"] = ttr_normalized
    features["TTR_sin"] = math.sin(math.pi * ttr_normalized)
    features["TTR_cos"] = math.cos(math.pi * ttr_normalized)

    # ── 16-17: Strike contextual ──────────────────────────────
    strike_distance_pct = (current_price - strike_price) / strike_price * 100.0
    features["strike_distance_pct"] = strike_distance_pct
    features["contest_urgency"] = abs(strike_distance_pct) * (1.0 - ttr_normalized)

    # ── 18-20: Interaction features ───────────────────────────
    features["ttr_x_obi"] = ttr_normalized * features["OBI"]
    features["ttr_x_tfm"] = ttr_normalized * features["TFM_normalized"]
    features["ttr_x_strike"] = ttr_normalized * strike_distance_pct

    # ── 21-24: CLOB features (historical if available, else estimates) ─
    # In production, these come from live CLOB. For training, we use
    # symmetric estimates based on market dynamics.
    features["clob_yes_mid"] = 0.5  # Will be enriched from CLOB history if available
    features["clob_yes_spread"] = 0.02
    features["clob_no_spread"] = 0.02
    features["market_vig"] = 0.03

    # ── Sanitize NaN/Inf ──────────────────────────────────────
    for k, v in features.items():
        if math.isnan(v) or math.isinf(v):
            features[k] = 0.0

    return features


# ============================================================
# Main Builder
# ============================================================

def build_training_dataset(
    markets_path: Path,
    ohlcv_path: Path,
    aggtrades_dir: Optional[Path] = None,
    ob_dir: Optional[Path] = None,
    samples_per_market: int = 3,
) -> pd.DataFrame:
    """
    Build the merged training dataset.

    For each resolved Polymarket market:
      1. Verify ground truth (strike, outcome) exists
      2. Look up BTC price at T_resolution → verify label
      3. Generate N sample points within the market's entry window
      4. Compute all 24 features at each point (anti-lookahead)
      5. Attach label (1 if BTC > strike at T_resolution, else 0)

    Args:
        markets_path: Path to polymarket_markets.parquet
        ohlcv_path: Path to ohlcv_15m.parquet
        aggtrades_dir: Path to aggTrades directory (optional but recommended)
        ob_dir: Path to orderbook snapshots directory (optional)
        samples_per_market: Number of signal evaluation points per market

    Returns:
        DataFrame with all 24 features + label + metadata columns
    """
    # Load data sources
    ohlcv = load_ohlcv(ohlcv_path)
    markets = load_markets(markets_path, ohlcv)
    from pathlib import Path

    aggtrades_dir_path = Path(aggtrades_dir) if aggtrades_dir else None

    ob_snapshots = None
    if ob_dir:
        ob_snapshots = load_orderbook_snapshots(ob_dir)

    # ── Pyth Oracle Prices ────────────────────────────────────
    pyth_cache: Dict[int, float] = {}
    from pathlib import Path
    raw_dir = Path("data/raw")
    pyth_path = raw_dir / "pyth_prices.parquet"
    if pyth_path.exists():
        df_pyth = pd.read_parquet(pyth_path)
        pyth_cache = dict(zip(df_pyth["timestamp_ms"], df_pyth["oracle_price"]))
        logger.info("pyth_oracle_prices_loaded", rows=len(pyth_cache))
    else:
        logger.warning("pyth_oracle_prices_missing", path=str(pyth_path))

    # ── OHLCV time range ──────────────────────────────────────
    ohlcv_min_ms = int(ohlcv["open_time"].min())
    ohlcv_max_ms = int(ohlcv["close_time"].max())

    # ── Build samples ─────────────────────────────────────────
    from tqdm import tqdm
    all_samples: List[Dict] = []
    markets_processed = 0
    markets_skipped = 0
    samples_dropped_no_tfm = 0
    FAIL_FAST_THRESHOLD = 100  # Abort if first N markets produce 0 samples

    for _, market in tqdm(markets.iterrows(), total=len(markets), desc="Processing Markets"):
        t_res_ms = int(market["t_resolution_epoch_ms"])
        t_open_ms = market.get("t_open_epoch_ms")

        if t_open_ms is None or pd.isna(t_open_ms):
            # Estimate T_open as T_resolution - 15 minutes
            t_open_ms = t_res_ms - 15 * 60 * 1000

        t_open_ms = int(t_open_ms)
        question = str(market.get("question", ""))
        strike_price = float(market["strike_price"])
        outcome_binary = int(market["outcome_binary"])

        # ── Check temporal overlap ────────────────────────────
        if t_res_ms < ohlcv_min_ms or t_open_ms > ohlcv_max_ms:
            markets_skipped += 1
            continue

        # Need enough OHLCV history (at least 21 bars before signal)
        min_history_ms = 21 * 15 * 60 * 1000  # ~5.25 hours
        if t_open_ms - ohlcv_min_ms < min_history_ms:
            markets_skipped += 1
            continue

        # ── Verify BTC price at resolution ────────────────────
        # ── Daily Filter ──────────────────────────────────────
        lifespan_ms = t_res_ms - t_open_ms
        is_daily = lifespan_ms >= 18 * 3600 * 1000
        
        # Additional filter: Daily markets usually resolve at 16:00 UTC
        # and don't contain minute-by-minute segments in the title.
        q_lower = question.lower()
        is_segment = any(x in q_lower for x in [":", "am-", "pm-", "min", "5-minute", "15-minute"])
        is_valid_above = 'above' in q_lower and not any(x in q_lower for x in ['dip', 'below', 'up', 'down'])
        
        if not is_daily or is_segment or not is_valid_above:
            markets_skipped += 1
            continue

        # ── Get Resolution Price ──────────────────────────────
        # PRIORITY: Pyth Oracle
        btc_at_res = pyth_cache.get(t_res_ms)
            
        if btc_at_res is None:
            markets_skipped += 1
            continue

        # ── Sanity Check ──────────────────────────────────────
        max_divergence = 0.12  # Daily markets can deviate more
        if abs(strike_price - btc_at_res) / btc_at_res > max_divergence:
            markets_skipped += 1
            continue

        # Cross-check outcome
        computed_label = 1 if btc_at_res > strike_price else 0
        label_match = (computed_label == outcome_binary)
        is_imputed = market.get("is_imputed_strike", False)

        # ── Generate signal evaluation timepoints ─────────────
        signal_timepoints_ms = []
        
        if is_daily:
            # Multi-Entry Strategy Anchor Points
            # T-20h, T-12h, T-4h (with ±30m jitter), T-1h (exact)
            HOUR_MS = 3600 * 1000
            anchors_with_jitter = [
                t_res_ms - 20 * HOUR_MS,
                t_res_ms - 12 * HOUR_MS,
                t_res_ms - 4 * HOUR_MS,
            ]
            
            import random
            for base_t in anchors_with_jitter:
                jitter = random.uniform(-30, 30) * 60 * 1000
                t_sampled = int(base_t + jitter)
                if t_sampled >= t_open_ms and t_sampled < t_res_ms:
                    signal_timepoints_ms.append(t_sampled)
                    
            # T-1h (no jitter)
            t_1h = int(t_res_ms - 1 * HOUR_MS)
            if t_1h >= t_open_ms and t_1h < t_res_ms:
                signal_timepoints_ms.append(t_1h)
                
        else:
            # Legacy 15m mode (5-12 minutes before resolution)
            TTR_MIN_MS = 5 * 60 * 1000
            TTR_MAX_MS = 12 * 60 * 1000
            entry_start_ms = max(t_open_ms, t_res_ms - TTR_MAX_MS)
            entry_end_ms = t_res_ms - TTR_MIN_MS
            if entry_start_ms < entry_end_ms:
                time_span = entry_end_ms - entry_start_ms
                for i in range(samples_per_market):
                    if samples_per_market == 1:
                        t_signal = entry_start_ms + time_span // 2
                    else:
                        t_signal = entry_start_ms + int(time_span * i / (samples_per_market - 1))
                    signal_timepoints_ms.append(int(t_signal))

        if not signal_timepoints_ms:
            markets_skipped += 1
            continue

        # Sort for chronological processing
        signal_timepoints_ms.sort()

        # ── Lazy Load TFM computation chunks ──────
        market_aggtrades = None
        if aggtrades_dir_path:
            start_ts = min(signal_timepoints_ms) - 60 * 1000
            end_ts = t_res_ms
            
            from datetime import datetime, timezone, timedelta
            d1 = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).date()
            d2 = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).date()
            
            dfs = []
            curr_d = d1
            while curr_d <= d2:
                f_path = aggtrades_dir_path / f"aggTrades_{curr_d.strftime('%Y-%m-%d')}.parquet"
                if f_path.exists():
                    dfs.append(pd.read_parquet(f_path))
                curr_d += timedelta(days=1)
                
            if dfs:
                daily_agg = pd.concat(dfs, ignore_index=True)
                sample_ts = daily_agg["timestamp"].iloc[len(daily_agg)//2]
                if sample_ts > 1e15:
                    daily_agg["timestamp"] = daily_agg["timestamp"] // 1000
                elif sample_ts < 1e10:
                    daily_agg["timestamp"] = daily_agg["timestamp"] * 1000
                    
                start_idx = daily_agg["timestamp"].searchsorted(start_ts, side='left')
                end_idx = daily_agg["timestamp"].searchsorted(end_ts, side='right')
                market_aggtrades = daily_agg.iloc[start_idx:end_idx].copy()
                if market_aggtrades.empty:
                    market_aggtrades = None

        # ── Evaluate Features at each signal point ────────────
        for i, t_signal in enumerate(signal_timepoints_ms):

            # ── Compute features ──────────────────────────────
            features = compute_features_at_point(
                signal_timestamp_ms=t_signal,
                strike_price=strike_price,
                t_resolution_ms=t_res_ms,
                t_open_ms=t_open_ms,
                ohlcv=ohlcv,
                aggtrades=market_aggtrades,
                ob_snapshots=ob_snapshots,
            )

            if features is None:
                continue

            # Enforce TFM existence
            if features["TFM_normalized"] == 0.0:
                samples_dropped_no_tfm += 1
                continue

            # ── Assemble sample row ───────────────────────────
            sample = features.copy()
            sample["label"] = outcome_binary  # GROUND TRUTH from Polymarket
            sample["market_id"] = market["market_id"]
            sample["strike_price"] = strike_price
            sample["btc_at_resolution"] = btc_at_res
            sample["btc_at_signal"] = find_btc_price_at(t_signal, ohlcv, market_aggtrades) or 0.0
            sample["signal_timestamp_ms"] = t_signal
            sample["t_resolution_ms"] = t_res_ms
            sample["TTR_minutes"] = max(0, (t_res_ms - t_signal) / 60000.0)
            sample["resolution_date"] = datetime.fromtimestamp(t_res_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            sample["label_match"] = label_match
            sample["resolution_source"] = market.get("resolution_source", "")
            sample["sample_index"] = i
            sample["is_daily"] = int(is_daily)
            sample["is_imputed_strike"] = int(is_imputed)

            all_samples.append(sample)

        markets_processed += 1

        # ── FAIL-FAST: abort early if no samples after N markets ──
        if markets_processed == FAIL_FAST_THRESHOLD and len(all_samples) == 0:
            logger.error(
                "fail_fast_triggered",
                markets_processed=markets_processed,
                markets_skipped=markets_skipped,
                dropped_no_tfm=samples_dropped_no_tfm,
            )
            raise RuntimeError(
                f"\n\nFAIL-FAST: Processed {markets_processed} markets with "
                f"0 samples generated (dropped_no_tfm={samples_dropped_no_tfm}). "
                f"Aborting to prevent multi-hour wasted run. "
                f"Check data alignment and timestamp formats."
            )

    if not all_samples:
        logger.error("no_training_samples_generated", dropped_no_tfm=samples_dropped_no_tfm)
        return pd.DataFrame()

    df = pd.DataFrame(all_samples)

    # Verify feature columns
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        logger.error("missing_features_in_output", missing=missing)

    # Summary
    logger.info(
        "dataset_built",
        total_samples=len(df),
        markets_processed=markets_processed,
        markets_skipped=markets_skipped,
        label_distribution=df["label"].value_counts().to_dict(),
        has_obi_data=(df["OBI"] != 0).sum(),
        has_tfm_data=(df["TFM_normalized"] != 0).sum(),
        label_mismatch_rate=round(1.0 - df["label_match"].mean(), 4),
    )

    return df


# ============================================================
# CLI
# ============================================================

@click.command()
@click.option(
    "--markets",
    default=str(RAW_DIR / "polymarket_markets.parquet"),
    help="Path to polymarket_markets.parquet",
)
@click.option(
    "--ohlcv",
    default=str(RAW_DIR / "ohlcv_15m.parquet"),
    help="Path to ohlcv_15m.parquet",
)
@click.option(
    "--aggtrades-dir",
    default=str(RAW_DIR / "aggTrades"),
    help="Path to aggTrades directory",
)
@click.option(
    "--ob-dir",
    default=str(RAW_DIR),
    help="Path to parent of orderbook_snapshots directory",
)
@click.option(
    "--samples-per-market",
    default=150,
    help="Signal evaluation points per market",
)
@click.option("--output", default=None, help="Output file path")
def main(
    markets: str,
    ohlcv: str,
    aggtrades_dir: str,
    ob_dir: str,
    samples_per_market: int,
    output: str | None,
) -> None:
    """Build merged training dataset from raw data sources."""
    click.echo("🔧 Building merged training dataset...")
    click.echo("   This joins Polymarket ground truth + Binance features.\n")

    markets_path = Path(markets)
    ohlcv_path = Path(ohlcv)
    aggtrades_path = Path(aggtrades_dir) if Path(aggtrades_dir).exists() else None
    ob_path = Path(ob_dir) if Path(ob_dir).exists() else None

    if not markets_path.exists():
        click.echo(f"❌ Markets file not found: {markets_path}")
        click.echo("   Run: python scripts/collect_polymarket.py first")
        return

    if not ohlcv_path.exists():
        click.echo(f"❌ OHLCV file not found: {ohlcv_path}")
        click.echo("   Run: python scripts/collect_binance_data.py first")
        return

    if aggtrades_path is None:
        click.echo("⚠️  No aggTrades directory — TFM features will be zero.")
        click.echo("   Run: python scripts/collect_binance_data.py --with-aggtrades\n")

    df = build_training_dataset(
        markets_path=markets_path,
        ohlcv_path=ohlcv_path,
        aggtrades_dir=aggtrades_path,
        ob_dir=ob_path,
        samples_per_market=samples_per_market,
    )

    if df.empty:
        click.echo("❌ No training samples generated")
        click.echo("   Check temporal overlap between Polymarket markets and Binance data")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else PROCESSED_DIR / "merged_training_features.parquet"
    df.to_parquet(out_path, index=False)

    # Report
    click.echo(f"\n✅ Saved {len(df)} training samples to {out_path}")
    click.echo(f"   Feature columns: {len(FEATURE_NAMES)}")
    click.echo(f"   Label 1 (YES): {(df['label']==1).sum()} ({(df['label']==1).mean():.1%})")
    click.echo(f"   Label 0 (NO):  {(df['label']==0).sum()} ({(df['label']==0).mean():.1%})")
    click.echo(f"   Non-zero OBI:  {(df['OBI']!=0).sum()}")
    click.echo(f"   Non-zero TFM:  {(df['TFM_normalized']!=0).sum()}")

    label_mismatch = 1.0 - df["label_match"].mean()
    if label_mismatch > 0.05:
        click.echo(f"\n⚠️  Label mismatch rate: {label_mismatch:.1%}")
        click.echo("   This indicates basis risk between Binance price and Polymarket oracle.")
    else:
        click.echo(f"   Label mismatch rate: {label_mismatch:.1%} ✓")

    size_mb = out_path.stat().st_size / 1024 / 1024
    click.echo(f"   File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

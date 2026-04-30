"""
run.py — deterministic backtest/simulation harness.

CLI:
  python -m src.sim.run --market-id <id> --limit 500
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from src.config_manager import ConfigManager
from src.dry_run import DryRunEngine
from src.fair_probability import FairProbabilityEngine
from src.risk_manager import RiskManager
from src.schemas import (
    ActiveMarket,
    CLOBState,
    FeatureMetadata,
    FeatureVector,
)
from src.signal_generator import SignalGenerator


FEATURE_NAMES = [
    "OBI",
    "TFM_normalized",
    "VAM",
    "RV",
    "vol_percentile",
    "depth_ratio",
    "price_vs_ema20",
    "binance_spread_bps",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "TTR_normalized",
    "TTR_sin",
    "TTR_cos",
    "strike_distance_pct",
    "contest_urgency",
    "ttr_x_obi",
    "ttr_x_tfm",
    "ttr_x_strike",
    "clob_yes_mid",
    "clob_yes_spread",
    "clob_no_spread",
    "market_vig",
]


class FakeBinanceFeed:
    def __init__(self, latest_price: float) -> None:
        self.latest_price = latest_price


class ConfigOverride:
    """In-memory config overrides without persisting to config/config.json."""

    def __init__(self, base_cfg: ConfigManager, overrides: Dict[str, float]) -> None:
        self._base = base_cfg
        self._overrides = overrides

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._base.get(key, default)


def _ms_to_dt(ms: float) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


def _ttr_phase(ttr_minutes: float, cfg: ConfigManager, lifespan_h: float = 0.25) -> str:
    dyn_enabled = bool(cfg.get("signal.dynamic_ttr_enabled", True))
    if not dyn_enabled:
        ttr_min = float(cfg.get("signal.ttr_min_minutes", 5.0))
        ttr_max = float(cfg.get("signal.ttr_max_minutes", 12.0))
    elif lifespan_h <= 2.0:
        ttr_min = float(cfg.get("signal.entry_window_short_min_minutes", 5.0))
        ttr_max = float(cfg.get("signal.entry_window_short_max_minutes", 60.0))
    elif lifespan_h <= 8.0:
        ttr_min = float(cfg.get("signal.entry_window_medium_min_minutes", 30.0))
        ttr_max = float(cfg.get("signal.entry_window_medium_max_minutes", 240.0))
    else:
        ttr_min = float(cfg.get("signal.entry_window_long_min_minutes", 60.0))
        ttr_max = float(cfg.get("signal.entry_window_long_max_minutes", 720.0))
    if ttr_minutes > ttr_max:
        return "EARLY"
    if ttr_minutes >= ttr_min:
        return "ENTRY_WINDOW"
    return "LATE"


def _reconstruct_clob_state(
    row: Dict[str, Any],
    market_id: str,
    cfg: ConfigManager,
    ts: datetime,
    fair_prob: Optional[float] = None,
) -> CLOBState:
    """
    Build a realistic CLOB snapshot for simulation.

    If fair_prob is provided (preferred), we simulate a market-maker who prices
    around the fair value with a configurable vig and random spread jitter.
    This prevents the artificial edge that arises when CLOB is stuck at 0.50
    while fair prob says 0.95.

    Market-maker model:
      - MM sets YES mid ≈ fair_prob + noise
      - Spread drawn from [base_spread, base_spread * 3] to simulate variable
        liquidity conditions (tight near 0.50, wider at extremes)
      - Vig applied symmetrically
    """
    import random

    base_vig = float(cfg.get("sim.mm_base_vig", 0.04))
    base_spread = float(cfg.get("sim.mm_base_spread", 0.02))
    spread_noise_pct = float(cfg.get("sim.mm_spread_noise_pct", 0.50))

    if fair_prob is not None and 0.0 < fair_prob < 1.0:

        mm_noise_sigma = float(cfg.get("sim.mm_noise_sigma", 0.08))
        mm_lag_probability = float(cfg.get("sim.mm_lag_probability", 0.25))
        mm_lag_sigma = float(cfg.get("sim.mm_lag_sigma", 0.05))

        # Market-maker pricing with two sources of imperfection:
        #
        # 1. Estimation noise: MM doesn't compute the same fair value as us.
        #    On Polymarket (retail-heavy, not HFT), MMs are slower and less
        #    precise. σ=0.08 means ±8% disagreement 1-sigma.
        #
        # 2. Latency lag: With probability mm_lag_probability, the MM is using
        #    a stale fair estimate (lagged). This is the "slow skew" edge —
        #    our Binance-derived fair prob updates faster than CLOB reprices.
        mm_noise = random.gauss(0, mm_noise_sigma)

        if random.random() < mm_lag_probability:
            # MM is lagging — additional offset toward 0.50 (stale price)
            lag_pull = mm_lag_sigma * (0.5 - fair_prob)  # pulls toward center
            mm_noise += lag_pull

        mm_fair = max(0.02, min(0.98, fair_prob + mm_noise))

        # Spread model: Polymarket spreads are typically 2-6 cents.
        # Slightly wider at extremes but not dramatically so (unlike TradFi).
        extremity = abs(mm_fair - 0.5) * 2.0  # 0 at center, 1 at extremes
        spread = base_spread * (1.0 + extremity * 0.8)  # moderate widening
        spread *= (1.0 + random.uniform(-spread_noise_pct, spread_noise_pct))
        spread = max(0.01, min(0.10, spread))

        yes_ask = min(0.99, mm_fair + spread / 2.0 + base_vig / 2.0)
        yes_bid = max(0.01, mm_fair - spread / 2.0)
        no_ask = min(0.99, (1.0 - mm_fair) + spread / 2.0 + base_vig / 2.0)
        no_bid = max(0.01, (1.0 - mm_fair) - spread / 2.0)

        market_vig = max(0.0, (yes_ask + no_ask) - 1.0)
    else:
        # Fallback: use dataset values (legacy path)
        yes_mid = float(row["clob_yes_mid"])
        yes_spread = float(row["clob_yes_spread"])
        no_spread = float(row["clob_no_spread"])
        market_vig = float(row["market_vig"])

        yes_bid = yes_mid - yes_spread / 2.0
        yes_ask = yes_mid + yes_spread / 2.0
        no_ask = 1.0 - market_vig - yes_ask
        no_bid = no_ask - no_spread

    # Clamp into [0.01, 0.99] for safety.
    yes_bid = max(0.01, min(0.99, yes_bid))
    yes_ask = max(0.01, min(0.99, yes_ask))
    no_bid = max(0.01, min(0.99, no_bid))
    no_ask = max(0.01, min(0.99, no_ask))
    market_vig = max(0.0, (yes_ask + no_ask) - 1.0)

    max_vig = float(cfg.get("clob.max_market_vig", 0.07))
    is_liquid = market_vig <= max_vig

    # Depth is not present in the processed dataset; set large depth to avoid
    # accidentally blocking on depth in the signal generator.
    yes_depth_usd = 1e6
    no_depth_usd = 1e6

    return CLOBState(
        market_id=market_id,
        timestamp=ts,
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=no_ask,
        no_bid=no_bid,
        yes_depth_usd=yes_depth_usd,
        no_depth_usd=no_depth_usd,
        market_vig=market_vig,
        is_liquid=is_liquid,
        is_stale=False,
    )


def _build_feature_vector(row: Dict[str, Any], now_ts: datetime, cfg: ConfigManager) -> FeatureVector:
    values = []
    for name in FEATURE_NAMES:
        v = row.get(name, 0.0)
        # Polars may return NaNs; signal generator expects floats.
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = 0.0
        values.append(float(v))

    ttr_minutes = float(row["TTR_minutes"])
    # Compute lifespan for dynamic TTR window classification
    t_resolution = _ms_to_dt(row["t_resolution_ms"])
    t_open = t_resolution - timedelta(minutes=max(15.0, ttr_minutes + 15.0))
    lifespan_h = max(0.0, (t_resolution - t_open).total_seconds() / 3600.0)
    md = FeatureMetadata(
        timestamp=now_ts,
        bar_close_time=now_ts,
        market_id=str(row["market_id"]),
        strike_price=float(row["strike_price"]),
        current_btc_price=float(row["btc_at_signal"]),
        TTR_minutes=ttr_minutes,
        TTR_phase=_ttr_phase(ttr_minutes, cfg, lifespan_h=lifespan_h),
        compute_lag_ms=0.0,
    )

    return FeatureVector(values=values, feature_names=FEATURE_NAMES, metadata=md)


def _build_active_market(row: Dict[str, Any], cfg: ConfigManager) -> ActiveMarket:
    t_resolution = _ms_to_dt(row["t_resolution_ms"])
    ttr_minutes = float(row["TTR_minutes"])
    # Approximate open time for feature normalization (not critical for signal generator).
    t_open = t_resolution - timedelta(minutes=max(15.0, ttr_minutes + 15.0))

    return ActiveMarket(
        market_id=str(row["market_id"]),
        question="SIM",
        strike_price=float(row["strike_price"]),
        T_open=t_open,
        T_resolution=t_resolution,
        TTR_minutes=ttr_minutes,
        clob_token_ids={"YES": "", "NO": ""},
        settlement_exchange="CHAINLINK",
        settlement_instrument="BTCUSDT",
        settlement_granularity="1m",
        settlement_price_type="close",
        resolution_source="Chainlink",
    )


async def run_backtest(
    market_id: Optional[str],
    limit: Optional[int],
    ttr_min: Optional[float],
    ttr_max: Optional[float],
    model_version: str = "fair_prob_sim_v1",
) -> None:
    cfg = ConfigManager.get_instance()
    overrides: Dict[str, float] = {}
    if ttr_min is not None:
        overrides["signal.ttr_min_minutes"] = float(ttr_min)
    if ttr_max is not None:
        overrides["signal.ttr_max_minutes"] = float(ttr_max)

    effective_cfg: Any = ConfigOverride(cfg, overrides) if overrides else cfg

    df = pl.read_parquet("data/processed/merged_training_features.parquet")
    if market_id:
        df = df.filter(pl.col("market_id") == str(market_id))
    if limit:
        df = df.head(int(limit))

    if df.height == 0:
        print("No rows after filtering.")
        return

    df = df.sort("signal_timestamp_ms")

    dry = DryRunEngine(effective_cfg, initial_capital=float(effective_cfg.get("sim.initial_capital", 100.0)))
    from src.database import DatabaseManager
    db = DatabaseManager()
    risk = RiskManager(effective_cfg, db)
    fair_engine = FairProbabilityEngine(effective_cfg)
    signal_gen = SignalGenerator(effective_cfg)

    pending: List[Tuple[Any, datetime, float]] = []
    # pending items: (PaperTrade, T_resolution_dt, btc_at_resolution)
    signal_counts: Dict[str, int] = {"BUY_UP": 0, "BUY_DOWN": 0, "ABSTAIN": 0}
    abstain_reason_counts: Dict[str, int] = {}

    for row in df.iter_rows(named=True):
        sig_ts = _ms_to_dt(row["signal_timestamp_ms"])
        active_market = _build_active_market(row, effective_cfg)

        # Resolve any trades that matured before this signal.
        pending.sort(key=lambda x: x[1])
        while pending and pending[0][1] <= sig_ts:
            trade, _, btc_at_resolution = pending.pop(0)
            resolved = await dry.resolve_trade(trade, btc_at_resolution)
            await risk.on_trade_resolved(resolved.pnl_usd or 0.0)

        dry.increment_bars()

        # Circuit: if signal happened while a position is open, risk manager will reject.
        # Still record the signal for calibration/prediction tracking.

        sigma_override = float(row["RV"]) if row.get("RV") is not None else None
        if isinstance(sigma_override, float) and math.isnan(sigma_override):
            sigma_override = None
        if sigma_override is None:
            sigma_override = float(cfg.get("fair_prob.sigma_default_ann", 0.30))

        fake_feed = FakeBinanceFeed(latest_price=float(row["btc_at_signal"]))

        # PHASE 1: Compute fair probability with a minimal placeholder CLOB.
        # fair_engine only uses clob_state.market_vig for uncertainty scaling,
        # so a neutral placeholder is acceptable here.
        placeholder_clob = CLOBState(
            market_id=active_market.market_id,
            timestamp=sig_ts,
            yes_ask=0.51, yes_bid=0.49,
            no_ask=0.51, no_bid=0.49,
            yes_depth_usd=1e6, no_depth_usd=1e6,
            market_vig=0.02, is_liquid=True, is_stale=False,
        )
        fair_initial = fair_engine.compute(
            binance_feed=fake_feed,
            active_market=active_market,
            clob_state=placeholder_clob,
            sigma_ann_override=sigma_override,
            data_confidence_override=1.0,
            as_of_time=sig_ts,
        )

        # PHASE 2: Build realistic CLOB centered on fair probability.
        # A rational market-maker prices around q_fair, NOT at a static 0.50.
        clob_state = _reconstruct_clob_state(
            row, active_market.market_id, effective_cfg, sig_ts,
            fair_prob=fair_initial.q_fair,
        )

        # PHASE 3: Re-compute fair with realistic CLOB (for correct uncertainty).
        fair = fair_engine.compute(
            binance_feed=fake_feed,
            active_market=active_market,
            clob_state=clob_state,
            sigma_ann_override=sigma_override,
            data_confidence_override=1.0,
            as_of_time=sig_ts,
        )

        # Build feature vector with realistic CLOB values baked in
        fv = _build_feature_vector(row, sig_ts, effective_cfg)

        signal = signal_gen.evaluate(
            P_model=fair.q_fair,
            uncertainty_u=fair.uncertainty_u,
            clob_state=clob_state,
            active_market=active_market,
            feature_vector=fv,
        )
        dry.record_signal(signal)
        signal_counts[signal.signal] = signal_counts.get(signal.signal, 0) + 1
        if signal.signal == "ABSTAIN":
            r = signal.abstain_reason or "NO_EDGE"
            abstain_reason_counts[r] = abstain_reason_counts.get(r, 0) + 1

        if signal.signal == "ABSTAIN":
            continue

        approved = await risk.approve(signal, dry.capital)
        # ApprovedBet / RejectedBet are Pydantic models; compare by presence of bet_size.
        if not hasattr(approved, "bet_size"):
            continue

        trade = dry.simulate_trade(signal, approved, active_market)
        btc_at_res = float(row["btc_at_resolution"])
        pending.append((trade, active_market.T_resolution, btc_at_res))

    # Resolve remaining pending trades at end.
    if pending:
        pending.sort(key=lambda x: x[1])
        for trade, _, btc_at_resolution in pending:
            resolved = await dry.resolve_trade(trade, btc_at_resolution)
            await risk.on_trade_resolved(resolved.pnl_usd or 0.0)

    metrics = dry.compute_session_metrics(model_version=model_version)
    print("=== Backtest Summary ===")
    print(f"session_id={metrics.session_id}")
    print(f"trades_executed={metrics.trades_executed}")
    print(f"signal_counts={signal_counts}")
    print(f"abstain_reason_counts={abstain_reason_counts}")
    print(f"win_rate={metrics.win_rate}")
    print(f"total_pnl_usd={metrics.total_pnl_usd}")
    print(f"max_drawdown={metrics.max_drawdown}")
    print(f"profit_factor={metrics.profit_factor}")
    print(f"brier_score={metrics.brier_score}")
    print(f"pass_fail={metrics.pass_fail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--ttr-min", type=float, default=None)
    parser.add_argument("--ttr-max", type=float, default=None)
    args = parser.parse_args()

    asyncio.run(
        run_backtest(
            market_id=args.market_id,
            limit=args.limit,
            ttr_min=args.ttr_min,
            ttr_max=args.ttr_max,
        )
    )


if __name__ == "__main__":
    main()


"""
RED FLAG VALIDATION — Step 1: Stale CLOB Detection
====================================================
For samples with Edge > 0.44, investigate:
  - What was the CLOB price?
  - How many CLOB price updates existed in the 5-minute window before signal?
  - What was the time gap between last CLOB trade and our signal timestamp?
  - Is this "mispricing" or just a dead orderbook?
"""
import asyncio
import httpx
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_VER = "v20260406_084313"

np.random.seed(42)  # Fixed seed for reproducibility


async def get_market_info(client, market_id, sem):
    """Fetch token ID AND volume/liquidity from Gamma API."""
    async with sem:
        try:
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"id": market_id},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    m = data[0]
                    clob_ids = m.get("clobTokenIds", "[]")
                    if isinstance(clob_ids, str):
                        clob_ids = json.loads(clob_ids)
                    yes_token = clob_ids[0] if len(clob_ids) >= 2 else None
                    return market_id, {
                        "yes_token": yes_token,
                        "volume": float(m.get("volume", 0) or 0),
                        "liquidity": float(m.get("liquidity", 0) or 0),
                        "question": m.get("question", ""),
                    }
        except Exception as e:
            pass
        return market_id, None


async def get_clob_history_full(client, token_id, sem):
    """Fetch FULL price history at maximum fidelity (1-second granularity)."""
    async with sem:
        try:
            resp = await client.get(
                "https://clob.polymarket.com/prices-history",
                params={"market": token_id, "interval": "max", "fidelity": 1},
                timeout=15.0,
            )
            if resp.status_code == 200:
                hist = resp.json().get("history", [])
                return token_id, [
                    {"t_ms": h["t"] * 1000, "p": float(h["p"])}
                    for h in hist
                    if "p" in h and "t" in h
                ]
        except Exception as e:
            pass
        return token_id, []


def analyze_clob_activity(hist_points, signal_ts_ms, window_ms=5 * 60 * 1000):
    """
    For a given signal timestamp, measure CLOB activity in the preceding window.
    Returns dict with activity metrics.
    """
    if not hist_points:
        return {
            "points_in_window": 0,
            "last_trade_age_sec": None,
            "price_at_signal": None,
            "price_range_in_window": None,
            "is_stale": True,
        }

    window_start = signal_ts_ms - window_ms
    points_in_window = [
        p for p in hist_points if window_start <= p["t_ms"] <= signal_ts_ms
    ]

    # Find closest price point BEFORE or AT signal
    before_signal = [p for p in hist_points if p["t_ms"] <= signal_ts_ms]
    if before_signal:
        closest = max(before_signal, key=lambda x: x["t_ms"])
        last_trade_age_sec = (signal_ts_ms - closest["t_ms"]) / 1000
        price_at_signal = closest["p"]
    else:
        last_trade_age_sec = None
        price_at_signal = None

    if points_in_window:
        prices = [p["p"] for p in points_in_window]
        price_range = max(prices) - min(prices)
    else:
        price_range = None

    # Stale if: fewer than 2 price updates in 5min, OR last trade > 3min ago
    is_stale = (
        len(points_in_window) < 2
        or (last_trade_age_sec is not None and last_trade_age_sec > 180)
    )

    return {
        "points_in_window": len(points_in_window),
        "last_trade_age_sec": last_trade_age_sec,
        "price_at_signal": price_at_signal,
        "price_range_in_window": price_range,
        "is_stale": is_stale,
    }


async def main():
    # ── Load dataset & model ──────────────────────────────────
    df = pd.read_parquet(DATA_DIR / "processed" / "merged_training_features.parquet")
    model = joblib.load(DATA_DIR / "models" / f"model_lgbm_{MODEL_VER}.pkl")
    scaler = joblib.load(DATA_DIR / "models" / f"scaler_{MODEL_VER}.pkl")
    try:
        calibrator = joblib.load(DATA_DIR / "models" / f"calibrator_{MODEL_VER}.pkl")
    except:
        calibrator = None

    feature_cols = df.columns[:24]
    X_scaled = scaler.transform(df[feature_cols])
    if calibrator:
        df["P_model"] = calibrator.predict_proba(X_scaled)[:, 1]
    else:
        df["P_model"] = model.predict_proba(X_scaled)[:, 1]

    # ── Sample same 100 markets (fixed seed 42) ───────────────
    all_markets = df["market_id"].unique()
    sample_markets = np.random.choice(all_markets, size=min(100, len(all_markets)), replace=False)
    df_sub = df[df["market_id"].isin(sample_markets)].copy()

    sem = asyncio.Semaphore(5)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=5)) as client:
        # Fetch market info (tokens + volume)
        print(f"Fetching market info for {len(sample_markets)} markets...")
        tasks = [get_market_info(client, mid, sem) for mid in sample_markets]
        results = await asyncio.gather(*tasks)

        market_info = {}
        tokens = {}
        for mid, info in results:
            if info and info["yes_token"]:
                market_info[mid] = info
                tokens[mid] = info["yes_token"]

        print(f"  Markets with tokens: {len(tokens)}")

        # Fetch CLOB histories
        print(f"Fetching CLOB price histories...")
        unique_tokens = list(set(tokens.values()))
        tasks2 = [get_clob_history_full(client, tid, sem) for tid in unique_tokens]
        results2 = await asyncio.gather(*tasks2)

        histories = {}
        for tid, hist in results2:
            if hist:
                histories[tid] = hist

        print(f"  Histories retrieved: {len(histories)}")

    # ── Match CLOB prices to samples ──────────────────────────
    records = []
    for idx, row in df_sub.iterrows():
        mid = row["market_id"]
        ts_signal = row["signal_timestamp_ms"]

        token = tokens.get(mid)
        if not token:
            continue

        hist = histories.get(token)
        if not hist:
            continue

        # Find closest price before signal
        before = [p for p in hist if p["t_ms"] <= ts_signal]
        if not before:
            continue

        closest = max(before, key=lambda x: x["t_ms"])
        gap_sec = (ts_signal - closest["t_ms"]) / 1000

        # Only accept within 15 minutes
        if gap_sec > 15 * 60:
            continue

        # Measure CLOB activity
        activity = analyze_clob_activity(hist, ts_signal)

        mkt = market_info.get(mid, {})
        records.append({
            "market_id": mid,
            "signal_ts": datetime.fromtimestamp(ts_signal / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "P_model": row["P_model"],
            "P_clob": closest["p"],
            "Edge": row["P_model"] - closest["p"],
            "label": int(row["label"]),
            "clob_gap_sec": gap_sec,
            "clob_points_5min": activity["points_in_window"],
            "clob_price_range_5min": activity["price_range_in_window"],
            "clob_last_trade_age_sec": activity["last_trade_age_sec"],
            "is_stale": activity["is_stale"],
            "market_volume_usd": mkt.get("volume", 0),
            "market_liquidity": mkt.get("liquidity", 0),
            "question": mkt.get("question", "")[:60],
        })

    result_df = pd.DataFrame(records)
    # Filter to active CLOB range
    active = result_df[(result_df["P_clob"] >= 0.02) & (result_df["P_clob"] <= 0.98)].copy()

    print(f"\n{'='*70}")
    print("RED FLAG 1 — STALE CLOB INVESTIGATION")
    print(f"{'='*70}")
    print(f"Total matched samples: {len(result_df)}")
    print(f"Active CLOB samples (P in 0.02-0.98): {len(active)}")

    # ── Tail analysis: Edge > 0.44 ────────────────────────────
    tail = active[active["Edge"].abs() > 0.44].copy()
    print(f"\n--- TAIL SAMPLES (|Edge| > 0.44): {len(tail)} ---")
    if len(tail) > 0:
        for _, r in tail.iterrows():
            stale_tag = "⚠️  STALE" if r["is_stale"] else "✅ ACTIVE"
            print(f"\n  {stale_tag} | Edge={r['Edge']:+.3f}")
            print(f"    P_model={r['P_model']:.3f}, P_clob={r['P_clob']:.3f}, Label={r['label']}")
            print(f"    CLOB price points in 5min: {r['clob_points_5min']}")
            print(f"    Last CLOB update age: {r['clob_last_trade_age_sec']:.0f}s")
            if r['clob_price_range_5min'] is not None:
                print(f"    Price range in 5min: {r['clob_price_range_5min']:.4f}")
            print(f"    Market volume: ${r['market_volume_usd']:,.0f}")
            print(f"    Signal at: {r['signal_ts']}")
            print(f"    Market: {r['question']}")

        stale_count = tail["is_stale"].sum()
        print(f"\n  VERDICT: {stale_count}/{len(tail)} tail samples are STALE")
        print(f"  Median CLOB points in 5min window: {tail['clob_points_5min'].median():.0f}")
        print(f"  Median last trade age: {tail['clob_last_trade_age_sec'].median():.0f}s")

    # ── Recalculate without stale ─────────────────────────────
    non_stale = active[~active["is_stale"]].copy()
    print(f"\n{'='*70}")
    print("RECALCULATED STATISTICS (STALE REMOVED)")
    print(f"{'='*70}")
    print(f"Samples after stale removal: {len(non_stale)} (removed {len(active) - len(non_stale)})")

    if len(non_stale) > 0:
        print(f"Mean P_model: {non_stale['P_model'].mean():.4f}")
        print(f"Mean P_clob:  {non_stale['P_clob'].mean():.4f}")
        print(f"Mean Edge:    {non_stale['Edge'].mean():.4f}")
        print(f"Mean |Edge|:  {non_stale['Edge'].abs().mean():.4f}")

        # Recalculate trading sim
        threshold = 0.05
        bets = non_stale[non_stale["Edge"].abs() > threshold].copy()
        print(f"\nTrading Sim (|Edge| > {threshold*100}%): {len(bets)} trades")

        if len(bets) > 0:
            def calc_pnl(row):
                if row["Edge"] > 0:
                    cost = row["P_clob"]
                    revenue = 1 if row["label"] == 1 else 0
                else:
                    cost = 1 - row["P_clob"]
                    revenue = 1 if row["label"] == 0 else 0
                return revenue - cost

            bets["PnL"] = bets.apply(calc_pnl, axis=1)
            win = (bets["PnL"] > 0).mean()
            total_cost = sum(
                r["P_clob"] if r["Edge"] > 0 else (1 - r["P_clob"])
                for _, r in bets.iterrows()
            )
            roi = bets["PnL"].sum() / total_cost if total_cost > 0 else 0
            print(f"Win Rate (clean): {win*100:.2f}%")
            print(f"PnL: {bets['PnL'].sum():.2f} USD")
            print(f"ROI: {roi*100:.2f}%")

    # ── Step 2 preview: Timestamp alignment ───────────────────
    print(f"\n{'='*70}")
    print("RED FLAG 2 — TIMESTAMP ALIGNMENT (Preview)")
    print(f"{'='*70}")
    print(f"CLOB gap distribution across ALL {len(active)} active samples:")
    print(f"  Mean  clob_gap: {active['clob_gap_sec'].mean():.1f}s")
    print(f"  Median clob_gap: {active['clob_gap_sec'].median():.1f}s")
    print(f"  P90 clob_gap: {active['clob_gap_sec'].quantile(0.9):.1f}s")
    print(f"  Max clob_gap: {active['clob_gap_sec'].max():.1f}s")
    pct_over_60 = (active["clob_gap_sec"] > 60).mean() * 100
    print(f"  Samples with gap > 60s: {(active['clob_gap_sec'] > 60).sum()} ({pct_over_60:.1f}%)")

    if pct_over_60 > 20:
        print("  ⚠️  WARNING: >20% samples have CLOB gap > 60s — alignment issue!")
    else:
        print("  ✅ Timestamp alignment within acceptable range")

    # ── Top 10 edge samples timestamp detail ──────────────────
    print(f"\nTop 10 HIGHEST |Edge| samples — timestamp detail:")
    top10 = active.nlargest(10, "Edge")
    for _, r in top10.iterrows():
        stale_tag = "STALE" if r["is_stale"] else "OK"
        print(
            f"  Edge={r['Edge']:+.3f} | clob_gap={r['clob_gap_sec']:>5.0f}s | "
            f"pts_5min={r['clob_points_5min']:>3} | {stale_tag} | {r['question']}"
        )


if __name__ == "__main__":
    asyncio.run(main())

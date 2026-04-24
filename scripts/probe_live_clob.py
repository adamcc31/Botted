"""
probe_live_clob.py — One-shot probe of live Polymarket BTC CLOB state.

Fetches all active BTC binary markets, gets YES/NO orderbooks,
computes fair probability, and shows real edge distribution.
Outputs a CSV with the results for MM model calibration.
"""

import asyncio
import json
import math
import re
from datetime import datetime, timezone
from typing import Optional

import httpx


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

STRIKE_PATTERNS = [
    r"\$([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"above\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"below\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"up\s+or\s+down\s+from\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
]


def extract_strike(text: str) -> Optional[float]:
    for pat in STRIKE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            price = float(m.group(1).replace(",", ""))
            if 1_000 < price < 1_000_000:
                return price
    return None


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fair_prob(spot: float, strike: float, ttr_min: float, sigma_ann: float = 0.50) -> float:
    """Black-Scholes digital option approximation."""
    tau_years = max(1e-8, ttr_min / (365.25 * 24 * 60))
    sigma_sqrt_t = sigma_ann * math.sqrt(tau_years)
    if sigma_sqrt_t < 1e-8:
        return 1.0 if spot >= strike else 0.0
    d2 = (math.log(spot / strike) + (-0.5 * sigma_ann**2) * tau_years) / sigma_sqrt_t
    return phi(d2)


def extract_yes_prob(market_data: dict) -> Optional[float]:
    outcomes_raw = market_data.get("outcomes")
    prices_raw = market_data.get("outcomePrices")
    try:
        outcomes = outcomes_raw if isinstance(outcomes_raw, list) else json.loads(outcomes_raw or "[]")
        prices = prices_raw if isinstance(prices_raw, list) else json.loads(prices_raw or "[]")
        for i, name in enumerate(outcomes):
            if str(name).upper() == "YES" and i < len(prices):
                p = float(prices[i])
                return p if 0 < p < 1 else None
    except Exception:
        pass
    return None


async def main():
    now = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
        # 1. Get BTC spot price (try Binance, fallback to CoinGecko)
        try:
            resp = await client.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
            resp.raise_for_status()
            spot = float(resp.json()["price"])
        except Exception:
            resp = await client.get("https://api.coingecko.com/api/v3/simple/price",
                                    params={"ids": "bitcoin", "vs_currencies": "usd"})
            spot = float(resp.json()["bitcoin"]["usd"])
        print(f"BTC Spot: ${spot:,.2f}\n")

        # 2. Get active BTC markets
        resp = await client.get(f"{GAMMA_API}/markets", params={
            "active": "true", "closed": "false", "order": "volume24hr",
            "ascending": "false", "limit": 200,
        })
        markets = resp.json()

        btc_markets = []
        for m in markets:
            q = m.get("question", "").lower()
            if not ("bitcoin" in q or "btc" in q):
                continue
            if not any(kw in q for kw in ["above", "up or down", "reach", "dip"]):
                continue
            if not m.get("active") or m.get("closed"):
                continue

            strike = extract_strike(m.get("question", ""))
            if strike is None:
                continue

            end_date = m.get("end_date_iso") or m.get("endDate", "")
            if not end_date:
                continue
            try:
                end_date = end_date.replace("Z", "+00:00")
                t_res = datetime.fromisoformat(end_date)
                if t_res.tzinfo is None:
                    t_res = t_res.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            ttr_min = (t_res - now).total_seconds() / 60.0
            if ttr_min <= 0:
                continue

            # Extract CLOB token IDs
            tokens = m.get("tokens", [])
            yes_token = ""
            no_token = ""
            for t in tokens:
                outcome = str(t.get("outcome", "")).upper()
                tid = t.get("token_id") or t.get("tokenId") or ""
                if outcome == "YES" and tid:
                    yes_token = tid
                elif outcome == "NO" and tid:
                    no_token = tid

            if not yes_token:
                clob_ids = m.get("clobTokenIds", [])
                if isinstance(clob_ids, str):
                    try:
                        clob_ids = json.loads(clob_ids)
                    except Exception:
                        clob_ids = []
                if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                    yes_token = clob_ids[0]
                    no_token = clob_ids[1]

            gamma_yes_prob = extract_yes_prob(m)

            btc_markets.append({
                "market_id": m.get("conditionId", m.get("id", "")),
                "question": m.get("question", ""),
                "strike": strike,
                "ttr_min": ttr_min,
                "yes_token": yes_token,
                "no_token": no_token,
                "gamma_yes_prob": gamma_yes_prob,
                "volume_24h": float(m.get("volume24hr", 0) or 0),
            })

        print(f"Found {len(btc_markets)} active BTC binary markets\n")

        # 3. Fetch CLOB books and compute edges
        print(f"{'Question':<55} {'Strike':>8} {'TTR':>6} {'Fair':>6} {'CLOB':>6} "
              f"{'YesAsk':>7} {'NoBid':>7} {'Vig':>6} {'EdgeY':>7} {'EdgeN':>7} {'MM_err':>7}")
        print("-" * 140)

        rows = []
        for mkt in sorted(btc_markets, key=lambda m: m["ttr_min"]):
            q_fair = fair_prob(spot, mkt["strike"], mkt["ttr_min"])

            # Fetch YES book
            yes_book = None
            no_book = None
            if mkt["yes_token"]:
                try:
                    r = await client.get(f"{CLOB_API}/book", params={"token_id": mkt["yes_token"]})
                    if r.status_code == 200:
                        yes_book = r.json()
                except Exception:
                    pass
                await asyncio.sleep(0.3)

            if mkt["no_token"]:
                try:
                    r = await client.get(f"{CLOB_API}/book", params={"token_id": mkt["no_token"]})
                    if r.status_code == 200:
                        no_book = r.json()
                except Exception:
                    pass
                await asyncio.sleep(0.3)

            yes_ask = 1.0
            yes_bid = 0.0
            no_ask = 1.0
            no_bid = 0.0

            if yes_book:
                asks = yes_book.get("asks", [])
                bids = yes_book.get("bids", [])
                if asks:
                    yes_ask = min(float(a["price"]) for a in asks)
                if bids:
                    yes_bid = max(float(b["price"]) for b in bids)

            if no_book:
                asks = no_book.get("asks", [])
                bids = no_book.get("bids", [])
                if asks:
                    no_ask = min(float(a["price"]) for a in asks)
                if bids:
                    no_bid = max(float(b["price"]) for b in bids)

            vig = yes_ask + no_ask - 1.0
            clob_mid = (yes_ask + yes_bid) / 2.0

            edge_yes = q_fair - yes_ask
            edge_no = (1.0 - q_fair) - no_ask
            mm_err = q_fair - clob_mid  # how much MM disagrees with our fair

            q_short = mkt["question"][:52] + "..." if len(mkt["question"]) > 55 else mkt["question"]
            print(f"{q_short:<55} {mkt['strike']:>8,.0f} {mkt['ttr_min']:>5.0f}m "
                  f"{q_fair:>5.3f} {clob_mid:>5.3f} {yes_ask:>6.3f} {no_bid:>6.3f} "
                  f"{vig:>5.3f} {edge_yes:>+6.3f} {edge_no:>+6.3f} {mm_err:>+6.3f}")

            rows.append({
                "market_id": mkt["market_id"],
                "question": mkt["question"],
                "strike": mkt["strike"],
                "ttr_min": round(mkt["ttr_min"], 1),
                "spot": spot,
                "fair_prob": round(q_fair, 4),
                "clob_yes_ask": yes_ask,
                "clob_yes_bid": yes_bid,
                "clob_no_ask": no_ask,
                "clob_no_bid": no_bid,
                "clob_mid": round(clob_mid, 4),
                "vig": round(vig, 4),
                "edge_yes": round(edge_yes, 4),
                "edge_no": round(edge_no, 4),
                "mm_error": round(mm_err, 4),
                "gamma_yes_prob": mkt["gamma_yes_prob"],
                "volume_24h": mkt["volume_24h"],
                "timestamp": now.isoformat(),
            })

        # 4. Summary statistics
        if rows:
            print(f"\n{'='*60}")
            mm_errors = [abs(r["mm_error"]) for r in rows if r["clob_mid"] > 0.01]
            edges = [max(r["edge_yes"], r["edge_no"]) for r in rows]
            vigs = [r["vig"] for r in rows]
            print(f"Markets analyzed: {len(rows)}")
            if mm_errors:
                print(f"MM error |fair - clob_mid|: mean={sum(mm_errors)/len(mm_errors):.4f} "
                      f"max={max(mm_errors):.4f} min={min(mm_errors):.4f}")
            print(f"Best edge (max of edge_yes, edge_no): mean={sum(edges)/len(edges):.4f} "
                  f"max={max(edges):.4f}")
            tradeable = [r for r in rows if max(r["edge_yes"], r["edge_no"]) > 0.05]
            print(f"Tradeable (edge > 5%): {len(tradeable)} / {len(rows)}")
            print(f"Vig: mean={sum(vigs)/len(vigs):.4f} max={max(vigs):.4f}")

            # Save CSV
            import csv
            out_path = "data/raw/clob_snapshots/probe_live_clob.csv"
            import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

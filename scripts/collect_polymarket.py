"""
collect_polymarket.py — Polymarket market history collector.

CRITICAL PURPOSE: Collects GROUND TRUTH for model training.
Without this data, there are NO LABELS (Y variable) to train on.

For each resolved market, we need:
  - strike_price (static, from question text)
  - T_open, T_resolution (timestamps)
  - outcome (YES/NO — i.e. did BTC end above strike?)
  - CLOB odds at various points in the market lifecycle

Output: data/raw/polymarket_markets.parquet

Usage:
  python scripts/collect_polymarket.py
  python scripts/collect_polymarket.py --limit 1000 --deep
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import click
import httpx
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"

STRIKE_PATTERNS = [
    r"above\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"below\s+\$?([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
    r"\$([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)",
]

MARKET_SEARCH_TERMS = [
    "bitcoin",
    "btc",
    "bitcoin above",
    "bitcoin price",
]


def extract_strike_price(text: str) -> Optional[float]:
    """Extract static strike price from market question text."""
    for pattern in STRIKE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                price = float(price_str)
                if 1_000 < price < 1_000_000:
                    return price
            except ValueError:
                continue
    return None


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp to UTC datetime."""
    if not ts:
        return None
    try:
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


async def collect_markets(
    limit: int = 500,
    deep: bool = False,
    min_dates: int = 0
) -> pd.DataFrame:
    """
    Collect historical Bitcoin Up/Down markets from Polymarket Gamma API.
    """
    logger.info(
        "polymarket_fetch_start", limit=limit, deep=deep, min_dates=min_dates
    )
    all_markets: List[Dict] = []
    seen_ids: set = set()
    unique_dates: set = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # ── Fetch Resolved Markets ────────────────────────────
        offset = 0
        batch_size = 100

        while offset < limit:
            try:
                resp = await client.get(
                    f"{GAMMA_API}/markets",
                    params={
                        "closed": "true",
                        "limit": min(batch_size, limit - offset),
                        "offset": offset,
                        "order": "endDate", # Sort by resolution date
                        "ascending": "false",
                    },
                )
                resp.raise_for_status()
                raw_markets = resp.json()

                if not isinstance(raw_markets, list):
                    raw_markets = raw_markets.get("data", [])
                    if not raw_markets:
                        raw_markets = raw_markets.get("markets", [])

                if not raw_markets:
                    logger.info("no_more_markets", offset=offset)
                    break

                for m in raw_markets:
                    question = m.get("question", "").lower()

                    # Filter for Bitcoin up/down markets
                    is_btc_market = any(
                        term in question for term in MARKET_SEARCH_TERMS
                    )
                    if not is_btc_market:
                        continue

                    mid = m.get("condition_id") or m.get("id", "")
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)

                    parsed = _parse_resolved_market(m)
                    if parsed:
                        all_markets.append(parsed)
                        if parsed.get("resolution_date"):
                            unique_dates.add(parsed["resolution_date"])

                offset += len(raw_markets)
                logger.info(
                    "polymarket_batch",
                    offset=offset,
                    btc_markets_found=len(all_markets),
                    unique_dates=len(unique_dates),
                )

                # Early exit if we reached target depth
                if min_dates > 0 and len(unique_dates) >= min_dates:
                    logger.info("target_depth_reached", target=min_dates, found=len(unique_dates))
                    break

                await asyncio.sleep(0.3)

            except httpx.HTTPError as e:
                logger.error("gamma_api_error", error=str(e), offset=offset)
                await asyncio.sleep(5.0)
                offset += batch_size
            except Exception as e:
                logger.error("collect_error", error=str(e))
                break

        # ── Fetch Active Markets (for current data) ───────────
        try:
            resp = await client.get(
                f"{GAMMA_API}/markets",
                params={"closed": "false", "limit": 100},
            )
            resp.raise_for_status()
            active = resp.json()

            if isinstance(active, list):
                for m in active:
                    question = m.get("question", "").lower()
                    if any(term in question for term in MARKET_SEARCH_TERMS):
                        mid = m.get("condition_id") or m.get("id", "")
                        if mid not in seen_ids:
                            parsed = _parse_resolved_market(m)
                            if parsed:
                                all_markets.append(parsed)

        except Exception as e:
            logger.warning("active_markets_error", error=str(e))

        # ── Deep Mode: Fetch Historical CLOB Prices ───────────
        if deep and all_markets:
            logger.info("deep_mode_fetching_clob_history", markets=len(all_markets))
            for i, market in enumerate(all_markets):
                if market.get("yes_token_id"):
                    clob_data = await _fetch_clob_history(
                        client, market["yes_token_id"]
                    )
                    if clob_data:
                        market.update(clob_data)

                if (i + 1) % 20 == 0:
                    logger.info("clob_history_progress", done=i + 1, total=len(all_markets))
                    await asyncio.sleep(1.0)

    if not all_markets:
        logger.error("no_btc_markets_found")
        return pd.DataFrame()

    df = pd.DataFrame(all_markets)
    for col in ["volume_usd", "liquidity", "liquidity_usd", "volume", "yes_final_price", "no_final_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df = df.drop_duplicates(subset=["market_id"]).sort_values(
        "t_resolution_utc", na_position="last"
    ).reset_index(drop=True)

    # Summary stats
    has_strike = df["strike_price"].notna()
    is_resolved = df["is_resolved"] == True

    logger.info(
        "polymarket_collection_complete",
        total_markets=len(df),
        resolved=is_resolved.sum(),
        with_strike_price=has_strike.sum(),
        with_outcome=df["outcome_binary"].notna().sum(),
    )

    return df


def _parse_resolved_market(m: dict) -> Optional[Dict]:
    """
    Parse a Gamma API market record into ground truth row.

    Extracts everything needed for label construction:
      - strike_price: static from question text
      - T_resolution: when the market resolves
      - outcome: which side won (YES/NO)
      - outcome_binary: 1 if YES won (BTC > strike), 0 if NO won
    """
    try:
        question = m.get("question", "")
        mid = m.get("condition_id") or m.get("id", "")

        if not mid:
            return None

        # ── Strike Price ──────────────────────────────────────
        strike = extract_strike_price(question)
        if strike is None:
            desc = m.get("description", "")
            strike = extract_strike_price(desc)
        # Some markets won't have parseable strikes — record anyway

        # ── Timestamps ────────────────────────────────────────
        end_date_str = (
            m.get("end_date_iso")
            or m.get("endDate")
            or m.get("end_date", "")
        )
        created_str = (
            m.get("created_at")
            or m.get("createdAt")
            or m.get("start_date", "")
        )

        t_resolution = parse_timestamp(end_date_str)
        t_open = parse_timestamp(created_str)

        # ── Token IDs ─────────────────────────────────────────
        tokens = m.get("tokens", [])
        yes_token = ""
        no_token = ""
        for t in tokens:
            outcome = t.get("outcome", "").upper()
            if outcome == "YES":
                yes_token = t.get("token_id", "")
            elif outcome == "NO":
                no_token = t.get("token_id", "")

        # Fallback
        if not yes_token:
            clob_ids = m.get("clobTokenIds", [])
            if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                yes_token = clob_ids[0]
                no_token = clob_ids[1]

        # ── Outcome / Resolution ──────────────────────────────
        is_resolved = m.get("closed", False) or m.get("resolved", False)
        outcome_str = m.get("outcome", "")  # "Yes" or "No" or ""

        # Outcome prices (final settlement — 1.0 = winner, 0.0 = loser)
        outcome_prices = m.get("outcomePrices", [])
        yes_final = None
        no_final = None
        if outcome_prices:
            if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                try:
                    yes_final = float(outcome_prices[0])
                    no_final = float(outcome_prices[1])
                except (ValueError, TypeError):
                    pass
            elif isinstance(outcome_prices, str):
                # Sometimes JSON string
                import json
                try:
                    prices = json.loads(outcome_prices)
                    if len(prices) >= 2:
                        yes_final = float(prices[0])
                        no_final = float(prices[1])
                except (json.JSONDecodeError, ValueError):
                    pass

        # Binary outcome: 1 = YES/UP won (BTC > strike), 0 = NO/DOWN won
        outcome_binary = None
        
        # 1. Parse by outcome prices + explicit outcomes array matching
        outcomes_arr = m.get("outcomes", [])
        if yes_final is not None and no_final is not None and outcomes_arr and len(outcomes_arr) >= 2:
            out_0 = outcomes_arr[0].lower() if isinstance(outcomes_arr[0], str) else ""
            out_1 = outcomes_arr[1].lower() if isinstance(outcomes_arr[1], str) else ""
            
            # Identify which index means "UP" or "YES"
            up_idx = 0 if out_0 in ("up", "yes") else 1 if out_1 in ("up", "yes") else None
            
            if up_idx is not None:
                # If the 'up' index won, outcome = 1
                up_price = yes_final if up_idx == 0 else no_final
                outcome_binary = 1 if up_price > 0.5 else 0
        
        # 2. Legacy fallback
        if outcome_binary is None and yes_final is not None:
             outcome_binary = 1 if yes_final > 0.5 else 0
        elif outcome_binary is None and outcome_str:
            if outcome_str.lower() in ("yes", "1", "true", "up"):
                outcome_binary = 1
            elif outcome_str.lower() in ("no", "0", "false", "down"):
                outcome_binary = 0

        # ── Resolution source ─────────────────────────────────
        rules = (
            m.get("description", "")
            + " "
            + str(m.get("resolution_source", ""))
        ).lower()
        
        full_text_lower = (question + " " + rules).lower()

        resolution_source = None
        if "pyth" in rules:
            resolution_source = "Pyth"
        elif "coinbase" in rules:
            resolution_source = "Coinbase"
        elif "coingecko" in rules:
            resolution_source = "CoinGecko"
        elif "binance" in rules:
            resolution_source = "Binance"

        # ── Filter: Duration & Text ────────────────────
        # Skip macro long-term markets (e.g. End of Year prediction)
        if t_resolution and t_open:
            lifespan_ms = t_resolution.timestamp() * 1000 - t_open.timestamp() * 1000
            
            # If the market lifespan is greater than 14 days, it's macro
            if lifespan_ms > 14 * 24 * 60 * 60 * 1000:
                return None
                
            # If the market lifespan is greater than 2 days (Daily markets are usually ~24h to 48h),
            # we check for "bitcoin" and binary price keywords.
            if lifespan_ms > 48 * 60 * 60 * 1000:
                 # Check if it's explicitly a "daily", "up or down", etc.
                 if not any(kw in full_text_lower for kw in ["daily", "hour", "15m", "minute", "price", "above", "below"]):
                     return None
        
        # Absolute hard limit: don't allow markets lasting > 14 days
        if t_resolution and t_open and (t_resolution.timestamp() - t_open.timestamp()) > 14 * 24 * 60 * 60:
            return None

        return {
            "market_id": mid,
            "question": question,
            "strike_price": strike,
            "t_open_utc": t_open.isoformat() if t_open else None,
            "t_resolution_utc": t_resolution.isoformat() if t_resolution else None,
            "t_open_epoch_ms": int(t_open.timestamp() * 1000) if t_open else None,
            "t_resolution_epoch_ms": int(t_resolution.timestamp() * 1000) if t_resolution else None,
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "is_resolved": is_resolved,
            "outcome_str": outcome_str,
            "outcome_binary": outcome_binary,
            "yes_final_price": yes_final,
            "no_final_price": no_final,
            "resolution_date": t_resolution.strftime("%Y-%m-%d") if t_resolution else None,
            "volume_usd": m.get("volume", 0),
            "liquidity_usd": m.get("liquidity", 0),
        }

    except Exception as e:
        logger.warning("parse_error", error=str(e))
        return None


async def _fetch_clob_history(
    client: httpx.AsyncClient,
    token_id: str,
) -> Optional[Dict]:
    """
    Fetch historical CLOB price for a resolved market.
    Returns mid-market price snapshots if available.
    """
    try:
        resp = await client.get(
            f"{CLOB_API}/prices-history",
            params={
                "tokenID": token_id,
                "interval": "max",
                "fidelity": 60,  # 1-minute resolution
            },
            timeout=15.0,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        history = data.get("history", [])

        if not history:
            return None

        # Extract key price points
        prices = [float(h.get("p", 0)) for h in history if h.get("p")]
        timestamps = [int(h.get("t", 0)) for h in history if h.get("t")]

        if not prices:
            return None

        return {
            "clob_price_count": len(prices),
            "clob_price_first": prices[0] if prices else None,
            "clob_price_last": prices[-1] if prices else None,
            "clob_price_min": min(prices),
            "clob_price_max": max(prices),
            "clob_price_mean": sum(prices) / len(prices),
        }

    except Exception as e:
        logger.debug("clob_history_error", error=str(e))
        return None


# ============================================================
# CLI
# ============================================================

@click.command()
@click.option("--limit", default=500, help="Maximum markets to collect")
@click.option("--deep", is_flag=True, help="Also fetch CLOB price history")
@click.option("--min-dates", default=0, help="Minimum unique resolution dates to fetch")
@click.option("--output", default=None, help="Output file path")
def main(limit: int, deep: bool, min_dates: int, output: str | None) -> None:
    """Collect Polymarket Bitcoin Up/Down market history (ground truth)."""
    click.echo(f"🔍 Collecting Polymarket BTC markets (limit: {limit}, deep: {deep}, min_dates: {min_dates})...")
    click.echo("   This provides GROUND TRUTH labels for model training.\n")

    df = asyncio.run(collect_markets(limit=limit, deep=deep, min_dates=min_dates))

    if df.empty:
        click.echo("❌ No BTC markets found")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else OUTPUT_DIR / "polymarket_markets.parquet"
    df.to_parquet(out_path, index=False)

    # Report
    has_strike = df["strike_price"].notna().sum()
    has_outcome = df["outcome_binary"].notna().sum()
    resolved = (df["is_resolved"] == True).sum()

    click.echo(f"✅ Saved {len(df)} markets to {out_path}")
    click.echo(f"   Resolved:          {resolved}")
    click.echo(f"   With strike price: {has_strike}")
    click.echo(f"   With outcome:      {has_outcome}")
    click.echo(f"   YES outcomes:      {(df['outcome_binary'] == 1).sum()}")
    click.echo(f"   NO  outcomes:      {(df['outcome_binary'] == 0).sum()}")

    if has_outcome > 0:
        yes_rate = (df["outcome_binary"] == 1).sum() / has_outcome
        click.echo(f"   YES win rate:      {yes_rate:.1%}")

    if has_strike == 0:
        click.echo("\n⚠️  WARNING: No strike prices extracted!")
        click.echo("   Check market question format or update STRIKE_PATTERNS regex.")


if __name__ == "__main__":
    main()

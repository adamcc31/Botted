"""
scripts/resolve_historical_labels.py
=====================================
Standalone script — DO NOT import into main.py or any production pipeline.

Resolves PENDING, BUY_UP, and BUY_DOWN actual_outcome labels in the SQLite signals table.
It queries the Gamma API to map condition IDs to YES tokens and end dates,
and queries the Polymarket CLOB prices-history API concurrently to resolve outcomes.

Usage (on Railway):
    python3 scripts/resolve_historical_labels.py [--db /path/to/trading.db] [--dry-run] [--limit N]

Strategy:
    1. Load unresolved signals (where actual_outcome is PENDING, BUY_UP, or BUY_DOWN).
    2. Extract all unique condition IDs (market_id).
    3. Query Gamma API in batches of 20 to map condition IDs to YES token IDs and market end dates.
    4. Group signals by condition ID.
    5. For each condition ID, fetch CLOB prices-history concurrently from the earliest signal time
       to the market end date using the YES token.
    6. For each signal in that market, resolve to WIN or LOSE based on whether the YES price
       reached the threshold (>= 0.80 for BUY_UP, <= 0.20 for BUY_DOWN).
    7. Commit results in batches to SQLite.
"""

import asyncio
import aiohttp
import sqlite3
import argparse
import logging
import time
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

WIN_THRESHOLD     = 0.80      # YES token price must reach this to be WIN
BATCH_SIZE        = 1_000     # SQLite commit batch size
RATE_LIMIT_RPS    = 10        # Max requests per second to CLOB/Gamma APIs
REQUEST_TIMEOUT   = 15        # Seconds per individual API request
MAX_RETRIES       = 3         # Retry on transient failures
RETRY_DELAY       = 2.0       # Base delay between retries

# API base URLs
GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_API = "https://clob.polymarket.com/prices-history"

# Default DB path (Railway volume)
DEFAULT_DB = "/app/data/trading.db"

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("resolve_labels")


# ── Database Operations ───────────────────────────────────────────────────────

def load_unresolved_signals(db_path: str, limit: int | None = None) -> list[dict]:
    """
    Fetch all signals that need outcome resolution.
    This includes PENDING and any legacy BUY_UP/BUY_DOWN in actual_outcome.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT
            signal_id,
            market_id,
            timestamp_utc,
            signal_type,
            actual_outcome,
            entry_odds,
            ttr_minutes
        FROM signals
        WHERE (actual_outcome IS NULL OR actual_outcome NOT IN ('WIN', 'LOSE', 'INVALID'))
          AND market_id IS NOT NULL
          AND market_id != ''
        ORDER BY timestamp_utc ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    log.info(f"Loaded {len(rows):,} unresolved signals from {db_path}")
    return rows


def batch_update_outcomes(
    db_path: str,
    results: list[tuple[str, str]],
    dry_run: bool = False,
) -> int:
    """
    Batch UPDATE actual_outcome in SQLite.
    """
    if not results:
        return 0

    if dry_run:
        log.info(f"[DRY-RUN] Would update {len(results):,} rows")
        return len(results)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    updated = 0
    for i in range(0, len(results), BATCH_SIZE):
        batch = results[i : i + BATCH_SIZE]
        cur.executemany(
            "UPDATE signals SET actual_outcome = ? WHERE signal_id = ?",
            batch,
        )
        conn.commit()
        updated += len(batch)
        log.debug(f"  Committed batch {i // BATCH_SIZE + 1}: {len(batch)} rows")

    conn.close()
    return updated


# ── Helper for Timestamps ─────────────────────────────────────────────────────

def _parse_timestamp(ts_str: str) -> int:
    """
    Convert timestamp_utc string from SQLite → Unix timestamp (int).
    Handles ISO formats like: '2026-04-28 10:01:59.225405+00:00'
    """
    ts_str = ts_str.replace(" ", "T")
    if "." in ts_str:
        base, rest = ts_str.split(".", 1)
        if "+" in rest:
            micro, tz = rest.split("+", 1)
            ts_str = f"{base}.{micro[:6]}+{tz}"
        elif rest.endswith("Z"):
            ts_str = f"{base}.{rest[:6]}Z"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.fromisoformat(ts_str[:26]).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


# ── Gamma API: Batch Fetch Market Info ───────────────────────────────────────

async def fetch_gamma_metadata_batch(
    session: aiohttp.ClientSession,
    batch_condition_ids: list[str],
    status: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Query Gamma API for a batch of condition_ids.
    """
    params = [("condition_ids", cid) for cid in batch_condition_ids]
    params.append((status, "true"))

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(
                    GAMMA_API,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait = RETRY_DELAY * (2 ** attempt)
                        await asyncio.sleep(wait)
                    else:
                        log.debug(f"Gamma API status {resp.status} for batch")
                        return []
            except Exception as e:
                wait = RETRY_DELAY * attempt
                await asyncio.sleep(wait)
    return []


async def fetch_all_market_metadata(
    session: aiohttp.ClientSession,
    unique_mids: list[str],
    rate_limiter: asyncio.Semaphore,
) -> dict[str, tuple[str, int]]:
    """
    Fetch metadata (YES token ID, end_ts) for all unique condition IDs.
    Queries closed markets first, and falls back to active markets if needed.
    """
    market_metadata = {}
    batches = [unique_mids[i : i + 20] for i in range(0, len(unique_mids), 20)]

    async def process_batch(batch, status):
        data = await fetch_gamma_metadata_batch(session, batch, status, rate_limiter)
        for m in data:
            cid = m.get("conditionId")
            clob_ids_str = m.get("clobTokenIds", "[]")
            end_date_str = m.get("endDate")
            
            try:
                clob_ids = json.loads(clob_ids_str) if isinstance(clob_ids_str, str) else clob_ids_str
            except Exception:
                clob_ids = []
            
            if cid and len(clob_ids) >= 2 and end_date_str:
                yes_token = clob_ids[0]
                try:
                    dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    end_ts = int(dt.timestamp())
                except Exception:
                    end_ts = None
                
                if yes_token and end_ts:
                    market_metadata[cid] = (yes_token, end_ts)

    # 1. Fetch closed markets
    log.info(f"Querying Gamma API for {len(unique_mids):,} unique markets (closed status)...")
    tasks = [process_batch(b, "closed") for b in batches]
    await asyncio.gather(*tasks)

    # 2. Fetch active markets for any missing IDs
    missing_mids = [m for m in unique_mids if m not in market_metadata]
    if missing_mids:
        log.info(f"Querying Gamma API for {len(missing_mids):,} remaining markets (active status)...")
        active_batches = [missing_mids[i : i + 20] for i in range(0, len(missing_mids), 20)]
        tasks = [process_batch(b, "active") for b in active_batches]
        await asyncio.gather(*tasks)

    log.info(f"Gamma Mapping: found metadata for {len(market_metadata):,}/{len(unique_mids):,} unique markets.")
    return market_metadata


# ── CLOB API: Price History Operations ────────────────────────────────────────

async def fetch_price_history(
    session: aiohttp.ClientSession,
    token_id: str,
    start_ts: int,
    end_ts: int,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Fetch price history candles for a token.
    Uses fidelity=1 since these are 5-minute Bitcoin markets.
    """
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": 1,
    }

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(
                    CLOB_API,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("history", [])
                    elif resp.status == 429:
                        wait = RETRY_DELAY * (2 ** attempt)
                        await asyncio.sleep(wait)
                    else:
                        return []
            except Exception:
                wait = RETRY_DELAY * attempt
                await asyncio.sleep(wait)
    return []


# ── Market Resolver Task ─────────────────────────────────────────────────────

async def resolve_market_signals(
    session: aiohttp.ClientSession,
    market_id: str,
    signals: list[dict],
    metadata: tuple[str, int] | None,
    semaphore: asyncio.Semaphore,
) -> list[tuple[str, str]]:
    """
    Worker function to fetch history and resolve all signals for a single market.
    """
    if not metadata:
        # Market not found on Gamma API -> Mark all signals as INVALID
        return [("INVALID", sig["signal_id"]) for sig in signals]

    token_id, end_ts = metadata

    # 1. Parse signal timestamps and find minimum startTs
    signal_data = []
    min_start_ts = None
    for sig in signals:
        try:
            ts = _parse_timestamp(sig["timestamp_utc"])
            signal_data.append((ts, sig))
            if min_start_ts is None or ts < min_start_ts:
                min_start_ts = ts
        except Exception:
            continue

    if not signal_data:
        return [("INVALID", sig["signal_id"]) for sig in signals]

    # Add small buffer to endTs to ensure we cover the closing resolution candles
    clob_end_ts = end_ts + 300

    # 2. Fetch history from earliest signal to end date
    history = await fetch_price_history(session, token_id, min_start_ts, clob_end_ts, semaphore)

    if not history:
        # CLOB returned no price history -> Mark as INVALID
        return [("INVALID", sig["signal_id"]) for sig in signals]

    # Sort history chronologically
    history = sorted(history, key=lambda x: x["t"])

    # 3. Resolve each signal based on subsequent history path
    resolutions = []
    for sig_ts, sig in signal_data:
        sig_id = sig["signal_id"]
        sig_type = sig["signal_type"] or "ABSTAIN"
        outcome_dir = sig["actual_outcome"]  # May contain legacy BUY_UP/BUY_DOWN direction

        # BUY_DOWN signals (or legacy BUY_DOWN outcomes) represent the NO token side
        is_no_side = (sig_type == "BUY_DOWN" or outcome_dir == "BUY_DOWN")

        # Filter candles occurring AT or AFTER the signal time
        subsequent_candles = [c for c in history if c["t"] >= sig_ts]

        if not subsequent_candles:
            # No future price history records -> Mark as INVALID
            resolutions.append(("INVALID", sig_id))
            continue

        resolved = False
        if is_no_side:
            # WIN if YES price <= 0.20 (which means NO price >= 0.80)
            for c in subsequent_candles:
                price = c.get("p", 1.0)
                if price <= (1.0 - WIN_THRESHOLD):
                    resolutions.append(("WIN", sig_id))
                    resolved = True
                    break
        else:
            # WIN if YES price >= 0.80
            for c in subsequent_candles:
                price = c.get("p", 0.0)
                if price >= WIN_THRESHOLD:
                    resolutions.append(("WIN", sig_id))
                    resolved = True
                    break

        if not resolved:
            resolutions.append(("LOSE", sig_id))

    return resolutions


# ── Main Orchestrator ──────────────────────────────────────────────────────────

async def run_resolution(
    db_path: str,
    dry_run: bool = False,
    limit: int | None = None,
    concurrency: int = 15,
) -> dict:
    """
    Main async orchestrator.
    """
    signals = load_unresolved_signals(db_path, limit=limit)
    if not signals:
        log.info("No unresolved signals found. Nothing to do.")
        return {"total": 0, "win": 0, "lose": 0, "invalid": 0, "updated": 0}

    total = len(signals)
    stats = {"win": 0, "lose": 0, "invalid": 0}
    results = []

    # Rate limiting semaphore to control concurrency
    rate_limiter = asyncio.Semaphore(concurrency)

    connector = aiohttp.TCPConnector(
        limit=concurrency + 5,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    headers = {
        "Accept":     "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PolymarketLabelResolver/2.0",
    }

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        # Step 1: Extract unique condition IDs
        unique_mids = list(set(sig["market_id"] for sig in signals))
        
        # Step 2: Fetch mapping from Gamma API
        market_metadata = await fetch_all_market_metadata(session, unique_mids, rate_limiter)

        # Step 3: Group signals by market ID
        from collections import defaultdict
        signals_by_market = defaultdict(list)
        for sig in signals:
            signals_by_market[sig["market_id"]].append(sig)

        log.info(f"Resolving price paths for {len(signals_by_market):,} markets using CLOB API...")
        t0 = time.monotonic()

        # Step 4: Run resolution concurrently per market
        tasks = []
        for mid, sig_list in signals_by_market.items():
            metadata = market_metadata.get(mid)
            tasks.append(resolve_market_signals(session, mid, sig_list, metadata, rate_limiter))

        # Gather resolutions
        log.info("Executing price path resolution tasks...")
        resolution_blocks = await asyncio.gather(*tasks)

        # Flatten outcomes
        for block in resolution_blocks:
            for outcome, signal_id in block:
                results.append((outcome, signal_id))
                stats[outcome.lower()] += 1

        # Step 5: Flush outcomes to database
        if results:
            log.info(f"Committing {len(results):,} resolved labels to SQLite...")
            updated = batch_update_outcomes(db_path, results, dry_run=dry_run)
            log.info(f"Successfully updated {updated:,} outcomes in the database.")

    elapsed = time.monotonic() - t0
    summary = {
        "total":    total,
        "win":      stats.get("win", 0),
        "lose":     stats.get("lose", 0),
        "invalid":  stats.get("invalid", 0),
        "updated":  len(results),
        "elapsed_min": round(elapsed / 60, 1),
    }
    return summary


# ── Post-Run Verification ──────────────────────────────────────────────────────

def verify_results(db_path: str) -> None:
    """Print outcome distribution after run for sanity check."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT actual_outcome, COUNT(*) as n
        FROM signals
        GROUP BY actual_outcome
        ORDER BY n DESC
    """)
    rows = cur.fetchall()
    conn.close()

    log.info("=== Outcome distribution after run ===")
    total = sum(r[1] for r in rows)
    for outcome, n in rows:
        pct = n / total * 100
        log.info(f"  {outcome or 'NULL':<20s}: {n:>8,}  ({pct:.1f}%)")
    log.info(f"  {'TOTAL':<20s}: {total:>8,}")


# ── Entry Point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve PENDING/unresolved signal labels via Polymarket CLOB API")
    p.add_argument("--db",          default=DEFAULT_DB,  help="Path to SQLite trading.db")
    p.add_argument("--dry-run",     action="store_true", help="Query API but don't write to DB")
    p.add_argument("--limit",       type=int, default=None, help="Limit number of signals processed (for testing)")
    p.add_argument("--concurrency", type=int, default=15,   help="Concurrent API requests (default: 15)")
    p.add_argument("--log-level",   default="INFO",      choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    log.info("=" * 65)
    log.info("  resolve_historical_labels.py — Polymarket Label Resolver")
    log.info("=" * 65)
    log.info(f"  DB path      : {args.db}")
    log.info(f"  Dry run      : {args.dry_run}")
    log.info(f"  Limit        : {args.limit or 'ALL'}")
    log.info(f"  Concurrency  : {args.concurrency}")
    log.info(f"  WIN threshold: {WIN_THRESHOLD}")
    log.info("=" * 65)

    if not Path(args.db).exists():
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Run async loop
    summary = asyncio.run(
        run_resolution(
            db_path=args.db,
            dry_run=args.dry_run,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    )

    log.info("")
    log.info("=" * 65)
    log.info("  RESOLUTION COMPLETE")
    log.info("=" * 65)
    log.info(f"  Total signals processed : {summary['total']:,}")
    log.info(f"  WIN                     : {summary['win']:,}")
    log.info(f"  LOSE                    : {summary['lose']:,}")
    log.info(f"  INVALID                 : {summary['invalid']:,}")
    log.info(f"  Elapsed                 : {summary['elapsed_min']} minutes")
    log.info("=" * 65)

    if not args.dry_run:
        verify_results(args.db)


if __name__ == "__main__":
    main()

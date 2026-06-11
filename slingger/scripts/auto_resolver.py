"""
scripts/auto_resolver.py — Slingger V6 Daemon Resolver
=======================================================
Automatically labels PENDING rows in shadow dry-run CSV files as WIN or LOSE
by querying the Polymarket CLOB API for post-signal outcome data.

CRITICAL CONSTRAINTS:
  - PERMANENT HARD CUTOFF: skip any file/session before 2026-06-06
  - Thread-safe CSV updates via file locking (filelock library)
  - Max 1 CLOB API request/second with exponential backoff
  - PENDING → WIN/LOSE only; never overwrite existing WIN/LOSE labels
  - All secrets from .env — never hardcoded

Usage:
    python scripts/auto_resolver.py             # Run once
    python scripts/auto_resolver.py --daemon    # Run as 6-hour daemon
    python scripts/auto_resolver.py --dry-run   # Simulate without writing
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import schedule
from dotenv import load_dotenv

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    # Fallback: minimal lock using a .lock file sentinel
    class FileLock:
        def __init__(self, path, timeout=60):
            self.path = path + ".lock"
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    FileLockTimeout = Exception

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SLINGGER_DIR = SCRIPT_DIR.parent  # slingger/
ROOT_DIR = SLINGGER_DIR.parent    # project root

DATA_DIR = SLINGGER_DIR / "data"
LOGS_DIR = SLINGGER_DIR / "logs"
ENV_PATH = ROOT_DIR / ".env"

# ─────────────────────────────────────────────────────────────────
# HARD BOUNDARY — NEVER CHANGE THIS VALUE
# ─────────────────────────────────────────────────────────────────
CUTOFF_DATE = date(2026, 6, 6)  # PERMANENT IGNORE — data sebelum ini beracun

# ─────────────────────────────────────────────────────────────────
# API Config
# ─────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 2       # Exponential: 2^n seconds between retries
REQUEST_TIMEOUT_SEC = 10
MIN_REQUEST_INTERVAL_SEC = 1.0  # Rate limit: 1 req/sec max

# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(dotenv_path=ENV_PATH, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        RotatingFileHandler(
            LOGS_DIR / "slingger_pipeline.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("auto_resolver")

# ─────────────────────────────────────────────────────────────────
# CLOB API Client
# ─────────────────────────────────────────────────────────────────
def get_clob_session() -> requests.Session:
    """Create a requests session with CLOB API authentication."""
    session = requests.Session()
    api_key = os.getenv("CLOB_API_KEY") or os.getenv("POLY_BUILDER_API_KEY")
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


_clob_session: Optional[requests.Session] = None
_last_request_time: float = 0.0


def _get_session() -> requests.Session:
    global _clob_session
    if _clob_session is None:
        _clob_session = get_clob_session()
    return _clob_session


def _throttle() -> None:
    """Enforce 1 request/second rate limit."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL_SEC:
        time.sleep(MIN_REQUEST_INTERVAL_SEC - elapsed)
    _last_request_time = time.monotonic()


def query_clob_market_trades(
    market_id: str,
    after_timestamp: datetime,
    base_url: Optional[str] = None,
) -> Optional[list[dict]]:
    """
    Query CLOB API for trade history on a market after a given timestamp.

    Returns:
        List of trade dicts with 'price', 'timestamp' keys,
        or None if the API call failed / data unavailable.
    """
    if base_url is None:
        base_url = os.getenv("CLOB_API_BASE_URL", "https://clob.polymarket.com")

    url = f"{base_url}/trades"
    params = {
        "market": market_id,
        "after": after_timestamp.isoformat(),
        "limit": 50,
    }

    session = _get_session()
    _throttle()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"[CLOB] Querying market={market_id} attempt={attempt}")
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SEC)

            if resp.status_code == 429:
                wait = BASE_BACKOFF_SEC ** attempt
                logger.warning(f"[CLOB] Rate limited (429). Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = BASE_BACKOFF_SEC ** attempt
                logger.warning(f"[CLOB] Server error {resp.status_code}. Waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # API returns {"data": [...]} or directly a list
            if isinstance(data, dict):
                return data.get("data", data.get("trades", []))
            elif isinstance(data, list):
                return data
            else:
                logger.warning(f"[CLOB] Unexpected response format: {type(data)}")
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"[CLOB] Request timed out (attempt {attempt}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[CLOB] Connection error (attempt {attempt}/{MAX_RETRIES}): {e}")
        except Exception as e:
            logger.error(f"[CLOB] Unexpected error (attempt {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(BASE_BACKOFF_SEC ** attempt)

    logger.error(f"[CLOB] All {MAX_RETRIES} attempts failed for market={market_id}")
    return None


# ─────────────────────────────────────────────────────────────────
# Resolution Logic
# ─────────────────────────────────────────────────────────────────
PRICE_MOVE_THRESHOLD = 0.02  # 2% minimum price movement to count as WIN


def resolve_outcome(
    market_id: str,
    predicted_side: str,  # "YES" or "NO"
    signal_timestamp: datetime,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Determine WIN/LOSE for a signal by checking CLOB price 5 minutes post-signal.

    The rule: if the YES price moved >= PRICE_MOVE_THRESHOLD in the direction
    of predicted_side within 5 minutes, the signal is WIN, else LOSE.

    Returns:
        "WIN"  — price moved in predicted direction
        "LOSE" — price moved against predicted direction
        None   — insufficient CLOB data, retry later
    """
    if dry_run:
        logger.info(f"[RESOLVE] DRY RUN — would query market={market_id} side={predicted_side}")
        return None

    # Query window: signal_time to signal_time + 5 minutes
    window_start = signal_timestamp
    window_end = signal_timestamp + timedelta(minutes=5)

    trades = query_clob_market_trades(market_id, after_timestamp=window_start)

    if trades is None:
        logger.warning(f"[RESOLVE] CLOB unavailable for market={market_id} — skip, retry later")
        return None

    if not trades:
        logger.info(f"[RESOLVE] No trades found in window for market={market_id}")
        return None

    # Filter trades within the 5-minute window
    window_trades = []
    for trade in trades:
        try:
            # Handle various timestamp formats from CLOB API
            ts_raw = trade.get("timestamp") or trade.get("created_at") or trade.get("time")
            if ts_raw is None:
                continue
            ts = pd.to_datetime(ts_raw, utc=True)
            if window_start <= ts <= window_end:
                window_trades.append(trade)
        except Exception:
            continue

    if not window_trades:
        logger.info(
            f"[RESOLVE] No trades within 5-min window [{window_start} → {window_end}] "
            f"for market={market_id}"
        )
        return None

    # Extract prices from window trades
    prices = []
    for trade in window_trades:
        try:
            price = float(trade.get("price", 0))
            if 0 < price < 1:
                prices.append(price)
        except (ValueError, TypeError):
            continue

    if not prices:
        logger.warning(f"[RESOLVE] No valid prices extracted for market={market_id}")
        return None

    # Determine direction: first price vs last price in window
    price_start = prices[0]
    price_end = prices[-1]
    price_delta = price_end - price_start

    logger.debug(
        f"[RESOLVE] market={market_id} side={predicted_side} "
        f"prices: {price_start:.4f} → {price_end:.4f} (Δ={price_delta:+.4f})"
    )

    # YES direction: price going UP is WIN for YES buyer
    # NO direction: price going DOWN (YES falls) is WIN for NO buyer
    if predicted_side == "YES":
        outcome = "WIN" if price_delta >= PRICE_MOVE_THRESHOLD else "LOSE"
    elif predicted_side == "NO":
        outcome = "WIN" if price_delta <= -PRICE_MOVE_THRESHOLD else "LOSE"
    else:
        logger.warning(f"[RESOLVE] Unknown predicted_side: {predicted_side}")
        return None

    return outcome


# ─────────────────────────────────────────────────────────────────
# CSV Discovery & Filtering
# ─────────────────────────────────────────────────────────────────
def find_resolvable_files() -> list[Path]:
    """
    Scan data/ for shadow dry-run CSVs that contain PENDING rows.
    Returns only files with dates >= CUTOFF_DATE.
    """
    pattern = "dry_run_shadow_*.csv"
    candidates = []

    for csv_path in sorted(DATA_DIR.glob(pattern)):
        # Extract date from filename: dry_run_shadow_YYYY-MM-DD_HHMMSS*.csv
        try:
            parts = csv_path.stem.split("_")
            # Format: dry_run_shadow_2026-06-06_154404[_resolved]
            # Date is at index 3 (0-indexed after split on _)
            date_str = None
            for part in parts:
                if len(part) == 10 and part.count("-") == 2:
                    date_str = part
                    break

            if date_str is None:
                logger.warning(f"[SCAN] Cannot parse date from: {csv_path.name} — skip")
                continue

            file_date = date.fromisoformat(date_str)

            # ── CRITICAL GATE — PERMANENT IGNORE ─────────────────
            if file_date < CUTOFF_DATE:
                logger.warning(
                    f"[SCAN] SKIPPED pre-cutoff session: {csv_path.name} "
                    f"(date={file_date} < cutoff={CUTOFF_DATE})"
                )
                continue  # PERMANENT IGNORE — data beracun

            candidates.append(csv_path)

        except Exception as e:
            logger.warning(f"[SCAN] Error parsing {csv_path.name}: {e} — skip")

    logger.info(f"[SCAN] Found {len(candidates)} eligible files post-cutoff")
    return candidates


# ─────────────────────────────────────────────────────────────────
# CSV Update (Thread-safe)
# ─────────────────────────────────────────────────────────────────
def update_row_outcome(csv_path: Path, row_index: int, outcome: str) -> None:
    """
    Thread-safe single-row update in a CSV file.
    Uses FileLock to prevent concurrent write corruption.
    """
    lock_path = str(csv_path) + ".lock"

    with FileLock(lock_path, timeout=60):
        # Re-read inside lock to get fresh state
        df = pd.read_csv(csv_path)

        # Determine label column name
        label_col = "label" if "label" in df.columns else "actual_outcome"

        # Safety: only update if still PENDING
        current_val = df.at[row_index, label_col]
        if current_val not in ("PENDING", None, ""):
            logger.warning(
                f"[UPDATE] Row {row_index} in {csv_path.name} "
                f"already labeled '{current_val}' — skipping overwrite"
            )
            return

        df.at[row_index, label_col] = outcome
        df.to_csv(csv_path, index=False)
        logger.info(
            f"[RESOLVE] market_id={df.at[row_index, 'market_id']} "
            f"outcome={outcome} → {csv_path.name}[row {row_index}]"
        )


# ─────────────────────────────────────────────────────────────────
# Main Resolver Run
# ─────────────────────────────────────────────────────────────────
def run_resolver(dry_run: bool = False) -> dict:
    """
    Main resolver pass: scan all eligible CSVs and resolve PENDING rows.

    Returns:
        dict with keys: files_scanned, rows_resolved, rows_skipped, rows_failed
    """
    logger.info("[RESOLVER] ========== Starting resolver pass ==========")
    stats = {"files_scanned": 0, "rows_resolved": 0, "rows_skipped": 0, "rows_failed": 0}

    csv_files = find_resolvable_files()
    stats["files_scanned"] = len(csv_files)

    for csv_path in csv_files:
        logger.info(f"[RESOLVER] Processing: {csv_path.name}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"[RESOLVER] Failed to read {csv_path.name}: {e}")
            stats["rows_failed"] += 1
            continue

        # Determine label column
        label_col = "label" if "label" in df.columns else "actual_outcome"
        if label_col not in df.columns:
            logger.warning(f"[RESOLVER] No label column in {csv_path.name} — skip file")
            continue

        # Find PENDING rows
        pending_mask = df[label_col].isin(["PENDING", None]) | df[label_col].isna()
        pending_rows = df[pending_mask]

        if len(pending_rows) == 0:
            logger.info(f"[RESOLVER] No PENDING rows in {csv_path.name}")
            continue

        logger.info(f"[RESOLVER] {len(pending_rows)} PENDING rows in {csv_path.name}")

        for idx, row in pending_rows.iterrows():
            try:
                market_id = str(row.get("market_id", ""))
                signal_dir = str(row.get("signal_direction", "YES"))

                # Parse signal timestamp
                ts_raw = row.get("timestamp")
                if pd.isna(ts_raw):
                    logger.warning(f"[RESOLVER] Row {idx}: missing timestamp — skip")
                    stats["rows_skipped"] += 1
                    continue

                signal_ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime()

                # Skip if signal is too recent (need 5-min + buffer to resolve)
                age_minutes = (datetime.now(timezone.utc) - signal_ts).total_seconds() / 60
                if age_minutes < 6:
                    logger.info(
                        f"[RESOLVER] Row {idx} too recent ({age_minutes:.1f} min) — skip"
                    )
                    stats["rows_skipped"] += 1
                    continue

                # Resolve outcome
                outcome = resolve_outcome(
                    market_id=market_id,
                    predicted_side=signal_dir,
                    signal_timestamp=signal_ts,
                    dry_run=dry_run,
                )

                if outcome is None:
                    logger.info(f"[RESOLVER] Row {idx}: CLOB data unavailable — defer")
                    stats["rows_skipped"] += 1
                else:
                    if not dry_run:
                        update_row_outcome(csv_path, idx, outcome)
                    stats["rows_resolved"] += 1

            except Exception as e:
                logger.error(f"[RESOLVER] Error processing row {idx}: {e}")
                stats["rows_failed"] += 1

    logger.info(
        f"[RESOLVER] ===== Pass complete: "
        f"files={stats['files_scanned']} "
        f"resolved={stats['rows_resolved']} "
        f"skipped={stats['rows_skipped']} "
        f"failed={stats['rows_failed']} ====="
    )
    return stats


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Slingger V6 Daemon Resolver")
    parser.add_argument("--daemon", action="store_true", help="Run as 6-hour recurring daemon")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing")
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=6.0,
        help="Daemon interval in hours (default: 6)",
    )
    args = parser.parse_args()

    if args.daemon:
        logger.info(f"[RESOLVER] Starting daemon mode (every {args.interval_hours}h)")

        # Run immediately on start
        run_resolver(dry_run=args.dry_run)

        # Schedule recurring runs
        schedule.every(args.interval_hours).hours.do(run_resolver, dry_run=args.dry_run)

        logger.info(f"[RESOLVER] Daemon active. Next run in {args.interval_hours}h.")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    else:
        # One-shot run
        stats = run_resolver(dry_run=args.dry_run)
        if stats["rows_failed"] > 0:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())

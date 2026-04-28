"""
retrofix_strikes.py
====================
Memperbaiki strike_price dan resolution_price di CSV training data
yang dikumpulkan saat Vatic Oracle sedang down (502 Bad Gateway).

Selama Vatic down, bot menggunakan Chainlink current price sebagai
approximation (CHAINLINK_RTDS_ACTIVE_WINDOW / CHAINLINK_RTDS_CURRENT).
Script ini mengganti nilai tersebut dengan harga Chainlink yang benar
menggunakan Vatic History API.

  strike_price     = first Chainlink tick >= epoch_ts        (window OPEN)
  resolution_price = first Chainlink tick >= epoch_ts + 300  (window CLOSE)

Cara pakai:
    pip install httpx pandas tqdm

    python retrofix_strikes.py \
        --input  /app/data/training_data.csv \
        --output /app/data/training_data_fixed.csv \
        --start  1777253821 \
        --end    0

    --start : Unix timestamp saat Vatic mulai down
              (dari log: 2026-04-27T01:57:01Z = 1777253821)
    --end   : Unix timestamp saat Vatic pulih
              (0 = gunakan sekarang)

Output:
    CSV baru dengan kolom:
    - strike_price       : nilai yang sudah diperbaiki (window open)
    - resolution_price   : harga penutupan window (window close)
    - oracle_source      : 'VATIC_RETROFIX' untuk baris yang difix
    - retrofix_delta     : selisih strike lama vs baru (untuk audit)
    - retrofix_status    : 'FIXED' | 'SKIPPED_ACCURATE' |
                           'SKIPPED_NO_DATA' | 'SKIPPED_OUTSIDE_WINDOW'
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from tqdm import tqdm

# PTB rate limiting — jeda antar request ke polymarket.com
PTB_REQUEST_DELAY_SECONDS = 1.0   # 1 detik antar slug = max ~60 req/menit
PTB_MAX_SLUGS = 50                 # Jika > 50 slugs unik, skip PTB sepenuhnya

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

VATIC_BASE = "https://api.vatic.trading"

# Source values yang dianggap approximation (perlu difix)
APPROX_SOURCES = {
    "CHAINLINK_RTDS_ACTIVE_WINDOW",
    "CHAINLINK_RTDS_CURRENT",
    "CHAINLINK_RTDS_CACHED",
    "BINANCE_APPROX",
    "",          # kosong = tidak ter-track
}

# Source values yang dianggap akurat (skip, tidak perlu difix)
ACCURATE_SOURCES = {
    "CHAINLINK_RTDS_SNAPSHOT",   # captured tepat di epoch boundary
    "CHAINLINK_FIRST_TICK",      # first tick at/after epoch
    "PTB_API",                   # Polymarket price-to-beat endpoint
    "VATIC",                     # dari Vatic langsung
    "VATIC_RETROFIX",            # sudah pernah difix
}

# Toleransi — jika selisih < threshold, tidak perlu difix
FIX_THRESHOLD_USD = 0.5

# Batch size untuk fetch Vatic (agar tidak overload API)
VATIC_FETCH_CHUNK_SECONDS = 300_000  # 1000 points × 300s = ~83 hours, fits in one request

# ──────────────────────────────────────────────
# VATIC API FUNCTIONS
# ──────────────────────────────────────────────

def check_vatic_health() -> bool:
    """
    Cek apakah Vatic API sudah pulih sebelum memulai proses.
    Return True jika healthy, False jika masih down.
    """
    print("Checking Vatic API health...")
    endpoints_to_try = [
        f"{VATIC_BASE}/health",
        f"{VATIC_BASE}/api/v1/targets/timestamp?asset=btc&type=5min&timestamp=1744027500",
    ]
    for url in endpoints_to_try:
        try:
            r = httpx.get(url, timeout=10.0)
            if r.status_code == 200:
                print(f"  Vatic healthy — {url} returned 200")
                return True
            elif r.status_code == 502:
                print(f"  Vatic still down (502) — {url}")
                return False
            else:
                print(f"  Unexpected status {r.status_code} — {url}")
        except Exception as e:
            print(f"  Connection failed: {e}")
    return False


def fetch_vatic_chainlink_range(
    start_ts: int,
    end_ts: int,
    asset: str = "btc"
) -> list[dict]:
    """
    Fetch Chainlink price history dari Vatic API untuk range waktu tertentu.
    Returns list of {timestamp: int, price: float} sorted by timestamp.

    Endpoint: GET /api/v1/history/chainlink/range
    Params:   asset=btc, start=<unix>, end=<unix>
    """
    all_ticks = []

    # Fetch dalam chunks agar tidak overload API
    current_start = start_ts
    while current_start < end_ts:
        chunk_end = min(current_start + VATIC_FETCH_CHUNK_SECONDS, end_ts)

        try:
            r = httpx.get(
                f"{VATIC_BASE}/api/v1/history/range",
                params={
                    "asset": asset,
                    "type": "5min",       # REQUIRED per docs — missing = 400 Bad Request
                    "start": current_start,
                    "end": chunk_end,
                },
                timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()

            # Handle Vatic API response format
            # Docs confirmed: {"history": [...], "count": N, ...}
            ticks = (
                data.get("history", [])  # Vatic primary field (confirmed from docs)
                or (data if isinstance(data, list) else [])
                or data.get("data", [])
                or data.get("prices", [])
            )

            if not ticks:
                print(f"  Warning: no ticks for range {current_start}-{chunk_end}")
            else:
                all_ticks.extend(ticks)
                print(f"  Fetched {len(ticks)} ticks [{current_start} → {chunk_end}]")

        except httpx.HTTPStatusError as e:
            print(f"  HTTP error fetching range {current_start}-{chunk_end}: {e}")
            if e.response.status_code == 502:
                print("  Vatic still returning 502 — aborting fetch")
                raise
        except Exception as e:
            print(f"  Error fetching range {current_start}-{chunk_end}: {e}")

        current_start = chunk_end

    # Normalize tick format dan sort
    # Vatic API response format (from docs):
    # {"history": [{"timestamp_start": 1744012800, "price": 82100, "source": "chainlink"}]}
    normalized = []
    for tick in all_ticks:
        ts = (
            tick.get("timestamp_start")   # Vatic v1 format (docs confirmed)
            or tick.get("timestamp")
            or tick.get("ts")
            or tick.get("time")
        )
        price = tick.get("price") or tick.get("value") or tick.get("close")
        if ts and price:
            normalized.append({"timestamp": int(ts), "price": float(price)})

    normalized.sort(key=lambda x: x["timestamp"])
    return normalized


def get_strike_at_epoch(ticks: list[dict], epoch_ts: int) -> float | None:
    """
    Ambil harga Chainlink PERTAMA yang timestamp-nya >= epoch_ts.
    Ini adalah definisi resmi dari 'Price to Beat' Polymarket:
    first price at/after the window boundary (window OPEN).

    Args:
        ticks     : list of {timestamp, price} sorted ascending
        epoch_ts  : Unix timestamp dari window boundary (divisible by 300)

    Returns:
        float price atau None jika tidak ada data
    """
    for tick in ticks:
        if tick["timestamp"] >= epoch_ts:
            return tick["price"]
    return None


def get_resolution_price(ticks: list[dict], epoch_ts: int) -> float | None:
    """
    Ambil harga Chainlink PERTAMA yang timestamp-nya >= epoch_ts + 300.
    Ini adalah harga penutupan window (window CLOSE / The Finish Line).

    Identik secara logika dengan get_strike_at_epoch, tapi menargetkan
    close time bukan open time. Polymarket menggunakan harga ini untuk
    menentukan apakah market resolve UP atau DOWN.

    Args:
        ticks     : list of {timestamp, price} sorted ascending
        epoch_ts  : Unix timestamp dari window OPEN (divisible by 300)
                    Close time = epoch_ts + 300

    Returns:
        float price atau None jika tidak ada data
    """
    close_ts = epoch_ts + 300
    for tick in ticks:
        if tick["timestamp"] >= close_ts:
            return tick["price"]
    return None


# ──────────────────────────────────────────────
# POLYMARKET PTB API — BATCH PRE-FETCH
# (rate-limited, dijalankan SEKALI sebelum main loop)
# ──────────────────────────────────────────────

def batch_prefetch_ptb(
    slugs: list[str],
    delay_seconds: float = PTB_REQUEST_DELAY_SECONDS,
    max_slugs: int = PTB_MAX_SLUGS,
) -> dict[str, float]:
    """
    Pre-fetch Price-to-Beat dari Polymarket API untuk semua slug unik,
    dengan rate limiting yang aman (1 request/detik).

    PENTING: Fungsi ini dipanggil SEKALI sebelum main loop,
    bukan di dalam loop per-row. Ini mencegah 429/403 dari Cloudflare.

    Args:
        slugs         : list slug unik yang perlu di-fetch
        delay_seconds : jeda antar request (default 1.0 detik)
        max_slugs     : jika jumlah slug > max_slugs, skip PTB sepenuhnya

    Returns:
        dict {slug: price} untuk slug yang berhasil di-fetch
    """
    if len(slugs) > max_slugs:
        print(f"  PTB pre-fetch: {len(slugs)} slugs > max {max_slugs} — "
              f"skipping PTB, using Vatic history only")
        return {}

    print(f"  PTB pre-fetch: fetching {len(slugs)} slugs "
          f"at {delay_seconds}s delay (~{len(slugs) * delay_seconds:.0f}s total)...")

    result: dict[str, float] = {}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; retrofix-script/1.0)"}

    for i, slug in enumerate(tqdm(slugs, desc="PTB pre-fetch")):
        try:
            url = f"https://polymarket.com/api/equity/price-to-beat/{slug}"
            r = httpx.get(url, timeout=10.0, headers=headers)

            if r.status_code == 200:
                data = r.json()
                price = (
                    data.get("price")
                    or data.get("value")
                    or data.get("strike")
                    or data.get("open_price")
                )
                if price:
                    result[slug] = float(price)

            elif r.status_code in (429, 403):
                print(f"\n  Rate limit hit at slug {i+1}/{len(slugs)} — "
                      f"stopping PTB pre-fetch, Vatic history used for remaining")
                break

        except Exception:
            pass  # Silent fail — Vatic history digunakan sebagai fallback

        # Jeda antar request — wajib, jangan dihapus
        if i < len(slugs) - 1:
            time.sleep(delay_seconds)

    print(f"  PTB pre-fetch complete: {len(result)}/{len(slugs)} slugs resolved")
    return result


def fetch_vatic_single_epoch(epoch_ts: int, asset: str = "btc") -> float | None:
    """
    Fetch strike price untuk satu epoch dari Vatic targets/timestamp endpoint.
    Digunakan sebagai fallback jika epoch tidak ada di range history.

    Endpoint: GET /api/v1/targets/timestamp
    Params:   asset=btc, type=5min, timestamp=<epoch_ts>

    Catatan dari docs: retries up to 4x dengan 1s delay untuk absorb
    ~1-5s publish lag di window boundaries. Sudah built-in di Vatic.
    """
    try:
        r = httpx.get(
            f"{VATIC_BASE}/api/v1/targets/timestamp",
            params={
                "asset": asset,
                "type": "5min",
                "timestamp": epoch_ts,
            },
            timeout=15.0,
        )
        if r.status_code == 200:
            data = r.json()
            price = data.get("price")
            if price:
                return float(price)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
# MAIN RETROFIX FUNCTION
# ──────────────────────────────────────────────

def retrofix_csv(
    csv_path: str,
    output_path: str,
    vatic_down_start: int,
    vatic_down_end: int,
    dry_run: bool = False,
) -> dict:
    """
    Main function untuk memperbaiki strike_price di CSV.

    Args:
        csv_path         : path ke CSV input
        output_path      : path ke CSV output
        vatic_down_start : Unix timestamp awal Vatic down
        vatic_down_end   : Unix timestamp akhir Vatic down (0 = sekarang)
        dry_run          : jika True, hanya print tanpa simpan

    Returns:
        dict dengan statistik: total, fixed, skipped, errors
    """
    if vatic_down_end == 0:
        vatic_down_end = int(time.time())

    print(f"\n{'='*60}")
    print(f"RETROFIX STRIKES")
    print(f"{'='*60}")
    print(f"Input  : {csv_path}")
    print(f"Output : {output_path}")
    print(f"Window : {datetime.fromtimestamp(vatic_down_start, tz=timezone.utc)} → "
          f"{datetime.fromtimestamp(vatic_down_end, tz=timezone.utc)}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # Load CSV
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    original_len = len(df)
    print(f"Loaded {original_len} rows from {csv_path}")

    # Identifikasi rows yang perlu difix
    has_strike_source_col = "strike_price_source" in df.columns

    def needs_fix(row) -> bool:
        # Harus dalam window Vatic down
        try:
            row_ts = int(row["timestamp"].timestamp())
        except Exception:
            return False
        if not (vatic_down_start <= row_ts <= vatic_down_end):
            return False

        # Jika kolom strike_price_source tersedia, gunakan sebagai filter utama
        if has_strike_source_col:
            strike_src = str(row.get("strike_price_source", "") or "")
            if strike_src in ACCURATE_SOURCES:
                return False  # Strike sudah akurat, skip
            # Jika strike_src kosong atau approximation, perlu difix
            return True

        # Jika kolom strike_price_source TIDAK ADA (dataset lama):
        # Asumsikan SEMUA rows dalam window Vatic down perlu difix,
        # karena kita tidak bisa tahu mana yang approximation.
        # oracle_source = CHAINLINK_LIVE hanya berarti spread filter bekerja,
        # bukan berarti strike_price akurat.
        return True

    rows_to_fix = df[df.apply(needs_fix, axis=1)]
    rows_outside = len(df) - len(df[df.apply(
        lambda r: vatic_down_start <= int(r["timestamp"].timestamp()) <= vatic_down_end,
        axis=1
    )])

    print(f"Rows in Vatic down window   : {len(df) - rows_outside}")
    print(f"Rows needing fix            : {len(rows_to_fix)}")
    print(f"Rows outside window (skip)  : {rows_outside}\n")

    if len(rows_to_fix) == 0:
        print("Nothing to fix. Exiting.")
        return {"total": original_len, "fixed": 0, "skipped": original_len, "errors": 0}

    # Fetch Chainlink data dari Vatic
    print(f"Fetching Chainlink history from Vatic API...")
    print(f"Range: {vatic_down_start} → {vatic_down_end} "
          f"({(vatic_down_end - vatic_down_start) / 3600:.1f} hours)\n")

    try:
        ticks = fetch_vatic_chainlink_range(vatic_down_start, vatic_down_end)
    except Exception as e:
        print(f"\nFATAL: Failed to fetch from Vatic: {e}")
        print("Run this script again after Vatic API fully recovers.")
        sys.exit(1)

    print(f"\nTotal ticks fetched: {len(ticks)}")
    if not ticks:
        print("ERROR: No ticks fetched. Cannot proceed.")
        sys.exit(1)

    # Tambahkan kolom retrofix jika belum ada
    if "retrofix_delta" not in df.columns:
        df["retrofix_delta"] = None
    if "retrofix_status" not in df.columns:
        df["retrofix_status"] = "SKIPPED_OUTSIDE_WINDOW"
    if "resolution_price" not in df.columns:
        df["resolution_price"] = None

    # Pre-fetch PTB prices SEKALI sebelum main loop (rate-limited, safe)
    # Ambil semua slug unik yang perlu difix
    unique_slugs = list(rows_to_fix["slug"].dropna().unique())
    print(f"\nPre-fetching PTB prices for {len(unique_slugs)} unique slugs...")
    ptb_cache = batch_prefetch_ptb(unique_slugs)
    print(f"PTB cache: {len(ptb_cache)} entries\n")

    # Process setiap row yang perlu difix
    stats = {"fixed": 0, "skipped_accurate": 0, "skipped_no_data": 0,
             "skipped_outside": 0, "errors": 0}

    print(f"Processing {len(rows_to_fix)} rows...")
    for idx, row in tqdm(rows_to_fix.iterrows(), total=len(rows_to_fix)):
        try:
            # Ambil epoch dari slug
            slug = str(row.get("slug", ""))
            try:
                epoch_ts = int(slug.split("-")[-1])
                assert epoch_ts > 1_700_000_000, f"epoch_ts looks wrong: {epoch_ts}"
                assert epoch_ts % 300 == 0, f"epoch not aligned to 5-min boundary: {epoch_ts}"
            except (ValueError, IndexError, AssertionError) as e:
                df.at[idx, "retrofix_status"] = f"SKIPPED_BAD_SLUG:{e}"
                stats["errors"] += 1
                continue

            old_strike = float(row.get("strike_price", 0) or 0)

            # Strike price (window OPEN) — lookup dari pre-fetched cache
            # PTB cache sudah diisi sebelum loop (batch_prefetch_ptb)
            # Tidak ada HTTP call di sini — zero rate limit risk
            correct_price = ptb_cache.get(slug)
            source = "PTB_API"

            # Fallback ke Vatic history
            if not correct_price:
                correct_price = get_strike_at_epoch(ticks, epoch_ts)
                source = "VATIC_RETROFIX"

            if correct_price is None:
                # Last resort: query Vatic per-epoch endpoint
                # Useful if epoch falls outside range history or in a gap
                correct_price = fetch_vatic_single_epoch(epoch_ts)
                if correct_price:
                    source = "VATIC_SINGLE_EPOCH"
                    resolution_price = get_resolution_price(ticks, epoch_ts)

            if correct_price is None:
                df.at[idx, "retrofix_status"] = "SKIPPED_NO_DATA"
                stats["skipped_no_data"] += 1
                continue

            # Resolution price (window CLOSE = epoch + 300)
            # Selalu dari Vatic history — PTB endpoint tidak punya data ini
            resolution_price = get_resolution_price(ticks, epoch_ts)

            delta = abs(correct_price - old_strike)

            # Jika selisih sangat kecil, sudah cukup akurat
            if delta < FIX_THRESHOLD_USD:
                df.at[idx, "retrofix_status"] = "SKIPPED_ACCURATE"
                df.at[idx, "retrofix_delta"] = round(delta, 4)
                # Tetap update resolution_price jika tersedia
                if resolution_price and not dry_run:
                    df.at[idx, "resolution_price"] = round(resolution_price, 5)
                stats["skipped_accurate"] += 1
                continue

            # Apply fix
            # Apply fix
            if not dry_run:
                df.at[idx, "strike_price"] = round(correct_price, 5)
                df.at[idx, "oracle_source"] = source
                if "strike_price_source" in df.columns:
                    df.at[idx, "strike_price_source"] = source
                
                if resolution_price:
                    df.at[idx, "resolution_price"] = round(resolution_price, 5)
                    
                    # --- NEW: Rekalkulasi Label (Win/Loss) ---
                    signal_dir = str(row.get("signal_direction", ""))
                    if signal_dir in ["BUY_UP", "BUY_DOWN"]:
                        if signal_dir == "BUY_UP":
                            is_win = resolution_price > correct_price
                        else:  # BUY_DOWN
                            is_win = resolution_price < correct_price
                        
                        # Timpa label lama dengan realitas Chainlink yang baru
                        df.at[idx, "signal_correct"] = "TRUE" if is_win else "FALSE"

            df.at[idx, "retrofix_delta"] = round(delta, 4)
            df.at[idx, "retrofix_status"] = f"FIXED_{source}"

            stats["fixed"] += 1

            if dry_run:
                res_str = f" | resolution={resolution_price:.2f}" if resolution_price else ""
                print(f"  [DRY RUN] Row {idx}: strike {old_strike:.2f} → {correct_price:.2f} "
                      f"(delta={delta:.2f}{res_str}, slug={slug}, source={source})")

        except Exception as e:
            df.at[idx, "retrofix_status"] = f"ERROR:{e}"
            stats["errors"] += 1
            continue

    # Print summary
    print(f"\n{'='*60}")
    print(f"RETROFIX SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows in CSV          : {original_len}")
    print(f"Fixed                      : {stats['fixed']}")
    print(f"Skipped (already accurate) : {stats['skipped_accurate']}")
    print(f"Skipped (no Vatic data)    : {stats['skipped_no_data']}")
    print(f"Errors                     : {stats['errors']}")
    print(f"{'='*60}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return stats

    # Save output
    df.to_csv(output_path, index=False)
    print(f"\nSaved fixed CSV to: {output_path}")

    # Audit report
    fixed_rows = df[df["retrofix_status"].str.startswith("FIXED", na=False)]
    if len(fixed_rows) > 0:
        deltas = fixed_rows["retrofix_delta"].dropna().astype(float)
        print(f"\nDelta stats for fixed rows:")
        print(f"  Min   : ${deltas.min():.2f}")
        print(f"  Max   : ${deltas.max():.2f}")
        print(f"  Mean  : ${deltas.mean():.2f}")
        print(f"  Median: ${deltas.median():.2f}")
        print(f"\nLargest corrections:")
        top = fixed_rows.nlargest(5, "retrofix_delta")[
            ["slug", "strike_price", "retrofix_delta", "retrofix_status"]
        ]
        print(top.to_string(index=False))

    return stats


# ──────────────────────────────────────────────
# ML TRAINING DATA FILTER HELPER
# ──────────────────────────────────────────────

def filter_for_ml_training(csv_path: str, output_path: str) -> None:
    """
    Helper untuk filter dataset yang sudah diretrofix menjadi
    training-ready dataset untuk XGBoost.

    Filter rules (sesuai Fase 3 plan):
    - entry_odds_source = 'CLOB_LIVE'
    - signal_correct IN ('TRUE', 'FALSE')
    - oracle_source IN ('CHAINLINK_LIVE', 'CHAINLINK_CACHED', 'VATIC_RETROFIX', 'PTB_API')
    - signal_direction IN ('BUY_UP', 'BUY_DOWN')
    - retrofix_status != ERROR:* (jika kolom ada)
    """
    print(f"\nFiltering {csv_path} for ML training...")
    df = pd.read_csv(csv_path)
    original = len(df)

    # Base filters
    mask = (
        (df["entry_odds_source"] == "CLOB_LIVE") &
        (df["signal_correct"].isin(["TRUE", "FALSE"])) &
        (df["signal_direction"].isin(["BUY_UP", "BUY_DOWN"]))
    )

    # Oracle source filter
    valid_oracle = {
        "CHAINLINK_LIVE", "CHAINLINK_CACHED",
        "VATIC_RETROFIX", "PTB_API",
        "CHAINLINK_RTDS_SNAPSHOT", "CHAINLINK_FIRST_TICK"
    }
    if "oracle_source" in df.columns:
        mask = mask & (df["oracle_source"].isin(valid_oracle))

    # Exclude retrofix errors
    if "retrofix_status" in df.columns:
        mask = mask & (~df["retrofix_status"].str.startswith("ERROR", na=False))

    df_filtered = df[mask].copy()

    # Derived features untuk XGBoost
    df_filtered["obi_tfm_product"] = (
        df_filtered["obi_value"].astype(float) *
        df_filtered["tfm_value"].astype(float)
    )
    df_filtered["obi_tfm_alignment"] = (
        (df_filtered["obi_tfm_product"] > 0).astype(int)
    )
    df_filtered["label"] = (df_filtered["signal_correct"] == "TRUE").astype(int)

    df_filtered.to_csv(output_path, index=False)

    print(f"Original: {original} rows")
    print(f"Training-ready: {len(df_filtered)} rows")
    print(f"Class balance: {df_filtered['label'].mean():.2%} positive")
    print(f"Saved to: {output_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrofix strike prices in training CSV after Vatic Oracle downtime"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV (e.g. /app/data/training_data.csv)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output fixed CSV"
    )
    parser.add_argument(
        "--start", type=int, default=1777253821,
        help="Unix timestamp when Vatic went down (default: 2026-04-27T01:57:01Z = 1777253821)"
    )
    parser.add_argument(
        "--end", type=int, default=0,
        help="Unix timestamp when Vatic recovered (0 = now)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview fixes without writing output file"
    )
    parser.add_argument(
        "--skip-health-check", action="store_true",
        help="Skip Vatic health check (use if health endpoint unavailable but API works)"
    )
    parser.add_argument(
        "--filter-ml", action="store_true",
        help="After retrofix, also generate ML training-ready dataset"
    )
    parser.add_argument(
        "--ml-output",
        help="Path for ML-filtered output (required if --filter-ml)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Health check
    if not args.skip_health_check:
        healthy = check_vatic_health()
        if not healthy:
            print("\nVatic API is still down. Run this script after Vatic recovers.")
            print("To skip this check: add --skip-health-check flag")
            sys.exit(1)
        print()

    # Retrofix
    stats = retrofix_csv(
        csv_path=args.input,
        output_path=args.output,
        vatic_down_start=args.start,
        vatic_down_end=args.end,
        dry_run=args.dry_run,
    )

    # Filter untuk ML training
    if args.filter_ml and not args.dry_run:
        if not args.ml_output:
            print("\nError: --ml-output required when --filter-ml is set")
            sys.exit(1)
        filter_for_ml_training(
            csv_path=args.output,
            output_path=args.ml_output,
        )

    sys.exit(0 if stats["errors"] == 0 else 1)
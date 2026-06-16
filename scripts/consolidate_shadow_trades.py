"""
consolidate_shadow_trades.py
============================
Konsolidasi shadow trade log dari Railway ke SHADOW_MASTER.csv

Cari shadow trade log dengan:
  1. dry_run_shadow_*.csv  (exporter output per session)
  2. v5_trades SQLite       (native V5 database)
  3. JSONL logs             (fallback)

Usage (setelah copy file dari Railway):
    python scripts/consolidate_shadow_trades.py [--source-dir PATH]

Jalankan setelah rebuild_clob_master.py
"""

import argparse
import json
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional


ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / "dataset/shadow_trades/SHADOW_MASTER.csv"

# Expected minimal columns yang harus ada di output akhir
EXPECTED_COLS = [
    'trade_id', 'market_id', 'session_date', 'token_side',
    'entry_odds', 'exit_odds', 'entry_fill_price', 'exit_fill_price',
    'entry_fill_time', 'exit_fill_time', 'ttr_at_entry', 'ttr_at_exit',
    'result', 'net_pnl', 'stake_usd', 'shares',
    'btc_vs_strike_pct', 'depth_at_entry',
    'swing_prob', 'source',
]


def load_shadow_csvs(source_dir: Path) -> Optional[pd.DataFrame]:
    """Load semua dry_run_shadow_*.csv dari directory."""
    files = sorted(source_dir.glob("dry_run_shadow_*.csv"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df['source'] = f.name
            dfs.append(df)
            print(f"  [CSV] {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  [CSV] ERROR {f.name}: {e}")

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_v5_sqlite(db_path: Path) -> Optional[pd.DataFrame]:
    """Load v5_trades table dari SQLite native database."""
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(db_path)
        df   = pd.read_sql("SELECT * FROM v5_trades", conn)
        conn.close()
        df['source'] = 'v5_sqlite'
        print(f"  [SQLite] {db_path.name}: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  [SQLite] ERROR {db_path.name}: {e}")
        return None


def load_jsonl(path: Path) -> Optional[pd.DataFrame]:
    """Load JSONL trade log."""
    if not path.exists():
        return None

    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    if not records:
        return None

    df = pd.DataFrame(records)
    df['source'] = path.name
    print(f"  [JSONL] {path.name}: {len(df):,} rows")
    return df


def main(source_dir: Path):
    print("=" * 60)
    print("SHADOW TRADE CONSOLIDATION")
    print("=" * 60)
    print(f"Source dir: {source_dir}")

    all_frames = []

    # ── 1. Try CSV exports ─────────────────────────────────────────────
    print("\n[1] Looking for dry_run_shadow CSVs...")
    csv_df = load_shadow_csvs(source_dir)
    if csv_df is not None:
        all_frames.append(csv_df)
    else:
        print("     None found.")

    # ── 2. Try V5 SQLite ───────────────────────────────────────────────
    print("\n[2] Looking for v5_trades SQLite...")
    for db_candidate in [
        source_dir / "v5_trades.db",
        source_dir / "v5_database.db",
        ROOT / "data/v5_trades.db",
        ROOT / "data/database.db",
    ]:
        sqlite_df = load_v5_sqlite(db_candidate)
        if sqlite_df is not None:
            all_frames.append(sqlite_df)
            break
    else:
        print("     None found.")

    # ── 3. Try JSONL ───────────────────────────────────────────────────
    print("\n[3] Looking for JSONL logs...")
    for jsonl_candidate in source_dir.glob("*.jsonl"):
        jsonl_df = load_jsonl(jsonl_candidate)
        if jsonl_df is not None:
            all_frames.append(jsonl_df)

    if not all_frames:
        print("\nERROR: No shadow trade data found.")
        print("Pastikan file sudah di-copy dari Railway ke direktori yang benar.")
        print(f"  Expected: {source_dir}")
        return

    # ── 4. Combine ─────────────────────────────────────────────────────
    master = pd.concat(all_frames, ignore_index=True, sort=False)
    print(f"\nTotal combined : {len(master):,} rows")

    # Dedup berdasarkan trade_id jika ada
    if 'trade_id' in master.columns:
        before = len(master)
        master = master.drop_duplicates(subset=['trade_id'])
        print(f"Dedup (trade_id): {before:,} --> {len(master):,}")

    # ── 5. Result distribution report ─────────────────────────────────
    print("\n" + "-" * 40)
    print("RESULT DISTRIBUTION")
    print("-" * 40)
    if 'result' in master.columns:
        dist = master['result'].value_counts()
        for result, count in dist.items():
            pct = count / len(master) * 100
            print(f"  {result:<25} {count:>5,}  ({pct:.1f}%)")

        hits   = master['result'].isin(['HIT']).sum()
        misses = master['result'].isin(['MISS']).sum()
        total_resolved = hits + misses
        if total_resolved > 0:
            hit_rate = hits / total_resolved * 100
            print(f"\nHIT rate (HIT vs MISS only): {hit_rate:.1f}%  ({hits}/{total_resolved})")
    else:
        print("  WARNING: 'result' column not found")

    # ── 6. PnL report ─────────────────────────────────────────────────
    if 'net_pnl' in master.columns:
        pnl = master['net_pnl'].dropna()
        print(f"\nPnL Summary:")
        print(f"  Total net PnL : ${pnl.sum():.2f}")
        print(f"  Avg per trade : ${pnl.mean():.2f}")
        print(f"  Best trade    : ${pnl.max():.2f}")
        print(f"  Worst trade   : ${pnl.min():.2f}")

    # ── 7. Save ────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT, index=False)
    print(f"\nSHADOW_MASTER saved: {OUTPUT}")
    print(f"   Rows   : {len(master):,}")
    print(f"   Columns: {list(master.columns)}")

    # ── 8. Gap analysis ────────────────────────────────────────────────
    if 'entry_fill_time' in master.columns:
        master['entry_fill_time'] = pd.to_datetime(master['entry_fill_time'], unit='s', errors='coerce')
        master_sorted = master.dropna(subset=['entry_fill_time']).sort_values('entry_fill_time')
        print(f"\n   Date range: {master_sorted['entry_fill_time'].min()} --> {master_sorted['entry_fill_time'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate shadow trades from Railway")
    parser.add_argument(
        "--source-dir",
        default=str(ROOT / "dataset/shadow_trades/raw"),
        help="Directory dengan file dari Railway (default: dataset/shadow_trades/raw)"
    )
    args = parser.parse_args()
    main(Path(args.source_dir))

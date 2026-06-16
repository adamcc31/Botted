"""
rebuild_clob_master.py
======================
Gabungkan CLOB_MASTER lama dengan batch baru dari Railway.

Usage:
    python scripts/rebuild_clob_master.py

Prerequisites (jalankan di Railway SSH lebih dulu):
    mkdir -p /tmp/clob_export
    find /app/data/exports -name "clob_log*.csv" | xargs -I{} cp {} /tmp/clob_export/
    # Lalu scp / railway volume copy ke dataset/clob_log/new_batch/
"""

import pandas as pd
import glob
import os
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
OLD_MASTER   = ROOT / "dataset/clob_log/CLOB_MASTER.csv"
NEW_BATCH    = ROOT / "dataset/clob_log/new_batch"
OUTPUT       = ROOT / "dataset/clob_log/CLOB_MASTER_v2.csv"
BACKUP       = ROOT / "dataset/clob_log/CLOB_MASTER_v1_backup.csv"

def main():
    # ── 1. Load old master ─────────────────────────────────────────────
    if not OLD_MASTER.exists():
        print(f"Old master not found: {OLD_MASTER}")
        return

    print("=" * 60)
    print("CLOB MASTER REBUILD")
    print("=" * 60)

    old = pd.read_csv(OLD_MASTER, low_memory=False)
    print(f"Old master    : {len(old):>10,} rows | "
          f"{old['market_id'].nunique():,} markets | "
          f"{old['timestamp'].min()} --> {old['timestamp'].max()}")

    # ── 2. Backup old master ───────────────────────────────────────────
    if not BACKUP.exists():
        shutil.copy2(OLD_MASTER, BACKUP)
        print(f"Backed up   : {BACKUP.name}")
    else:
        print(f"Backup exists: {BACKUP.name} (skip)")

    # ── 3. Load new batch files ────────────────────────────────────────
    new_files = sorted(NEW_BATCH.glob("clob_log*.csv"))
    if not new_files:
        print(f"\nNo new files found in {NEW_BATCH}")
        print("Pastikan sudah copy file dari Railway ke dataset/clob_log/new_batch/")
        return

    print(f"\nNew batch files: {len(new_files)} files")
    new_dfs = []
    for f in new_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df['source_file'] = f.name
            new_dfs.append(df)
            print(f"  {f.name}: {len(df):>8,} rows")
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")

    if not new_dfs:
        print("No valid new files loaded. Abort.")
        return

    new_data = pd.concat(new_dfs, ignore_index=True)
    print(f"\nNew data total : {len(new_data):>10,} rows | "
          f"{new_data['market_id'].nunique():,} markets")

    # ── 4. Schema alignment report ────────────────────────────────────
    old_cols = set(old.columns)
    new_cols = set(new_data.columns)
    common   = old_cols & new_cols
    only_old = old_cols - new_cols - {'source_file'}
    only_new = new_cols - old_cols - {'source_file'}

    print(f"\nSchema         : {len(common)} common cols")
    if only_old:
        print(f"  Old-only cols: {sorted(only_old)}")
    if only_new:
        print(f"  New-only cols: {sorted(only_new)}")

    # ── 5. Concat + dedup ──────────────────────────────────────────────
    master = pd.concat([old, new_data], ignore_index=True, sort=False)
    before = len(master)

    if 'timestamp' in master.columns and 'market_id' in master.columns:
        master = master.drop_duplicates(subset=['timestamp', 'market_id'])
    else:
        print("Warning: 'timestamp' or 'market_id' column missing -- skipping dedup")

    if 'timestamp' in master.columns:
        master = master.sort_values('timestamp').reset_index(drop=True)

    removed = before - len(master)
    print(f"\nDedup          : {before:,} --> {len(master):,} ({removed:,} duplicates removed)")

    # ── 6. Save ────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT, index=False)

    print(f"\nCLOB_MASTER_v2 saved!")
    print(f"   Path   : {OUTPUT}")
    print(f"   Rows   : {len(master):,}")
    print(f"   Markets: {master['market_id'].nunique():,}")
    if 'timestamp' in master.columns:
        print(f"   Range  : {master['timestamp'].min()} --> {master['timestamp'].max()}")

    # ── 7. Summary comparison ──────────────────────────────────────────
    print("\n" + "-" * 60)
    print("PERBANDINGAN DATASET")
    print("-" * 60)
    print(f"{'METRIC':<25} {'LAMA':>15} {'BARU (v2)':>15}")
    print("-" * 60)
    print(f"{'Total rows':<25} {len(old):>15,} {len(master):>15,}")
    print(f"{'Unique markets':<25} {old['market_id'].nunique():>15,} {master['market_id'].nunique():>15,}")
    print("-" * 60)

if __name__ == "__main__":
    main()

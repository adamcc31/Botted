import pandas as pd
import glob
import os
from pathlib import Path

INPUT_DIR  = "dataset/clob_log"
OUTPUT_DIR = "dataset/clob_log"
OUTPUT_FILE = f"{OUTPUT_DIR}/CLOB_MASTER.csv"

files = sorted(glob.glob(f"{INPUT_DIR}/*clob_log.csv"))
print(f"[INFO] Found {len(files)} files to merge")

dfs = []
schema_issues = []

for f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
        print(f"  [OK] {os.path.basename(f)}: {len(df):,} rows | cols: {len(list(df.columns))}")
    except Exception as e:
        print(f"  [FAIL] {os.path.basename(f)}: FAILED — {e}")
        schema_issues.append(f)

if not dfs:
    raise RuntimeError("No files loaded. Check dataset/clob_log/ directory.")

# Schema alignment check
col_sets = [set(df.columns) for df in dfs]
common_cols = set.intersection(*col_sets)
all_cols    = set.union(*col_sets)
extra_cols  = all_cols - common_cols

if extra_cols:
    print(f"\n[WARNING] Schema mismatch across files.")
    print(f"  Common columns ({len(common_cols)}): {sorted(common_cols)}")
    print(f"  Extra columns not in all files: {sorted(extra_cols)}")
    print(f"  Action: Using common columns only. Extra cols will be NaN.")

master = pd.concat(dfs, ignore_index=True, sort=False)

# Deduplication
before = len(master)
if 'timestamp' in master.columns and 'market_id' in master.columns:
    master = master.drop_duplicates(subset=['timestamp', 'market_id'])
elif 'timestamp' in master.columns:
    master = master.drop_duplicates(subset=['timestamp'])
after = len(master)
print(f"\n[INFO] Deduplication: {before:,} -> {after:,} rows "
      f"({before - after:,} duplicates removed)")

# Sort by timestamp
if 'timestamp' in master.columns:
    master = master.sort_values('timestamp').reset_index(drop=True)

master.to_csv(OUTPUT_FILE, index=False)
print(f"\n[SUCCESS] Master saved: {OUTPUT_FILE}")
print(f"  Total rows:    {len(master):,}")
print(f"  Total markets: {master['market_id'].nunique():,}" 
      if 'market_id' in master.columns else "  market_id column not found")
print(f"  Columns:       {list(master.columns)}")
print(f"  Date range:    {master['timestamp'].min()} -> {master['timestamp'].max()}"
      if 'timestamp' in master.columns else "")
print(f"  File size:     {os.path.getsize(OUTPUT_FILE)/1024/1024:.1f} MB")

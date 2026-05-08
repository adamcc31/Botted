"""Build DATASET_MASTER_08-05-2026.csv with all dry-run data."""
import sys, glob, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
from model_training.dataset import merge_datasets, sanitize_dataset

# --- Build new master dataset ---
V4_TRAIN = Path('dataset/processed/predator_v4_train.csv')
RAW_DIR = Path('dataset/raw')
MASTER_OUT = Path('dataset/DATASET_MASTER_08-05-2026.csv')

df_v4 = pd.read_csv(V4_TRAIN, low_memory=False)
print("V4 Training:", len(df_v4), "rows")

dry_run_files = sorted(glob.glob(str(RAW_DIR / 'dry_run_*.csv')))
print("Dry-run files found:", len(dry_run_files))
for f in dry_run_files:
    print(" ", Path(f).name)

df_master = merge_datasets(df_v4, dry_run_files)
df_master = sanitize_dataset(df_master)

df_master.to_csv(MASTER_OUT, index=False)
print("Master dataset saved:", MASTER_OUT, "(", len(df_master), "rows )")
lbl = df_master["label"]
print("Label dist: WIN=", round(lbl.mean()*100, 1), "%  LOSS=", round((1-lbl.mean())*100, 1), "%")
ts = pd.to_datetime(df_master["timestamp"], utc=True)
print("Date range:", ts.min(), "to", ts.max())

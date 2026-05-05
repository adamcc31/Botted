import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def download_clob_log():
    clob_path = Path("dataset/clob_log.csv")
    if not clob_path.exists():
        print("clob_log.csv not found, pulling from railway via SSH...")
        os.makedirs("dataset", exist_ok=True)
        # Using Option B logic: python concatenation via SSH to avoid spinner interference in stdout
        # We use a list comprehension to keep it a single line valid for python -c
        python_cmd = (
            "import glob, sys; "
            "files = sorted(glob.glob('/app/data/exports/*/clob_log.csv')); "
            "[sys.stdout.write(''.join(open(f).readlines()[1:] if i > 0 else open(f).readlines())) "
            "for i, f in enumerate(files)]"
        )
        # Execute railway ssh and redirect to file
        os.system(f'railway ssh "python3 -c \\"{python_cmd}\\"" > dataset/clob_log.csv')
        print("Download completed: dataset/clob_log.csv")

def main():
    download_clob_log()
    
    print("Loading datasets...")
    dry_path = "dataset/processed/dry_run_master_clean.csv"
    clob_path = "dataset/clob_log.csv"
    
    if not os.path.exists(dry_path):
        print(f"Error: {dry_path} not found.")
        return
        
    dry = pd.read_csv(dry_path, low_memory=False)
    clob = pd.read_csv(clob_path, low_memory=False)
    
    # Pre-processing: timestamps and mapping entry_timestamp
    dry['entry_timestamp'] = pd.to_datetime(dry['timestamp'], utc=True)
    clob['timestamp'] = pd.to_datetime(clob['timestamp'], utc=True)
    
    print(f"Dataset sizes: dry={len(dry)}, clob={len(clob)}")
    
    # Implementation: O(N log N) confirmed join
    clob = clob.sort_values(['market_id', 'timestamp'])
    dry = dry.sort_values(['market_id', 'entry_timestamp'])
    
    clob_grouped = clob.groupby('market_id')
    
    def get_max_high(row):
        # Only process BUY signals
        if row['signal_direction'] not in ['BUY_UP', 'BUY_DOWN']:
            return np.nan
            
        mkt = clob_grouped.get_group(row['market_id']) \
              if row['market_id'] in clob_grouped.groups else None
        if mkt is None:
            return np.nan
            
        # Strict No-Look-Ahead: timestamp > entry_timestamp
        future = mkt[mkt['timestamp'] > row['entry_timestamp']]
        if future.empty:
            return np.nan
            
        if row['signal_direction'] == 'BUY_UP':
            return future['yes_bid'].max()
        else: # BUY_DOWN
            return future['no_bid'].max()

    print("Calculating max_high_bid (this may take a moment)...")
    dry['max_high_bid'] = dry.apply(get_max_high, axis=1)
    
    # Save modified dataframe
    out_path = "dataset_diagnostics.csv"
    dry.to_csv(out_path, index=False)
    print(f"Saved modified dataframe to {out_path}\n")
    
    # --- REPORTING ---
    df = dry.copy()
    
    # A. Threshold Sweep (Analisis Class Balance)
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print("--- HIT RATE BY THRESHOLD ---")
    for t in thresholds:
        hit_rate = (df['max_high_bid'] >= t).mean()
        print(f"Threshold {t}: {hit_rate:.2%} hit rate")
        
    # B. The 'Lost Alpha' Conversion Rate — Split per Arah Signal
    # Auto-detect encoding actual_outcome (string "LOSS" atau integer 0)
    # Mapping for this specific dataset where 'label' 0 is LOSS
    if 'label' in df.columns and (df['actual_outcome'].astype(str) != 'LOSS').all():
        df['actual_outcome_encoded'] = df['label']
    else:
        df['actual_outcome_encoded'] = df['actual_outcome']

    if df['actual_outcome_encoded'].dtype == object:
        loss_df = df[df['actual_outcome_encoded'] == 'LOSS']
    else:
        loss_df = df[df['actual_outcome_encoded'] == 0]

    print("\n--- LOSS CONVERTED TO WIN (LOST ALPHA) ---")
    print(f"Total LOSS trades: {len(loss_df)}")

    # Overall
    print("\n  [ALL DIRECTIONS]")
    for t in thresholds:
        rate = (loss_df['max_high_bid'] >= t).mean()
        print(f"  Threshold {t}: {rate:.2%} of LOSS trades could have been won")

    # Split per arah
    for direction in ['BUY_UP', 'BUY_DOWN']:
        dir_loss = loss_df[loss_df['signal_direction'] == direction]
        print(f"\n  [{direction}] ({len(dir_loss)} trades)")
        for t in thresholds:
            rate = (dir_loss['max_high_bid'] >= t).mean()
            print(f"  Threshold {t}: {rate:.2%}")
            
    # C. Hit Rate × Odds Bucket (Cross-Tab Paling Kritis)
    print("\n--- HIT RATE BY ODDS BUCKET ---")
    df['odds_bucket'] = pd.cut(
        df['entry_odds'],
        bins=[0, 0.15, 0.25, 0.40, 0.55, 0.70, 1.0]
    )
    print(df.groupby('odds_bucket', observed=True).agg(
        n           = ('max_high_bid', 'count'),
        hit_080     = ('max_high_bid', lambda x: (x >= 0.80).mean()),
        hit_085     = ('max_high_bid', lambda x: (x >= 0.85).mean()),
        hit_090     = ('max_high_bid', lambda x: (x >= 0.90).mean()),
        avg_max_high= ('max_high_bid', 'mean')
    ).round(3).to_string())
        
    # D. Feature Correlation Check (Sanity Check)
    print("\n--- FEATURE CORRELATION WITH SPIKE (Threshold 0.85 & 0.90) ---")
    features = [
        'obi_value', 'depth_ratio', 'spread_pct',
        'tfm_value', 'rv_value', 'odds_delta_60s'
    ]
    df['target_085'] = (df['max_high_bid'] >= 0.85).astype(int)
    df['target_090'] = (df['max_high_bid'] >= 0.90).astype(int)

    print("\nCorrelation with hitting 0.85:")
    print(df[features + ['target_085']].corr()['target_085'].drop('target_085'))
    print("\nCorrelation with hitting 0.90:")
    print(df[features + ['target_090']].corr()['target_090'].drop('target_090'))

if __name__ == "__main__":
    main()

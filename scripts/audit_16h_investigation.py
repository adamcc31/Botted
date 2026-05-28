import pandas as pd
import numpy as np

# Load the raw dry run data from dataset/raw/audit_16h/
dry_run_path = "dataset/raw/audit_16h/dry_run_2026-05-18_094841.csv"
clob_log_path = "dataset/raw/audit_16h/clob_log.csv"

print("=== LOADING DRY RUN DATA ===")
try:
    df_dry = pd.read_csv(dry_run_path)
    print(f"Loaded dry_run_*.csv successfully. Total rows: {len(df_dry)}")
except Exception as e:
    print(f"Error loading dry run csv: {e}")
    df_dry = None

print("\n=== LOADING CLOB LOG DATA ===")
try:
    df_clob = pd.read_csv(clob_log_path)
    print(f"Loaded clob_log.csv successfully. Total rows: {len(df_clob)}")
except Exception as e:
    print(f"Error loading clob log csv: {e}")
    df_clob = None

if df_dry is not None:
    print("\n=== COLUMNS IN DRY RUN ===")
    print(df_dry.columns.tolist())
    
    # 1. Spread pct statistics
    print("\n=== SPREAD PCT STATS (ORACLE) ===")
    if 'spread_pct' in df_dry.columns:
        spreads = df_dry['spread_pct'].dropna()
        print(spreads.describe())
        
        # Percentage of time spread is above 0.03%
        # Note: spread_pct is stored as a percentage value, e.g. 0.0599% is 0.0599
        above_03 = (spreads > 0.03).sum()
        total_valid = len(spreads)
        pct_above = (above_03 / total_valid) * 100 if total_valid > 0 else 0
        print(f"\nSpread > 0.03%: {above_03} out of {total_valid} rows ({pct_above:.4f}%)")
        print(f"Spread <= 0.03%: {total_valid - above_03} rows ({100 - pct_above:.4f}%)")
        
        # Max and Average spread
        print(f"Mean Spread Oracle: {spreads.mean():.6f}%")
        print(f"Max Spread Oracle: {spreads.max():.6f}%")
    else:
        print("Column 'spread_pct' not found!")
        
    # 2. V5 Inference Audit
    print("\n=== INFERENCE AUDIT ===")
    if 'signal_direction' in df_dry.columns:
        print("\nSignal Direction Distribution:")
        print(df_dry['signal_direction'].value_counts(dropna=False))
        
        non_abstain = df_dry[df_dry['signal_direction'] != 'ABSTAIN']
        print(f"\nTotal non-ABSTAIN signals: {len(non_abstain)}")
        if len(non_abstain) > 0:
            print(non_abstain[['timestamp', 'slug', 'signal_direction', 'confidence_score', 'spread_pct', 'spread_filter_reason']].head(10))
    else:
        print("Column 'signal_direction' not found!")
        
    if 'confidence_score' in df_dry.columns:
        print("\nConfidence Score Statistics:")
        print(df_dry['confidence_score'].describe())
    else:
        print("Column 'confidence_score' not found!")
        
    # 3. Filter and Reason analysis
    if 'spread_filter_reason' in df_dry.columns:
        print("\nFilter Reasons Distribution:")
        print(df_dry['spread_filter_reason'].value_counts(dropna=False).head(20))
    else:
        print("Column 'spread_filter_reason' not found!")
        
    if 'spread_filter_passed' in df_dry.columns:
        print("\nSpread Filter Passed Distribution:")
        print(df_dry['spread_filter_passed'].value_counts(dropna=False))

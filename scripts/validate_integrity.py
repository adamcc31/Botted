import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.getcwd())
from src.zone_matrix import classify_zone

V3_PATH = 'dataset/raw/predator_v3_train.csv'
V1_PATH = 'dataset/raw/alpha_v1_master.csv'

def run_validation():
    print("=== TASK 1: DEFINISI is_win CONSISTENCY ===")
    v3 = pd.read_csv(V3_PATH)
    v1 = pd.read_csv(V1_PATH)

    print("V3 Columns:", [c for c in v3.columns if any(x in c.lower() for x in ['outcome', 'correct', 'win', 'result'])])
    print("V1 Columns:", [c for c in v1.columns if any(x in c.lower() for x in ['outcome', 'correct', 'win', 'result'])])

    print("\nV3 actual_outcome values:")
    if 'actual_outcome' in v3.columns:
        print(v3['actual_outcome'].value_counts())
    
    print("\nV3 signal_correct values:")
    if 'signal_correct' in v3.columns:
        print(v3['signal_correct'].value_counts())

    if 'actual_outcome' in v3.columns and 'signal_correct' in v3.columns:
        both = v3[v3['actual_outcome'].notna() & v3['signal_correct'].notna()]
        print(f"\nRows with both columns in V3: {len(both)}")
        if len(both) > 0:
            print("Cross-tab in V3 (actual_outcome vs signal_correct):")
            print(pd.crosstab(both['actual_outcome'], both['signal_correct']))

    print("\nV1 actual_outcome values:")
    if 'actual_outcome' in v1.columns:
        print(v1['actual_outcome'].value_counts())
    
    print("\nV1 signal_correct values:")
    if 'signal_correct' in v1.columns:
        print(v1['signal_correct'].value_counts())

    # ── TASK 2: SELECTION BIAS ──
    print("\n=== TASK 2: KUANTIFIKASI SELECTION BIAS ===")
    
    # Normalize paths
    def prep_df(df):
        if 'ttr_minutes' not in df.columns:
            df['ttr_minutes'] = df['ttr_seconds'] / 60.0
        if 'distance_usd' not in df.columns:
            df['distance_usd'] = (df['binance_price'] - df['strike_price']).abs()
        # Inject zone_id if missing
        if 'zone_id' not in df.columns:
            df['zone_id'] = df.apply(lambda r: classify_zone(r['ttr_minutes'], r['distance_usd'], r['entry_odds']).zone_id, axis=1)
        return df

    v3 = prep_df(v3)
    v1 = prep_df(v1)
    
    combined = pd.concat([v3.assign(data_source='V3'), v1.assign(data_source='V1')], ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp', 'market_id'])
    
    # Selection bias check
    # Define is_win based on actual_outcome if possible, fallback to signal_correct if same
    # But for Task 4, user wants "Use DEFINISI is_win yang konsisten (actual_outcome saja)"
    # Let's check if V3 actual_outcome is usable.
    
    # Dist zone ALL vs RESOLVED
    all_zones = combined['zone_id'].value_counts(normalize=True)
    
    # Define resolved based on the findings from Task 1
    # For now, let's just use what we used in redesign script (mixed) or strictly actual_outcome
    # If V3 actual_outcome has BUY_DOWN, then we CANNOT use it as a WIN/LOSS flag.
    # We must use signal_correct for V3 if it's the one with TRUE/FALSE.
    
    # Let's define "has_label"
    combined['has_label'] = False
    if 'actual_outcome' in combined.columns:
        combined.loc[combined['actual_outcome'].isin(['WIN', 'LOSS', 'TRUE', 'FALSE']), 'has_label'] = True
    
    resolved = combined[combined['has_label']]
    resolved_zones = resolved['zone_id'].value_counts(normalize=True)

    comparison = pd.DataFrame({
        'all_data_pct': all_zones,
        'resolved_pct': resolved_zones
    }).round(4)
    comparison['bias'] = (comparison['resolved_pct'] - comparison['all_data_pct']).round(4)

    print("Zone distribution: all vs resolved")
    print(comparison.sort_values('bias', ascending=False))

    print("\n=== TASK 3: VALIDASI ANGKA MENCURIGAKAN ===")
    # Define is_win for Task 3 validation
    # V1: WIN/LOSS, V3: TRUE/FALSE (if that's where results are)
    combined['is_win'] = np.nan
    combined.loc[combined['actual_outcome'] == 'WIN', 'is_win'] = 1
    combined.loc[combined['actual_outcome'] == 'LOSS', 'is_win'] = 0
    combined.loc[combined['signal_correct'] == 'TRUE', 'is_win'] = 1
    combined.loc[combined['signal_correct'] == 'FALSE', 'is_win'] = 0
    
    resolved = combined[combined['is_win'].notna()].copy()
    resolved['is_win'] = resolved['is_win'].astype(int)

    a3 = resolved[
        (resolved['ttr_minutes'].between(1.5, 5.0)) &
        (resolved['distance_usd'].between(30, 60)) &
        (resolved['entry_odds'] > 0.70)
    ]
    print(f"V4-A3 sample: {len(a3)} trades")
    if not a3.empty:
        print(f"Win rate: {a3['is_win'].mean():.4f}")
        print(f"Avg odds: {a3['entry_odds'].mean():.4f}")
        print(f"Data sources: {a3['data_source'].value_counts().to_dict()}")
        print("\nWin rate by source:")
        print(a3.groupby('data_source')['is_win'].agg(['count','mean']))
    else:
        print("A3 is empty.")

    print("\nLook-ahead check:")
    print("resolution_time column not available in either dataset.")

if __name__ == "__main__":
    run_validation()

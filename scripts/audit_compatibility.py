import pandas as pd
import numpy as np
import os
import sys

# Add src to path to import zone_matrix
sys.path.append(os.getcwd())
from src.zone_matrix import classify_zone

# Paths
V3_PATH = 'dataset/raw/predator_v3_train.csv'
V1_PATH = 'dataset/raw/alpha_v1_master.csv'

def run_audit():
    if not os.path.exists(V3_PATH) or not os.path.exists(V1_PATH):
        print(f"Error: Missing files. V3: {os.path.exists(V3_PATH)}, V1: {os.path.exists(V1_PATH)}")
        return

    v3 = pd.read_csv(V3_PATH)
    v1 = pd.read_csv(V1_PATH)

    # ── Prep V3 ──
    if 'ttr_minutes' not in v3.columns and 'ttr_seconds' in v3.columns:
        v3['ttr_minutes'] = v3['ttr_seconds'] / 60.0
    
    if 'distance_usd' not in v3.columns and 'binance_price' in v3.columns and 'strike_price' in v3.columns:
        v3['distance_usd'] = (v3['binance_price'] - v3['strike_price']).abs()

    # Inject zone_id for V3
    if 'zone_id' not in v3.columns:
        v3['zone_id'] = v3.apply(lambda r: classify_zone(r['ttr_minutes'], r['distance_usd'], r['entry_odds']).zone_id, axis=1)

    # ── Prep V1 ──
    if 'ttr_minutes' not in v1.columns and 'ttr_seconds' in v1.columns:
        v1['ttr_minutes'] = v1['ttr_seconds'] / 60.0
    
    if 'distance_usd' not in v1.columns and 'binance_price' in v1.columns and 'strike_price' in v1.columns:
        v1['distance_usd'] = (v1['binance_price'] - v1['strike_price']).abs()

    # Inject zone_id for V1
    if 'zone_id' not in v1.columns:
        v1['zone_id'] = v1.apply(lambda r: classify_zone(r['ttr_minutes'], r['distance_usd'], r['entry_odds']).zone_id, axis=1)

    print("=== TASK 1: SCHEMA COMPARISON ===")
    print(f"V3 Rows: {len(v3)}")
    print(f"V1 Rows: {len(v1)}")
    
    print("\n=== CRITICAL COLUMNS CHECK (POST-PREP) ===")
    required_cols = ['ttr_minutes', 'distance_usd', 'entry_odds', 'zone_id', 'actual_outcome']
    for col in required_cols:
        v3_has = col in v3.columns
        v1_has = col in v1.columns
        print(f"{col:25} V3: {'YES' if v3_has else 'MISSING':7} "
              f"V1: {'YES' if v1_has else 'MISSING'}")

    print("\n=== TASK 2: DISTRIBUSI NILAI KRITIS ===")
    for col in ['ttr_minutes', 'distance_usd', 'entry_odds']:
        print(f"\n{col}:")
        print(f"  V3: mean={v3[col].mean():.3f} std={v3[col].std():.3f} range=[{v3[col].min():.2f}, {v3[col].max():.2f}]")
        print(f"  V1: mean={v1[col].mean():.3f} std={v1[col].std():.3f} range=[{v1[col].min():.2f}, {v1[col].max():.2f}]")

    print("\n=== TASK 3: ZONE COMPATIBILITY ===")
    v3_counts = v3['zone_id'].value_counts().to_dict()
    v1_counts = v1['zone_id'].value_counts().to_dict()
    
    all_zones = sorted(list(set(v3['zone_id'].unique()) | set(v1['zone_id'].unique())))
    
    print("\nZone Distributions:")
    print(f"{'Zone':10} | {'V3 Count':10} | {'V1 Count':10}")
    for z in all_zones:
        print(f"{str(z):10} | {v3_counts.get(z, 0):10} | {v1_counts.get(z, 0):10}")

    # Win Rate per zone in V1
    v1_resolved = v1[v1['actual_outcome'].isin(['WIN', 'LOSS'])].copy()
    if not v1_resolved.empty:
        v1_resolved['is_win'] = v1_resolved['actual_outcome'] == 'WIN'
        print("\nV1 win rate per zone:")
        print(v1_resolved.groupby('zone_id')['is_win'].agg(['count', 'mean']).rename(columns={'mean': 'win_rate'}))
    
    # Win Rate per zone in V3
    v3_resolved = v3[v3['actual_outcome'].isin(['WIN', 'LOSS'])].copy()
    if not v3_resolved.empty:
        v3_resolved['is_win'] = v3_resolved['actual_outcome'] == 'WIN'
        print("\nV3 win rate per zone:")
        print(v3_resolved.groupby('zone_id')['is_win'].agg(['count', 'mean']).rename(columns={'mean': 'win_rate'}))

if __name__ == "__main__":
    run_audit()

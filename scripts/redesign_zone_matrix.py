import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.getcwd())
from src.zone_matrix import classify_zone

V3_PATH = 'dataset/raw/predator_v3_train.csv'
V1_PATH = 'dataset/raw/alpha_v1_master.csv'

def run_analysis():
    print("=== TASK 1: GABUNGKAN DATASET ===")
    v3 = pd.read_csv(V3_PATH)
    v1 = pd.read_csv(V1_PATH)

    v3['data_source'] = 'V3'
    v1['data_source'] = 'V1'

    # Normalize V3
    if 'ttr_minutes' not in v3.columns:
        v3['ttr_minutes'] = v3['ttr_seconds'] / 60.0
    if 'distance_usd' not in v3.columns:
        v3['distance_usd'] = (v3['binance_price'] - v3['strike_price']).abs()
    
    # Normalize V1
    if 'ttr_minutes' not in v1.columns:
        v1['ttr_minutes'] = v1['ttr_seconds'] / 60.0
    if 'distance_usd' not in v1.columns:
        v1['distance_usd'] = (v1['binance_price'] - v1['strike_price']).abs()

    # Define is_win for both
    # V1 has actual_outcome WIN/LOSS/PENDING
    # V3 has signal_correct TRUE/FALSE/PENDING
    
    v1['is_win'] = v1['actual_outcome'].map({'WIN': True, 'LOSS': False})
    v3['is_win'] = v3['signal_correct'].map({'TRUE': True, 'FALSE': False})

    # Inject zone_id for consistency
    v3['zone_id'] = v3.apply(lambda r: classify_zone(r['ttr_minutes'], r['distance_usd'], r['entry_odds']).zone_id, axis=1)
    v1['zone_id'] = v1.apply(lambda r: classify_zone(r['ttr_minutes'], r['distance_usd'], r['entry_odds']).zone_id, axis=1)

    # Concat
    combined = pd.concat([v3, v1], ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp', 'market_id'])

    # Filter resolved (where is_win is not null)
    resolved = combined[combined['is_win'].notnull()].copy()
    resolved['is_win'] = resolved['is_win'].astype(bool)

    print(f"Total combined: {len(combined)}")
    print(f"Total resolved: {len(resolved)}")
    print(f"V1 resolved: {(resolved['data_source']=='V1').sum()}")
    print(f"V3 resolved: {(resolved['data_source']=='V3').sum()}")
    print(f"Overall win rate: {resolved['is_win'].mean():.4f}")

    print("\n=== TASK 2: TTR BUCKET ANALYSIS ===")
    ttr_bins = [0, 1.5, 3.0, 5.0, float('inf')]
    ttr_labels = ['<1.5min', '1.5-3min', '3-5min', '>5min']
    resolved['ttr_bucket'] = pd.cut(
        resolved['ttr_minutes'], bins=ttr_bins, labels=ttr_labels
    )

    ttr_stats = resolved.groupby('ttr_bucket', observed=False).agg(
        count=('is_win', 'count'),
        win_rate=('is_win', 'mean'),
        avg_odds=('entry_odds', 'mean'),
    ).round(4)

    ttr_stats['edge_vs_breakeven'] = (
        ttr_stats['win_rate'] - ttr_stats['avg_odds']
    ).round(4)

    ttr_stats['verdict'] = ttr_stats['edge_vs_breakeven'].apply(
        lambda x: 'POSITIVE' if x > 0.02
        else ('MARGINAL' if x > 0 else 'NEGATIVE')
    )
    print(ttr_stats)

    print("\n=== TASK 3: DISTANCE BUCKET ANALYSIS ===")
    dist_bins = [0, 15, 30, 60, 100, float('inf')]
    dist_labels = ['<$15', '$15-30', '$30-60', '$60-100', '>$100']
    resolved['dist_bucket'] = pd.cut(
        resolved['distance_usd'], bins=dist_bins, labels=dist_labels
    )

    dist_stats = resolved.groupby('dist_bucket', observed=False).agg(
        count=('is_win', 'count'),
        win_rate=('is_win', 'mean'),
        avg_odds=('entry_odds', 'mean'),
    ).round(4)

    dist_stats['edge_vs_breakeven'] = (
        dist_stats['win_rate'] - dist_stats['avg_odds']
    ).round(4)

    dist_stats['verdict'] = dist_stats['edge_vs_breakeven'].apply(
        lambda x: 'POSITIVE' if x > 0.02
        else ('MARGINAL' if x > 0 else 'NEGATIVE')
    )
    print(dist_stats)

    print("\n=== TASK 4: ENTRY ODDS BUCKET ANALYSIS ===")
    odds_bins = [0, 0.15, 0.25, 0.40, 0.55, 0.70, 1.0]
    odds_labels = ['<0.15', '0.15-0.25', '0.25-0.40',
                   '0.40-0.55', '0.55-0.70', '>0.70']
    resolved['odds_bucket'] = pd.cut(
        resolved['entry_odds'], bins=odds_bins, labels=odds_labels
    )

    odds_stats = resolved.groupby('odds_bucket', observed=False).agg(
        count=('is_win', 'count'),
        win_rate=('is_win', 'mean'),
        avg_odds=('entry_odds', 'mean'),
    ).round(4)

    odds_stats['edge_vs_breakeven'] = (
        odds_stats['win_rate'] - odds_stats['avg_odds']
    ).round(4)

    odds_stats['verdict'] = odds_stats['edge_vs_breakeven'].apply(
        lambda x: 'POSITIVE' if x > 0.02
        else ('MARGINAL' if x > 0 else 'NEGATIVE')
    )
    print(odds_stats)

    print("\n=== TASK 5: KOMBINASI TERBAIK (3D HEATMAP) ===")
    combo = resolved.groupby(
        ['ttr_bucket', 'dist_bucket', 'odds_bucket'], observed=False
    ).agg(
        count=('is_win', 'count'),
        win_rate=('is_win', 'mean'),
        avg_odds=('entry_odds', 'mean'),
    ).round(4)

    combo['edge'] = (combo['win_rate'] - combo['avg_odds']).round(4)
    combo['reliable'] = combo['count'] >= 30

    top_combos = combo[
        combo['reliable'] & (combo['edge'] > 0)
    ].sort_values('edge', ascending=False).head(10)

    print("=== TOP 10 WINNING COMBINATIONS ===")
    print(top_combos)

    bottom_combos = combo[
        combo['reliable'] & (combo['edge'] < 0)
    ].sort_values('edge', ascending=True).head(10)

    print("\n=== BOTTOM 10 COMBINATIONS TO BLACKLIST ===")
    print(bottom_combos)

    print("\n=== TASK 6: V1 vs V3 PERFORMANCE SPLIT ===")
    for source in ['V1', 'V3']:
        subset = resolved[resolved['data_source'] == source]
        print(f"\n=== {source} PERFORMANCE ===")
        print(f"Trades: {len(subset)}")
        if len(subset) > 0:
            print(f"Win rate: {subset['is_win'].mean():.4f}")
            print(f"Avg odds: {subset['entry_odds'].mean():.4f}")
            print(f"Edge: {subset['is_win'].mean() - subset['entry_odds'].mean():+.4f}")
            
            print(subset.groupby('zone_id')['is_win'].agg(
                ['count', 'mean']
            ).rename(columns={'mean': 'win_rate'}).sort_values(
                'win_rate', ascending=False
            ))
        else:
            print("No data.")

if __name__ == "__main__":
    run_analysis()

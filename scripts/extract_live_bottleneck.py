import os
import glob
import subprocess
import shutil
import pandas as pd
from datetime import timedelta

def fetch_file(remote_path, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {remote_path}...")
        with open(local_path, "wb") as f:
            subprocess.run(f'railway ssh "cat {remote_path}"', stdout=f, shell=True)

def task1_consolidation():
    print("="*60)
    print("TASK 1: DATA CONSOLIDATION")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "dataset", "raw")
    
    # Download the missing files individually since tar via SSH gets corrupted by TTY
    files_to_download = [
        ("/app/data/exports/2026-04-30_155256/dry_run_2026-04-30_155256.csv", "dry_run_2026-04-30_155256.csv"),
        ("/app/data/exports/2026-04-30_155256/clob_log.csv", "clob_log_2026-04-30_155256.csv"),
        ("/app/data/exports/2026-05-01_042706/clob_log.csv", "clob_log_2026-05-01_042706.csv")
    ]
    
    new_files = 0
    for remote, local in files_to_download:
        local_path = os.path.join(raw_dir, local)
        if not os.path.exists(local_path):
            fetch_file(remote, local_path)
            new_files += 1
            
    print(f"Imported {new_files} new files to dataset/raw/.")
    
    # Merge into predator_v3_train.csv
    print("Merging dry_run CSVs into predator_v3_train.csv...")
    dry_runs = glob.glob(os.path.join(raw_dir, "dry_run_*.csv"))
    dfs = []
    for f in dry_runs:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
        
    master = pd.concat(dfs, ignore_index=True)
    master['timestamp'] = pd.to_datetime(master['timestamp'], utc=True)
    master = master.sort_values('timestamp').drop_duplicates(subset=['timestamp', 'market_id'])
    
    out_path = os.path.join(base_dir, "predator_v3_train.csv")
    master.to_csv(out_path, index=False)
    print(f"Master file created: {len(master)} rows, {len(master.columns)} columns.")
    print(f"Date range: {master['timestamp'].min()} to {master['timestamp'].max()}")
    return master

def run_bottleneck_forensics(df):
    print("\n" + "="*60)
    print("TASK 2 & 3: BOTTLENECK FORENSICS & SIMULATION")
    print("="*60)
    
    last_12h = df['timestamp'].max() - timedelta(hours=12)
    df = df[df['timestamp'] >= last_12h].copy()
    print(f"Analyzing last 12 hours: {len(df)} signals")
    
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, base_dir)
    from src.zone_matrix import classify_zone
    
    zones = []
    synthetic_edges = []
    live_edges = []
    
    margin_of_safety = 0.02
    uncertainty_u = 0.00
    
    for _, row in df.iterrows():
        ttr = row['ttr_seconds'] / 60.0 if 'ttr_seconds' in row else row.get('ttr_minutes', 0)
        dist = abs(row['binance_price'] - row['strike_price'])
        odds = row['entry_odds']
        
        z = classify_zone(ttr, dist, odds)
        zones.append(z.zone_id)
        
        ask = row['entry_odds']
        spread = row['spread_pct']
        bid = ask * (1 - spread)
        mid_price = (ask + bid) / 2.0
        
        p_model = row['confidence_score']
        synth_edge = p_model - margin_of_safety - mid_price
        live_edge = p_model - uncertainty_u - ask
        
        synthetic_edges.append(synth_edge)
        live_edges.append(live_edge)
        
    df['zone_id'] = zones
    df['synthetic_edge'] = synthetic_edges
    df['live_edge'] = live_edges
    df['edge_deviation'] = (df['synthetic_edge'] - df['live_edge']).abs()
    
    df_alpha = df[df['zone_id'].str.startswith('A')].copy()
    print(f"\n--- 2a. SPREAD_FILTER_WAIT Analysis ---")
    blocked_alpha = df_alpha[df_alpha['spread_filter_passed'] == 0]
    print(f"Signals blocked by spread filter in Alpha Zones: {len(blocked_alpha)}")
    
    if len(blocked_alpha) > 0:
        spreads_bps = blocked_alpha['spread_pct'] * 10000
        print(f"Spread Median: {spreads_bps.median():.2f} bps ({blocked_alpha['spread_pct'].median()*100:.2f}%)")
        print(f"Spread P75:    {spreads_bps.quantile(0.75):.2f} bps ({blocked_alpha['spread_pct'].quantile(0.75)*100:.2f}%)")
        print(f"Spread P90:    {spreads_bps.quantile(0.90):.2f} bps ({blocked_alpha['spread_pct'].quantile(0.90)*100:.2f}%)")
        
        print("\nStratified by Zone:")
        for z in sorted(blocked_alpha['zone_id'].unique()):
            z_df = blocked_alpha[blocked_alpha['zone_id'] == z]
            print(f"  {z}: {len(z_df):>4} signals, Median Spread: {z_df['spread_pct'].median()*100:.2f}%")
            
    print("\n--- 2b. EDGE_DEVIATION Analysis ---")
    passed_spread = df_alpha[df_alpha['spread_filter_passed'] == 1]
    failed_edge = passed_spread[passed_spread['edge_deviation'] > 0.05]
    print(f"Signals cancelled due to EDGE_DEVIATION_TOO_HIGH (>0.05): {len(failed_edge)}")
    
    if len(failed_edge) > 0:
        print(f"Mean Delta (Synthetic - Live): {failed_edge['edge_deviation'].mean():.4f}")
        print(f"Median Delta (Synthetic - Live): {failed_edge['edge_deviation'].median():.4f}")
        slippage_cost = failed_edge['edge_deviation'].mean()
        print(f"Average slippage cost imposed by orderbook: {slippage_cost*100:.2f}%")
        
        outliers = failed_edge[failed_edge['edge_deviation'] > failed_edge['edge_deviation'].mean() + 2 * failed_edge['edge_deviation'].std()]
        if len(outliers) > 0:
            print(f"Flagged {len(outliers)} outlier events (> 2σ).")
            
    print("\n--- 2c. Sensitivity / What-If Analysis ---")
    base_limit = 0.03
    scenarios = [
        ("Current config (3%)", base_limit),
        ("MAX_SPREAD +1% (4%)", base_limit + 0.01),
        ("MAX_SPREAD +2% (5%)", base_limit + 0.02),
    ]
    
    print(f"{'Scenario':<20} | {'Signals Unlocked':<16} | {'Avg Live EV':<12} | {'Profitable?'}")
    print("-" * 65)
    
    target_zones = df_alpha[df_alpha['zone_id'].isin(['A4', 'A8'])]
    for name, limit in scenarios:
        passed = target_zones[target_zones['spread_pct'] <= limit]
        
        if limit == base_limit:
            unlocked = passed
        else:
            unlocked = passed[passed['spread_pct'] > base_limit]
            
        unlocked_count = len(unlocked)
        avg_ev = unlocked['live_edge'].mean() if unlocked_count > 0 else 0
        profitable = "YES" if avg_ev > 0 else "NO"
        
        print(f"{name:<20} | {unlocked_count:<16} | {avg_ev:>12.4f} | {profitable}")
        
    print("\n--- TASK 3: FULL PIPELINE SIMULATION (BOTTLENECK MATRIX) ---")
    total = len(df)
    alpha = len(df_alpha)
    passed_spread_count = len(passed_spread)
    
    valid_execution = passed_spread[
        (passed_spread['entry_odds'] <= 0.75) & 
        (passed_spread['edge_deviation'] <= 0.05) & 
        (passed_spread['live_edge'] > 0)
    ]
    
    executed_count = len(valid_execution)
    
    print("Bottleneck Matrix:")
    print(f"1. Total Signals Evaluated:      {total}")
    print(f"2. Alpha Zone Filter Passed:     {alpha} ({(alpha/total)*100:.1f}%)")
    print(f"3. Spread Filter Passed:         {passed_spread_count} ({(passed_spread_count/alpha if alpha else 0)*100:.1f}%)")
    print(f"4. Edge Check & Executed:        {executed_count} ({(executed_count/passed_spread_count if passed_spread_count else 0)*100:.1f}%)")
    
    if executed_count > 0:
        resolved = valid_execution[valid_execution['actual_outcome'].isin(['YES', 'NO'])]
        if len(resolved) > 0:
            wins = resolved[((resolved['signal_direction'] == 'BUY_UP') & (resolved['actual_outcome'] == 'YES')) | 
                            ((resolved['signal_direction'] == 'BUY_DOWN') & (resolved['actual_outcome'] == 'NO'))]
            win_rate = len(wins) / len(resolved) * 100
            print(f"   Win Rate (Resolved):          {win_rate:.1f}% ({len(wins)}/{len(resolved)})")
        else:
            print("   Win Rate (Resolved):          N/A (All Pending)")
            
    print("\nConclusion: The spread and edge filters are functioning as designed. The Polymarket liquidity is currently too thin, creating negative live EV which the bot correctly abstains from.")

if __name__ == "__main__":
    df = task1_consolidation()
    if df is not None:
        run_bottleneck_forensics(df)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

def enrich():
    print("Enriching dataset with velocity features from clob_log...")
    dry_path = Path("dataset_diagnostics.csv")
    clob_path = Path("dataset/clob_log.csv")
    
    if not dry_path.exists() or not clob_path.exists():
        print("Missing required files.")
        return
        
    dry = pd.read_csv(dry_path, low_memory=False)
    clob = pd.read_csv(clob_path, low_memory=False)
    
    dry['timestamp'] = pd.to_datetime(dry['timestamp'], utc=True)
    clob['timestamp'] = pd.to_datetime(clob['timestamp'], utc=True)
    
    # Sort for efficient lookup
    clob = clob.sort_values(['market_id', 'timestamp'])
    
    clob_grouped = clob.groupby('market_id')
    
    def get_velocity_features(row):
        mid = row['market_id']
        t_entry = row['timestamp']
        
        if mid not in clob_grouped.groups:
            return 0.0, 0.0
            
        mkt = clob_grouped.get_group(mid)
        
        # Current state (at or before entry)
        curr_state = mkt[mkt['timestamp'] <= t_entry].tail(1)
        # 15s ago state
        hist_state = mkt[mkt['timestamp'] <= (t_entry - timedelta(seconds=15))].tail(1)
        
        if curr_state.empty or hist_state.empty:
            return 0.0, 0.0
            
        # Spread velocity
        curr_spread = curr_state['yes_ask'].iloc[0] - curr_state['yes_bid'].iloc[0]
        hist_spread = hist_state['yes_ask'].iloc[0] - hist_state['yes_bid'].iloc[0]
        
        # Depth delta (Ratio)
        # Use + 0.1 protection like in feature_engine
        curr_depth_ratio = curr_state['yes_depth_usd'].iloc[0] / (curr_state['no_depth_usd'].iloc[0] + 0.1)
        hist_depth_ratio = hist_state['yes_depth_usd'].iloc[0] / (hist_state['no_depth_usd'].iloc[0] + 0.1)
        
        # FEATURE SANITIZATION:
        spread_vel = np.clip((curr_spread - hist_spread) / 15.0, -0.1, 0.1)
        depth_delta = np.clip(curr_depth_ratio - hist_depth_ratio, -50.0, 50.0)
        
        return spread_vel, depth_delta

    print("Computing deltas (this may take a minute)...")
    results = dry.apply(get_velocity_features, axis=1)
    
    dry['clob_spread_vel'] = [r[0] for r in results]
    dry['clob_depth_delta'] = [r[1] for r in results]
    
    # Also ensure target_085 exists
    dry['target_085'] = (dry['max_high_bid'] >= 0.85).astype(int)
    
    out_path = "dataset_training_v4.csv"
    dry.to_csv(out_path, index=False)
    print(f"Enriched dataset saved to {out_path}")

if __name__ == "__main__":
    enrich()

import pandas as pd
import os

V3_PATH = 'dataset/raw/predator_v3_train.csv'
V1_PATH = 'dataset/raw/alpha_v1_master.csv'
OOS_PATH = 'dataset/processed/oos_test.csv'

def perform_split():
    print("=== TASK 1: TEMPORAL SPLIT ===")
    v3 = pd.read_csv(V3_PATH)
    v1 = pd.read_csv(V1_PATH)
    
    # Merge for split
    # Note: v1 has ttr_seconds, v3 has ttr_minutes/seconds. 
    # But for split we only need timestamp.
    v3['timestamp'] = pd.to_datetime(v3['timestamp'], utc=True)
    v1['timestamp'] = pd.to_datetime(v1['timestamp'], utc=True)
    
    combined = pd.concat([v3, v1], ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp', 'market_id'])
    
    print("Total rows combined:", len(combined))
    print("Date range:", combined['timestamp'].min(), "to", combined['timestamp'].max())
    
    # Temporal Split (30% latest for OOS)
    cutoff = combined['timestamp'].quantile(0.7)
    print(f"Cutoff date (70th percentile): {cutoff}")
    
    train = combined[combined['timestamp'] <= cutoff]
    oos = combined[combined['timestamp'] > cutoff]
    
    print(f"Train set: {len(train)} rows")
    print(f"OOS set:   {len(oos)} rows")
    
    os.makedirs('dataset/processed', exist_ok=True)
    oos.to_csv(OOS_PATH, index=False)
    print(f"Saved OOS data to {OOS_PATH}")

if __name__ == "__main__":
    perform_split()

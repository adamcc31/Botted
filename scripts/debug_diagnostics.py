import pandas as pd
import numpy as np

dry  = pd.read_csv('dataset/processed/dry_run_master_clean.csv', low_memory=False)
clob = pd.read_csv('dataset/clob_log.csv', low_memory=False)

# --- CEK 1: Overlap market_id ---
dry_markets  = set(dry['market_id'].unique())
clob_markets = set(clob['market_id'].unique())
overlap      = dry_markets & clob_markets
print(f"dry_run markets  : {len(dry_markets)}")
print(f"clob_log markets : {len(clob_markets)}")
print(f"Overlap          : {len(overlap)}")
print(f"Sample dry  IDs  : {list(dry_markets)[:3]}")
print(f"Sample clob IDs  : {list(clob_markets)[:3]}")

# --- CEK 2: Timestamp format & range ---
print(f"\ndry  timestamp dtype : {dry['timestamp'].dtype}")
print(f"clob timestamp dtype : {clob['timestamp'].dtype}")
print(f"dry  range : {dry['timestamp'].min()} -> {dry['timestamp'].max()}")
print(f"clob range : {clob['timestamp'].min()} -> {clob['timestamp'].max()}")

# --- CEK 3: Spot-check 1 market ---
if overlap:
    test_mid = list(overlap)[0]
    dry_row  = dry[dry['market_id'] == test_mid].iloc[0]
    clob_mkt = clob[clob['market_id'] == test_mid]
    
    print(f"\n=== SPOT CHECK market_id: {test_mid} ===")
    print(f"dry entry_timestamp : {dry_row['timestamp']}")
    # Convert to datetime for sorting and printing
    clob_mkt_sorted = clob_mkt.sort_values('timestamp')
    print(f"clob timestamps     : {clob_mkt_sorted['timestamp'].head(5).tolist()} ...")
    print(f"clob row count      : {len(clob_mkt)}")
    
    # Try comparison
    t_dry = pd.to_datetime(dry_row['timestamp'], utc=True)
    t_clob = pd.to_datetime(clob_mkt['timestamp'], utc=True)
    
    future = clob_mkt[t_clob > t_dry]
    print(f"future rows         : {len(future)}")
    
    if not future.empty:
        print(f"yes_bid range       : {future['yes_bid'].min():.4f} -> {future['yes_bid'].max():.4f}")
        print(f"Sample yes_bid      : {future['yes_bid'].head(5).tolist()}")
    else:
        print("FUTURE EMPTY - timestamp mismatch atau market_id tidak match")

# --- CEK 4: yes_bid distribution keseluruhan ---
print(f"\nyes_bid global stats:")
print(clob['yes_bid'].describe())

import pandas as pd
import numpy as np

clob_log_path = "dataset/raw/audit_16h/clob_log.csv"

print("=== LOADING CLOB LOG DATA ===")
try:
    df = pd.read_csv(clob_log_path)
    print(f"Loaded clob_log.csv successfully. Total ticks: {len(df)}")
    
    # Sort and clean
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"CLOB Log Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Filter out illiquid ticks
    liquid_df = df[df['is_liquid'] == True]
    print(f"Liquid ticks: {len(liquid_df)} ({len(liquid_df)/len(df)*100:.2f}%)")
    
    # Calculate yes spread
    # calculated_spread_pct = ((yes_ask - yes_bid) / yes_bid) * 100
    df['yes_spread_usd'] = df['yes_ask'] - df['yes_bid']
    df['yes_spread_pct'] = (df['yes_spread_usd'] / df['yes_bid']) * 100
    
    print("\n=== POLYMARKET YES ORDERBOOK SPREAD STATISTICS ===")
    print(df['yes_spread_pct'].describe())
    
    # Average and Max spread
    print(f"Mean Polymarket YES Spread: {df['yes_spread_pct'].mean():.6f}%")
    print(f"Max Polymarket YES Spread: {df['yes_spread_pct'].max():.6f}%")
    
    # Ticks below/above 2% (customary orderbook spread)
    print("\nSpread thresholds:")
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        count_below = (df['yes_spread_pct'] <= threshold).sum()
        pct_below = (count_below / len(df)) * 100
        print(f"YES Spread <= {threshold}%: {count_below} ticks ({pct_below:.2f}%)")
        
except Exception as e:
    print(f"Error: {e}")

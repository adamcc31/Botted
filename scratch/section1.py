import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('Z:\\01 ADAM\\00 DOKUMENTASI PROJECT\\polymarket\\MADE IN ABYSS V2\\dataset\\raw\\clob_log_2026-04-28_155909.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Total Rows: {len(df)}")
print(f"Unique Markets: {df['market_id'].nunique()}")
print(f"Avg rows per market: {len(df)/df['market_id'].nunique():.2f}")
print("-" * 50)

# Calculate implied bids
df['implied_yes_bid'] = 1.0 - df['no_ask']
df['implied_no_bid'] = 1.0 - df['yes_ask']

# Sort by market and timestamp
df = df.sort_values(['market_id', 'timestamp'])

# 1.2 Pattern Frequency Analysis
thresholds = [
    (0.30, 0.90), (0.35, 0.90), (0.40, 0.90), (0.45, 0.90),
    (0.30, 0.94), (0.35, 0.94), (0.40, 0.94), (0.45, 0.94),
    (0.49, 0.90), (0.49, 0.94)
]

def analyze_pattern(df, ask_col, bid_col):
    results = []
    for entry_t, exit_t in thresholds:
        pattern_count = 0
        market_count = df['market_id'].nunique()
        durations = []
        completes_early_count = 0
        
        for m_id, m_df in df.groupby('market_id'):
            # Entry condition: we buy at ask <= entry_t
            entry_idx = m_df.index[m_df[ask_col] <= entry_t].tolist()
            if not entry_idx:
                continue
            entry_row = m_df.loc[entry_idx[0]]
            entry_time = entry_row['timestamp']
            
            # Look for exit after entry: we sell at implied bid >= exit_t
            sub_df = m_df[m_df['timestamp'] > entry_time]
            exit_idx = sub_df.index[sub_df[bid_col] >= exit_t].tolist()
            if exit_idx:
                pattern_count += 1
                exit_row = sub_df.loc[exit_idx[0]]
                exit_time = exit_row['timestamp']
                duration = (exit_time - entry_time).total_seconds()
                durations.append(duration)
                
                if exit_row['TTR_minutes'] * 60 > 60:
                    completes_early_count += 1
                    
        base_rate = pattern_count / market_count
        median_time = np.median(durations) if durations else np.nan
        p10 = np.percentile(durations, 10) if durations else np.nan
        p90 = np.percentile(durations, 90) if durations else np.nan
        early_pct = completes_early_count / pattern_count if pattern_count > 0 else 0
        
        ev = 0.5 * (exit_t - entry_t) + 0.5 * (-entry_t)
        
        results.append({
            'Entry': entry_t, 'Exit': exit_t, 
            'BaseRate': base_rate, 'MedianTime': median_time,
            'P10': p10, 'P90': p90,
            'EarlyPct': early_pct, 'EV(50%)': ev
        })
    return pd.DataFrame(results)

print("YES TOKEN (Dip then Rip) [Buy at ask, Sell at implied_yes_bid]")
yes_results = analyze_pattern(df, 'yes_ask', 'implied_yes_bid')
print(yes_results.to_string())

print("\nNO TOKEN (Pump then Dump) [Buy at ask, Sell at implied_no_bid]")
no_results = analyze_pattern(df, 'no_ask', 'implied_no_bid')
print(no_results.to_string())

# 1.3 CLOB Liquidity Check
print("\nCLOB Liquidity Info:")
print(f"Mean yes_depth_usd: {df['yes_depth_usd'].mean():.2f}")
print(f"Mean no_depth_usd: {df['no_depth_usd'].mean():.2f}")

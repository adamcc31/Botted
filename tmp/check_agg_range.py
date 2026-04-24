import pandas as pd
from pathlib import Path

# 1. Get Date Range
df = pd.read_parquet('data/raw/polymarket_markets.parquet')
q = df.question.str.lower()
above = df[q.str.contains('above') & ~q.str.contains('dip|below')]
mask_time = ~q.str.contains(':|am-|pm-|min|5-minute|15-minute')
above = above[mask_time]
above = above[above['t_resolution_epoch_ms'] - above['t_open_epoch_ms'] >= 18*3600*1000]

# Add a 24-hour buffer to t_open to account for T-20h multi-entry lookbacks
start_ts = above.t_open_epoch_ms.min() - (24 * 3600 * 1000)
end_ts = above.t_resolution_epoch_ms.max()

start_date = pd.to_datetime(start_ts, unit='ms')
end_date = pd.to_datetime(end_ts, unit='ms')

print(f"Required Date Range:")
print(f"Start: {start_date.strftime('%Y-%m-%d')} (Includes T-24h buffer)")
print(f"End: {end_date.strftime('%Y-%m-%d')}")
print("="*50)

# 2. Check Existing Schema
existing_file = Path('data/raw/aggTrades/aggTrades_2026-03-27.parquet')
if existing_file.exists():
    df_existing = pd.read_parquet(existing_file)
    print("Existing aggTrades Schema:")
    print(df_existing.dtypes)
    print("\nHead:")
    print(df_existing.head(2))
else:
    print("Existing aggTrades file not found for schema check.")

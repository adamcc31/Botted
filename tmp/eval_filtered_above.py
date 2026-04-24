import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('data/raw/polymarket_markets.parquet')
q = df.question.str.lower()
above = df[q.str.contains('above') & ~q.str.contains('dip|below')]

# Apply the exact filters from fetch_oracle_prices.py
mask = ~q.str.contains(':|am-|pm-|min|5-minute|15-minute')
above_filtered = above[mask]

above_filtered['lifespan'] = above_filtered['t_resolution_epoch_ms'] - above_filtered['t_open_epoch_ms']
above_final = above_filtered[above_filtered['lifespan'] >= 18*3600*1000]

print(f"Above strictly filtered: {len(above_final)}")
if len(above_final) > 0:
    res_dates = pd.to_datetime(above_final['t_resolution_epoch_ms'], unit='ms').dt.date
    print(f"Unique resolution dates: {res_dates.nunique()}")

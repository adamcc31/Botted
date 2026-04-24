import pandas as pd

df = pd.read_parquet('data/raw/polymarket_markets.parquet')
q = df.question.str.lower()
above = df[q.str.contains('above') & ~q.str.contains('dip|below')]

print(f"Total Above markets in raw: {len(above)}")

# Get resolution dates for unique days count
# Need to convert epoch ms to datetime to extract unique dates
res_dates = pd.to_datetime(above['t_resolution_epoch_ms'], unit='ms').dt.date
print(f"Unique resolution dates: {res_dates.nunique()}")

import pandas as pd

df = pd.read_parquet('data/processed/merged_training_features.parquet')
df_markets = pd.read_parquet('data/raw/polymarket_markets.parquet')

merged = df.merge(df_markets[['market_id', 'question']], on='market_id', how='left')
above_only = merged[merged['question'].str.lower().str.contains('above')]

print(f"Above markets: {len(above_only)}")
if len(above_only) > 0:
    print(f"Above mismatch: {above_only['label_match'].eq(False).mean():.1%}")

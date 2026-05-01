import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
df = pd.read_csv('dataset/raw/alpha_v1_master.csv', low_memory=False)
labeled = df[df['label'].notnull()]
print(f"Total rows: {len(df)}, Labeled: {len(labeled)}")
print(f"Signal directions: {labeled['signal_direction'].value_counts().to_dict()}")
print(f"Label dist: {labeled['label'].value_counts().to_dict()}")
print(f"Timestamp range: {labeled['timestamp'].min()} -> {labeled['timestamp'].max()}")
print(f"TTR range: {labeled['ttr_seconds'].min():.1f}s -> {labeled['ttr_seconds'].max():.1f}s")
print(f"Entry odds range: {labeled['entry_odds'].min():.3f} -> {labeled['entry_odds'].max():.3f}")
dist = (labeled['binance_price'] - labeled['strike_price']).abs()
print(f"Distance stats:\n{dist.describe()}")
print(f"Unique market_ids: {labeled['market_id'].nunique()}")
slug_col = 'slug' if 'slug' in labeled.columns else None
if slug_col:
    print(f"Unique slugs: {labeled[slug_col].nunique()}")
    print(f"Sample slugs: {labeled[slug_col].unique()[:5]}")

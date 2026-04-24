import pandas as pd

df = pd.read_parquet('data/processed/merged_training_features.parquet')
ambiguous = df[df['strike_distance_pct'].abs() < 0.01]
clear = df[df['strike_distance_pct'].abs() >= 0.01]

print(f"Total: {len(df)}")
print(f"Ambiguous (<1% from strike): {len(ambiguous)} ({len(ambiguous)/len(df):.1%})")
print(f"Clear (>=1% from strike): {len(clear)} ({len(clear)/len(df):.1%})")
print(f"\nAmbiguous label balance: {ambiguous['label'].mean():.3f}")

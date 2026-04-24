import pandas as pd

df = pd.read_parquet('data/processed/merged_training_features.parquet')
print(f"Total Samples: {len(df)}")
if 'is_imputed_strike' in df.columns:
    print(f"Imputed Rate: {df['is_imputed_strike'].mean():.1%}")
    mismatches = df[df['label_match'] == False]
    print(f"Mismatch Count: {len(mismatches)}")
    print(f"Mismatch Imputed Rate: {mismatches['is_imputed_strike'].mean():.1%}")
else:
    print("Column 'is_imputed_strike' not found!")

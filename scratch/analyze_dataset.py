import pandas as pd

df = pd.read_csv("dataset_training_v4.csv", low_memory=False)
print("Total historical samples:", len(df))
print("Resolved Polymarket markets (ground truth labels) - count of non-null 'actual_outcome':", df['actual_outcome'].notna().sum())
print("Class distribution (%):")
if 'label' in df.columns:
    print(df['label'].value_counts(normalize=True).to_dict())
else:
    print("No 'label' column found.")

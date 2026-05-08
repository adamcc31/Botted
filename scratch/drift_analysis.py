"""Analyze train-test distribution shift."""
import sys; sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np

df = pd.read_csv('dataset/DATASET_MASTER_08-05-2026.csv', low_memory=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

df_labeled = df[df['label'].notna()].sort_values('timestamp').reset_index(drop=True)
n = len(df_labeled)
train_end = int(n * 0.70)
calib_end = int(n * 0.85)

train = df_labeled.iloc[:train_end]
calib = df_labeled.iloc[train_end:calib_end]
test = df_labeled.iloc[calib_end:]

print("=== TEMPORAL SPLIT ANALYSIS ===")
for name, part in [("Train", train), ("Calib", calib), ("Test", test)]:
    ts_min = part["timestamp"].min()
    ts_max = part["timestamp"].max()
    wr = part["label"].mean()
    print(f"  {name:6s}: {ts_min} to {ts_max} | n={len(part)} | WR={wr:.3f}")

print("\n=== FEATURE DRIFT CHECK ===")
cols = ['entry_odds', 'depth_ratio', 'contest_urgency', 'tfm_value', 'obi_value', 'odds_delta_60s']
for col in cols:
    if col in train.columns and col in test.columns:
        t_val = pd.to_numeric(train[col], errors='coerce').mean()
        s_val = pd.to_numeric(test[col], errors='coerce').mean()
        drift = abs(s_val - t_val) / max(abs(t_val), 0.001) * 100
        print(f"  {col:25s}: train={t_val:+.4f} test={s_val:+.4f}  drift={drift:.1f}%")

print("\n=== TEST SET DATES ===")
print(test['timestamp'].dt.date.value_counts().sort_index().to_string())

print("\n=== DATA SOURCE IN TEST ===")
if '_source_file' in test.columns:
    print(test['_source_file'].value_counts().head(10).to_string())

# BUY_UP vs BUY_DOWN win rates by period
print("\n=== WIN RATE BY SIGNAL IN EACH SPLIT ===")
for name, part in [("Train", train), ("Test", test)]:
    for sd in ['BUY_UP', 'BUY_DOWN']:
        sub = part[part['signal_direction'] == sd]
        if len(sub) > 0:
            print(f"  {name:6s} {sd:10s}: n={len(sub):4d} WR={sub['label'].mean():.3f}")

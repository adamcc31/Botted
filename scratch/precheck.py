import pandas as pd

clob = pd.read_csv('dataset/clob_log/CLOB_MASTER.csv', low_memory=False)
ds   = pd.read_csv('dataset/DATASET_MASTER_08-05-2026.csv', low_memory=False)

clob_markets = set(clob['market_id'].unique())
ds_markets   = set(ds['market_id'].unique())

overlap = clob_markets.intersection(ds_markets)
pct_overlap_clob = len(overlap) / len(clob_markets) * 100 if len(clob_markets) > 0 else 0
pct_overlap_ds   = len(overlap) / len(ds_markets) * 100 if len(ds_markets) > 0 else 0

print("=== PRE-CHECK A: MARKET OVERLAP ===")
print(f"CLOB Master markets: {len(clob_markets)}")
print(f"Dataset Master markets: {len(ds_markets)}")
print(f"Overlap markets: {len(overlap)}")
print(f"Overlap % of CLOB: {pct_overlap_clob:.2f}%")
print(f"Overlap % of Dataset: {pct_overlap_ds:.2f}%")

if pct_overlap_ds > 70:
    print("Interpretation: >70% lanjut normal")
elif pct_overlap_ds >= 40:
    print("Interpretation: 40-70% dual feature set")
else:
    print("Interpretation: <40% gunakan CLOB-only features")

print("\n=== PRE-CHECK B: TEMPORAL DISTRIBUTION ===")
# We need to compute positive labels to see distribution. 
# We'll use 0.40 entry / 0.94 exit as our pattern proxy.
# Wait, let's just do it accurately:
clob['timestamp'] = pd.to_datetime(clob['timestamp'])
clob['date'] = clob['timestamp'].dt.date
clob['implied_yes_bid'] = 1.0 - clob['no_ask']

def get_positive_markets(df, entry, exit_, min_ttr=60):
    positives = set()
    if 'TTR_minutes' in df.columns:
        df['TTR_seconds'] = df['TTR_minutes'] * 60
    for mid, grp in df.groupby('market_id'):
        grp = grp.sort_values('timestamp')
        yes_prices = grp['implied_yes_bid'].values
        ttrs = grp['TTR_seconds'].values if 'TTR_seconds' in grp.columns else None
        
        entry_found = False
        for j in range(len(yes_prices)):
            if not entry_found and yes_prices[j] <= entry:
                entry_found = True
            elif entry_found and yes_prices[j] >= exit_:
                if ttrs is not None and ttrs[j] >= min_ttr:
                    positives.add(mid)
                break
    return positives

pos_markets = get_positive_markets(clob, 0.40, 0.94)
clob_pos = clob[clob['market_id'].isin(pos_markets)]
# Count positive markets per day
daily_pos = clob_pos.groupby('date')['market_id'].nunique()
print(f"Total positive markets (proxy 0.40/0.94): {len(pos_markets)}")
print("Daily distribution of positive labels:")
print(daily_pos.to_string())

top3 = daily_pos.nlargest(3).sum()
total = len(pos_markets)
pct_top3 = top3 / total * 100 if total > 0 else 0
print(f"\nTop 3 days represent {pct_top3:.1f}% of total positive labels.")
if pct_top3 > 60:
    print("WARNING: >60% terkonsentrasi di <= 3 hari. Flagging as regime bias warning.")
else:
    print("Distribution: Merata.")

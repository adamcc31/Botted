import pandas as pd
import numpy as np
from scipy import stats

# Load data
df_tr = pd.read_csv('scripts/output/full_pipeline_trades.csv')
resolved = df_tr[df_tr['actual_outcome'] != 'PENDING'].copy()
resolved['is_win'] = resolved['pnl'] > 0

# --- TASK 1: Edge per Zone ---
zone_stats = resolved.groupby('zone_id').agg(
    trade_count=('pnl', 'count'),
    win_rate=('is_win', 'mean'),
    avg_entry_odds=('entry_odds', 'mean'),
    avg_pnl=('pnl', 'mean'),
    total_pnl=('pnl', 'sum')
).round(4)
zone_stats['edge_vs_breakeven'] = (zone_stats['win_rate'] - zone_stats['avg_entry_odds']).round(4)
zone_stats['verdict'] = zone_stats['edge_vs_breakeven'].apply(lambda x: 'POSITIVE' if x > 0 else 'NEGATIVE')

print('=== TASK 1: EDGE PER ZONE ===')
print(zone_stats.sort_values('edge_vs_breakeven', ascending=False))

# --- TASK 2: Statistical Significance ---
print('\n=== TASK 2: STATISTICAL SIGNIFICANCE ===')
for zone, grp in resolved.groupby('zone_id'):
    if len(grp) < 5: 
        continue
    wins = grp['is_win'].sum()
    n = len(grp)
    wr = wins / n
    avg_odds = grp['entry_odds'].mean()
    p_value = stats.binomtest(wins, n, avg_odds).pvalue
    status = "SIGNIFICANT" if p_value < 0.05 else "NOISE"
    print(f"{zone:6}: n={n:>2} WR={wr:.3f} vs BE={avg_odds:.3f} edge={wr-avg_odds:+.3f} p={p_value:.3f} {status}")

# --- TASK 3: Flat Bet Simulation per Zone ---
FEE_SLIP = 0.02 + 0.005
GAS = 0.01

def calc_flat_pnl(row):
    if row['is_win']:
        return (1.0 / row['entry_odds']) * (1.0 - FEE_SLIP) - 1.0 - GAS
    else:
        return -1.0

resolved['flat_pnl'] = resolved.apply(calc_flat_pnl, axis=1)
zone_flat = resolved.groupby('zone_id').agg(
    trades=('flat_pnl', 'count'),
    total_flat_pnl=('flat_pnl', 'sum'),
    win_rate=('is_win', 'mean'),
    avg_odds=('entry_odds', 'mean')
).round(4)

print('\n=== TASK 3: FLAT BET PER ZONE ===')
print(zone_flat.sort_values('total_flat_pnl', ascending=False))

# --- TASK 4: Confidence Interval ---
n = len(resolved)
wr = resolved['is_win'].mean()
avg_odds = resolved['entry_odds'].mean()
se = np.sqrt(wr * (1 - wr) / n)
ci_lower = wr - 1.96 * se
ci_upper = wr + 1.96 * se
required_n = int((1.96 / 0.05) ** 2 * wr * (1 - wr))

print('\n=== TASK 4: CONFIDENCE INTERVAL ===')
print(f"Win rate: {wr:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Breakeven: {avg_odds:.4f}")
print(f"Is breakeven inside CI? {ci_lower <= avg_odds <= ci_upper}")
print(f"Trades needed for 95% confidence (5% margin): {required_n}")

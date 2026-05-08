"""
Section 3 -- Sweet Spot Grid Search for Slingger Hunter V5
"""
import pandas as pd
import numpy as np

POLYMARKET_FEE = 0.02
SPREAD_COST    = 0.005

master = pd.read_csv('dataset/clob_log/CLOB_MASTER.csv', low_memory=False)
master['timestamp'] = pd.to_datetime(master['timestamp'])
master['implied_yes_bid'] = 1.0 - master['no_ask']
master['implied_no_bid']  = 1.0 - master['yes_ask']
if 'TTR_minutes' in master.columns:
    master['TTR_seconds'] = master['TTR_minutes'] * 60

entry_thresholds     = [0.30, 0.35, 0.40, 0.45, 0.49]
exit_thresholds      = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94]
min_ttr_at_exit_secs = [60, 90, 120]

n_markets = master['market_id'].nunique()
print(f"Total markets: {n_markets}")

# --- Pre-compute per-market trajectories ---
market_groups = {}
for mid, grp in master.groupby('market_id'):
    grp = grp.sort_values('timestamp')
    market_groups[mid] = {
        'yes_prices': grp['implied_yes_bid'].values,
        'no_prices':  grp['implied_no_bid'].values,
        'ttrs':       grp['TTR_seconds'].values if 'TTR_seconds' in grp.columns else None,
        'yes_depth':  grp['yes_depth_usd'].values,
    }

# --- Compute depth at entry zone ---
def get_median_depth_at_entry(entry_lo, entry_hi=None):
    if entry_hi is None:
        entry_hi = entry_lo + 0.05
    zone = master[master['yes_ask'].between(entry_lo - 0.03, entry_lo + 0.03)]
    if len(zone) == 0:
        return 0.0
    return zone['yes_depth_usd'].median()

def compute_ev(entry, exit_, pattern_rate, early_rate):
    p_complete    = pattern_rate * early_rate
    p_mid_resolve = pattern_rate * (1 - early_rate)
    
    profit_complete  = (exit_ - entry) * (1 - POLYMARKET_FEE) - SPREAD_COST
    profit_resolve_w = (1.0 - entry) * (1 - POLYMARKET_FEE) - SPREAD_COST
    loss_resolve_l   = -entry - SPREAD_COST
    
    ev = (p_complete    * profit_complete +
          p_mid_resolve * (0.5 * profit_resolve_w + 0.5 * loss_resolve_l))
    return ev

results = []

for entry in entry_thresholds:
    for exit_ in exit_thresholds:
        if exit_ <= entry:
            continue
        for min_ttr in min_ttr_at_exit_secs:
            yes_patterns = 0
            yes_early    = 0
            no_patterns  = 0
            no_early     = 0
            
            for mid, data in market_groups.items():
                # YES: dip then rip
                yes_p = data['yes_prices']
                ttrs  = data['ttrs']
                entry_found = False
                for j in range(len(yes_p)):
                    if not entry_found and yes_p[j] <= entry:
                        entry_found = True
                    elif entry_found and yes_p[j] >= exit_:
                        yes_patterns += 1
                        if ttrs is not None and ttrs[j] >= min_ttr:
                            yes_early += 1
                        break
                
                # NO: pump then dump
                no_p = data['no_prices']
                entry_found = False
                for j in range(len(no_p)):
                    if not entry_found and no_p[j] <= entry:
                        entry_found = True
                    elif entry_found and no_p[j] >= exit_:
                        no_patterns += 1
                        if ttrs is not None and ttrs[j] >= min_ttr:
                            no_early += 1
                        break
            
            total_patterns = yes_patterns + no_patterns
            total_early    = yes_early + no_early
            base_rate  = total_patterns / (2 * n_markets)  # denominator = 2*n since YES+NO
            early_rate = total_early / max(total_patterns, 1)
            
            ev = compute_ev(entry, exit_, base_rate, early_rate)
            
            med_depth = get_median_depth_at_entry(entry)
            risk_reward = (exit_ - entry) / entry if entry > 0 else 0
            
            results.append({
                'entry':          entry,
                'exit':           exit_,
                'min_ttr':        min_ttr,
                'base_rate':      round(base_rate, 4),
                'early_pct':      round(early_rate, 4),
                'est_pos_labels': total_early,
                'ev_per_trade':   round(ev, 6),
                'med_depth_entry': round(med_depth, 1),
                'max_viable_bet': round(min(med_depth * 0.5, 50), 1),
                'risk_reward':    round(risk_reward, 3),
            })

df = pd.DataFrame(results).sort_values('ev_per_trade', ascending=False)
print("\n=== FULL GRID (sorted by EV) ===")
print(df.to_string(index=False))

print("\n=== TOP 10 ===")
print(df.head(10).to_string(index=False))

# Select optimal: highest EV with est_pos_labels >= 150
candidates = df[df['est_pos_labels'] >= 150]
if len(candidates) > 0:
    optimal = candidates.iloc[0]
    print(f"\n=== OPTIMAL PARAMS (pos_labels >= 150) ===")
else:
    optimal = df.iloc[0]
    print(f"\n=== OPTIMAL PARAMS (relaxed, best EV) ===")

print(f"  Entry:          {optimal['entry']}")
print(f"  Exit:           {optimal['exit']}")
print(f"  Min TTR:        {optimal['min_ttr']}s")
print(f"  EV/trade:       {optimal['ev_per_trade']}")
print(f"  Base rate:      {optimal['base_rate']}")
print(f"  Early %:        {optimal['early_pct']}")
print(f"  Est pos labels: {optimal['est_pos_labels']}")
print(f"  Median depth:   ${optimal['med_depth_entry']}")
print(f"  Risk/Reward:    {optimal['risk_reward']}")

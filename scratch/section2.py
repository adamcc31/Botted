import pandas as pd
import numpy as np

master = pd.read_csv('dataset/clob_log/CLOB_MASTER.csv', low_memory=False)

print("=== MASTER DATASET PROFILE ===")
print(f"Total rows:          {len(master):,}")
print(f"Unique markets:      {master['market_id'].nunique():,}")
print(f"Avg rows/market:     {len(master)/master['market_id'].nunique():.1f}")
print(f"\nNumeric summary:")
print(master[['yes_ask','yes_bid','no_ask','no_bid','yes_depth_usd','no_depth_usd']].describe())

# Check implied bid validity
master['implied_yes_bid'] = 1.0 - master['no_ask']
master['implied_no_bid']  = 1.0 - master['yes_ask']
print(f"\nImplied YES bid stats:\n{master['implied_yes_bid'].describe()}")

def analyze_patterns(master, entry_thresholds, exit_thresholds, min_ttr_seconds=60):
    results = []
    
    # Convert TTR to seconds if in minutes
    if 'TTR_minutes' in master.columns:
        master['TTR_seconds'] = master['TTR_minutes'] * 60
    
    markets = master.groupby('market_id')
    
    for entry in entry_thresholds:
        for exit_ in exit_thresholds:
            yes_patterns, no_patterns = 0, 0
            yes_early, no_early = 0, 0
            yes_times, no_times = [], []
            
            for mid, grp in markets:
                grp = grp.sort_values('timestamp')
                yes_prices = grp['implied_yes_bid'].values
                no_prices  = grp['implied_no_bid'].values
                ttrs       = grp['TTR_seconds'].values if 'TTR_seconds' in grp.columns else None
                
                # YES: Dip then Rip
                entry_found = False
                entry_time  = None
                for j in range(len(yes_prices)):
                    if not entry_found and yes_prices[j] <= entry:
                        entry_found = True
                        entry_time  = ttrs[j] if ttrs is not None else None
                    elif entry_found and yes_prices[j] >= exit_:
                        yes_patterns += 1
                        exit_ttr = ttrs[j] if ttrs is not None else None
                        if exit_ttr is not None:
                            yes_times.append(entry_time - exit_ttr)
                            if exit_ttr >= min_ttr_seconds:
                                yes_early += 1
                        break
                
                # NO: Pump then Dump (same logic on no_prices)
                entry_found = False
                entry_time  = None
                for j in range(len(no_prices)):
                    if not entry_found and no_prices[j] <= entry:
                        entry_found = True
                        entry_time  = ttrs[j] if ttrs is not None else None
                    elif entry_found and no_prices[j] >= exit_:
                        no_patterns += 1
                        exit_ttr = ttrs[j] if ttrs is not None else None
                        if exit_ttr is not None:
                            no_times.append(entry_time - exit_ttr)
                            if exit_ttr >= min_ttr_seconds:
                                no_early += 1
                        break
            
            n_markets = master['market_id'].nunique()
            results.append({
                'entry': entry, 'exit': exit_,
                'yes_base_rate': yes_patterns/n_markets,
                'yes_early_pct': yes_early/max(yes_patterns,1),
                'yes_med_time':  np.median(yes_times) if yes_times else None,
                'no_base_rate':  no_patterns/n_markets,
                'no_early_pct':  no_early/max(no_patterns,1),
                'no_med_time':   np.median(no_times) if no_times else None,
            })
    
    return pd.DataFrame(results)

entry_thresholds = [0.30, 0.35, 0.40, 0.45, 0.49]
exit_thresholds  = [0.80, 0.85, 0.90, 0.92, 0.94, 0.97]
print("\n=== PATTERN FREQUENCY (MASTER DATASET) ===")
results = analyze_patterns(master, entry_thresholds, exit_thresholds)
print(results.to_string())

# DEPTH AT TARGET ZONES
entry_zone = master[master['yes_ask'].between(0.35, 0.49)]
exit_zone  = master[master['implied_yes_bid'].between(0.85, 0.97)]

print("\n=== DEPTH AT ENTRY ZONE (yes_ask 0.35-0.49) ===")
print(entry_zone['yes_depth_usd'].describe())
print(f"% of snapshots with depth >= $50:  {(entry_zone['yes_depth_usd'] >= 50).mean()*100:.1f}%")
print(f"% of snapshots with depth >= $100: {(entry_zone['yes_depth_usd'] >= 100).mean()*100:.1f}%")

print("\n=== DEPTH AT EXIT ZONE (implied_yes_bid 0.85-0.97) ===")
print(exit_zone['yes_depth_usd'].describe())
print(f"% of snapshots with depth >= $50:  {(exit_zone['yes_depth_usd'] >= 50).mean()*100:.1f}%")

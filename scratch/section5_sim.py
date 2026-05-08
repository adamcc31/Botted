"""
Section 5 -- End-to-End Simulation
4 scenarios, emergency exit comparison, full metrics
"""
import pandas as pd
import numpy as np

POLYMARKET_FEE = 0.02
SPREAD_COST    = 0.005
STARTING_CAP   = 1000.0
KELLY_FRAC     = 0.15
MAX_POS_PCT    = 0.05

OPTIMAL_ENTRY = 0.49
OPTIMAL_EXIT  = 0.80
MIN_TTR       = 60

# Load OOF data
oof = pd.read_csv('scratch/slingger_v5_oof.csv', low_memory=False)
master = pd.read_csv('dataset/clob_log/CLOB_MASTER.csv', low_memory=False)
master['timestamp'] = pd.to_datetime(master['timestamp'])
master['implied_yes_bid'] = 1.0 - master['no_ask']
master['implied_no_bid']  = 1.0 - master['yes_ask']
if 'TTR_minutes' in master.columns:
    master['TTR_seconds'] = master['TTR_minutes'] * 60

def emergency_exit_decision(current_price, entry_price, fee=0.02, spread=0.005):
    ev_exit_now = (current_price - entry_price) * (1 - fee) - spread
    ev_hold     = (current_price * (1.0 - entry_price) * (1 - fee)
                   - (1 - current_price) * entry_price - spread)
    decision    = 'EXIT_NOW' if ev_exit_now >= ev_hold else 'HOLD_TO_MATURITY'
    return decision, max(ev_exit_now, ev_hold)

def simulate_scenario(oof_df, master_df, label, prob_threshold=None,
                       alignment_filter=False, conservative=False,
                       emergency_mode='EXIT_NOW'):
    """
    Simulate swing trades. Return metrics dict.
    """
    capital = STARTING_CAP
    trades  = []
    peak_capital = capital
    max_dd  = 0.0
    bankrupt = False
    
    # Pre-compute per-market trajectories
    market_groups = {}
    for mid, grp in master_df.groupby('market_id'):
        grp = grp.sort_values('timestamp')
        market_groups[mid] = {
            'yes_prices': grp['implied_yes_bid'].values,
            'no_prices':  grp['implied_no_bid'].values,
            'ttrs':       grp['TTR_seconds'].values if 'TTR_seconds' in grp.columns else None,
            'yes_depth':  grp['yes_depth_usd'].values,
            'no_depth':   grp['no_depth_usd'].values,
        }
    
    for _, row in oof_df.iterrows():
        mid  = row['market_id']
        prob = row['oof_prob']
        lbl  = row['label']
        side = row.get('token_side', 'NONE')
        
        if bankrupt:
            break
        
        # Filter: scenario-specific
        if prob_threshold and prob < prob_threshold:
            continue
        if conservative and prob < 0.65:
            continue
        
        if mid not in market_groups:
            continue
        
        data = market_groups[mid]
        
        # Determine which prices to use
        if side == 'YES':
            prices = data['yes_prices']
            depths = data['yes_depth']
        elif side == 'NO':
            prices = data['no_prices']
            depths = data['no_depth']
        else:
            # For negative labels, still simulate as if we entered
            prices = data['yes_prices']
            depths = data['yes_depth']
        
        ttrs = data['ttrs']
        
        # Stake sizing
        stake = min(capital * MAX_POS_PCT, capital * KELLY_FRAC * prob)
        stake = min(stake, 50.0)  # hard cap
        stake = round(stake, 2)
        
        if stake < 1.0:
            continue
        
        # Simulate entry
        entry_filled = False
        entry_price  = None
        entry_idx    = None
        for j in range(len(prices)):
            if prices[j] <= OPTIMAL_ENTRY:
                # Check depth
                if depths[j] >= stake:
                    entry_filled = True
                    entry_price  = prices[j]
                    entry_idx    = j
                    break
        
        if not entry_filled:
            trades.append({'result': 'NO_ENTRY', 'pnl': 0})
            continue
        
        # Simulate exit
        shares = stake / entry_price
        exit_filled = False
        exit_price  = None
        emergency   = False
        
        for j in range(entry_idx + 1, len(prices)):
            # Check exit target
            if prices[j] >= OPTIMAL_EXIT:
                exit_filled = True
                exit_price  = prices[j]
                pnl = shares * ((exit_price - entry_price) * (1 - POLYMARKET_FEE) - SPREAD_COST)
                trades.append({'result': 'HIT', 'pnl': pnl})
                capital += pnl
                break
            
            # Check emergency
            if ttrs is not None and ttrs[j] < MIN_TTR and not emergency:
                emergency = True
                decision, ev = emergency_exit_decision(prices[j], entry_price)
                
                if emergency_mode == 'EXIT_NOW' and decision == 'EXIT_NOW':
                    pnl = shares * ((prices[j] - entry_price) * (1 - POLYMARKET_FEE) - SPREAD_COST)
                    trades.append({'result': 'EMERGENCY_EXIT', 'pnl': pnl})
                    capital += pnl
                    exit_filled = True
                    break
                elif emergency_mode == 'HOLD_TO_MATURITY' or decision == 'HOLD_TO_MATURITY':
                    # Hold to resolution: 50/50 win/loss
                    if np.random.random() < 0.5:
                        pnl = shares * ((1.0 - entry_price) * (1 - POLYMARKET_FEE) - SPREAD_COST)
                    else:
                        pnl = shares * (-entry_price - SPREAD_COST)
                    trades.append({'result': 'HOLD_TO_MATURITY', 'pnl': pnl})
                    capital += pnl
                    exit_filled = True
                    break
        
        if not exit_filled:
            # Expired without exit — resolve at 50/50
            if np.random.random() < 0.5:
                pnl = shares * ((1.0 - entry_price) * (1 - POLYMARKET_FEE) - SPREAD_COST)
            else:
                pnl = shares * (-entry_price - SPREAD_COST)
            trades.append({'result': 'RESOLVE', 'pnl': pnl})
            capital += pnl
        
        # Track drawdown
        peak_capital = max(peak_capital, capital)
        dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        max_dd = max(max_dd, dd)
        
        if capital <= 0:
            bankrupt = True
    
    trades_df = pd.DataFrame(trades)
    total_attempts = len(trades_df)
    entries = trades_df[trades_df['result'] != 'NO_ENTRY']
    hits    = trades_df[trades_df['result'] == 'HIT']
    
    pnls    = entries['pnl'].values if len(entries) > 0 else np.array([0])
    net_pnl = pnls.sum()
    win_rate = (pnls > 0).mean() if len(pnls) > 0 else 0
    
    # Sharpe (annualized from per-trade)
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = pnls.mean() / pnls.std()
    else:
        sharpe = 0
    
    # Calmar
    calmar = (net_pnl / STARTING_CAP) / max(max_dd, 0.01)
    
    return {
        'scenario':               label,
        'total_attempts':         total_attempts,
        'entry_fill_rate':        f"{len(entries)/max(total_attempts,1)*100:.1f}%",
        'pattern_completion_rate': f"{len(hits)/max(len(entries),1)*100:.1f}%",
        'resolution_rate':        f"{len(trades_df[trades_df['result'].isin(['RESOLVE','HOLD_TO_MATURITY'])])/max(len(entries),1)*100:.1f}%",
        'avg_profit_completed':   f"${hits['pnl'].mean():.2f}" if len(hits) > 0 else "$0.00",
        'net_pnl':                f"${net_pnl:.2f}",
        'win_rate':               f"{win_rate*100:.1f}%",
        'max_dd':                 f"{max_dd*100:.1f}%",
        'sharpe':                 f"{sharpe:.4f}",
        'calmar':                 f"{calmar:.2f}",
        'bankrupt':               'YES' if bankrupt else 'NO',
        'final_capital':          f"${capital:.2f}",
    }

np.random.seed(42)

# Scenario A: Baseline (no filter)
print("Running Scenario A: Baseline...")
res_a = simulate_scenario(oof, master, "A: Baseline")

# Scenario B: Trajectory filter (prob > 0.55)
print("Running Scenario B: Trajectory filter...")
res_b = simulate_scenario(oof, master, "B: Trajectory (>0.55)", prob_threshold=0.55)

# Scenario C: Combined (prob > 0.55 + alignment)
print("Running Scenario C: Combined...")
res_c = simulate_scenario(oof, master, "C: Combined (>0.55+align)", prob_threshold=0.55, alignment_filter=True)

# Scenario D: Conservative (prob > 0.65)
print("Running Scenario D: Conservative...")
res_d = simulate_scenario(oof, master, "D: Conservative (>0.65)", conservative=True)

results = pd.DataFrame([res_a, res_b, res_c, res_d])
print("\n=== SIMULATION RESULTS ===")
print(results.to_string(index=False))

# Emergency comparison
print("\n=== EMERGENCY EXIT COMPARISON ===")
np.random.seed(42)
res_exit_now = simulate_scenario(oof, master, "EXIT_NOW", prob_threshold=0.55, emergency_mode='EXIT_NOW')
np.random.seed(42)
res_hold     = simulate_scenario(oof, master, "HOLD_TO_MATURITY", prob_threshold=0.55, emergency_mode='HOLD_TO_MATURITY')

emerg_df = pd.DataFrame([res_exit_now, res_hold])
emerg_df = emerg_df[['scenario', 'net_pnl', 'win_rate', 'max_dd', 'sharpe', 'calmar', 'final_capital']]
print(emerg_df.to_string(index=False))

# Determine winner
pnl_exit = float(res_exit_now['net_pnl'].replace('$',''))
pnl_hold = float(res_hold['net_pnl'].replace('$',''))
winner = "EXIT_NOW" if pnl_exit >= pnl_hold else "HOLD_TO_MATURITY"
margin = abs(pnl_exit - pnl_hold)
print(f"\nEmergency winner: {winner} by ${margin:.2f}")

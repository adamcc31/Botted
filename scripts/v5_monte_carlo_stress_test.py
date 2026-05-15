import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import pickle
import random
from src.utils import compute_position_size

def run_monte_carlo_v2():
    # 1. Load Data
    oof_path = 'scratch/slingger_v5_oof.csv'
    if not os.path.exists(oof_path):
        print(f"Error: {oof_path} not found.")
        return
    
    df = pd.read_csv(oof_path)
    
    # 2. Load Calibrator
    cal_path = 'models/slingger_hunter_v5/calibrator.pkl'
    if not os.path.exists(cal_path):
        print(f"Error: {cal_path} not found.")
        return
        
    with open(cal_path, 'rb') as f:
        calibrator = pickle.load(f)
    
    # Apply Calibration
    probs = df['oof_prob'].values.reshape(-1, 1)
    df['cal_prob'] = calibrator.predict_proba(probs)[:, 1]
    
    # 3. Filter: Threshold 0.65 + Unique per Market (1 Trade per Market)
    df_filtered = df[df['cal_prob'] >= 0.65].copy()
    df_strict = df_filtered.groupby('market_id').first().reset_index()
    
    trades_pool_base = df_strict[['cal_prob', 'label']].to_dict('records')
    n_trades_total = len(trades_pool_base)
    
    print(f"Pool size (Threshold 0.65): {n_trades_total} unique markets")
    
    # 4. Simulation Parameters (MANDAT: $50 capital, $20 cap)
    INIT_CAPITAL = 50.0
    ENTRY = 0.49
    EXIT = 0.80
    FEE = 0.02
    SPREAD = 0.005
    KELLY_FRAC = 0.15 # Metadata value
    MAX_PCT_CAP = 0.25 
    
    # Note: src.utils.compute_position_size already implements Half-Kelly (0.5x) 
    # and $20.0 hard ceiling from previous task.
    
    RUIN_THRESHOLD = 5.0
    ITERATIONS = 1000
    
    final_capitals = []
    max_drawdowns = []
    ruin_count = 0
    
    print(f"Running {ITERATIONS} Monte Carlo iterations with $50 capital and $20 cap...")
    
    for i in range(ITERATIONS):
        trades_pool = trades_pool_base.copy()
        random.shuffle(trades_pool)
        
        capital = INIT_CAPITAL
        peak_capital = INIT_CAPITAL
        max_dd = 0.0
        bankrupt = False
        
        for trade in trades_pool:
            if capital < RUIN_THRESHOLD:
                bankrupt = True
                break
            
            # Sizing via src.utils (now with 0.5 Kelly and $20 ceiling)
            res = compute_position_size(
                capital=capital,
                swing_prob=trade['cal_prob'],
                entry_odds=ENTRY,
                exit_odds=EXIT,
                kelly_fraction=KELLY_FRAC,
                max_pct=MAX_PCT_CAP,
                fee=FEE,
                spread=SPREAD
            )
            
            stake = res['stake_usd']
            if stake <= 0:
                continue
                
            # Simulate Outcome
            if trade['label'] == 1:
                # Win (HIT target)
                pnl = res['max_win_usd']
            else:
                # Loss (Timed out / matured at lower price)
                # In this swing strategy, label 0 means it never hit 0.80.
                # Conservatively, we treat it as loss of stake minus spread.
                pnl = res['max_loss_usd']
            
            capital += pnl
            
            # Track Drawdown
            if capital > peak_capital:
                peak_capital = capital
            
            if peak_capital > 0:
                dd = (peak_capital - capital) / peak_capital
                if dd > max_dd:
                    max_dd = dd
        
        if bankrupt or capital < RUIN_THRESHOLD:
            ruin_count += 1
            capital = min(capital, RUIN_THRESHOLD)
            
        final_capitals.append(capital)
        max_drawdowns.append(max_dd)
        
    # 5. Report Metrics
    avg_final = np.mean(final_capitals)
    median_final = np.median(final_capitals)
    median_dd = np.median(max_drawdowns)
    worst_dd = np.max(max_drawdowns)
    ruin_pct = (ruin_count / ITERATIONS) * 100
    
    print("-" * 60)
    print("STRICT MONTE CARLO RESULTS ($50 CAP, $20 CEILING)")
    print("-" * 60)
    print(f"Initial Capital:         ${INIT_CAPITAL:.2f}")
    print(f"Avg Final Capital:       ${avg_final:.2f}")
    print(f"Median Final Capital:    ${median_final:.2f}")
    print(f"Median Max Drawdown:     {median_dd*100:.2f}%")
    print(f"Worst Case Max Drawdown: {worst_dd*100:.2f}%")
    print(f"Risk of Ruin (<$5):      {ruin_pct:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    run_monte_carlo_v2()

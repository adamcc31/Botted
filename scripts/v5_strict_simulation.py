import pandas as pd
import numpy as np
import pickle
import os

def strict_ev_simulation():
    # 1. Load Data
    oof_path = 'scratch/slingger_v5_oof.csv'
    if not os.path.exists(oof_path):
        print(f"Error: {oof_path} not found.")
        return
        
    df = pd.read_csv(oof_path)
    print(f"Loaded OOF Data: {len(df)} rows (Market snapshots)")

    # 2. Load Calibrator
    cal_path = 'models/slingger_hunter_v5/calibrator.pkl'
    with open(cal_path, 'rb') as f:
        calibrator = pickle.load(f)
    
    # Apply Calibration
    probs = df['oof_prob'].values.reshape(-1, 1)
    df['cal_prob'] = calibrator.predict_proba(probs)[:, 1]

    # 3. Parameters
    ENTRY_ODDS = 0.49
    EXIT_ODDS  = 0.80
    FEE        = 0.02
    SPREAD     = 0.005
    BET_SIZE   = 10.0 # Standard unit for EV calculation
    
    # Math:
    # Gross Payout Multiplier = (0.80 / 0.49) = 1.6326
    # Net Payout (if win) = Bet * (1.6326 - 1) * (1 - 0.02) = Bet * 0.6326 * 0.98 = Bet * 0.62
    # Loss = -Bet
    
    gross_payout_mult = EXIT_ODDS / ENTRY_ODDS
    net_win_mult = (gross_payout_mult - 1) * (1 - FEE)
    
    print(f"\nSimulation Parameters:")
    print(f"  Entry: {ENTRY_ODDS} | Exit: {EXIT_ODDS} | Fee: {FEE*100}% | Spread: ${SPREAD}")
    print(f"  Net Win Multiplier: {net_win_mult:.4f}x")
    print("-" * 80)
    print(f"{'Threshold':<10} {'Markets':<10} {'Wins':<10} {'Losses':<10} {'WinRate':<10} {'Total PnL':<10} {'Avg EV %':<10}")
    print("-" * 80)

    # 4. Iteration per Threshold
    results = []
    for threshold in [0.50, 0.60, 0.65, 0.70]:
        # FILTER: Only markets meeting threshold
        # IMPORTANT: Logic for 1 Trade per Market_ID
        # Since oof.csv is already summarized at market level, we just filter.
        # But to be ABSOLUTELY strict as mandated:
        eligible_markets = df[df['cal_prob'] >= threshold].copy()
        
        # Ensure unique trade per market_id (taking the first occurrence)
        strict_trades = eligible_markets.groupby('market_id').first().reset_index()
        
        n_trades = len(strict_trades)
        if n_trades == 0:
            print(f"{threshold:<10.2f} {'0':<10} {'0':<10} {'0':<10} {'0.0%':<10} {'$0.00':<10} {'0.0%':<10}")
            continue
            
        wins = int(strict_trades['label'].sum())
        losses = n_trades - wins
        win_rate = wins / n_trades
        
        # Realized PnL Calculation
        # PnL = (Wins * Bet * NetWinMult) - (Losses * Bet) - (TotalTrades * Spread)
        total_pnl = (wins * BET_SIZE * net_win_mult) - (losses * BET_SIZE) - (n_trades * SPREAD)
        avg_ev_pct = (total_pnl / (n_trades * BET_SIZE)) * 100
        
        print(f"{threshold:<10.2f} {n_trades:<10} {wins:<10} {losses:<10} {win_rate:<10.1%} ${total_pnl:<10.2f} {avg_ev_pct:<10.2f}%")

    print("-" * 80)
    print("MANDATE: Single Trade per Market_ID enforced via groupby('market_id').first()")

if __name__ == "__main__":
    strict_ev_simulation()

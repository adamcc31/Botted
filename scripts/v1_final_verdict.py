import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import joblib
import json
import random
from model_training.features import build_features
from src.utils import compute_position_size

def main():
    print("=" * 80)
    print("ALPHA V1 FINAL VERDICT: MACRO REGIME SCAN & MONTE CARLO STRESS SIMULATION")
    print("=" * 80)

    # 1. Load Dataset and Models
    master_path = 'dataset/raw/alpha_v1_master.csv'
    cv_path = 'models/alpha_v1/cv_results.json'
    model_path = 'models/alpha_v1/model.pkl'
    
    if not all(os.path.exists(p) for p in [master_path, cv_path, model_path]):
        print("Error: Missing dataset or model files.")
        return

    df = pd.read_csv(master_path, low_memory=False)
    df = df[df['label'].notnull()].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    with open(cv_path, 'r') as f:
        cv_results = json.load(f)
        
    model_bundle = joblib.load(model_path)
    base_model = model_bundle['base_model']
    platt = model_bundle['platt']
    feature_names = model_bundle['feature_names']

    # Build predictions
    market_end_time = df.groupby("market_id")["timestamp"].max().sort_values()
    ordered_markets = market_end_time.index.tolist()
    n_markets = len(ordered_markets)
    
    test_ratio = 0.15
    calib_ratio = 0.30
    
    test_n_markets  = int(n_markets * test_ratio)
    calib_n_markets = int(n_markets * calib_ratio)
    train_n_markets = n_markets - calib_n_markets - test_n_markets
    
    train_markets = set(ordered_markets[:train_n_markets])
    calib_markets = set(ordered_markets[train_n_markets:train_n_markets + calib_n_markets])
    test_markets = set(ordered_markets[train_n_markets + calib_n_markets:])
    
    df_train = df[df["market_id"].isin(train_markets)].copy().sort_values("timestamp").reset_index(drop=True)
    df_calib = df[df["market_id"].isin(calib_markets)].copy().sort_values("timestamp").reset_index(drop=True)
    df_test = df[df["market_id"].isin(test_markets)].copy().sort_values("timestamp").reset_index(drop=True)
    
    df_calib_feat = build_features(df_calib)
    df_test_feat = build_features(df_test)
    X_calib = df_calib_feat[feature_names].values.astype(np.float32)
    X_test = df_test_feat[feature_names].values.astype(np.float32)
    
    raw_train = np.array(cv_results['oof_preds'])
    raw_calib = base_model.predict_proba(X_calib)[:, 1]
    raw_test = base_model.predict_proba(X_test)[:, 1]
    
    df_train['raw_prob'] = raw_train
    df_calib['raw_prob'] = raw_calib
    df_test['raw_prob'] = raw_test
    
    df_pred = pd.concat([df_train, df_calib, df_test], ignore_index=True)
    df_pred = df_pred.sort_values("timestamp").reset_index(drop=True)
    
    raw_probs = df_pred['raw_prob'].values
    cal_probs = platt.predict(raw_probs)
    df_pred['cal_prob'] = np.clip(cal_probs, 0.30, 0.72)

    # -------------------------------------------------------------------------
    # TASK 1: MACRO REGIME FILTERING & EV SCAN
    # -------------------------------------------------------------------------
    print("\n[TASK 1] MACRO REGIME FILTERING & EV SCAN (TTR >= 60 Minutes)")
    print("-" * 80)
    
    # FILTER MUTLAK: TTR >= 60 minutes (3600 seconds)
    df_macro = df_pred[df_pred['ttr_seconds'] >= 3600].copy()
    print(f"Total trades with TTR >= 60 mins: {len(df_macro)} rows (from {len(df_pred)} total)")
    
    EXIT_ODDS = 1.00 # Hold to expiry
    FEE = 0.02
    SPREAD = 0.005
    BET_SIZE = 10.0

    df_macro['gross_return_ratio'] = (EXIT_ODDS - df_macro['entry_odds']) / df_macro['entry_odds']
    df_macro['net_win_mult'] = df_macro['gross_return_ratio'] * (1 - FEE)
    df_macro['ev_pct'] = (df_macro['cal_prob'] * df_macro['net_win_mult'] - (1 - df_macro['cal_prob'])) * 100
    df_macro['ev'] = df_macro['ev_pct'] / 100
    df_macro['pnl'] = np.where(
        df_macro['label'] == 1,
        BET_SIZE * df_macro['net_win_mult'] - SPREAD,
        -BET_SIZE - SPREAD
    )

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    has_viable_threshold = False
    best_threshold = None
    best_ev_pct = 0.0

    print(f"{'Threshold':<10} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'Avg EV %':<10} | {'Net PnL ($)':<12}")
    print("-" * 80)

    for t in thresholds:
        # We also enforce that EV must be positive to count as a realistic trade
        sub = df_macro[(df_macro['cal_prob'] > t) & (df_macro['ev'] >= 0.04)]
        n_trades = len(sub)
        if n_trades == 0:
            print(f"> {t:<8.2f} | {'0':<8} | {'0':<6} | {'0':<6} | {'0.00%':<10} | {'0.00%':<10} | {'$0.00':<12}")
            continue
            
        wins = int(sub['label'].sum())
        losses = n_trades - wins
        wr = wins / n_trades * 100
        avg_ev = sub['ev_pct'].mean()
        net_pnl = sub['pnl'].sum()
        
        print(f"> {t:<8.2f} | {n_trades:<8} | {wins:<6} | {losses:<6} | {wr:<9.2f}% | {avg_ev:<+9.2f}% | ${net_pnl:<+11.2f}")
        
        if avg_ev >= 5.0 and n_trades >= 5:
            has_viable_threshold = True
            if avg_ev > best_ev_pct:
                best_ev_pct = avg_ev
                best_threshold = t
    print("-" * 80)

    # -------------------------------------------------------------------------
    # TASK 2: MONTE CARLO SURVIVAL (IF VIABLE)
    # -------------------------------------------------------------------------
    print("\n[TASK 2] MONTE CARLO SURVIVAL TEST")
    print("-" * 80)
    
    if not has_viable_threshold:
        print("Verdict: NO VIABLE THRESHOLD FOUND (No threshold with EV >= 5% and Trades >= 5)")
        print("Verdict status: NO-GO ❌")
        return
        
    print(f"Viable Threshold Found: > {best_threshold:.2f} with Avg EV = {best_ev_pct:.2f}%")
    df_filtered_trades = df_macro[(df_macro['cal_prob'] > best_threshold) & (df_macro['ev'] >= 0.04)].copy()
    trades_pool_filtered = df_filtered_trades[['entry_odds', 'cal_prob', 'label', 'market_id']].to_dict('records')
    
    INIT_CAPITAL = 50.0
    RUIN_THRESHOLD = 5.0
    ITERATIONS = 1000
    KELLY_FRAC = 0.25
    MAX_PCT_CAP = 0.10
    
    final_capitals = []
    max_drawdowns = []
    ruin_count = 0
    
    for i in range(1, ITERATIONS + 1):
        trades_pool = trades_pool_filtered.copy()
        random.shuffle(trades_pool)
        
        capital = INIT_CAPITAL
        peak_capital = INIT_CAPITAL
        max_dd = 0.0
        bankrupt = False
        
        for trade in trades_pool:
            if capital < RUIN_THRESHOLD:
                bankrupt = True
                break
                
            res = compute_position_size(
                capital=capital,
                swing_prob=trade['cal_prob'],
                entry_odds=trade['entry_odds'],
                exit_odds=EXIT_ODDS,
                kelly_fraction=KELLY_FRAC,
                max_pct=MAX_PCT_CAP,
                fee=FEE,
                spread=SPREAD
            )
            
            stake = res['stake_usd']
            if stake <= 0:
                continue
                
            if trade['label'] == 1:
                pnl = res['max_win_usd']
            else:
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
        
    avg_final = np.mean(final_capitals)
    median_final = np.median(final_capitals)
    median_dd = np.median(max_drawdowns)
    worst_dd = np.max(max_drawdowns)
    ruin_pct = (ruin_count / ITERATIONS) * 100
    
    print(f"Filtered trades pool size: {len(trades_pool_filtered)} transactions")
    print(f"Initial Capital:         ${INIT_CAPITAL:.2f}")
    print(f"Avg Final Capital:       ${avg_final:.2f} ({((avg_final - INIT_CAPITAL)/INIT_CAPITAL)*100:+.2f}%)")
    print(f"Median Final Capital:    ${median_final:.2f}")
    print(f"Median Max Drawdown:     {median_dd*100:.2f}%")
    print(f"Worst Case Max Drawdown: {worst_dd*100:.2f}%")
    print(f"Risk of Ruin (<$5):      {ruin_pct:.2f}%")
    print("-" * 80)
    
    # -------------------------------------------------------------------------
    # TASK 3: THE VERDICT (GO OR NO-GO)
    # -------------------------------------------------------------------------
    print("\n[TASK 3] THE VERDICT")
    print("-" * 80)
    if avg_final >= 55.0 and median_dd < 0.15 and len(trades_pool_filtered) >= 5:
        print("Verdict: GO ✅")
        print("Reason: Solid positive PnL and controlled volatility meet the quality criteria.")
    else:
        print("Verdict: NO-GO ❌")
        print(f"Reason: Final capital ${avg_final:.2f} is below $55.00 or Max Drawdown {median_dd*100:.2f}% exceeds 15%.")

if __name__ == "__main__":
    main()

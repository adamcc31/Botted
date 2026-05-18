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
    print("=" * 70)
    print("ALPHA V1 PERFORMANCE AUDIT: FEATURES, EV TABLES & REALISTIC STRESS TEST")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load Dataset and Models
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # TASK 1: FEATURE IMPORTANCE AUDIT
    # -------------------------------------------------------------------------
    print("\n[TASK 1] FEATURE IMPORTANCE AUDIT")
    print("-" * 50)
    importances = base_model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print(f"{'Feature Name':<25} | {'Importance (Gain / Weight)':<25}")
    print("-" * 55)
    for name, imp in feat_imp:
        print(f"{name:<25} | {imp:<25.6f} ({imp*100:.2f}%)")
    print("-" * 55)

    # -------------------------------------------------------------------------
    # Generate Calibrated Probabilities for all 2,213 rows
    # -------------------------------------------------------------------------
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
    # TASK 2: EV CALIBRATION TABLE (THE SWEET SPOT)
    # -------------------------------------------------------------------------
    print("\n[TASK 2] EXPECTED VALUE (EV) CALIBRATION TABLES")
    
    EXIT_ODDS = 0.80
    FEE = 0.02
    SPREAD = 0.005
    BET_SIZE = 10.0 # Standard unit bet size for EV table

    # Calculate trade-level payout multipliers, expected values, and PnL
    # gross_return_ratio = (exit_odds - entry_odds) / entry_odds
    # net_win_mult = gross_return_ratio * (1 - fee)
    df_pred['gross_return_ratio'] = (EXIT_ODDS - df_pred['entry_odds']) / df_pred['entry_odds']
    df_pred['net_win_mult'] = df_pred['gross_return_ratio'] * (1 - FEE)
    
    # Expected Value %: (p_cal * net_win_mult - (1 - p_cal)) * 100
    df_pred['ev_pct'] = (df_pred['cal_prob'] * df_pred['net_win_mult'] - (1 - df_pred['cal_prob'])) * 100
    df_pred['ev'] = df_pred['ev_pct'] / 100
    
    # PnL for standard $10 bet:
    # Win: BET_SIZE * net_win_mult - SPREAD
    # Loss: -BET_SIZE - SPREAD
    df_pred['pnl'] = np.where(
        df_pred['label'] == 1,
        BET_SIZE * df_pred['net_win_mult'] - SPREAD,
        -BET_SIZE - SPREAD
    )

    # --- Table A: Pure Probability Filter ---
    print("\n=== TABLE A: PURE PROBABILITY FILTER (cal_prob > Threshold) ===")
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'Avg EV %':<10} | {'Net PnL ($)':<12}")
    print("-" * 75)
    
    thresholds = [0.00, 0.50, 0.55, 0.60, 0.65, 0.70]
    sweet_spot_threshold = 0.60
    
    for t in thresholds:
        sub = df_pred[df_pred['cal_prob'] > t]
        n_trades = len(sub)
        if n_trades == 0:
            print(f"{t:<10.2f} | {'0':<8} | {'0':<6} | {'0':<6} | {'0.00%':<10} | {'0.00%':<10} | {'$0.00':<12}")
            continue
            
        wins = int(sub['label'].sum())
        losses = n_trades - wins
        wr = wins / n_trades * 100
        avg_ev = sub['ev_pct'].mean()
        net_pnl = sub['pnl'].sum()
        
        print(f"> {t:<8.2f} | {n_trades:<8} | {wins:<6} | {losses:<6} | {wr:<9.2f}% | {avg_ev:<+9.2f}% | ${net_pnl:<+11.2f}")
    print("-" * 75)

    # --- Table B: Production Filter (cal_prob > Threshold AND ev >= 0.04) ---
    print("\n=== TABLE B: PRODUCTION FILTER (cal_prob > Threshold AND EV >= 4%) ===")
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'Avg EV %':<10} | {'Net PnL ($)':<12}")
    print("-" * 75)
    
    for t in thresholds:
        sub = df_pred[(df_pred['cal_prob'] > t) & (df_pred['ev'] >= 0.04)]
        n_trades = len(sub)
        if n_trades == 0:
            print(f"{t:<10.2f} | {'0':<8} | {'0':<6} | {'0':<6} | {'0.00%':<10} | {'0.00%':<10} | {'$0.00':<12}")
            continue
            
        wins = int(sub['label'].sum())
        losses = n_trades - wins
        wr = wins / n_trades * 100
        avg_ev = sub['ev_pct'].mean()
        net_pnl = sub['pnl'].sum()
        
        print(f"> {t:<8.2f} | {n_trades:<8} | {wins:<6} | {losses:<6} | {wr:<9.2f}% | {avg_ev:<+9.2f}% | ${net_pnl:<+11.2f}")
    print("-" * 75)

    # -------------------------------------------------------------------------
    # TASK 3: REALISTIC MONTE CARLO (THRESHOLD FILTERED)
    # -------------------------------------------------------------------------
    print("\n[TASK 3] REALISTIC MONTE CARLO STRESS TEST")
    
    def simulate_mc(threshold_val):
        df_filtered_trades = df_pred[(df_pred['cal_prob'] > threshold_val) & (df_pred['ev'] >= 0.04)].copy()
        trades_pool_filtered = df_filtered_trades[['entry_odds', 'cal_prob', 'label', 'market_id']].to_dict('records')
        n_filtered_trades = len(trades_pool_filtered)
        
        if n_filtered_trades == 0:
            return None, n_filtered_trades
            
        # Simulation parameters
        INIT_CAPITAL = 50.0
        RUIN_THRESHOLD = 5.0
        ITERATIONS = 1000
        KELLY_FRAC = 0.25  # V1 Quarter-Kelly
        MAX_PCT_CAP = 0.10 # V1 10% max bankroll cap
        
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
        
        return {
            "avg_final": avg_final,
            "median_final": median_final,
            "median_dd": median_dd,
            "worst_dd": worst_dd,
            "ruin_pct": ruin_pct,
            "init_capital": INIT_CAPITAL
        }, n_filtered_trades

    for t_val in [0.60, 0.65]:
        print("-" * 70)
        print(f"RUNNING MONTE CARLO FOR THRESHOLD > {t_val} (AND EV >= 4%)")
        print("-" * 70)
        metrics, n_trades = simulate_mc(t_val)
        if metrics is None:
            print(f"No trades meet the threshold > {t_val} and EV >= 4%")
            continue
            
        print(f"Filtered trades pool size: {n_trades} transactions")
        print(f"Initial Capital:         ${metrics['init_capital']:.2f}")
        print(f"Avg Final Capital:       ${metrics['avg_final']:.2f} ({((metrics['avg_final'] - metrics['init_capital'])/metrics['init_capital'])*100:+.2f}%)")
        print(f"Median Final Capital:    ${metrics['median_final']:.2f}")
        print(f"Median Max Drawdown:     {metrics['median_dd']*100:.2f}%")
        print(f"Worst Case Max Drawdown: {metrics['worst_dd']*100:.2f}%")
        print(f"Risk of Ruin (<$5):      {metrics['ruin_pct']:.2f}%")
        print("-" * 70)


if __name__ == "__main__":
    main()

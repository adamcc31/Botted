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

def run_v1_monte_carlo():
    print("=" * 60)
    print("ALPHA V1 MONTE CARLO STRESS TEST SIMULATION")
    print("=" * 60)

    # 1. Load Dataset
    master_path = 'dataset/raw/alpha_v1_master.csv'
    if not os.path.exists(master_path):
        print(f"Error: {master_path} not found.")
        return
        
    print(f"Loading master dataset from {master_path}...")
    df = pd.read_csv(master_path, low_memory=False)
    
    # Filter rows with non-null labels (resolved bets)
    df = df[df['label'].notnull()].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df)
    print(f"Total labeled transactions: {n_total}")
    
    # 2. Replicate Chronological Split (Market-Level) to align with CV
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
    
    print(f"Chronological Splits:")
    print(f"  Train Set:     {len(df_train)} rows ({len(train_markets)} markets)")
    print(f"  Calib Set:     {len(df_calib)} rows ({len(calib_markets)} markets)")
    print(f"  Test Set:      {len(df_test)} rows ({len(test_markets)} markets)")
    
    # 3. Load CV Results and trained model
    cv_path = 'models/alpha_v1/cv_results.json'
    model_path = 'models/alpha_v1/model.pkl'
    
    if not os.path.exists(cv_path) or not os.path.exists(model_path):
        print(f"Error: {cv_path} or {model_path} not found.")
        return
        
    print("\nLoading CV results and model artifacts...")
    with open(cv_path, 'r') as f:
        cv_results = json.load(f)
        
    model_bundle = joblib.load(model_path)
    base_model = model_bundle['base_model']
    platt = model_bundle['platt'] # Isotonic calibrator
    feature_names = model_bundle['feature_names']
    
    # Check OOF labels match df_train['label'] exactly
    oof_labels = np.array(cv_results['oof_labels'])
    if not (df_train['label'].values == oof_labels).all():
        print("Warning: OOF labels do not match df_train labels exactly!")
        
    # 4. Generate Raw Predictions for Calib and Test splits
    print("Generating out-of-sample predictions...")
    
    df_calib_feat = build_features(df_calib)
    df_test_feat = build_features(df_test)
    
    X_calib = df_calib_feat[feature_names].values.astype(np.float32)
    X_test = df_test_feat[feature_names].values.astype(np.float32)
    
    # Raw prediction probabilities (before calibrator)
    raw_train = np.array(cv_results['oof_preds'])
    raw_calib = base_model.predict_proba(X_calib)[:, 1]
    raw_test = base_model.predict_proba(X_test)[:, 1]
    
    # Concat raw predictions chronologically matching the dataset order
    df_train['raw_prob'] = raw_train
    df_calib['raw_prob'] = raw_calib
    df_test['raw_prob'] = raw_test
    
    df_pred = pd.concat([df_train, df_calib, df_test], ignore_index=True)
    df_pred = df_pred.sort_values("timestamp").reset_index(drop=True)
    
    # Apply calibrator platt (IsotonicRegression) on all raw predictions
    raw_probs = df_pred['raw_prob'].values
    cal_probs = platt.predict(raw_probs)
    
    # Clip probabilities to [0.30, 0.72] to match production ML gate clipping
    df_pred['cal_prob'] = np.clip(cal_probs, 0.30, 0.72)
    
    trades_pool_base = df_pred[['entry_odds', 'cal_prob', 'label', 'market_id']].to_dict('records')
    print(f"Total trades pool compiled: {len(trades_pool_base)} transactions")
    print(f"Calibrated probability range: [{df_pred['cal_prob'].min():.4f}, {df_pred['cal_prob'].max():.4f}]")
    
    # 5. Simulation Parameters
    INIT_CAPITAL = 50.0
    RUIN_THRESHOLD = 5.0
    ITERATIONS = 1000
    
    EXIT_ODDS = 0.80
    FEE = 0.02
    SPREAD = 0.005
    KELLY_FRAC = 0.25  # V1 Quarter-Kelly
    MAX_PCT_CAP = 0.10 # V1 10% max bankroll cap
    
    final_capitals = []
    max_drawdowns = []
    ruin_count = 0
    
    print(f"\nRunning {ITERATIONS} Monte Carlo iterations with $50 capital, $20 cap, and Quarter-Kelly Sizing...")
    
    for i in range(1, ITERATIONS + 1):
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
                
            # Kelly Sizing using utils.compute_position_size
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
                
            # Simulate Outcome
            if trade['label'] == 1:
                # Win (HIT target exit 0.80)
                pnl = res['max_win_usd']
            else:
                # Loss (Timed out / never reached 0.80)
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
        
    # 6. Report Metrics
    avg_final = np.mean(final_capitals)
    median_final = np.median(final_capitals)
    median_dd = np.median(max_drawdowns)
    worst_dd = np.max(max_drawdowns)
    ruin_pct = (ruin_count / ITERATIONS) * 100
    
    print("-" * 65)
    print("ALPHA V1 MONTE CARLO STRESS TEST RESULTS")
    print("-" * 65)
    print(f"Initial Capital:         ${INIT_CAPITAL:.2f}")
    print(f"Avg Final Capital:       ${avg_final:.2f}")
    print(f"Median Final Capital:    ${median_final:.2f}")
    print(f"Median Max Drawdown:     {median_dd*100:.2f}%")
    print(f"Worst Case Max Drawdown: {worst_dd*100:.2f}%")
    print(f"Risk of Ruin (<$5):      {ruin_pct:.2f}%")
    print("-" * 65)

if __name__ == "__main__":
    run_v1_monte_carlo()

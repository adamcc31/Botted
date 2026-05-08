"""
Section 4 -- Trajectory Model: Slingger Hunter V5
  4.1 Label Engineering
  4.2 Feature Engineering
  4.3 Model Training (GroupKFold)
  4.4 Feature Importance Sanity Check
  4.5 Save Artifacts
"""
import pandas as pd
import numpy as np
import json, pickle, os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ============================================================
# OPTIMAL PARAMS from Section 3
# ============================================================
OPTIMAL_ENTRY = 0.49
OPTIMAL_EXIT  = 0.80
MIN_TTR       = 60  # seconds

POLYMARKET_FEE = 0.02
SPREAD_COST    = 0.005

# ============================================================
# Load data
# ============================================================
print("Loading CLOB_MASTER...")
master = pd.read_csv('dataset/clob_log/CLOB_MASTER.csv', low_memory=False)
master['timestamp'] = pd.to_datetime(master['timestamp'])
master['date'] = master['timestamp'].dt.date
master['implied_yes_bid'] = 1.0 - master['no_ask']
master['implied_no_bid']  = 1.0 - master['yes_ask']
if 'TTR_minutes' in master.columns:
    master['TTR_seconds'] = master['TTR_minutes'] * 60

print(f"Master: {len(master):,} rows | {master['market_id'].nunique():,} markets")

# Load DATASET_MASTER for join
print("Loading DATASET_MASTER...")
ds = pd.read_csv('dataset/DATASET_MASTER_08-05-2026.csv', low_memory=False)
ds_markets = set(ds['market_id'].unique())
print(f"DATASET_MASTER: {len(ds):,} rows | {len(ds_markets):,} markets")

# ============================================================
# 4.1 LABEL ENGINEERING
# ============================================================
print("\n" + "="*60)
print("4.1 LABEL ENGINEERING")
print("="*60)

def create_swing_labels(master_df, entry_thresh, exit_thresh, min_ttr_seconds=60):
    records = []
    for mid, grp in master_df.groupby('market_id'):
        grp = grp.sort_values('timestamp')
        yes_prices = grp['implied_yes_bid'].values
        no_prices  = grp['implied_no_bid'].values
        ttrs       = grp['TTR_seconds'].values if 'TTR_seconds' in grp.columns else None
        
        # Try YES side (dip then rip)
        yes_label = 0
        yes_entry_price = None
        yes_exit_price  = None
        yes_entry_ttr   = None
        yes_exit_ttr    = None
        yes_duration    = None
        entry_found = False
        entry_idx   = None
        for j in range(len(yes_prices)):
            if not entry_found and yes_prices[j] <= entry_thresh:
                entry_found     = True
                entry_idx       = j
                yes_entry_price = yes_prices[j]
                yes_entry_ttr   = ttrs[j] if ttrs is not None else None
            elif entry_found and yes_prices[j] >= exit_thresh:
                yes_exit_price = yes_prices[j]
                yes_exit_ttr   = ttrs[j] if ttrs is not None else None
                if ttrs is not None and ttrs[j] >= min_ttr_seconds:
                    yes_label = 1
                    yes_duration = yes_entry_ttr - yes_exit_ttr if yes_entry_ttr and yes_exit_ttr else None
                break
        
        # Try NO side (pump then dump)
        no_label = 0
        no_entry_price = None
        no_exit_price  = None
        no_entry_ttr   = None
        no_exit_ttr    = None
        no_duration    = None
        entry_found = False
        for j in range(len(no_prices)):
            if not entry_found and no_prices[j] <= entry_thresh:
                entry_found    = True
                no_entry_price = no_prices[j]
                no_entry_ttr   = ttrs[j] if ttrs is not None else None
            elif entry_found and no_prices[j] >= exit_thresh:
                no_exit_price = no_prices[j]
                no_exit_ttr   = ttrs[j] if ttrs is not None else None
                if ttrs is not None and ttrs[j] >= min_ttr_seconds:
                    no_label = 1
                    no_duration = no_entry_ttr - no_exit_ttr if no_entry_ttr and no_exit_ttr else None
                break
        
        # Pick the better side (or both as separate rows)
        if yes_label == 1:
            records.append({
                'market_id': mid, 'label': 1, 'token_side': 'YES',
                'entry_ttr': yes_entry_ttr, 'exit_ttr': yes_exit_ttr,
                'pattern_duration_s': yes_duration,
                'entry_price': yes_entry_price, 'exit_price': yes_exit_price
            })
        elif no_label == 1:
            records.append({
                'market_id': mid, 'label': 1, 'token_side': 'NO',
                'entry_ttr': no_entry_ttr, 'exit_ttr': no_exit_ttr,
                'pattern_duration_s': no_duration,
                'entry_price': no_entry_price, 'exit_price': no_exit_price
            })
        else:
            # Negative: neither side completed
            records.append({
                'market_id': mid, 'label': 0, 'token_side': 'NONE',
                'entry_ttr': None, 'exit_ttr': None,
                'pattern_duration_s': None,
                'entry_price': None, 'exit_price': None
            })
    
    return pd.DataFrame(records)

labels_df = create_swing_labels(master, OPTIMAL_ENTRY, OPTIMAL_EXIT, MIN_TTR)
n_pos = (labels_df['label'] == 1).sum()
n_neg = (labels_df['label'] == 0).sum()
print(f"n_positive: {n_pos}")
print(f"n_negative: {n_neg}")
print(f"Class balance: 1:{n_neg/max(n_pos,1):.1f}")

# Temporal distribution of positives
labels_with_date = labels_df.merge(
    master.groupby('market_id')['date'].first().reset_index(),
    on='market_id', how='left'
)
daily_pos = labels_with_date[labels_with_date['label']==1].groupby('date')['market_id'].count()
print(f"\nDaily positive label distribution:")
print(daily_pos.to_string())

# ============================================================
# 4.2 FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print("4.2 FEATURE ENGINEERING")
print("="*60)

features_list = []
for mid, grp in master.groupby('market_id'):
    grp = grp.sort_values('timestamp')
    if len(grp) < 2:
        continue
    
    # T=0 snapshot (first row)
    t0 = grp.iloc[0]
    
    # First 30s of data
    t0_ts = grp['timestamp'].iloc[0]
    first_30s = grp[grp['timestamp'] <= t0_ts + pd.Timedelta(seconds=30)]
    
    yes_price_t0 = t0['implied_yes_bid']
    no_price_t0  = t0['implied_no_bid']
    
    # Price velocity in first 30s
    if len(first_30s) >= 2:
        price_vel = (first_30s['implied_yes_bid'].iloc[-1] - first_30s['implied_yes_bid'].iloc[0])
        dt = (first_30s['timestamp'].iloc[-1] - first_30s['timestamp'].iloc[0]).total_seconds()
        price_velocity_30s = price_vel / max(dt, 1)
    else:
        price_velocity_30s = 0.0
    
    # Depth trend in first 30s
    if len(first_30s) >= 2:
        depth_start = first_30s['yes_depth_usd'].iloc[0]
        depth_end   = first_30s['yes_depth_usd'].iloc[-1]
        depth_trend_30s = 1 if depth_end > depth_start else (-1 if depth_end < depth_start else 0)
    else:
        depth_trend_30s = 0
    
    # Depth imbalance
    yd = t0['yes_depth_usd']
    nd = t0['no_depth_usd']
    depth_imbalance = (yd - nd) / max(yd + nd, 1e-6)
    
    # CLOB spread
    clob_spread_t0 = t0['yes_ask'] - t0['implied_yes_bid'] if 'yes_ask' in t0.index else np.nan
    
    # TTR at signal
    ttr_at_signal = t0.get('TTR_seconds', np.nan)
    
    # Market hour and day
    market_hour_utc = t0['timestamp'].hour
    day_of_week     = t0['timestamp'].weekday()
    
    # btc_realized_vol_prior_30m: estimate from CLOB price changes within first market data
    # We use the std of implied_yes_bid changes as a proxy
    if len(grp) >= 5:
        price_changes = grp['implied_yes_bid'].diff().dropna()
        btc_vol_proxy = price_changes.std()
    else:
        btc_vol_proxy = np.nan
    
    row = {
        'market_id':                 mid,
        'yes_price_t0':              yes_price_t0,
        'no_price_t0':               no_price_t0,
        'clob_spread_t0':            clob_spread_t0,
        'yes_depth_t0':              yd,
        'no_depth_t0':               nd,
        'depth_imbalance_t0':        depth_imbalance,
        'price_velocity_30s':        price_velocity_30s,
        'depth_trend_30s':           depth_trend_30s,
        'btc_realized_vol_prior_30m': btc_vol_proxy,
        'ttr_at_signal':             ttr_at_signal,
        'market_hour_utc':           market_hour_utc,
        'day_of_week':               day_of_week,
    }
    features_list.append(row)

features_df = pd.DataFrame(features_list)

# Join with DATASET_MASTER for overlapping markets
if 'strike_price' in ds.columns and 'btc_price' in ds.columns:
    ds_agg = ds.groupby('market_id').agg({
        'strike_price': 'first',
        'btc_price': 'first',
    }).reset_index()
    # Compute btc_vs_strike_pct
    ds_agg['btc_vs_strike_pct'] = (ds_agg['btc_price'] - ds_agg['strike_price']) / ds_agg['strike_price']
    features_df = features_df.merge(ds_agg[['market_id', 'btc_vs_strike_pct']], on='market_id', how='left')
else:
    features_df['btc_vs_strike_pct'] = np.nan

# Additional DS features if available
for col in ['obi_value', 'rv_value']:
    if col in ds.columns:
        ds_feat = ds.groupby('market_id')[col].first().reset_index()
        features_df = features_df.merge(ds_feat, on='market_id', how='left')
    else:
        features_df[col] = np.nan

# Merge labels
full_df = features_df.merge(labels_df[['market_id', 'label', 'token_side']], on='market_id', how='inner')
print(f"Full dataset: {len(full_df)} rows")
print(f"Positive: {(full_df['label']==1).sum()} | Negative: {(full_df['label']==0).sum()}")

FEATURE_COLS = [
    'yes_price_t0', 'no_price_t0', 'clob_spread_t0',
    'yes_depth_t0', 'no_depth_t0', 'depth_imbalance_t0',
    'price_velocity_30s', 'depth_trend_30s',
    'btc_realized_vol_prior_30m',
    # btc_vs_strike_pct REMOVED: 0% coverage in training = constant after impute
    # Would cause train/live distribution mismatch. Can reintroduce when
    # intramarket_logger.py collects it natively.
    'ttr_at_signal', 'market_hour_utc', 'day_of_week',
]

# Check feature coverage
print("\nFeature coverage (% non-null):")
for col in FEATURE_COLS:
    if col in full_df.columns:
        cov = full_df[col].notna().mean() * 100
        flag = " <-- FLAG: <70%" if cov < 70 else ""
        print(f"  {col:35s}: {cov:.1f}%{flag}")
    else:
        print(f"  {col:35s}: MISSING")

# ============================================================
# 4.3 MODEL TRAINING
# ============================================================
print("\n" + "="*60)
print("4.3 MODEL TRAINING (GroupKFold)")
print("="*60)

X = full_df[FEATURE_COLS].values.astype(np.float32)
y = full_df['label'].values.astype(int)
groups = full_df['market_id'].values

# Impute
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

n_pos_train = y.sum()
n_neg_train = (y == 0).sum()
scale_pos_weight = n_neg_train / max(n_pos_train, 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

params = {
    'n_estimators':         300,
    'max_depth':            4,
    'learning_rate':        0.05,
    'subsample':            0.8,
    'colsample_bytree':     0.8,
    'scale_pos_weight':     scale_pos_weight,
    'eval_metric':          'auc',
    'early_stopping_rounds': 20,
    'random_state':         42,
}

gkf = GroupKFold(n_splits=5)
oof_probs  = np.zeros(len(y))
fold_aucs  = []
fold_sizes = []
models     = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
    model = XGBClassifier(**params)
    model.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              verbose=False)
    oof_probs[va_idx] = model.predict_proba(X[va_idx])[:, 1]
    
    try:
        fold_auc = roc_auc_score(y[va_idx], oof_probs[va_idx])
    except ValueError:
        fold_auc = 0.5  # single class in fold
    
    fold_aucs.append(fold_auc)
    fold_sizes.append({'pos': int(y[va_idx].sum()), 'neg': int((y[va_idx]==0).sum())})
    models.append(model)
    print(f"Fold {fold+1}: AUC={fold_auc:.4f} | pos={y[va_idx].sum()} neg={(y[va_idx]==0).sum()}")

overall_auc = roc_auc_score(y, oof_probs)
print(f"\nOOF AUC:  {overall_auc:.4f}")
print(f"Std Dev:  {np.std(fold_aucs):.4f}")
print(f"Range:    {min(fold_aucs):.4f} - {max(fold_aucs):.4f}")

# Gate check
if overall_auc < 0.52:
    gate_result = "STOP"
    print("\n[GATE] STOP: Insufficient signal. Recheck after 30 days collection.")
elif overall_auc < 0.55:
    gate_result = "WEAK SIGNAL POC"
    print("\n[GATE] PROCEED (WEAK SIGNAL POC)")
else:
    gate_result = "PROCEED"
    print("\n[GATE] PROCEED (normal)")

# ============================================================
# 4.4 FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("4.4 FEATURE IMPORTANCE (gain)")
print("="*60)

# Use last fold model for importance
final_model = models[-1]
importance = final_model.get_booster().get_score(importance_type='gain')
imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Map f0,f1,... to feature names
feature_map = {f'f{i}': name for i, name in enumerate(FEATURE_COLS)}
print(f"\nTop {min(15, len(imp_sorted))} features by gain:")
for rank, (fkey, gain) in enumerate(imp_sorted[:15], 1):
    fname = feature_map.get(fkey, fkey)
    print(f"  #{rank:2d}: {fname:35s} gain={gain:.4f}")

# Check btc_realized_vol_prior_30m rank
vol_key = f'f{FEATURE_COLS.index("btc_realized_vol_prior_30m")}'
vol_rank = None
for i, (fkey, _) in enumerate(imp_sorted):
    if fkey == vol_key:
        vol_rank = i + 1
        break
if vol_rank and vol_rank <= 10:
    print(f"\nbtc_realized_vol_prior_30m: Rank #{vol_rank} (TOP 10) -- Model captures volatility regime.")
elif vol_rank:
    print(f"\nbtc_realized_vol_prior_30m: Rank #{vol_rank} -- Not top 10. Volatility may not be primary predictor.")
else:
    print(f"\nbtc_realized_vol_prior_30m: Not used by model (zero splits).")

# ============================================================
# 4.5 SAVE ARTIFACTS
# ============================================================
print("\n" + "="*60)
print("4.5 SAVE ARTIFACTS")
print("="*60)

if gate_result == "STOP":
    print("Skipping model save (GATE = STOP)")
else:
    MODEL_DIR = Path("models/slingger_hunter_v5")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train final model on all data
    final_model_full = XGBClassifier(**{k:v for k,v in params.items() if k != 'early_stopping_rounds'})
    final_model_full.set_params(n_estimators=final_model.best_iteration + 1 if hasattr(final_model, 'best_iteration') and final_model.best_iteration else 300)
    final_model_full.fit(X, y, verbose=False)
    
    # 1. Save model
    final_model_full.save_model(str(MODEL_DIR / "model.json"))
    
    # 2. Save calibrator (temperature scaling via LogisticRegression on OOF probs)
    temp_scaler = LogisticRegression(C=1.0, solver='lbfgs')
    temp_scaler.fit(oof_probs.reshape(-1, 1), y)
    with open(MODEL_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(temp_scaler, f)
    
    # 3. Save imputer
    with open(MODEL_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    
    # 4. Save metadata
    metadata = {
        "model_name":         "slingger_hunter_v5",
        "version":            "5.0.1",
        "strategy":           "intramarket_swing",
        "created_at":         datetime.utcnow().isoformat(),
        "training_rows":      int(len(X)),
        "n_positive_labels":  int(y.sum()),
        "n_markets":          int(len(set(groups))),
        "oof_roc_auc":        float(overall_auc),
        "fold_aucs":          [float(a) for a in fold_aucs],
        "fold_std":           float(np.std(fold_aucs)),
        "features":           FEATURE_COLS,
        "optimal_entry_odds": OPTIMAL_ENTRY,
        "optimal_exit_odds":  OPTIMAL_EXIT,
        "min_ttr_seconds":    MIN_TTR,
        "kelly_fraction":     0.15,
        "max_position_pct":   0.05,
        "polymarket_fee":     POLYMARKET_FEE,
        "training_dataset":   "dataset/clob_log/CLOB_MASTER.csv",
        "predecessor":        "shadow_predator_v4 (DualXGBoostGate)",
        "gate_result":        gate_result,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[SUCCESS] Saved to {MODEL_DIR}")
    for p in MODEL_DIR.iterdir():
        print(f"  {p.name}: {p.stat().st_size/1024:.1f} KB")

# Save OOF predictions for simulation
full_df['oof_prob'] = oof_probs
full_df.to_csv('scratch/slingger_v5_oof.csv', index=False)
print(f"\nOOF predictions saved to scratch/slingger_v5_oof.csv")
print(f"Gate result: {gate_result}")

"""Strict Simulation Script."""
import sys; sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, json, pickle
from pathlib import Path
from model_training.config import V5_FEATURES_FULL
from model_training.dataset import get_market_groups
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

df = pd.read_csv('dataset/DATASET_MASTER_08-05-2026.csv', low_memory=False)
with open('models/v5_final/metadata.json','r') as f: meta = json.load(f)
best_params = meta['model_params']
from model_training.features import build_features
from model_training.dataset import impute_features
df, imp = impute_features(df, strategy='median')
df_feat = build_features(df)
X_all = df_feat[V5_FEATURES_FULL].values.astype(np.float32)
y_all = df_feat['label'].values.astype(np.int32)
groups_all = get_market_groups(df_feat)

gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(y_all))
oof_labels = np.zeros(len(y_all))
for tr_idx, va_idx in gkf.split(X_all, y_all, groups_all):
    m = XGBClassifier(**best_params)
    m.fit(X_all[tr_idx], y_all[tr_idx], eval_set=[(X_all[va_idx], y_all[va_idx])], verbose=False)
    oof_preds[va_idx] = m.predict_proba(X_all[va_idx])[:, 1]
    oof_labels[va_idx] = y_all[va_idx]

from model_training.trainer import TemperatureScaling
temp_scaler = TemperatureScaling()
temp_scaler.fit(oof_preds, oof_labels)
oof_cal = temp_scaler.predict_proba(oof_preds)
oof_cal = np.clip(oof_cal, 0.001, 0.999)

configs = [
    ('V4_CLOB_ON', True, 0.25),
    ('V5_CLOB_OFF', False, 0.25)
]

print("=" * 80)
print(" STRICT SIMULATION (RUIN CONDITION ENFORCED)")
print("=" * 80)
print(f"{'Config':<15} {'Trades':>6} {'WR%':>6} {'PnL($)':>10} {'MaxDD($)':>10} {'Sharpe':>7} {'Calmar':>7} {'Bankrupt?'}")
print("-" * 80)

contrarian_trades = []

for cname, clob_gate, kelly_f in configs:
    cap = 1000.0
    peak = cap
    trades = []
    max_dd = 0.0
    bankrupt = False
    
    for i in range(len(df_feat)):
        if cap <= 0:
            bankrupt = True
            break
            
        row = df_feat.iloc[i]
        p_win = float(oof_cal[i])
        entry_odds = float(row.get('entry_odds', 0.5))
        sig = str(row.get('signal_direction', '')).strip().upper()

        if sig not in ('BUY_UP', 'BUY_DOWN'): continue
        if clob_gate and row.get('clob_alignment', -1) != 1: continue
        ttr = float(row.get('ttr_seconds', 150))
        if not (60 <= ttr <= 250): continue
        if entry_odds <= 0 or entry_odds >= 1.0: continue

        b = (1.0 / entry_odds) - 1.0
        b_adj = b * 0.98
        ev = p_win * b_adj - (1.0 - p_win)
        if ev <= 0.05: continue

        fk = (b_adj * p_win - (1 - p_win)) / b_adj
        stake_pct = min(max(kelly_f * fk, 0.0), 0.05)
        if stake_pct <= 0: continue

        stake = cap * stake_pct
        label = int(row.get('label', 0))
        is_contr = int(row.get('clob_alignment', 0)) == 0

        if label == 1:
            pnl = (stake * b * 0.98) - 0.005
        else:
            pnl = -stake - 0.005

        cap += pnl
        peak = max(peak, cap)
        max_dd = max(max_dd, peak - cap)
        
        t_data = {'pnl': pnl, 'label': label, 'is_contr': is_contr, 'p_win': p_win, 'ev': ev}
        trades.append(t_data)
        if not clob_gate and is_contr:
            contrarian_trades.append(t_data)

    nt = len(trades)
    wr = sum(1 for t in trades if t['label']==1)/nt if nt else 0
    tpnl = cap - 1000.0
    
    pnl_arr = np.array([t['pnl'] for t in trades]) if nt else np.array([0])
    sh = float(np.mean(pnl_arr) / max(np.std(pnl_arr), 1e-9)) if nt else 0
    
    ret = tpnl / 1000.0
    dd_pct = max_dd / 1000.0 if max_dd > 0 else 1.0
    calmar = ret / dd_pct if not bankrupt else -1.0
    
    status = 'YES' if bankrupt else 'NO'
    print(f"{cname:<15} {nt:>6} {wr*100:>6.1f} {tpnl:>10.2f} {max_dd:>10.2f} {sh:>7.3f} {calmar:>7.3f} {status:>9}")

print("=" * 80)
print(" CONTRARIAN TRADES VALIDATION (Strict OOF)")
print("=" * 80)
if contrarian_trades:
    c_pnl = sum(t['pnl'] for t in contrarian_trades)
    c_wr = sum(1 for t in contrarian_trades if t['label']==1)/len(contrarian_trades)
    c_ev = np.mean([t['ev'] for t in contrarian_trades])
    c_pnl_arr = np.array([t['pnl'] for t in contrarian_trades])
    c_sh = float(np.mean(c_pnl_arr) / max(np.std(c_pnl_arr), 1e-9))
    print(f"Trades: {len(contrarian_trades)}")
    print(f"Win Rate: {c_wr*100:.1f}%")
    print(f"OOF PnL: ${c_pnl:.2f}")
    print(f"Sharpe: {c_sh:.3f}")
    print(f"Avg EV: {c_ev:.4f}")
else:
    print("No contrarian trades executed.")

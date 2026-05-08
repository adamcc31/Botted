"""
Quantitative Simulation Engine — V5 Final
==========================================
Simulasi komparatif untuk mengukur dampak removal CLOB alignment gate
dan penambahan fee accounting + dynamic Kelly.

Konfigurasi yang diuji:
  1. ORIGINAL  : CLOB gate ON,  Quarter-Kelly, no fees
  2. NO_CLOB   : CLOB gate OFF, Quarter-Kelly, no fees
  3. FEE_ADJ   : CLOB gate OFF, Quarter-Kelly, WITH 2% winner fee + spread
  4. HALF_KELLY : CLOB gate OFF, Half-Kelly (ECE < 0.08), WITH fees
  5. OPTIMAL   : CLOB gate OFF, adaptive Kelly, WITH fees, tighter EV
"""
import sys, pickle, json, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict

from model_training.config import (
    V5_FEATURES_FULL, TARGET_COL, SPLIT_CFG, XGBConfig,
)
from model_training.dataset import (
    chronological_split, get_market_groups,
    impute_features, apply_imputer,
)
from model_training.features import build_features
from model_training.trainer import TemperatureScaling, compute_ece

# ============================================================================
# Load model + data
# ============================================================================
MODEL_DIR = Path('models/v5_final')
MASTER_CSV = Path('dataset/DATASET_MASTER_08-05-2026.csv')
FEATURES = V5_FEATURES_FULL

print("=" * 70)
print("  QUANTITATIVE SIMULATION ENGINE -- V5 FINAL")
print("  Dataset: DATASET_MASTER_08-05-2026.csv")
print("=" * 70)

# Load saved model & calibrator
with open(MODEL_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(MODEL_DIR / "calibrator.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open(MODEL_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Model loaded: {metadata['best_experiment']}")
print(f"Temperature: {calibrator.temperature:.4f}")
print(f"Features: {metadata['n_features']}")

# Load and prepare data
df = pd.read_csv(MASTER_CSV, low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df_train, df_calib, df_test = chronological_split(
    df, calib_ratio=SPLIT_CFG.calib_holdout_ratio, test_ratio=0.15,
)
df_train, imputer_vals = impute_features(df_train, strategy="median")
df_test = apply_imputer(df_test, imputer_vals)
df_test_feat = build_features(df_test)

# Predict
X_test = df_test_feat[FEATURES].values.astype(np.float32)
raw_proba = model.predict_proba(X_test)[:, 1]
y_prob = calibrator.predict_proba(raw_proba)
y_prob = np.clip(y_prob, 0.001, 0.999)
y_true = df_test_feat[TARGET_COL].values.astype(int)

print(f"\nTest set: {len(df_test_feat)} rows")
print(f"Signals: {df_test_feat['signal_direction'].value_counts().to_dict()}")
print(f"Label dist: WIN={y_true.mean():.3f}")


# ============================================================================
# Simulation Engine with configurable gates and fees
# ============================================================================
def simulate(
    df,
    y_prob,
    config_name="default",
    initial_capital=1000.0,
    ev_threshold=0.05,
    kelly_fraction=0.25,
    max_stake_pct=0.05,
    clob_gate=True,
    ttr_min=60,
    ttr_max=250,
    fee_pct=0.0,           # Fee on winnings (e.g. 0.02 = 2%)
    spread_cost=0.0,       # Flat spread cost per trade in USDC
    min_p_win=0.0,         # Minimum p_win threshold
    max_p_win=1.0,         # Maximum p_win threshold (clamp overconfident)
):
    """Vectorized-friendly simulation with configurable execution gates."""
    n = len(df)
    capital = initial_capital
    peak = initial_capital
    trades = []
    rejects = {}
    max_dd = 0.0
    consec_loss = 0
    max_consec = 0

    for i in range(n):
        row = df.iloc[i]
        p_win = float(y_prob[i])
        entry_odds = float(row.get("entry_odds", 0.5))
        signal_dir = str(row.get("signal_direction", "")).strip().upper()

        # Gate 0: Must be active signal
        if signal_dir not in ("BUY_UP", "BUY_DOWN"):
            rejects["ABSTAIN_SKIP"] = rejects.get("ABSTAIN_SKIP", 0) + 1
            continue

        # Gate 1: CLOB alignment (optional)
        if clob_gate:
            if row.get("clob_alignment", -1) != 1:
                rejects["CLOB_COUNTER"] = rejects.get("CLOB_COUNTER", 0) + 1
                continue

        # Gate 2: TTR window
        ttr = float(row.get("ttr_seconds", 150))
        if not (ttr_min <= ttr <= ttr_max):
            rejects["TTR_RANGE"] = rejects.get("TTR_RANGE", 0) + 1
            continue

        # Gate 3: P_win sanity
        p_win = np.clip(p_win, min_p_win, max_p_win)

        # Gate 4: EV threshold (fee-adjusted)
        if entry_odds <= 0 or entry_odds >= 1.0:
            rejects["INVALID_ODDS"] = rejects.get("INVALID_ODDS", 0) + 1
            continue

        b = (1.0 / entry_odds) - 1.0  # net payout ratio
        b_adj = b * (1.0 - fee_pct)   # fee-adjusted payout
        ev = p_win * b_adj - (1.0 - p_win)

        if ev <= ev_threshold:
            rejects["EV_LOW"] = rejects.get("EV_LOW", 0) + 1
            continue

        # Gate 5: Kelly sizing
        full_kelly = (b_adj * p_win - (1 - p_win)) / b_adj
        stake_pct = kelly_fraction * full_kelly
        stake_pct = min(max(stake_pct, 0.0), max_stake_pct)

        if stake_pct <= 0:
            rejects["KELLY_ZERO"] = rejects.get("KELLY_ZERO", 0) + 1
            continue

        stake = capital * stake_pct

        # Execute
        label = int(row.get("label", 0))
        if label == 1:
            gross_pnl = stake * b
            fee = gross_pnl * fee_pct
            pnl = gross_pnl - fee - spread_cost
            consec_loss = 0
        else:
            pnl = -stake - spread_cost
            consec_loss += 1
            max_consec = max(max_consec, consec_loss)

        capital += pnl
        peak = max(peak, capital)
        dd = peak - capital
        max_dd = max(max_dd, dd)

        trades.append({
            "p_win": p_win, "entry_odds": entry_odds, "ev": ev,
            "stake": stake, "pnl": pnl, "label": label,
            "capital_after": capital, "signal": signal_dir,
            "clob_alignment": int(row.get("clob_alignment", 0)),
        })

    # Aggregates
    n_trades = len(trades)
    if n_trades > 0:
        wins = sum(1 for t in trades if t["label"] == 1)
        win_rate = wins / n_trades
        total_pnl = capital - initial_capital
        roi = (total_pnl / initial_capital) * 100
        pnl_arr = np.array([t["pnl"] for t in trades])
        sharpe = float(np.mean(pnl_arr) / max(np.std(pnl_arr), 1e-9))
        avg_ev = np.mean([t["ev"] for t in trades])
        avg_stake = np.mean([t["stake"] for t in trades])

        # Contrarian analysis
        contrarian_trades = [t for t in trades if t["clob_alignment"] == 0]
        n_contrarian = len(contrarian_trades)
        contrarian_wr = sum(1 for t in contrarian_trades if t["label"] == 1) / max(n_contrarian, 1)
        contrarian_pnl = sum(t["pnl"] for t in contrarian_trades)
    else:
        win_rate = total_pnl = roi = sharpe = avg_ev = avg_stake = 0
        n_contrarian = contrarian_wr = contrarian_pnl = 0

    return {
        "config": config_name,
        "n_signals": n,
        "n_trades": n_trades,
        "n_rejected": sum(rejects.values()),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2) if n_trades else 0,
        "roi_pct": round(roi, 2) if n_trades else 0,
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "max_consec_loss": max_consec,
        "avg_ev": round(avg_ev, 4) if n_trades else 0,
        "avg_stake": round(avg_stake, 2) if n_trades else 0,
        "final_capital": round(capital, 2),
        "rejects": rejects,
        "n_contrarian": n_contrarian,
        "contrarian_wr": round(contrarian_wr, 3),
        "contrarian_pnl": round(contrarian_pnl, 2),
        "trades": trades,
    }


# ============================================================================
# Run all configurations
# ============================================================================
configs = [
    {
        "config_name": "1_ORIGINAL (CLOB gate ON)",
        "clob_gate": True,
        "kelly_fraction": 0.25,
        "fee_pct": 0.0,
        "spread_cost": 0.0,
        "ev_threshold": 0.05,
    },
    {
        "config_name": "2_NO_CLOB (gate OFF)",
        "clob_gate": False,
        "kelly_fraction": 0.25,
        "fee_pct": 0.0,
        "spread_cost": 0.0,
        "ev_threshold": 0.05,
    },
    {
        "config_name": "3_FEE_ADJUSTED (2% fee)",
        "clob_gate": False,
        "kelly_fraction": 0.25,
        "fee_pct": 0.02,
        "spread_cost": 0.005,
        "ev_threshold": 0.05,
    },
    {
        "config_name": "4_HALF_KELLY (ECE<0.08)",
        "clob_gate": False,
        "kelly_fraction": 0.50,
        "fee_pct": 0.02,
        "spread_cost": 0.005,
        "ev_threshold": 0.05,
    },
    {
        "config_name": "5_OPTIMAL (tuned EV+Kelly)",
        "clob_gate": False,
        "kelly_fraction": 0.35,
        "fee_pct": 0.02,
        "spread_cost": 0.005,
        "ev_threshold": 0.08,
        "min_p_win": 0.30,
        "max_p_win": 0.75,
    },
]

all_results = []
for cfg in configs:
    r = simulate(
        df_test_feat, y_prob,
        config_name=cfg["config_name"],
        clob_gate=cfg.get("clob_gate", True),
        kelly_fraction=cfg.get("kelly_fraction", 0.25),
        fee_pct=cfg.get("fee_pct", 0.0),
        spread_cost=cfg.get("spread_cost", 0.0),
        ev_threshold=cfg.get("ev_threshold", 0.05),
        min_p_win=cfg.get("min_p_win", 0.0),
        max_p_win=cfg.get("max_p_win", 1.0),
    )
    all_results.append(r)

# ============================================================================
# Print comparison table
# ============================================================================
print("\n")
print("=" * 120)
print(" QUANTITATIVE SIMULATION COMPARISON -- DATASET_MASTER_08-05-2026")
print("=" * 120)
header = f"{'Config':<35} {'Trades':>6} {'WinRate':>8} {'PnL':>10} {'ROI':>8} {'MaxDD':>8} {'Sharpe':>7} {'AvgEV':>7} {'Contr':>6} {'CntrWR':>7} {'CntrPnL':>9}"
print(header)
print("-" * 120)

for r in all_results:
    line = (
        f"{r['config']:<35} "
        f"{r['n_trades']:>6} "
        f"{r['win_rate']*100:>7.1f}% "
        f"${r['total_pnl']:>+9.2f} "
        f"{r['roi_pct']:>+7.2f}% "
        f"${r['max_drawdown']:>7.2f} "
        f"{r['sharpe']:>7.3f} "
        f"{r['avg_ev']:>7.4f} "
        f"{r['n_contrarian']:>6} "
        f"{r['contrarian_wr']*100:>6.1f}% "
        f"${r['contrarian_pnl']:>+8.2f}"
    )
    print(line)

print("=" * 120)

# ============================================================================
# Detailed rejection breakdown
# ============================================================================
print("\n--- REJECTION BREAKDOWN ---")
for r in all_results:
    print(f"\n  {r['config']}:")
    total_rej = r['n_rejected']
    for gate, count in sorted(r['rejects'].items(), key=lambda x: -x[1]):
        pct = count / max(total_rej, 1) * 100
        print(f"    {gate:<30}: {count:>5} ({pct:.1f}%)")

# ============================================================================
# Quant Analysis: Contrarian trade deep dive
# ============================================================================
print("\n")
print("=" * 70)
print("  CONTRARIAN TRADE ANALYSIS (clob_alignment=0)")
print("=" * 70)

# Use config 2 (NO_CLOB) for analysis
r2 = all_results[1]
contrarian = [t for t in r2["trades"] if t["clob_alignment"] == 0]
aligned = [t for t in r2["trades"] if t["clob_alignment"] == 1]

if contrarian:
    c_pnl = sum(t["pnl"] for t in contrarian)
    c_wr = sum(1 for t in contrarian if t["label"] == 1) / len(contrarian)
    c_avg_ev = np.mean([t["ev"] for t in contrarian])
    c_avg_p = np.mean([t["p_win"] for t in contrarian])
    print(f"  Contrarian trades: {len(contrarian)}")
    print(f"  Win rate: {c_wr:.1%}")
    print(f"  PnL: ${c_pnl:+.2f}")
    print(f"  Avg EV: {c_avg_ev:.4f}")
    print(f"  Avg P(win): {c_avg_p:.4f}")
else:
    print("  No contrarian trades in test set")

if aligned:
    a_pnl = sum(t["pnl"] for t in aligned)
    a_wr = sum(1 for t in aligned if t["label"] == 1) / len(aligned)
    a_avg_ev = np.mean([t["ev"] for t in aligned])
    a_avg_p = np.mean([t["p_win"] for t in aligned])
    print(f"\n  Aligned trades: {len(aligned)}")
    print(f"  Win rate: {a_wr:.1%}")
    print(f"  PnL: ${a_pnl:+.2f}")
    print(f"  Avg EV: {a_avg_ev:.4f}")
    print(f"  Avg P(win): {a_avg_p:.4f}")

# ============================================================================
# P_win bucket analysis
# ============================================================================
print("\n")
print("=" * 70)
print("  P(WIN) BUCKET ANALYSIS -- All trades (NO_CLOB config)")
print("=" * 70)

all_trades = r2["trades"]
if all_trades:
    buckets = [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.0)]
    print(f"  {'P(win) range':<15} {'N':>5} {'WinRate':>8} {'PnL':>10} {'AvgEV':>8}")
    print("  " + "-" * 50)
    for lo, hi in buckets:
        bucket = [t for t in all_trades if lo <= t["p_win"] < hi]
        if bucket:
            bwr = sum(1 for t in bucket if t["label"] == 1) / len(bucket)
            bpnl = sum(t["pnl"] for t in bucket)
            bev = np.mean([t["ev"] for t in bucket])
            print(f"  [{lo:.2f}, {hi:.2f})   {len(bucket):>5} {bwr:>7.1%} ${bpnl:>+9.2f} {bev:>8.4f}")

# ============================================================================
# Recommendation
# ============================================================================
best_config = max(all_results, key=lambda x: x["total_pnl"])
print("\n")
print("=" * 70)
print("  RECOMMENDATION")
print("=" * 70)
print(f"  Best config: {best_config['config']}")
print(f"  PnL: ${best_config['total_pnl']:+.2f} | ROI: {best_config['roi_pct']:+.2f}%")
print(f"  Win Rate: {best_config['win_rate']:.1%} | Sharpe: {best_config['sharpe']:.3f}")
print(f"  Trades: {best_config['n_trades']} | MaxDD: ${best_config['max_drawdown']:.2f}")

# Save best config details
output = {
    "recommended_config": best_config["config"],
    "clob_alignment_gate": False,
    "temperature": calibrator.temperature,
    "results": [
        {k: v for k, v in r.items() if k != "trades"}
        for r in all_results
    ],
}
with open(MODEL_DIR / "simulation_comparison.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Results saved to: {MODEL_DIR / 'simulation_comparison.json'}")
print("=" * 70)

"""
investigate_dip.py — Investigate the "Death Valley 0.30" anomaly.

What makes XGBoost predict 25-35% win probability on contracts
that NEVER win? This script profiles those samples.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model_training.config import SELECTED_FEATURES, TARGET_COL
from model_training.features import build_features
from model_training.dataset import load_from_csv

# ── Load data & model ──
DATA_PATH = BASE_DIR / "dataset" / "raw" / "alpha_v1_master.csv"
MODEL_PATH = BASE_DIR / "models" / "alpha_v1" / "model.pkl"

print("=" * 70)
print("INVESTIGATION: Death Valley 0.25 - 0.35 Anomaly")
print("=" * 70)

df = load_from_csv(DATA_PATH)
df = df[df[TARGET_COL].notna()].copy()
df = build_features(df)

bundle = joblib.load(MODEL_PATH)
base_model = bundle["base_model"]
isotonic = bundle["platt"]  # actually IsotonicRegression

# ── Predict ──
X = df[SELECTED_FEATURES].values.astype(np.float32)
# Handle NaN
for i in range(X.shape[1]):
    med = np.nanmedian(X[:, i])
    X[np.isnan(X[:, i]), i] = med

raw_proba = base_model.predict_proba(X)[:, 1]
p_cal = isotonic.predict(raw_proba)
p_cal = np.clip(p_cal, 0.001, 0.999)

df["p_win"] = p_cal
df["y_true"] = df[TARGET_COL].astype(int)

# ── Split into zones ──
mask_valley = (df["p_win"] >= 0.25) & (df["p_win"] <= 0.35)
mask_below = df["p_win"] < 0.25
mask_above = df["p_win"] > 0.35

df_valley = df[mask_valley]
df_below = df[mask_below]
df_above = df[mask_above]

print(f"\nTotal samples: {len(df)}")
print(f"  Valley (0.25-0.35): {len(df_valley)} samples")
print(f"  Below  (<0.25):     {len(df_below)} samples")
print(f"  Above  (>0.35):     {len(df_above)} samples")

KEY_FEATS = ["entry_odds", "contest_urgency", "depth_ratio", "tfm_value", "obi_vol_interaction"]

# ── Win rate per zone ──
print(f"\n{'─' * 70}")
print("WIN RATE PER ZONE:")
print(f"{'─' * 70}")
for name, subset in [("Valley 0.25-0.35", df_valley),
                       ("Below <0.25", df_below),
                       ("Above >0.35", df_above)]:
    if len(subset) > 0:
        wr = subset["y_true"].mean()
        print(f"  {name:<20}: {wr:.2%} ({int(subset['y_true'].sum())}/{len(subset)} wins)")
    else:
        print(f"  {name:<20}: N/A (0 samples)")

# ── Descriptive stats for Valley ──
print(f"\n{'─' * 70}")
print("VALLEY (0.25-0.35) — Descriptive Statistics:")
print(f"{'─' * 70}")
if len(df_valley) > 0:
    print(df_valley[KEY_FEATS].describe().to_string())
else:
    print("  No samples in valley range!")

# ── Comparison: Valley vs Rest ──
print(f"\n{'─' * 70}")
print("COMPARISON: Valley Mean vs Below Mean vs Above Mean")
print(f"{'─' * 70}")

comparison = pd.DataFrame({
    "Valley (0.25-0.35)": df_valley[KEY_FEATS].mean() if len(df_valley) > 0 else 0,
    "Below (<0.25)": df_below[KEY_FEATS].mean() if len(df_below) > 0 else 0,
    "Above (>0.35)": df_above[KEY_FEATS].mean() if len(df_above) > 0 else 0,
})
print(comparison.to_string())

# ── P_win distribution ──
print(f"\n{'─' * 70}")
print("P_WIN DISTRIBUTION (histogram bins):")
print(f"{'─' * 70}")
bins = [0, 0.1, 0.2, 0.25, 0.30, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
df["p_bin"] = pd.cut(df["p_win"], bins=bins)
dist = df.groupby("p_bin").agg(
    count=("y_true", "count"),
    wins=("y_true", "sum"),
    win_rate=("y_true", "mean"),
).reset_index()
print(dist.to_string(index=False))

# ── Valley samples: signal direction ──
if "signal_direction" in df.columns and len(df_valley) > 0:
    print(f"\n{'─' * 70}")
    print("VALLEY — Signal Direction Distribution:")
    print(f"{'─' * 70}")
    print(df_valley["signal_direction"].value_counts().to_string())

# ── Valley: raw XGB proba vs Isotonic output ──
print(f"\n{'─' * 70}")
print("RAW XGB PROBA vs ISOTONIC P_WIN (Valley samples):")
print(f"{'─' * 70}")
if len(df_valley) > 0:
    valley_idx = df_valley.index
    raw_valley = raw_proba[valley_idx]
    cal_valley = p_cal[valley_idx]
    print(f"  Raw XGB proba range:  [{raw_valley.min():.4f}, {raw_valley.max():.4f}]")
    print(f"  Raw XGB proba mean:   {raw_valley.mean():.4f}")
    print(f"  Isotonic P_WIN range: [{cal_valley.min():.4f}, {cal_valley.max():.4f}]")
    print(f"  Isotonic P_WIN mean:  {cal_valley.mean():.4f}")
    print(f"  Actual win rate:      {df_valley['y_true'].mean():.4f}")

print(f"\n{'=' * 70}")
print("INVESTIGATION COMPLETE")
print(f"{'=' * 70}")

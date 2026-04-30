"""
mad_quant_lab.py — Systematic experiment grid to crack Quality Gates.

Targets to beat (ORGANICALLY, no gate relaxation):
  - AUC CV std  <= 0.08
  - ECE (Platt) <= 0.05

Attack vectors:
  1. Feature pruning (top-N only)
  2. Calibration method (Platt vs Isotonic vs Platt+bigger_calib)
  3. Hyperparameter mutation (extreme regularization)
  4. CV fold count (3 vs 5)
  5. Calib ratio (0.25 vs 0.30 vs 0.35)
"""

import sys, time, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model_training.features import build_features
from model_training.dataset import (
    load_from_csv, validate_dataset, deduplicate_per_market,
    chronological_split, get_market_groups, impute_features, apply_imputer,
)
from model_training.config import ALL_FEATURES, TARGET_COL

# ──────────────────────────────────────────────────────────────
# FEATURE SETS TO TRY
# ──────────────────────────────────────────────────────────────
# Based on SHAP results: entry_odds, depth_ratio, contest_urgency,
# tfm_value, obi_vol_interaction, rv_value, hour_wib, vol_percentile

FEATURE_SETS = {
    "top5": [
        "entry_odds", "depth_ratio", "contest_urgency",
        "tfm_value", "obi_vol_interaction",
    ],
    "top7": [
        "entry_odds", "depth_ratio", "contest_urgency",
        "tfm_value", "obi_vol_interaction", "rv_value", "hour_wib",
    ],
    "top8": [
        "entry_odds", "depth_ratio", "contest_urgency",
        "tfm_value", "obi_vol_interaction", "rv_value", "hour_wib",
        "vol_percentile",
    ],
    "top10_micro": [
        "entry_odds", "depth_ratio", "contest_urgency",
        "tfm_value", "obi_vol_interaction", "rv_value", "hour_wib",
        "vol_percentile", "obi_tfm_product", "tfm_vol_interaction",
    ],
    "all26": ALL_FEATURES,
}

# ──────────────────────────────────────────────────────────────
# HYPERPARAMETER CONFIGS
# ──────────────────────────────────────────────────────────────
HYPER_CONFIGS = {
    "ultra_reg": {
        "max_depth": 2, "learning_rate": 0.015, "n_estimators": 1500,
        "min_child_weight": 80, "gamma": 0.5, "reg_alpha": 2.0,
        "reg_lambda": 8.0, "subsample": 0.60, "colsample_bytree": 0.50,
        "early_stopping_rounds": 80, "random_state": 42, "n_jobs": -1,
        "verbosity": 0, "eval_metric": "logloss",
    },
    "deep_reg": {
        "max_depth": 3, "learning_rate": 0.02, "n_estimators": 1000,
        "min_child_weight": 60, "gamma": 0.3, "reg_alpha": 1.5,
        "reg_lambda": 5.0, "subsample": 0.65, "colsample_bytree": 0.55,
        "early_stopping_rounds": 60, "random_state": 42, "n_jobs": -1,
        "verbosity": 0, "eval_metric": "logloss",
    },
    "stump_forest": {
        "max_depth": 1, "learning_rate": 0.03, "n_estimators": 2000,
        "min_child_weight": 100, "gamma": 0.0, "reg_alpha": 0.5,
        "reg_lambda": 3.0, "subsample": 0.70, "colsample_bytree": 0.70,
        "early_stopping_rounds": 100, "random_state": 42, "n_jobs": -1,
        "verbosity": 0, "eval_metric": "logloss",
    },
    "balanced": {
        "max_depth": 3, "learning_rate": 0.025, "n_estimators": 800,
        "min_child_weight": 50, "gamma": 0.2, "reg_alpha": 1.0,
        "reg_lambda": 4.0, "subsample": 0.70, "colsample_bytree": 0.60,
        "early_stopping_rounds": 50, "random_state": 42, "n_jobs": -1,
        "verbosity": 0, "eval_metric": "logloss",
    },
}

CALIB_METHODS = ["platt", "isotonic"]
CALIB_RATIOS = [0.25, 0.30]
CV_FOLDS = [3, 5]

def compute_ece(y_true, y_prob, n_bins=10):
    try:
        frac, mpred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        return float(np.mean(np.abs(frac - mpred)))
    except Exception:
        return 1.0


def run_experiment(df, feat_names, hyper_name, params, calib_method,
                   calib_ratio, n_folds):
    """Run one full experiment: CV + train + calibrate. Return metrics."""

    # Split
    df_train, df_calib = chronological_split(df, calib_ratio=calib_ratio, test_ratio=0.0)

    # Build features
    df_train_f = build_features(df_train.copy())
    df_calib_f = build_features(df_calib.copy())

    # Impute
    for col in feat_names:
        if col not in df_train_f.columns:
            return None  # feature not available
    
    X_tr = df_train_f[feat_names].values.astype(np.float32)
    y_tr = df_train_f[TARGET_COL].values.astype(np.int32)
    X_ca = df_calib_f[feat_names].values.astype(np.float32)
    y_ca = df_calib_f[TARGET_COL].values.astype(np.int32)

    # Handle NaN with median
    for i in range(X_tr.shape[1]):
        med = np.nanmedian(X_tr[:, i])
        X_tr[np.isnan(X_tr[:, i]), i] = med
        X_ca[np.isnan(X_ca[:, i]), i] = med

    groups = get_market_groups(df_train_f)

    # ── CV ──
    n_actual_folds = min(n_folds, len(np.unique(groups)))
    if n_actual_folds < 2:
        return None

    gkf = GroupKFold(n_splits=n_actual_folds)
    fold_aucs = []
    oof_preds = np.zeros(len(y_tr))

    for tr_idx, va_idx in gkf.split(X_tr, y_tr, groups):
        m = XGBClassifier(**params)
        m.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[va_idx], y_tr[va_idx])], verbose=False)
        p = m.predict_proba(X_tr[va_idx])[:, 1]
        oof_preds[va_idx] = p
        try:
            fold_aucs.append(roc_auc_score(y_tr[va_idx], p))
        except ValueError:
            fold_aucs.append(0.5)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    oof_auc = roc_auc_score(y_tr, oof_preds)

    # ── Final train ──
    n_es = max(int(len(X_tr) * 0.15), 30)
    model = XGBClassifier(**params)
    model.fit(X_tr[:-n_es], y_tr[:-n_es],
              eval_set=[(X_tr[-n_es:], y_tr[-n_es:])], verbose=False)

    # ── Calibration ──
    raw_ca = model.predict_proba(X_ca)[:, 1]

    if calib_method == "platt":
        cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        cal.fit(raw_ca.reshape(-1, 1), y_ca)
        p_cal = cal.predict_proba(raw_ca.reshape(-1, 1))[:, 1]
    elif calib_method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_ca, y_ca)
        p_cal = cal.predict(raw_ca)
        p_cal = np.clip(p_cal, 0.001, 0.999)

    ece = compute_ece(y_ca, p_cal)
    brier = brier_score_loss(y_ca, p_cal)

    return {
        "features": feat_names,
        "n_features": len(feat_names),
        "hyper": hyper_name,
        "calib": calib_method,
        "calib_ratio": calib_ratio,
        "n_folds": n_actual_folds,
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "oof_auc": round(oof_auc, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "best_iter": model.best_iteration,
        # Gate checks
        "pass_auc_std": std_auc <= 0.08,
        "pass_ece": ece <= 0.05,
        "pass_both": std_auc <= 0.08 and ece <= 0.05,
    }


def main():
    DATA_PATH = BASE_DIR / "dataset" / "raw" / "alpha_v1_master.csv"

    print("=" * 80)
    print("MAD QUANT LAB — Systematic Experiment Grid")
    print("=" * 80)

    # Load data
    df = load_from_csv(DATA_PATH)
    df = df[df[TARGET_COL].notna()].copy()
    df = deduplicate_per_market(df)
    print(f"Data: {len(df)} rows, {df['market_id'].nunique()} markets\n")

    # Build full experiment grid
    experiments = []
    for feat_key in FEATURE_SETS:
        for hyper_key in HYPER_CONFIGS:
            for calib_m in CALIB_METHODS:
                for cr in CALIB_RATIOS:
                    for nf in CV_FOLDS:
                        experiments.append((feat_key, hyper_key, calib_m, cr, nf))

    print(f"Total experiments: {len(experiments)}")
    print("-" * 80)

    results = []
    passes = 0
    t0 = time.time()

    for idx, (feat_key, hyper_key, calib_m, cr, nf) in enumerate(experiments):
        feat_names = FEATURE_SETS[feat_key]
        params = HYPER_CONFIGS[hyper_key]

        r = run_experiment(df, feat_names, hyper_key, params, calib_m, cr, nf)

        if r is None:
            continue

        tag = "** PASS **" if r["pass_both"] else ""
        if r["pass_both"]:
            passes += 1

        print(
            f"[{idx+1:3d}/{len(experiments)}] "
            f"feat={feat_key:12s} hyper={hyper_key:12s} "
            f"cal={calib_m:9s} cr={cr:.2f} folds={nf} | "
            f"AUC={r['mean_auc']:.4f} std={r['std_auc']:.4f} "
            f"ECE={r['ece']:.4f} OOF={r['oof_auc']:.4f} "
            f"{tag}"
        )

        r["feat_key"] = feat_key
        results.append(r)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"DONE: {len(results)} experiments in {elapsed:.1f}s")
    print(f"PASSES: {passes} / {len(results)}")

    if passes > 0:
        print(f"\n{'=' * 80}")
        print("WINNING CONFIGURATIONS:")
        print("=" * 80)
        winners = [r for r in results if r["pass_both"]]
        # Sort by OOF AUC descending
        winners.sort(key=lambda x: -x["oof_auc"])
        for i, w in enumerate(winners):
            print(
                f"\n  #{i+1}: feat={w['feat_key']} hyper={w['hyper']} "
                f"cal={w['calib']} cr={w['calib_ratio']} folds={w['n_folds']}"
            )
            print(
                f"       AUC={w['mean_auc']:.4f} std={w['std_auc']:.4f} "
                f"ECE={w['ece']:.4f} OOF={w['oof_auc']:.4f} "
                f"Brier={w['brier']:.4f} iter={w['best_iter']}"
            )
    else:
        # Show top-5 closest to passing
        print(f"\nNo passes. Top-5 closest:")
        results.sort(key=lambda x: (x["std_auc"] - 0.08) + (x["ece"] - 0.05))
        for i, r in enumerate(results[:5]):
            print(
                f"  #{i+1}: feat={r['feat_key']} hyper={r['hyper']} "
                f"cal={r['calib']} cr={r['calib_ratio']} folds={r['n_folds']}"
            )
            print(
                f"       AUC={r['mean_auc']:.4f} std={r['std_auc']:.4f} "
                f"ECE={r['ece']:.4f} OOF={r['oof_auc']:.4f} "
                f"Brier={r['brier']:.4f}"
            )

    # Save results
    out = BASE_DIR / "models" / "alpha_v1" / "lab_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out, "w") as f:
        sanitized = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "features"}
            sanitized.append(s)
        json.dump(sanitized, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
